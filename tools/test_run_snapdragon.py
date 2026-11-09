"""Model-free profile/launcher contract checks; no Janas app is launched."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("run_snapdragon", Path(__file__).with_name("run-snapdragon.py"))
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_hexagon_work_threshold_is_optional_bounded_and_reproduced(self):
        for value in (0, 16777216):
            self.profile["case"]["npu_min_macs"] = value
            env = launcher.launch_environment(self.load(), {"JANAS_NPU_MIN_MACS": "99"})
            self.assertEqual(env.get("JANAS_NPU_MIN_MACS"), str(value) if value else None)
        for value in (-1, True, 1.5, 1 << 63):
            self.profile["case"]["npu_min_macs"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model.jns"
        self.model.write_bytes(b"model fixture\n")
        self.profile = {"schema": 1, "model": self.identity(self.model), "mtp": None,
            "case": {"profile": "cpu", "speculation": "lookup", "threads": 4,
                     "cache_gib": 24, "draft_tokens": 2, "prompt_on": "cpu",
                     "drafts_on": "cpu", "reply_on": "cpu"},
            "settings": {"ctx": 4096, "reserve_gib": 12, "seed": 123, "temperature": 0}}
        self.path = self.root / "selected.json"

    def identity(self, path):
        return {"path": str(path), "size": path.stat().st_size,
                "first_mib_sha256": hashlib.sha256(path.read_bytes()[:1 << 20]).hexdigest()}

    def load(self):
        self.path.write_text(json.dumps(self.profile), encoding="utf-8")
        return launcher.load_profile(self.path, self.root)

    def test_mojo_default_c_profile_clears_inherited_module_without_mutation(self):
        profile = self.load()
        self.assertEqual(profile["case"]["mojo"], "off")
        inherited = {"JANAS_MOJO": "1", "JANAS_MOJO_LIBRARY": "hostile-module",
                     "UNCHANGED": "preserved"}
        before = inherited.copy()
        env = launcher.launch_environment(profile, inherited)
        self.assertNotIn("JANAS_MOJO", env)
        self.assertNotIn("JANAS_MOJO_LIBRARY", env)
        self.assertEqual(env["UNCHANGED"], "preserved")
        self.assertEqual(inherited, before)
        for key in ("JANAS_MOJO", "JANAS_MOJO_LIBRARY"):
            self.assertIn(key, launcher.PERFORMANCE_ENV)
            self.assertIn(key, launcher.PROFILE_ENV)

    def test_mojo_opt_in_uses_resolved_project_module_and_selected_panel_policy(self):
        module = self.root / "module.dll"
        module.write_bytes(b"fixture module; never loaded")
        self.profile["case"]["mojo"] = "on"
        self.profile["runtime"] = {"mojo_library": module.name}
        for policy, tiles in (("auto", None), ("experimental", "1")):
            self.profile["case"]["panels"] = policy
            with self.subTest(policy=policy):
                profile = self.load()
                self.assertEqual(profile["runtime"]["mojo_library"], module.resolve())
                env = launcher.launch_environment(profile,
                    {"JANAS_MOJO": "0", "JANAS_MOJO_LIBRARY": "wrong",
                     "JANAS_CPU_TILES": "0"})
                self.assertEqual(env["JANAS_MOJO"], "1")
                self.assertEqual(env["JANAS_MOJO_LIBRARY"], str(module.resolve()))
                self.assertEqual(env.get("JANAS_CPU_TILES"), tiles)

    def test_mojo_opt_in_requires_existing_file_cpu_and_enabled_panels(self):
        module = self.root / "module.dll"
        module.write_bytes(b"fixture module; never loaded")
        self.profile["case"]["mojo"] = "on"
        self.profile["runtime"] = {"mojo_library": str(module)}
        self.load()
        for value in (None, True, "", "missing-module.dll", str(self.root), "bad\0module"):
            self.profile["runtime"]["mojo_library"] = value
            with self.subTest(module=value), self.assertRaises(ValueError):
                self.load()
        self.profile["runtime"]["mojo_library"] = str(module)
        for key, value in (("profile", "cpu+gpu"), ("panels", "off")):
            original = self.profile["case"].get(key)
            self.profile["case"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
            if original is None:
                self.profile["case"].pop(key)
            else:
                self.profile["case"][key] = original

    def test_mojo_mode_is_explicit_and_off_does_not_require_a_module(self):
        self.profile["runtime"] = {"mojo_library": "missing-module.dll"}
        self.assertEqual(self.load()["case"]["mojo"], "off")
        for value in (True, None, 1, "auto", "ON"):
            self.profile["case"]["mojo"] = value
            with self.subTest(mode=value), self.assertRaisesRegex(ValueError, "case.mojo"):
                self.load()

    def test_hybrid_controls_round_trip_and_reject_invalid_caps(self):
        self.profile["case"].update(hybrid_policy="cost", lookup_drafts=8,
                                    mtp_drafts=4, hybrid_probe_percent=2)
        profile = self.load()
        env = launcher.launch_environment(profile, {"JANAS_HYBRID_POLICY": "hostile",
            "JANAS_LOOKUP_DRAFTS": "99", "JANAS_MTP_DRAFTS": "99",
            "JANAS_HYBRID_PROBE_PERCENT": "99"})
        for name, value in {"JANAS_HYBRID_POLICY": "cost", "JANAS_LOOKUP_DRAFTS": "8",
                            "JANAS_MTP_DRAFTS": "4", "JANAS_HYBRID_PROBE_PERCENT": "2"}.items():
            self.assertEqual(env[name], value)
            self.assertIn(name, launcher.PROFILE_ENV)
        command = launcher.launch_command(profile, "server", Path("server.exe"),
                                           "127.0.0.1", 8081, False)
        for name, value in (("--hybrid-policy", "cost"), ("--lookup-drafts", "8"),
                            ("--mtp-drafts", "4"), ("--hybrid-probe-percent", "2")):
            self.assertEqual(command[command.index(name)+1], value)
        for key, value in (("hybrid_policy", "invalid"), ("lookup_drafts", 0),
                           ("lookup_drafts", 17), ("mtp_drafts", True),
                           ("hybrid_probe_percent", -1), ("hybrid_probe_percent", 101)):
            old = self.profile["case"][key]
            self.profile["case"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load()
            self.profile["case"][key] = old

    def test_s2_profile_controls_are_opt_in_bounded_and_replayed(self):
        original = self.load()
        command = launcher.launch_command(original, "server", Path("server.exe"), "127.0.0.1", 8089, False)
        self.assertNotIn("--s2-exact", command)
        self.profile["mtp"] = self.identity(self.model)
        self.profile["case"].update(speculation="mtp", s2_exact="on", s2_branches=1, s2_max_nodes=8, s2_min_gain_percent=5)
        profile = self.load()
        hostile = {"JANAS_S2_EXACT": "off", "JANAS_S2_BRANCHES": "99"}
        env = launcher.launch_environment(profile, hostile)
        self.assertEqual(hostile["JANAS_S2_BRANCHES"], "99")
        self.assertEqual(env["JANAS_S2_EXACT"], "on")
        self.assertEqual(env["JANAS_S2_BRANCHES"], "1")
        command = launcher.launch_command(profile, "server", Path("server.exe"), "127.0.0.1", 8089, False)
        for name, value in (("--s2-exact", "on"), ("--s2-branches", "1"), ("--s2-max-nodes", "8"), ("--s2-min-gain-percent", "5")):
            self.assertEqual(command[command.index(name)+1], value)
        for key,value in (("s2_exact", True), ("s2_branches", 3), ("s2_max_nodes", 0), ("s2_min_gain_percent", 101)):
            old=self.profile["case"][key];self.profile["case"][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): self.load()
            self.profile["case"][key]=old

    def test_s2_on_requires_additive_cli_capability_but_off_accepts_old_app(self):
        flags="--speculation --draft-tokens --quality --prompt-on --drafts-on --reply-on --spec-planner --spec-explore --draft-vocab --draft-vocab-rows"
        response=subprocess.CompletedProcess([],0,flags,"")
        with patch.object(launcher.subprocess,"run",return_value=response):
            launcher.validate_cli(Path("app.exe"),{"JANAS_S2_EXACT":"off"})
            with self.assertRaisesRegex(ValueError,"--s2-exact"):
                launcher.validate_cli(Path("app.exe"),{"JANAS_S2_EXACT":"on"})
        response=subprocess.CompletedProcess([],0,flags+" --s2-exact --s2-branches --s2-max-nodes --s2-min-gain-percent","")
        with patch.object(launcher.subprocess,"run",return_value=response):
            launcher.validate_cli(Path("app.exe"),{"JANAS_S2_EXACT":"on"})

    def test_profile_is_explicit(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parser().parse_args([])

    def test_old_profile_defaults_to_auto_without_inherited_overrides(self):
        profile = self.load()
        self.assertEqual(profile["case"]["prefetch"], "auto")
        self.assertEqual((profile["case"]["gpu_min_tokens"], profile["case"]["npu_min_tokens"]), (0, 0))
        inherited = {"JANAS_PREFETCH": "10,2", "JANAS_PREFETCH_PREFILL": "2",
            "JANAS_GPU_MIN_N": "128", "JANAS_NPU_MIN_N": "128",
            "JANAS_KV": "16", "JANAS_ATTN": "int16", "JANAS_CPU_TILES": "1"}
        before = inherited.copy()
        env = launcher.launch_environment(profile, inherited)
        self.assertNotIn("JANAS_PREFETCH", env)
        self.assertNotIn("JANAS_PREFETCH_PREFILL", env)
        self.assertNotIn("JANAS_GPU_MIN_N", env)
        self.assertNotIn("JANAS_NPU_MIN_N", env)
        self.assertNotIn("JANAS_KV", env)
        self.assertNotIn("JANAS_ATTN", env)
        self.assertNotIn("JANAS_CPU_TILES", env)
        self.assertEqual(inherited, before)

    def test_old_profile_uses_explicit_legacy_controls_and_preserves_parent_env(self):
        profile = self.load()
        expected = {"spec_planner": "legacy", "spec_explore": 20,
            "draft_vocab": "legacy", "draft_vocab_rows": 0,
            "gdn_chain": 0, "gdn_journal": 0, "gdn_k_only": 0, "panels": "auto"}
        for key, value in expected.items():
            self.assertEqual(profile["case"][key], value)
        controlled = {"JANAS_SPEC_PLANNER": "legacy", "JANAS_SPEC_EXPLORE": "20",
            "JANAS_DRAFT_VOCAB_POLICY": "legacy", "JANAS_DRAFT_VOCAB_ROWS": "0",
            "JANAS_GDN_CHAIN": "0", "JANAS_GDN_JOURNAL": "0", "JANAS_GDN_K_ONLY": "0"}
        inherited = {key: "hostile inherited value" for key in controlled}
        inherited.update(JANAS_CPU_TILES="1", KEEP_ME="parent value",
                         JANAS_API_KEY="unreported parent credential")
        before = inherited.copy()
        env = launcher.launch_environment(profile, inherited)
        for key, value in controlled.items():
            self.assertEqual(env[key], value)
            self.assertIn(key, launcher.PERFORMANCE_ENV)
            self.assertIn(key, launcher.PROFILE_ENV)
        self.assertNotIn("JANAS_CPU_TILES", env)
        self.assertEqual(env["KEEP_ME"], "parent value")
        self.assertEqual(env["JANAS_API_KEY"], "unreported parent credential")
        self.assertEqual(inherited, before)

    def test_planner_controls_are_bounded_and_replayed_in_environment_and_cli(self):
        for planner in ("legacy", "cycle", "fixed"):
            for explore in (0, 20, 100):
                with self.subTest(planner=planner, explore=explore):
                    self.profile["case"].update(spec_planner=planner, spec_explore=explore)
                    profile = self.load()
                    env = launcher.launch_environment(profile,
                        {"JANAS_SPEC_PLANNER": "other", "JANAS_SPEC_EXPLORE": "99"})
                    self.assertEqual(env["JANAS_SPEC_PLANNER"], planner)
                    self.assertEqual(env["JANAS_SPEC_EXPLORE"], str(explore))
                    command = launcher.launch_command(profile, "server", Path("server.exe"),
                                                      "127.0.0.1", 8081, False)
                    self.assertEqual(command[command.index("--spec-planner") + 1], planner)
                    self.assertEqual(command[command.index("--spec-explore") + 1], str(explore))

    def test_draft_vocab_policies_and_rows_replay_exactly(self):
        for policy in ("legacy", "adaptive"):
            for rows in (0, 8192, 16384, 32768):
                with self.subTest(policy=policy, rows=rows):
                    self.profile["case"].update(draft_vocab=policy, draft_vocab_rows=rows)
                    profile = self.load()
                    env = launcher.launch_environment(profile,
                        {"JANAS_DRAFT_VOCAB_POLICY": "other", "JANAS_DRAFT_VOCAB_ROWS": "1"})
                    self.assertEqual(env["JANAS_DRAFT_VOCAB_POLICY"], policy)
                    self.assertEqual(env["JANAS_DRAFT_VOCAB_ROWS"], str(rows))
                    command = launcher.launch_command(profile, "chat", Path("chat.exe"),
                                                      "127.0.0.1", 8080, False)
                    self.assertEqual(command[command.index("--draft-vocab") + 1], policy)
                    self.assertEqual(command[command.index("--draft-vocab-rows") + 1], str(rows))

    def test_new_control_validation_rejects_wrong_types_and_unknown_values(self):
        invalid = {"spec_planner": ("unknown", [], None, True),
            "spec_explore": (-1, 101, True, 20.0, "20", float("nan")),
            "draft_vocab": ("unknown", [], None, True),
            "draft_vocab_rows": (1, 8193, 65536, True, 8192.0, "8192"),
            "panels": ("unknown", [], None, True)}
        for key in ("gdn_chain", "gdn_journal", "gdn_k_only"):
            invalid[key] = (-1, 2, True, 1.0, "1", None)
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    self.profile["case"][key] = value
                    with self.assertRaisesRegex(ValueError, "case." + key):
                        self.load()
                    del self.profile["case"][key]

    def test_recurrent_controls_enforce_dependencies(self):
        keys = ("gdn_chain", "gdn_journal", "gdn_k_only")
        for values in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)):
            with self.subTest(values=values):
                self.profile["case"].update(zip(keys, values))
                env = launcher.launch_environment(self.load(),
                    {"JANAS_GDN_CHAIN": "9", "JANAS_GDN_JOURNAL": "9", "JANAS_GDN_K_ONLY": "9"})
                self.assertEqual(tuple(env["JANAS_" + key.upper()] for key in keys),
                                 tuple(str(value) for value in values))
        for values in ((0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)):
            with self.subTest(values=values):
                self.profile["case"].update(zip(keys, values))
                with self.assertRaisesRegex(ValueError, "requires"):
                    self.load()

    def test_panel_policies_replace_inherited_tiling(self):
        for policy, value in (("auto", None), ("off", "0"), ("experimental", "1")):
            with self.subTest(policy=policy):
                self.profile["case"]["panels"] = policy
                inherited = {"JANAS_CPU_TILES": "hostile", "UNCHANGED": "yes"}
                env = launcher.launch_environment(self.load(), inherited)
                self.assertEqual(env.get("JANAS_CPU_TILES"), value)
                self.assertEqual(inherited["JANAS_CPU_TILES"], "hostile")
                self.assertEqual(env["UNCHANGED"], "yes")

    def test_manual_accelerator_thresholds_override_only_when_positive(self):
        self.profile["case"].update(gpu_min_tokens=1, npu_min_tokens=8)
        env = launcher.launch_environment(self.load(),
            {"JANAS_GPU_MIN_N": "128", "JANAS_NPU_MIN_N": "128"})
        self.assertEqual((env["JANAS_GPU_MIN_N"], env["JANAS_NPU_MIN_N"]), ("1", "8"))
        for key in ("gpu_min_tokens", "npu_min_tokens"):
            for value in (-1, 257, True, "8"):
                self.profile["case"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load()
            self.profile["case"][key] = 0

    def test_prefetch_policies_are_phase_isolated(self):
        expected = {"auto": (None, None), "off": ("0", "0"),
                    "decode-off": ("0", None), "prompt-off": (None, "0")}
        for policy, values in expected.items():
            with self.subTest(policy=policy):
                self.profile["case"]["prefetch"] = policy
                env = launcher.launch_environment(self.load(),
                    {"JANAS_PREFETCH": "10,2", "JANAS_PREFETCH_PREFILL": "2"})
                self.assertEqual(tuple(env.get(k) for k in ("JANAS_PREFETCH", "JANAS_PREFETCH_PREFILL")), values)

    def test_invalid_prefetch_policies_are_rejected(self):
        for policy in ("bogus", [], None):
            self.profile["case"]["prefetch"] = policy
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "case.prefetch"):
                self.load()

    def test_identity_changes_are_rejected(self):
        self.model.write_bytes(b"other fixture\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.load()

    def test_invalid_modes_and_capacities_are_rejected(self):
        for key, value in (("speculation", "bogus"), ("threads", True),
                           ("draft_tokens", 17), ("cache_gib", float("nan")),
                           ("prompt_on", "npu"), ("profile", [])):
            with self.subTest(key=key):
                old = self.profile["case"][key]
                self.profile["case"][key] = value
                with self.assertRaises(ValueError):
                    self.load()
                self.profile["case"][key] = old

    def test_forced_mtp_requires_a_frozen_head(self):
        for mode in ("mtp", "hybrid"):
            self.profile["case"]["speculation"] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "mtp identity"):
                self.load()

    def test_hybrid_requires_and_loads_the_frozen_mtp_head_for_each_app(self):
        mtp = self.root / "model-mtp.jns"
        mtp.write_bytes(b"mtp fixture\n")
        self.profile["mtp"] = self.identity(mtp)
        self.profile["case"]["speculation"] = "hybrid"
        profile = self.load()
        for app in ("chat", "server"):
            with self.subTest(app=app):
                command = launcher.launch_command(profile, app, Path(app + ".exe"),
                                                  "127.0.0.1", 8080, False)
                self.assertEqual(command[command.index("--speculation") + 1], "hybrid")
                self.assertEqual(command[command.index("--mtp") + 1], str(mtp))
                self.assertNotIn("--no-mtp", command)
        mtp.write_bytes(b"bad fixture\n")
        with self.assertRaises(ValueError):
            self.load()

    def test_chat_command_reproduces_explicit_settings(self):
        profile = self.load()
        command = launcher.launch_command(profile, "chat", self.root / "chat.exe", "127.0.0.1", 8080, False)
        for key, value in (("--speculation", "lookup"), ("--draft-tokens", "2"),
                           ("--cache", "24"), ("--quality", "high"), ("--bits", "6"),
                           ("--head", "6"), ("--ctx", "4096"), ("--seed", "123"),
                           ("--spec-planner", "legacy"), ("--spec-explore", "20"),
                           ("--draft-vocab", "legacy"), ("--draft-vocab-rows", "0")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("--no-mtp", command)
        self.assertIn("--no-tune", command)
        self.assertIn("--no-preload", command)

    def test_server_uses_selected_mode_and_keeps_sampling_with_requests(self):
        profile = self.load()
        command = launcher.launch_command(profile, "server", self.root / "server.exe", "127.0.0.1", 8081, True)
        self.assertEqual(command[command.index("--speculation") + 1], "lookup")
        self.assertIn("--test", command)
        self.assertNotIn("--temp", command)
        self.assertNotIn("--no-tune", command)

    def test_npu_runtime_is_pinned_and_original_weights_are_preserved(self):
        hx = self.root / "hexagon"
        (hx / "bin").mkdir(parents=True)
        (hx / "htp").mkdir()
        for name in ("ggml-base.dll", "ggml-hexagon.dll"):
            (hx / "bin" / name).write_bytes(b"fixture")
        self.profile["case"].update(profile="cpu+npu", prompt_on="npu")
        self.profile["runtime"] = {"hexagon_root": str(hx)}
        profile = self.load()
        env = launcher.launch_environment(profile, {"PATH": "old", "JANAS_EXPERTS": "1", "JANAS_HEAD_BITS": "4"})
        self.assertEqual(env["JANAS_HEXAGON_ROOT"], str(hx))
        self.assertEqual(env["ADSP_LIBRARY_PATH"], str(hx / "htp"))
        self.assertEqual(env["PATH"], str(hx / "bin") + os.pathsep + "old")
        self.assertEqual(env["JANAS_MISS_SKIP"], "0")
        self.assertEqual(env["JANAS_HEAD_BITS"], "0")
        self.assertEqual(env["JANAS_EXPERT_BITS"], "6")
        self.assertNotIn("JANAS_EXPERTS", env)

    def test_dry_run_starts_no_app_and_does_not_emit_inherited_credentials(self):
        self.load()
        exe = self.root / "not-an-executable.exe"
        exe.write_text("No process should run this fixture.", encoding="utf-8")
        env = os.environ.copy()
        env["JANAS_API_KEY"] = "launcher-test-secret"
        result = subprocess.run([sys.executable, "-B", str(Path(launcher.__file__)),
            "--profile", str(self.path), "--exe", str(exe), "--dry-run"],
            env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("launcher-test-secret", result.stdout)
        report = json.loads(result.stdout)
        self.assertEqual(report["affinity_threads"], 4)
        self.assertFalse(report["accelerator_execution_verified"])

    def test_old_app_is_rejected_without_starting_a_session(self):
        response = subprocess.CompletedProcess([], 2, "", "usage: chat --no-spec")
        with patch.object(launcher.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(ValueError, "Rebuild"):
                launcher.validate_cli(Path("old.exe"), {})

    def test_apps_with_only_old_profile_flags_are_rejected(self):
        old = "--speculation --draft-tokens --quality --prompt-on --drafts-on --reply-on"
        response = subprocess.CompletedProcess([], 0, old, "")
        with patch.object(launcher.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(ValueError, "spec-planner"):
                launcher.validate_cli(Path("old.exe"), {})
        updated = old + " --spec-planner --spec-explore --draft-vocab --draft-vocab-rows"
        response.stdout = updated
        with patch.object(launcher.subprocess, "run", return_value=response):
            launcher.validate_cli(Path("updated.exe"), {})
        response.stdout = updated.replace(" --draft-vocab ", " ")
        with patch.object(launcher.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(ValueError, "--draft-vocab$"):
                launcher.validate_cli(Path("missing-policy.exe"), {})

    def test_dry_run_records_selected_controls_without_inherited_values(self):
        self.profile["case"].update(spec_planner="cycle", spec_explore=0,
            draft_vocab="adaptive", draft_vocab_rows=8192,
            gdn_chain=1, gdn_journal=1, gdn_k_only=1, panels="off")
        self.load()
        exe = self.root / "not-an-executable.exe"
        exe.write_text("No child app is launched.", encoding="utf-8")
        env = os.environ.copy()
        env.update({key: "inherited-secret-test" for key in (
            "JANAS_SPEC_PLANNER", "JANAS_SPEC_EXPLORE", "JANAS_DRAFT_VOCAB_POLICY",
            "JANAS_DRAFT_VOCAB_ROWS", "JANAS_GDN_CHAIN", "JANAS_GDN_JOURNAL",
            "JANAS_GDN_K_ONLY", "JANAS_CPU_TILES", "JANAS_API_KEY")})
        result = subprocess.run([sys.executable, "-B", str(Path(launcher.__file__)),
            "--profile", str(self.path), "--exe", str(exe), "--dry-run"],
            env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("inherited-secret-test", result.stdout)
        recorded = json.loads(result.stdout)["environment"]
        expected = {"JANAS_SPEC_PLANNER": "cycle", "JANAS_SPEC_EXPLORE": "0",
            "JANAS_DRAFT_VOCAB_POLICY": "adaptive", "JANAS_DRAFT_VOCAB_ROWS": "8192",
            "JANAS_GDN_CHAIN": "1", "JANAS_GDN_JOURNAL": "1", "JANAS_GDN_K_ONLY": "1",
            "JANAS_CPU_TILES": "0"}
        for key, value in expected.items():
            self.assertEqual(recorded[key], value)


if __name__ == "__main__":
    unittest.main()

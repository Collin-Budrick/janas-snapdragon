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

    def test_profile_is_explicit(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parser().parse_args([])

    def test_old_profile_defaults_to_auto_without_inherited_overrides(self):
        profile = self.load()
        self.assertEqual(profile["case"]["prefetch"], "auto")
        self.assertEqual((profile["case"]["gpu_min_tokens"], profile["case"]["npu_min_tokens"]), (0, 0))
        inherited = {"JANAS_PREFETCH": "10,2", "JANAS_PREFETCH_PREFILL": "2",
            "JANAS_GPU_MIN_N": "128", "JANAS_NPU_MIN_N": "128",
            "JANAS_KV": "16", "JANAS_ATTN": "int16"}
        before = inherited.copy()
        env = launcher.launch_environment(profile, inherited)
        self.assertNotIn("JANAS_PREFETCH", env)
        self.assertNotIn("JANAS_PREFETCH_PREFILL", env)
        self.assertNotIn("JANAS_GPU_MIN_N", env)
        self.assertNotIn("JANAS_NPU_MIN_N", env)
        self.assertNotIn("JANAS_KV", env)
        self.assertNotIn("JANAS_ATTN", env)
        self.assertEqual(inherited, before)

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
        self.profile["case"]["speculation"] = "mtp"
        with self.assertRaisesRegex(ValueError, "mtp identity"):
            self.load()

    def test_chat_command_reproduces_explicit_settings(self):
        profile = self.load()
        command = launcher.launch_command(profile, "chat", self.root / "chat.exe", "127.0.0.1", 8080, False)
        for key, value in (("--speculation", "lookup"), ("--draft-tokens", "2"),
                           ("--cache", "24"), ("--quality", "high"), ("--bits", "6"),
                           ("--head", "6"), ("--ctx", "4096"), ("--seed", "123")):
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


if __name__ == "__main__":
    unittest.main()

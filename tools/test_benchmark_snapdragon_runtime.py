"""Protocol/receipt fixtures only. No model payloads or child executables run."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import struct
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

TOOL_PATH = Path(__file__).with_name("benchmark-snapdragon-runtime.py")
SPEC = importlib.util.spec_from_file_location("benchmark_snapdragon_runtime", TOOL_PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class RuntimeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.baseline = self.folder / "baseline-janas-bench.exe"
        self.candidate = self.folder / "candidate.exe"
        self.baseline.write_bytes(b"fixture control executable; never executed")
        self.candidate.write_bytes(b"fixture candidate executable; never executed")
        self.main = self.folder / "fixture-main.jns"
        self.mtp = self.folder / "fixture-mtp.jns"
        self.main.write_bytes(b"tiny identity fixture, not model weights")
        self.mtp.write_bytes(b"tiny sidecar identity fixture")
        self.pins = self.folder / "pins.json"
        self.receipt = self.folder / "receipt.json"
        models = {role: {"filename": path.name, "bytes": path.stat().st_size,
                         "sha256": tool.sha(path)}
                  for role, path in (("main", self.main), ("mtp", self.mtp))}
        self.pin_value = {"schema": 1, "models": models,
                          "published_baseline": {"sha256": tool.sha(self.baseline)}}
        self.receipt_value = {"complete": True, "files": {
            value["filename"]: {"sha256": value["sha256"], "sha256_verified": True}
            for value in models.values()}}
        self.pins.write_text(json.dumps(self.pin_value))
        self.receipt.write_text(json.dumps(self.receipt_value))
        self.argv = ["--baseline-exe", str(self.baseline), "--bench-exe", str(self.candidate),
                     "--model", str(self.main), "--mtp", str(self.mtp),
                     "--model-pins", str(self.pins), "--model-receipt", str(self.receipt)]

    def args(self, mode="paired", extra=()):
        return tool.parser().parse_args([mode, *self.argv, *extra])

    def inputs(self):
        with mock.patch.object(tool, "source_identity", return_value={"fixture": "source receipt"}):
            return tool.prepare_inputs(self.args())

    def records(self, run):
        c = run["control"]
        setup = {"schema": 1, "kind": "setup", "quality": "high", "requested_threads": 8,
                 "seed": 20260926, "context_tokens": 2048, "generation_limit": 128,
                 "draft_tokens_cap": c["draft_tokens"], "speculation": c["speculation"],
                 "speculation_resolved": "mtp" if c["speculation"] == "auto" else c["speculation"],
                 "cache_bytes_requested": 32 << 30, "reserve_bytes": 12 << 30, "gpu": "", "npu": ""}
        if run["binary"] == "B":
            setup.update({k: c[k] for k in ("spec_planner", "spec_explore", "draft_vocab", "draft_vocab_rows",
                                          "hybrid_policy", "lookup_drafts", "mtp_drafts", "hybrid_probe_percent")})
        if c["s2_exact"] == "on":
            setup.update({k: c[k] for k in ("s2_exact", "s2_branches", "s2_max_nodes", "s2_min_gain_percent")})
        setup.update({"fair_retain_controller": run["controller_protocol"] == "retained_across_fresh_conversations",
                          "fair_controller_protocol": run["controller_protocol"],
                          "conversation_cache_reuse": False,
                          "history_index_scope": "canonical_tokens_of_current_sample",
                          "target_sampler_seed_scope": "reset_each_sample"})
        samples = []
        for index in range(1 + run["warm_rounds"]):
            samples.append({"schema": 1, "kind": "sample", "index": index,
                            "phase": "warm" if index else "fresh_process",
                            **{k: run["expected"][k] for k in ("input_tokens", "output_tokens", "output_hash")},
                            "prompt_tokens": run["expected"]["input_tokens"], "cached_tokens": 0,
                            "input_threads": 8, "output_threads": 8,
                            "speculation": c["speculation"], "speculation_resolved": setup["speculation_resolved"],
                            "output_tokens_per_second": 12.8, "output_seconds": 10.0,
                            "total_seconds": 11.0, "first_token_seconds": 1.2,
                            "input_seconds": 1.0, "accelerator_counters": {},
                            "speculation_metrics": {"cycles": 8}})
            samples[-1]["fair_controller_protocol"] = run["controller_protocol"]
            if c["s2_exact"] == "on":
                samples[-1]["s2_metrics"] = dict(schema=1, cycles=8, alternate_builds=0,
                    alternate_nodes=0, cached_uses=0, pruned=8, probes=0)
            if c["hybrid_policy"] == "cost":
                samples[-1]["hybrid_metrics"] = {"schema": 1, "decisions": 8}
        return [setup, *samples]

    def test_s2_defaults_are_off_and_old_baseline_cli_stays_compatible(self):
        plan = tool.build_plan(self.args(extra=["--candidate-case", "s2-exact"]), self.inputs())
        for run in plan["contract"]["plan"]:
            if run["binary"] == "A":
                self.assertNotIn("--s2-exact", run["command"])
                self.assertEqual(run["performance_controls"]["JANAS_S2_EXACT"], "off")
            else:
                self.assertEqual(run["command"][run["command"].index("--s2-exact")+1], "on")
                self.assertEqual(run["control"]["speculation"], "mtp")
            self.assertEqual(tool.validate_records(self.records(run), run), [])

    def test_s2_invalid_controls_and_off_auxiliary_changes_have_no_effect(self):
        for bad in ({"s2_exact": True}, {"s2_branches": 0}, {"s2_branches": 3},
                    {"s2_max_nodes": 33}, {"s2_min_gain_percent": -1},
                    {"s2_min_gain_percent": True}, {"s2_exact": "on", "speculation": "none"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError): tool.valid_control(bad)
        self.assertEqual(tool.effective_control({}), tool.effective_control({"s2_branches": 1, "s2_max_nodes": 32}))
        self.assertNotEqual(tool.effective_control({}), tool.effective_control({"s2_exact": "on"}))

    def test_s2_on_requires_actual_setup_and_diagnostics(self):
        plan = tool.build_plan(self.args(extra=["--candidate-case", "s2-exact"]), self.inputs())
        run = next(r for r in plan["contract"]["plan"] if r["binary"] == "B")
        records = self.records(run)
        records[0]["s2_branches"] = 1
        self.assertTrue(any("s2_branches" in e for e in tool.validate_records(records, run)))
        records = self.records(run); records[1].pop("s2_metrics")
        self.assertTrue(any("S2 diagnostics" in e for e in tool.validate_records(records, run)))

    def test_s2_execute_and_resume_real_plumbing_with_old_off_baseline(self):
        self.exercise_execution_plumbing(candidate_case="s2-exact")

    def test_public_paths_and_frozen_prompt_hashes(self):
        self.assertEqual(tool.ROOT, TOOL_PATH.resolve().parents[1])
        for path, _tokens, digest, _output in tool.PROMPTS.values():
            self.assertTrue(path.is_relative_to(tool.ROOT / "tools/benchmark-prompts"))
            self.assertEqual(tool.sha(path), digest)
        self.assertEqual(tool.load_pins(tool.PINS_FILE)["quality_reference"]["logit_fnv1a64"],
                         "c70a79fcb00b7e3d")

    def test_plan_never_spawns_or_enters_execute(self):
        for mode in ("screen", "paired"):
            output = io.StringIO()
            with (mock.patch.object(tool.subprocess, "Popen", side_effect=AssertionError("child launched")),
                  mock.patch.object(tool.ctypes, "WinDLL", side_effect=AssertionError("plan requested Windows execution state"), create=True),
                  mock.patch.object(tool, "execute", side_effect=AssertionError("execute entered")),
                  mock.patch.object(tool, "native_host", side_effect=AssertionError("host execution queried")),
                  mock.patch.object(tool, "source_identity", return_value={"fixture": True}),
                  contextlib.redirect_stdout(output)):
                self.assertEqual(tool.main([mode, *self.argv]), 0)
            manifest = json.loads(output.getvalue())
            self.assertGreater(manifest["planned_processes"], 0)
            self.assertIn("launches no executable", manifest["execution_note"])

    def test_paired_abba_five_warm_per_binary_and_all_workloads(self):
        manifest = tool.build_plan(self.args(), self.inputs())
        runs = manifest["contract"]["plan"]
        self.assertEqual((manifest["planned_processes"], manifest["planned_samples"]), (12, 42))
        for workload in tool.PROMPTS:
            group = [r for r in runs if r["workload"] == workload]
            self.assertEqual([r["binary"] for r in group], list("ABBA"))
            self.assertEqual([r["warm_rounds"] for r in group], [3, 3, 2, 2])
            for binary in "AB":
                self.assertEqual(sum(r["warm_rounds"] for r in group if r["binary"] == binary), 5)
            self.assertEqual(group[0]["expected"]["output_tokens"], 128)
            self.assertNotIn("--spec-planner", group[0]["command"])
            self.assertIn("--spec-planner", group[1]["command"])

    def test_screen_adds_baseline_and_keeps_requested_case_order(self):
        args = self.args("screen", ["--cases", "hybrid4,legacy"])
        manifest = tool.build_plan(args, self.inputs())
        runs = manifest["contract"]["plan"]
        self.assertEqual((manifest["planned_processes"], manifest["planned_samples"]), (6, 18))
        self.assertEqual([r["case"] for r in runs[:3]], ["baseline", "hybrid4", "legacy"])
        self.assertTrue(all(r["warm_rounds"] == 2 for r in runs))

    def test_controller_retention_is_symmetric_for_current_baseline(self):
        inputs = self.inputs()
        controls = [({}, False), ({"gdn_chain": 1, "gdn_journal": 1}, False),
                    ({"spec_planner": "fixed"}, False),
                    ({"spec_planner": "cycle"}, True),
                    ({"speculation": "hybrid"}, True),
                    ({"draft_vocab": "adaptive"}, True)]
        for overrides, candidate_retain in controls:
            control = tool.valid_control(overrides)
            for binary in ("A", "B"):
                with self.subTest(overrides=overrides, binary=binary):
                    run = tool.run_spec(inputs, "code", "fixture", control, binary, 2, 0)
                    wanted = True
                    self.assertEqual(run["command"].count("--fair-retain-controller"), int(wanted))
                    self.assertEqual(run["controller_protocol"],
                                     "retained_across_fresh_conversations" if wanted else "fresh_chat_per_sample")
                    value = {k: v for k, v in run.items() if k != "run_sha256"}
                    self.assertEqual(run["run_sha256"], tool.digest(value))

    def test_control_validation_rejects_invalid_shapes_and_values(self):
        for value in ([], {"unknown": 1}, {"draft_tokens": True}, {"draft_tokens": 17},
                      {"spec_explore": -1}, {"spec_explore": float("nan")},
                      {"draft_vocab_rows": 123}, {"gdn_chain": False},
                      {"gdn_chain": 0, "gdn_journal": 1}, {"gdn_journal": 0, "gdn_k_only": 1}, {"panels": "magic"},
                      {"hybrid_policy": "unknown"}, {"lookup_drafts": 0},
                      {"mtp_drafts": 17}, {"hybrid_probe_percent": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tool.valid_control(value)
        self.assertEqual(tool.valid_control({"draft_tokens": 16})["draft_tokens"], 16)

    def test_argument_validation(self):
        for extra in (["--workloads", "code,code"], ["--workloads", "unknown"],
                      ["--timeout", "nan"], ["--cooldown", "-1"], ["--execute"], ["--resume"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                tool.check_args(self.args(extra=extra))
        with self.assertRaises(ValueError):
            tool.check_args(tool.parser().parse_args([]))
        with self.assertRaises(ValueError):
            tool.build_plan(self.args("screen", ["--cases", "legacy,legacy"]), self.inputs())

    def test_environment_removes_controls_without_mutating_parent(self):
        inherited = {"PATH": "fixture-path", "JANAS_KV": "inherited", "janas_attn": "inherited",
                     "JANAS_CPU_TILES": "1", "JANAS_DRAFT_VOCAB": "1", "XDG_CACHE_HOME": "old"}
        before = dict(inherited)
        env, record = tool.environment(self.folder / "cache", tool.DEFAULT_CONTROL, inherited)
        self.assertEqual(inherited, before)
        self.assertEqual(env["PATH"], "fixture-path")
        for key in ("JANAS_KV", "janas_attn", "JANAS_CPU_TILES", "JANAS_DRAFT_VOCAB"):
            self.assertNotIn(key, env)
        self.assertIsNone(record["JANAS_CPU_TILES"])
        self.assertEqual(record["cleared"]["JANAS_ATTN"], None)
        self.assertEqual(env["XDG_CACHE_HOME"], (self.folder / "cache").as_posix())

    def test_baseline_manifest_binds_exe_and_override_is_labelled(self):
        receipt = self.folder / "baseline.json"
        receipt.write_text(json.dumps({"source_commit": "fixture-commit", "baseline": {
            "path": self.baseline.name, "sha256": tool.sha(self.baseline)}}))
        args = self.args(extra=["--baseline-manifest", str(receipt)])
        value, proof = tool.baseline_input(args, self.pin_value)
        self.assertTrue(value["matches_published_reference"])
        self.assertEqual(value["source_commit"], "fixture-commit")
        self.assertEqual(proof["sha256"], tool.sha(receipt))
        args.baseline_exe = None
        selected, _ = tool.baseline_input(args, self.pin_value)
        self.assertEqual(selected["path"], str(self.baseline.resolve()))
        args.baseline_exe = self.baseline
        self.baseline.write_bytes(b"different explicitly selected control")
        with self.assertRaises(ValueError):
            tool.baseline_input(args, self.pin_value)
        value, proof = tool.baseline_input(self.args(), self.pin_value)
        self.assertFalse(value["matches_published_reference"])
        self.assertIsNone(proof)
        self.assertIn("override", value["identity_source"])

    def test_model_receipt_required_and_expected_pins_are_not_proof(self):
        inputs = self.inputs()
        self.assertEqual(inputs["models"][0]["bytes"], self.main.stat().st_size)
        self.receipt_value["files"][self.main.name]["sha256_verified"] = False
        self.receipt.write_text(json.dumps(self.receipt_value))
        with self.assertRaises(ValueError):
            self.inputs()
        self.pins.write_text(json.dumps({"schema": True, "models": []}))
        with self.assertRaises(ValueError):
            tool.load_pins(self.pins)

    def test_plan_without_private_receipt_never_full_hashes_payload(self):
        args = self.args()
        args.model_receipt = None
        args.verify_model_sha256 = True  # Even this flag must be inert in plan mode.
        original_sha = tool.sha
        def no_payload_hash(path):
            if Path(path) in (self.main, self.mtp):
                raise AssertionError("plan full-hashed a payload")
            return original_sha(path)
        with (mock.patch.object(tool, "sha", side_effect=no_payload_hash),
              mock.patch.object(tool, "source_identity", return_value={"fixture": True})):
            inputs = tool.prepare_inputs(args)
        self.assertIsNone(inputs["model_receipt"])
        self.assertFalse(inputs["model_verification"]["complete"])
        self.assertEqual(inputs["model_verification"]["method"], "unverified_plan_only")
        self.assertIsNone(inputs["models"][0]["full_sha256_previously_verified"])
        args.execute = True
        args.verify_model_sha256 = False
        args.output = self.folder / "unverified-output"
        with self.assertRaises(ValueError):
            tool.check_args(args)

    def test_full_verify_and_resume_attestation_on_tiny_identity_fixtures(self):
        args = self.args()
        args.model_receipt = None
        args.verify_model_sha256 = args.execute = True
        args.output = self.folder / "attestation"
        with (mock.patch.object(tool, "source_identity", return_value={"fixture": True}),
              contextlib.redirect_stderr(io.StringIO())):
            inputs = tool.prepare_inputs(args)
        proof = inputs["model_verification"]
        self.assertTrue(proof["complete"])
        self.assertEqual(proof["method"], "computed_streaming_sha256")
        self.assertEqual(proof["files"][self.main.name]["sha256"], tool.sha(self.main))
        args.output.mkdir()
        (args.output / "model-verification.json").write_text(json.dumps(proof))
        args.resume = True
        original_sha = tool.sha
        def no_payload_rescan(path):
            if Path(path) in (self.main, self.mtp):
                raise AssertionError("resume full-hashed an unchanged payload")
            return original_sha(path)
        with (mock.patch.object(tool, "sha", side_effect=no_payload_rescan),
              mock.patch.object(tool, "source_identity", return_value={"fixture": True})):
            resumed = tool.prepare_inputs(args)
            self.assertEqual(resumed["model_verification"], proof)
            self.main.write_bytes(b"x" * self.main.stat().st_size)
            with self.assertRaises(ValueError):
                tool.prepare_inputs(args)

    def test_sha_uses_bounded_streaming_reads(self):
        stream = mock.MagicMock()
        stream.__enter__.return_value = stream
        stream.read.side_effect = [b"a", b"b", b""]
        with mock.patch.object(Path, "open", return_value=stream):
            self.assertEqual(tool.sha("not-a-model"), hashlib.sha256(b"ab").hexdigest())
        self.assertEqual(stream.read.call_args_list, [mock.call(1 << 20)] * 3)

    def test_strict_json_rejects_nonfinite_duplicates_and_malformed_records(self):
        path = self.folder / "output.log"
        for value in ('{"a":NaN}', '{"a":Infinity}', '{"a":1e999}',
                      '{"a":1,"a":2}', '{not-json}'):
            path.write_text(value + "\n")
            with self.subTest(value=value), self.assertRaises(ValueError):
                tool.strict_records(path)
        path.write_text('human log\n  {"schema":1,"kind":"setup"}\n')
        self.assertEqual(tool.strict_records(path), [{"schema": 1, "kind": "setup"}])

    def test_actual_results_must_match_frozen_output_and_timing_protocol(self):
        run = tool.build_plan(self.args(), self.inputs())["contract"]["plan"][1]
        records = self.records(run)
        self.assertEqual(tool.validate_records(records, run), [])
        for field, value in (("output_hash", "wrong"), ("input_tokens", 275),
                             ("cached_tokens", 1), ("output_tokens_per_second", True),
                             ("first_token_seconds", 0), ("input_seconds", -1),
                             ("accelerator_counters", {"npu_products": 1})):
            bad = copy.deepcopy(records)
            bad[1][field] = value
            with self.subTest(field=field):
                self.assertTrue(tool.validate_records(bad, run))
        bad = copy.deepcopy(records)
        bad[0]["profile_phases_enabled"] = True
        self.assertTrue(tool.validate_records(bad, run))

    def test_invalid_run_cannot_qualify_a_rate_comparison(self):
        manifest = tool.build_plan(self.args(extra=["--workloads", "code"]), self.inputs())
        records = []
        for run in manifest["contract"]["plan"]:
            records.append({**run, "status": "ok", "records": self.records(run),
                            "process_wall_seconds": 1.0})
        good = tool.summary(manifest, records)
        self.assertTrue(good["comparisons"][0]["complete_and_output_verified"])
        records[0]["status"] = "invalid"
        bad = tool.summary(manifest, records)
        self.assertFalse(bad["comparisons"][0]["complete_and_output_verified"])
        self.assertIsNone(bad["comparisons"][0]["candidate_to_baseline_median_rate_ratio"])

    def test_controller_setup_and_sample_metadata_must_match_actual_protocol(self):
        manifest = tool.build_plan(self.args(extra=["--candidate-case", "cycle-mtp4"]), self.inputs())
        run = manifest["contract"]["plan"][1]
        records = self.records(run)
        self.assertEqual(tool.validate_records(records, run), [])
        for field, value in (("fair_retain_controller", False),
                             ("fair_controller_protocol", "fresh_chat_per_sample"),
                             ("conversation_cache_reuse", True),
                             ("history_index_scope", "previous_conversation"),
                             ("target_sampler_seed_scope", "continued")):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.subTest(field=field):
                self.assertTrue(tool.validate_records(bad, run))
            del bad[0][field]
            self.assertTrue(tool.validate_records(bad, run))
        bad = copy.deepcopy(records)
        bad[1]["fair_controller_protocol"] = "fresh_chat_per_sample"
        self.assertTrue(tool.validate_records(bad, run))
        del bad[1]["fair_controller_protocol"]
        self.assertTrue(tool.validate_records(bad, run))

    def test_current_baseline_requires_same_retained_controller_protocol(self):
        run = tool.build_plan(self.args(), self.inputs())["contract"]["plan"][0]
        records = self.records(run)
        self.assertEqual(tool.validate_records(records, run), [])
        for field, value in (("fair_retain_controller", False),
                             ("fair_controller_protocol", "fresh_chat_per_sample"),
                             ("conversation_cache_reuse", True),
                             ("history_index_scope", "previous_conversation"),
                             ("target_sampler_seed_scope", "continued")):
            bad = copy.deepcopy(records)
            bad[0][field] = value
            with self.subTest(field=field):
                self.assertTrue(tool.validate_records(bad, run))
            del bad[0][field]
            self.assertTrue(tool.validate_records(bad, run))

    def test_baseline_controls_preserve_promoted_gdn_and_custom_overrides(self):
        controls = self.folder / "baseline-controls.json"
        controls.write_text(json.dumps({"gdn_chain": 0, "gdn_journal": 0, "gdn_k_only": 0}))
        manifest = tool.build_plan(self.args(extra=["--baseline-controls", str(controls),
            "--candidate-case", "hybrid-cost8"]), self.inputs())
        baseline, candidate = manifest["contract"]["plan"][:2]
        self.assertEqual([baseline["control"][k] for k in ("gdn_chain", "gdn_journal", "gdn_k_only")], [0,0,0])
        self.assertEqual([candidate["control"][k] for k in ("gdn_chain", "gdn_journal", "gdn_k_only")], [1,1,1])
        self.assertEqual(candidate["control"]["draft_tokens"], 8)
        self.assertEqual(candidate["control"]["mtp_drafts"], 4)
        self.assertIn("--hybrid-policy", candidate["command"])
        self.assertEqual(tool.DEFAULT_CONTROL["gdn_k_only"], 1)
        records = self.records(candidate)
        self.assertEqual(tool.validate_records(records, candidate), [])
        del records[1]["hybrid_metrics"]
        self.assertIn("cost-aware hybrid metrics missing", tool.validate_records(records, candidate))

    def test_identical_calibration_seeds_and_original_preserved(self):
        path = self.folder / "drafts-0123456789abcdef.bin"
        data = struct.pack("<20d", *([10.0] * 10 + [7.0] * 10))
        path.write_bytes(data)
        frozen = tool.calibration_input(path)
        self.assertEqual(frozen["state"], "seeded")
        for label in ("A", "B"):
            cache = self.folder / label
            initial = tool.seed_calibration(cache, frozen)
            target = Path(initial["private_path"])
            self.assertEqual(target.read_bytes(), data)
            target.write_bytes(b"private per-process training")
            tool.verify_calibration_source(frozen)
        self.assertEqual(path.read_bytes(), data)
        path.write_bytes(data[:-1])
        with self.assertRaises((RuntimeError, ValueError)):
            tool.verify_calibration_source(frozen)

    def test_calibration_absence_is_frozen_and_invalid_counts_refused(self):
        path = self.folder / "drafts-0123456789abcdef.bin"
        absent = tool.calibration_input(path)
        self.assertEqual(absent["state"], "absent")
        cache = self.folder / "absent-cache"
        self.assertIsNone(tool.seed_calibration(cache, absent)["private_path"])
        self.assertFalse(cache.exists())
        path.write_bytes(struct.pack("<20d", *([0.0] * 10 + [1.0] * 10)))
        with self.assertRaises(ValueError):
            tool.calibration_input(path)
        with self.assertRaises(ValueError):
            tool.calibration_input(self.folder / "credentials.bin")
        path.write_bytes(struct.pack("<20d", *([1.0] * 20)))
        with self.assertRaises(RuntimeError):
            tool.verify_calibration_source(absent)

    def test_attention_calibration_preserves_legacy_columns_and_original(self):
        path = self.folder / "attn.txt"
        data = b"Snapdragon_X|8|16/2/256|0|2048 4294967295 1024\n"
        path.write_bytes(data)
        frozen = tool.calibration_input(path, "attention")
        self.assertEqual(frozen["kind"], "attention")
        for arm in ("attention-A", "attention-B"):
            seeded = tool.seed_calibration(self.folder / arm, frozen)
            self.assertEqual(Path(seeded["private_path"]).read_bytes(), data)
        self.assertEqual(path.read_bytes(), data)
        args = self.args(extra=["--attention-calibration-file", str(path)])
        with mock.patch.object(tool, "source_identity", return_value={"fixture": True}):
            inputs = tool.prepare_inputs(args)
        self.assertEqual(inputs["attention_calibration"], frozen)
        for data in (b"key 1\n", b"key|8 -1 1024\n", b"key|8 4294967296 1024\n"):
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                tool.calibration_input(path, "attention")

    def test_private_attention_retiming_is_detected_after_child(self):
        path = self.folder / "attn.txt"
        data = b"Snapdragon_X|8|16/2/256|0|2048 4294967295 1024\n"
        path.write_bytes(data)
        frozen = tool.calibration_input(path, "attention")
        cache = self.folder / "attention-child"
        seeded = tool.seed_calibration(cache, frozen)
        self.assertTrue(tool.attention_calibration_result(cache, frozen)["unchanged"])
        private = Path(seeded["private_path"])
        private.write_bytes(data + b"Other_CPU|8|16/2/256|0|2048 1024 1024\n")
        report = tool.attention_calibration_result(cache, frozen)
        self.assertFalse(report["unchanged"])
        self.assertTrue(report["auto_retiming_detected"])
        private.unlink()
        self.assertFalse(tool.attention_calibration_result(cache, frozen)["unchanged"])
        self.assertEqual(path.read_bytes(), data)

    def exercise_execution_plumbing(self, same_binary=False, candidate_case="hybrid-cost8"):
        # A fake process replaces the model executable, but the actual CLI,
        # execution loop, Win32 affinity calls, child_run, logs, parsers,
        # seeding, validation, summaries and final preservation all execute.
        attention = self.folder / "attn.txt"
        attention.write_bytes(b"Snapdragon_X|8|16/2/256|0|2048 1024 1024\n")
        drafts = self.folder / "drafts-0123456789abcdef.bin"
        drafts.write_bytes(struct.pack("<20d", *([10.0] * 10 + [7.0] * 10)))
        output = self.folder / "executed"
        kernel = mock.Mock()
        kernel.GetCurrentProcess.return_value = 77
        execution = {"state": tool.ES_CONTINUOUS}
        def execution_request(flags):
            previous = execution["state"]
            if flags & tool.ES_CONTINUOUS:
                execution["state"] = flags
            return previous
        kernel.SetThreadExecutionState.side_effect = execution_request
        def affinity(handle, process, system):
            value = getattr(handle, "value", handle)
            process._obj.value = 255 if value == 123 else 4095
            system._obj.value = 4095
            return 1
        kernel.GetProcessAffinityMask.side_effect = affinity
        kernel.SetProcessAffinityMask.return_value = 1
        current = {}
        actual_child_run = tool.child_run
        def child(run, folder, timeout):
            self.assertEqual(run["calibration"]["sha256"], tool.sha(drafts))
            self.assertEqual(run["attention_calibration"]["sha256"], tool.sha(attention))
            current["run"] = run
            return actual_child_run(run, folder, timeout)
        def process(command, **kwargs):
            run = current["run"]
            cache = Path(kwargs["env"]["XDG_CACHE_HOME"]) / "janas"
            self.assertEqual((cache / drafts.name).read_bytes(), drafts.read_bytes())
            self.assertEqual((cache / "attn.txt").read_bytes(), attention.read_bytes())
            for record in self.records(run):
                kwargs["stdout"].write(json.dumps(record) + "\n")
            result = mock.Mock()
            result._handle = 123
            result.returncode = 0
            result.wait.return_value = 0
            result.poll.return_value = 0
            return result
        args = ["paired", *self.argv, "--execute", "--workloads", "code", "--cooldown", "0",
                "--candidate-case", candidate_case, "--calibration-file", str(drafts),
                "--attention-calibration-file", str(attention), "--output", str(output)]
        if same_binary:
            self.candidate.write_bytes(self.baseline.read_bytes())
            args.append("--allow-same-binary-controls")
        with (mock.patch.object(tool, "native_host", return_value=True),
              mock.patch.object(tool, "source_identity", return_value={"fixture": True}),
              mock.patch.object(tool.ctypes, "WinDLL", return_value=kernel, create=True),
              mock.patch.object(tool.subprocess, "Popen", side_effect=process),
              mock.patch.object(tool, "child_run", side_effect=child),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(tool.main(args), 0)
        records = tool.read_json(output / "results.json")
        comparison = tool.read_json(output / "manifest.json")["contract"]["comparison"]
        kind = "same_binary_different_controls" if same_binary else "different_binaries"
        self.assertEqual(comparison["kind"], kind)
        self.assertEqual(comparison["allow_same_binary_controls"], same_binary)
        self.assertTrue(all(v["comparison_kind"] == kind for v in records))
        self.assertEqual([v["binary"] for v in records], list("ABBA"))
        self.assertTrue(all(v["status"] == "ok" for v in records))
        self.assertTrue(all(v["attention_calibration_final"]["unchanged"] for v in records))
        preserved = tool.read_json(output / "calibration-preserved.json")
        self.assertTrue(preserved["original_unchanged"])
        self.assertTrue(preserved["attention_original_unchanged"])
        self.assertEqual(preserved["sha256"], tool.sha(drafts))
        self.assertEqual(preserved["attention_sha256"], tool.sha(attention))
        hold = tool.read_json(output / "execution-state.json")
        self.assertTrue(hold["enter_passed"] and hold["restore_passed"] and hold["eligible"])
        self.assertEqual(execution["state"], tool.ES_CONTINUOUS)
        with (mock.patch.object(tool, "native_host", return_value=True),
              mock.patch.object(tool.ctypes, "WinDLL", return_value=kernel, create=True),
              mock.patch.object(tool, "source_identity", return_value={"fixture": True}),
              mock.patch.object(tool.subprocess, "Popen", side_effect=AssertionError("completed resume relaunched child")),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(tool.main(args + ["--resume"]), 0)

    def test_execute_cli_exercises_real_child_plumbing_with_private_frozen_seeds(self):
        self.exercise_execution_plumbing()

    def test_same_binary_different_controls_execute_and_resume_require_opt_in(self):
        self.exercise_execution_plumbing(same_binary=True)

    def test_identical_binary_rejected_without_flag_and_identical_effective_controls_rejected_with_flag(self):
        self.candidate.write_bytes(self.baseline.read_bytes())
        args = self.args(extra=["--execute", "--output", str(self.folder / "unused")])
        with self.assertRaisesRegex(ValueError, "equals selected baseline"):
            tool.prepare_inputs(args)
        for overrides in ({}, {"hybrid_probe_percent": 0}, {"speculation": "auto"}):
            candidate = self.folder / "same-controls.json"
            candidate.write_text(json.dumps(overrides))
            args = self.args(extra=["--allow-same-binary-controls", "--candidate-controls", str(candidate)])
            with mock.patch.object(tool, "source_identity", return_value={"fixture": True}):
                inputs = tool.prepare_inputs(args)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "different effective"):
                tool.build_plan(args, inputs)
        screen = self.args("screen", ["--allow-same-binary-controls"])
        with self.assertRaisesRegex(ValueError, "paired mode"):
            tool.check_args(screen)

    def execution_fixture(self, *, fail_call=None, child_error=None, previous=None):
        previous = tool.ES_CONTINUOUS if previous is None else previous
        output = self.folder / ("hold-" + str(len(list(self.folder.glob("hold-*")))))
        output.mkdir()
        tool.write_json(output / "summary.json", {"promotion": {"passed": True}})
        args = types.SimpleNamespace(output=output)
        manifest = {"contract": {"inputs": {
            "calibration": tool.calibration_input(None),
            "attention_calibration": tool.calibration_input(None, "attention")}}}
        state = {"current": previous, "calls": []}
        kernel = mock.Mock()
        def request(flags):
            state["calls"].append(flags)
            before = state["current"]
            if len(state["calls"]) == fail_call:
                return 0
            if flags & tool.ES_CONTINUOUS:
                state["current"] = flags
            return before
        kernel.SetThreadExecutionState.side_effect = request
        child = mock.Mock(side_effect=child_error, return_value=0)
        def execute_body(_manifest, _args, hold):
            hold.own_output(output, "fixture_owned_new_directory")
            hold.enter()
            return child()
        raised = None
        with mock.patch.object(tool.ctypes, "WinDLL", return_value=kernel, create=True), \
             mock.patch.object(tool, "_execute", side_effect=execute_body):
            try:
                tool.execute(manifest, args)
            except BaseException as error:
                raised = error
        return state, child, raised, tool.read_json(output / "execution-state.json"), tool.read_json(output / "summary.json")

    def test_execution_state_contract_is_identical_for_all_arms(self):
        manifest = tool.build_plan(self.args(), self.inputs())
        policy = manifest["contract"]["execution_state"]
        self.assertEqual(policy["protocol"], tool.EXECUTION_STATE_TAG)
        self.assertEqual(policy["required_flags"], 0x80000003)
        self.assertTrue(policy["retain_previous_requirements"])
        self.assertTrue(all(r["execution_state_protocol"] == policy["protocol"] for r in manifest["contract"]["plan"]))

    def test_execution_state_enters_and_restores_previous_requirements_on_success(self):
        state, child, error, hold, summary = self.execution_fixture()
        self.assertIsNone(error)
        child.assert_called_once_with()
        self.assertEqual(state["calls"], [3, 0x80000003, 0x80000000])
        self.assertEqual(state["current"], 0x80000000)
        self.assertTrue(hold["eligible"] and hold["enter_passed"] and hold["restore_passed"])
        self.assertTrue(summary["promotion"]["execution_state_gate_passed"])

    def test_execution_state_retains_existing_away_mode_and_system_request(self):
        previous = tool.ES_CONTINUOUS | 0x40 | tool.ES_SYSTEM_REQUIRED
        state, child, error, hold, summary = self.execution_fixture(previous=previous)
        self.assertIsNone(error)
        self.assertEqual(state["calls"], [3, previous | 3, previous])
        self.assertEqual(state["current"], previous)
        self.assertEqual(hold["previous_state"], previous)
        self.assertEqual(hold["requested_state"], previous | 3)

    def test_execution_state_restores_after_child_error_and_cancellation(self):
        for error in (RuntimeError("fixture child failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                state, child, raised, hold, summary = self.execution_fixture(child_error=error)
                self.assertIs(raised, error)
                self.assertEqual(state["current"], tool.ES_CONTINUOUS)
                self.assertTrue(hold["restore_passed"])
                self.assertEqual(hold["benchmark_error"]["type"], type(error).__name__)
                self.assertFalse(hold["eligible"] or summary["promotion"]["passed"])

    def test_execution_state_initial_request_failure_prevents_child(self):
        state, child, error, hold, summary = self.execution_fixture(fail_call=1)
        child.assert_not_called()
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("returned NULL", str(error))
        self.assertEqual(state["calls"], [3])
        self.assertFalse(hold["enter_passed"] or hold["eligible"])
        self.assertTrue(hold["restore_not_required"])
        self.assertFalse(summary["promotion"]["passed"])

    def test_execution_state_acquisition_failure_prevents_child_and_restores_prior(self):
        previous = tool.ES_CONTINUOUS | 0x40
        state, child, error, hold, summary = self.execution_fixture(fail_call=2, previous=previous)
        child.assert_not_called()
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(state["current"], previous)
        self.assertTrue(hold["restore_passed"])
        self.assertFalse(hold["eligible"])

    def test_execution_state_release_failure_is_explicit_and_blocks_promotion(self):
        state, child, error, hold, summary = self.execution_fixture(fail_call=3)
        child.assert_called_once_with()
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("restoration failed", str(error))
        self.assertTrue(hold["restore_attempted"])
        self.assertFalse(hold["restore_passed"] or hold["eligible"])
        self.assertIn("restore_error", hold)
        self.assertFalse(summary["promotion"]["passed"])
        self.assertFalse(summary["promotion"]["execution_state_gate_passed"])

    def test_calibration_drift_is_raised_after_restoration_and_blocks_promotion(self):
        def changed(_value):
            raise RuntimeError("fixture frozen calibration drift")
        with mock.patch.object(tool, "verify_calibration_source", side_effect=changed):
            state, child, error, hold, summary = self.execution_fixture()
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("calibration preservation failed", str(error))
        self.assertEqual(state["current"], tool.ES_CONTINUOUS)
        self.assertTrue(hold["restore_passed"])
        self.assertFalse(hold["eligible"] or summary["promotion"]["passed"])
        self.assertFalse(summary["promotion"]["calibration_gate_passed"])

    def test_restoration_and_calibration_errors_are_both_reported(self):
        with mock.patch.object(tool, "verify_calibration_source", side_effect=RuntimeError("fixture calibration drift")):
            state, child, error, hold, summary = self.execution_fixture(fail_call=3)
        self.assertIn("restoration failed", str(error))
        self.assertIn("calibration preservation failed", str(error))
        self.assertIn("restore_error", hold)
        self.assertIn("calibration_error", hold)
        self.assertFalse(summary["promotion"]["passed"])

    def invalid_output_fixture(self, mode):
        output = self.folder / ("old-output-" + mode)
        output.mkdir()
        attention = self.folder / "attn.txt"
        attention.write_bytes(b"Snapdragon_X|8|16/2/256|0|2048 1024 1024\n")
        extra = ["--execute", "--output", str(output), "--attention-calibration-file", str(attention)]
        if mode != "existing":
            extra.append("--resume")
        args = self.args(extra=extra)
        with mock.patch.object(tool, "source_identity", return_value={"fixture": True}):
            inputs = tool.prepare_inputs(args)
        manifest = tool.build_plan(args, inputs)
        old = copy.deepcopy(manifest)
        if mode == "contract":
            old["contract"]["settings"]["cache_gib"] = 7
            old["contract_sha256"] = tool.digest(old["contract"])
        tool.write_json(output / "manifest.json", old)
        tool.write_json(output / "model-verification.json", inputs["model_verification"])
        for name in ("execution-state.json", "summary.json", "calibration-preserved.json"):
            (output / name).write_bytes(b"preserve existing valid receipt bytes\n")
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with mock.patch.object(tool, "native_host", return_value=True), \
             mock.patch.object(tool, "source_identity", return_value={"fixture": True}), \
             mock.patch.object(tool.ctypes, "WinDLL", side_effect=AssertionError("invalid output requested execution state"), create=True), \
             mock.patch.object(tool, "completed_run", side_effect=RuntimeError("fixture invalid completed prefix")), \
             mock.patch.object(tool.subprocess, "Popen", side_effect=AssertionError("invalid output launched child")):
            with self.assertRaises((RuntimeError, FileExistsError)):
                tool.execute(manifest, args)
        self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)

    def test_existing_nonresume_output_keeps_all_original_receipts(self):
        self.invalid_output_fixture("existing")

    def test_invalid_resume_contract_keeps_all_original_receipts(self):
        self.invalid_output_fixture("contract")

    def test_invalid_resume_prefix_keeps_all_original_receipts(self):
        self.invalid_output_fixture("prefix")

    def test_promotion_requires_all_workloads_five_warm_and_guardrails(self):
        manifest = tool.build_plan(self.args(), self.inputs())
        records = []
        for run in manifest["contract"]["plan"]:
            samples = self.records(run)
            for sample in samples[1:]:
                sample["output_tokens_per_second"] = 14 if run["binary"] == "B" else 12.8
            records.append({**run, "status": "ok", "records": samples, "process_wall_seconds": 1.0})
        self.assertTrue(tool.summary(manifest, records)["promotion"]["passed"])
        for record in records:
            if record["binary"] == "B" and record["workload"] == "prose":
                for sample in record["records"][1:]:
                    sample["first_token_seconds"] = 1.3
        self.assertFalse(tool.summary(manifest, records)["promotion"]["passed"])


if __name__ == "__main__":
    unittest.main()

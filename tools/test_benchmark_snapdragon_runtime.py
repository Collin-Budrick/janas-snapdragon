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
            if c["hybrid_policy"] == "cost":
                samples[-1]["hybrid_metrics"] = {"schema": 1, "decisions": 8}
        return [setup, *samples]

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

"""Model-free protocol checks for the public native validation driver."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("validate-snapdragon-runtime.py")


def load():
    spec = importlib.util.spec_from_file_location("validate_snapdragon_fixture", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.v = load()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.v.OUT = self.root / "results"
        self.v.OUT.mkdir()

    def result(self, name, text, code=0):
        stdout = self.v.OUT / (name + ".stdout.log")
        stderr = self.v.OUT / (name + ".stderr.log")
        stdout.write_text(text, encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        return {"name": name, "stdout": str(stdout), "stderr": str(stderr),
                "passed": code == 0, "exit_code": code}

    def quality_record(self, batch):
        return dict(finite_logits=True, hardware_execution_verified=True, positions=64,
                    batch=batch, context=1024, cache_mib=24576, vocab=self.v.VOCAB,
                    threads=8, profile="cpu", profile_effective="cpu", expert_bits=6,
                    min_quality=2, tuning=False, expert_skip=False, head_requantized=False,
                    gpu_selected=0, npu_selected=0, comparison="compared_to_cpu",
                    logit_fnv1a64=self.v.EXPECTED_DIGEST, top1_agreement=1,
                    nll_delta=0, logit_max_abs=0, logit_mae=0, logit_rmse=0,
                    mean_kl_ref_to_candidate=0, nll=3.39, perplexity=29.76,
                    reference_nll=3.39, reference_perplexity=29.76,
                    accelerator_counters=dict.fromkeys(("npu_products", "npu_ffns",
                        "npu_layer_passes", "gpu_product_submissions", "gpu_products",
                        "gpu_whole_passes"), 0))

    def test_source_relative_root_and_plan_never_launches_or_creates_output(self):
        out = self.root / "uncreated"
        argv = [str(SCRIPT), "--plan", "--stage", "quality", "--out", str(out),
                "--model", str(self.root / "main.jns"),
                "--reference", str(self.root / "reference.qref")]
        with patch.object(sys, "argv", argv), patch.object(self.v.subprocess, "Popen",
                side_effect=AssertionError("plan launched a process")), contextlib.redirect_stdout(io.StringIO()) as text:
            self.assertEqual(self.v.main(), 0)
        plan = json.loads(text.getvalue())
        self.assertEqual(self.v.ROOT, SCRIPT.resolve().parents[1])
        self.assertFalse(plan["launches_executables"])
        self.assertFalse(out.exists())
        self.assertEqual([r["batch"] for r in plan["cases"]], [1, 3, 5, 8, 9, 17])
        self.assertTrue(all(str(self.root / "main.jns") in r["command"] for r in plan["cases"]))
        self.assertEqual(plan["reference"], str((self.root / "reference.qref").resolve()))
        self.assertNotIn("frozen-chat80b-20261108", json.dumps(plan))

    def test_inherited_controls_are_cleared_without_mutating_parent(self):
        with patch.dict(os.environ, {"JANAS_UNKNOWN_CONTROL": "parent", "JANAS_KV": "float"}):
            env = self.v.environment()
            self.assertNotIn("JANAS_UNKNOWN_CONTROL", env)
            self.assertNotIn("JANAS_KV", env)
            self.assertEqual(os.environ["JANAS_UNKNOWN_CONTROL"], "parent")
            self.assertTrue(all(env[k] == "1" for k in (
                "JANAS_CPU_TILES", "JANAS_GDN_CHAIN", "JANAS_GDN_JOURNAL", "JANAS_GDN_K_ONLY")))

    def test_receipt_and_frozen_tokens_are_public_and_self_consistent(self):
        receipt = self.v.load_receipt(self.v.RECEIPT)
        self.assertEqual(receipt["models"]["main"]["bytes"], 48411922432)
        corpus = json.loads(self.v.CODE.read_text(encoding="utf-8"))
        raw = struct.pack("<" + "i" * len(corpus["tokens"]), *corpus["tokens"])
        self.assertEqual(len(corpus["tokens"]), 276)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), corpus["int32_le_sha256"])
        self.assertEqual(hashlib.sha256(raw[:640]).hexdigest(),
                         self.v.EXPECTED_INPUTS["session-code160.int32"])
        self.assertNotIn("C:\\Users", json.dumps(corpus))

    def test_ambiguous_or_incomplete_receipts_are_rejected(self):
        path = self.root / "receipt.json"
        for text in ('{"schema":1,"schema":1}', '{"schema":true}',
                     '{"schema":1,"models":{}}', '{"schema":NaN}'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.v.load_receipt(path)

    def test_reference_generation_refuses_to_overwrite(self):
        self.v.QREF = self.root / "cpu64.qref"
        self.v.QREF.write_bytes(b"preserve existing input")
        with patch.object(self.v, "run", side_effect=AssertionError("unexpected model run")):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                self.v.create_reference()
        self.assertEqual(self.v.QREF.read_bytes(), b"preserve existing input")

    def test_execute_on_non_windows_rejects_before_creating_output(self):
        out = self.root / "uncreated"
        fake_os = types.SimpleNamespace(name="posix", getpid=os.getpid, environ=os.environ)
        with patch.object(self.v, "os", fake_os), patch.object(sys, "argv",
                [str(SCRIPT), "--execute", "--stage", "units", "--out", str(out)]), \
                patch.object(self.v.subprocess, "Popen", side_effect=AssertionError("spawned")), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.v.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(out.exists())

    def test_quality_rejects_drift_and_nonfinite_even_on_exit_zero(self):
        for change in ({}, {"nll_delta": 0.01}, {"nll": float("nan")}):
            calls = []
            def fake(name, argv, env, **unused):
                batch = int(argv[argv.index("--batch") + 1])
                record = self.quality_record(batch)
                record.update(change)
                calls.append(batch)
                return self.result(name, json.dumps(record) + "\n")
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.quality(), not change)
            self.assertEqual(calls, [1, 3, 5, 8, 9, 17] if not change else [1])

    def test_eval_requires_actual_rejection_prefix_marker(self):
        for wrong in (False, True):
            def fake(name, argv, env, expected_exit=(0,), **unused):
                if name == "eval-cli-capabilities":
                    result = self.result(name, "usage block=B reject=R strict=1\n", 2)
                    result["passed"] = 2 in expected_exit
                    return result
                block = int(next(a[6:] for a in argv if isinstance(a, str) and a.startswith("block=")))
                reject = int(next(a[7:] for a in argv if isinstance(a, str) and a.startswith("reject=")))
                return self.result(name, f"blocks of {block} keeping {block-reject+wrong}: passes\n"
                    f"blocks of {block}: 32/32 positions bit-identical to token by token, max |diff| 0;\n"
                    "32/32 input positions covered\n"
                    "top-1 agreement 32/32, max |diff| 0.000, mean |diff| 0.0000\n")
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.evaluation(), not wrong)

    def test_session_requires_width17_and_does_not_claim_unused_adaptive_head(self):
        for wrong_width in (False, True):
            def fake(name, argv, env, **unused):
                values = dict(a.split("=", 1) for a in argv if isinstance(a, str) and "=" in a)
                planner, mode = values["planner"], values["speculation"]
                drafts, rows = int(values["drafts"]), int(values["rows"])
                adaptive = values["draft_vocab"] == "adaptive"
                text = (f"settings: compute=8 io=4 planner={planner} speculation={mode} drafts={drafts} "
                        f"draft_vocab={values['draft_vocab']} rows={rows} quality=HIGH expert_bits=6 head_bits=0\n")
                for i in range(4):
                    selected = i % 2 == 1
                    mtp = selected and not (adaptive and mode == "hybrid")
                    resolved = {"legacy": 0, "cycle": 1, "fixed": 2}[planner] if selected else 0
                    width = (16 if wrong_width else 17) if selected and drafts == 16 else 5 if selected else 1
                    text += (f"runtime: mode={4 if selected and mode=='hybrid' else 2 if selected else 0} "
                        f"planner={resolved} cycles=36 max_width={width} "
                        f"lookup_attempts={4 if selected and mode=='hybrid' else 0} mtp_drafts={32 if mtp else 0} "
                        f"head_calls={32 if mtp else 0} repacks={2 if mtp and adaptive else 0} "
                        "head_policy=0 head_rows=16384 head_epoch=2 verified\n")
                    if selected and values.get("hybrid_policy") == "cost":
                        text += "hybrid: policy=cost decisions=12 offered=8 lookup=1 mtp=11 probes=1 cooldown=0 accepted=1 verified\n"
                for marker in ("greedy: turns vs all at once", "greedy + speculation: turn 1",
                    "greedy + speculation: turn 2", "sampling, speculation off vs on: turn 1",
                    "sampling, speculation off vs on: turn 2"):
                    text += marker + " identical\n"
                return self.result(name, text + "sampling differs from greedy yes\nllm_session: ok\n")
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.session(), not wrong_width)
            records = json.loads((self.v.OUT / "sessions.json").read_text())["results"]
            hybrid = next(r for r in records if r["name"] == "session-hybrid-cycle8-adaptive")
            self.assertFalse(hybrid["adaptive_execution_verified"])

    def test_cost_hybrid_plan_separates_normal_and_forced_probe_validation(self):
        cases = self.v.plan("session")["cases"]
        normal = next(c for c in cases if c["name"] == "hybrid-cost8")
        probe = next(c for c in cases if c["name"] == "hybrid-cost8-probe")
        for case in (normal, probe):
            self.assertIn("hybrid_policy=cost", case["command"])
            self.assertIn("lookup_drafts=8", case["command"])
            self.assertIn("mtp_drafts=4", case["command"])
            self.assertEqual(case["planner"], "legacy")
        self.assertIn("hybrid_probe_percent=2", normal["command"])
        self.assertIn("hybrid_probe_percent=100", probe["command"])

    def test_semantic_hybrid_requires_all_sources_both_samplers_and_actual_lookup(self):
        for lookup_seen in (False, True):
            def fake(name, argv, env, **unused):
                records = []
                for sampler in ("greedy", "seeded"):
                    for mode in ("plain", "mtp4", "hybrid-cost8"):
                        for turn, phase in enumerate(["copy"] * 4 + ["switch", "return"]):
                            records.append(dict(schema=1, kind="semantic", sampler=sampler, mode=mode,
                                turn=turn, phase=phase, identity_verified=True, input_threads=8, output_threads=8))
                records.append(dict(schema=1, kind="summary", identity_verified=True,
                                    lookup_exercised=lookup_seen, fallback_exercised=True))
                return self.result(name, "\n".join(json.dumps(v) for v in records))
            with patch.object(self.v, "run", fake):
                self.assertEqual(self.v.semantic_hybrid(), lookup_seen)
        planned = self.v.plan("hybrid")["cases"]
        self.assertEqual(len(planned), 1)
        self.assertIn("probe=2", planned[0]["command"])

    def test_attention_calibration_seeds_identical_private_copies_without_touching_original(self):
        path = self.root / "attn.txt"
        data = b"Snapdragon_X|8|16/2/256|0|2048 4294967295 1024\n"
        path.write_bytes(data)
        frozen = self.v.attention_input(path)
        for name in ("first", "second"):
            seed = self.v.seed_attention_calibration(self.root / name, frozen)
            copied = Path(seed["private_path"])
            self.assertEqual(copied.read_bytes(), data)
            copied.write_bytes(b"private changes remain private")
        self.assertEqual(path.read_bytes(), data)
        path.write_bytes(b"Snapdragon_X|8|16/2/256|0|2048 1024 1024\n")
        with self.assertRaises(RuntimeError):
            self.v.seed_attention_calibration(self.root / "third", frozen)

    def test_model_files_with_same_basename_are_validated_by_role(self):
        self.v.MODEL = self.root / "main" / "weights.jns"
        self.v.MTP = self.root / "mtp" / "weights.jns"
        for path, data in ((self.v.MODEL, b"a"), (self.v.MTP, b"bb")):
            path.parent.mkdir()
            path.write_bytes(data)
        self.v.MODEL_PINS = {"main": (1, "a" * 64), "mtp": (2, "b" * 64)}
        records = self.v.model_metadata()
        self.assertEqual([r["size"] for r in records], [1, 2])
        self.assertTrue(all(not r["sha256_recomputed_by_runner"] for r in records))


if __name__ == "__main__":
    unittest.main()

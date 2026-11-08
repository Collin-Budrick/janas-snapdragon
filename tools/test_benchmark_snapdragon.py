"""Model-free tests of benchmark scheduling and evidence handling."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "benchmark_snapdragon", Path(__file__).with_name("benchmark-snapdragon.py"))
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def row(case, rate=10, signature=None, prompts=None):
    return {"stage": "refine", "case": dataclasses.asdict(case), "status": "ok",
            "warm_median_tokens_per_second": rate,
            "output_signature": signature or ["same", 64],
            "comparison_prompt_ids": [0] if prompts is None else prompts}


class BenchmarkTests(unittest.TestCase):
    def test_new_controls_are_explicit_recorded_and_do_not_mutate_parent(self):
        args = argparse.Namespace(bench=Path("bench.exe"), model=Path("model.jns"),
            mtp=Path("mtp.jns"), seed=1, gen=64, ctx=2048, rounds=1,
            warmups=0, reserve_gib=12, timeout=30, n_prompts=1)
        response = subprocess.CompletedProcess([], 1, "", "model-free fixture")
        hostile = {name: "inherited-hostile" for name in bench.EXPERIMENT_ENV}
        hostile["JANAS_CPU_TILES"] = "1"
        base = bench.Case("cpu", "none", 8, 32, 4)
        selected = dataclasses.replace(base, speculation="hybrid", spec_planner="cycle",
            spec_explore=0, draft_vocab="adaptive", draft_vocab_rows=16384,
            gdn_chain=1, gdn_journal=1, gdn_k_only=1)
        for case in (base, selected):
            for panels, tile_value in (("auto", None), ("off", "0"), ("experimental", "1")):
                with self.subTest(case=case, panels=panels), tempfile.TemporaryDirectory() as tmp, \
                     patch.dict(os.environ, hostile), \
                     patch.object(bench.subprocess, "run", return_value=response) as run:
                    case = dataclasses.replace(case, panels=panels)
                    result = bench.run_case(args, case, Path("prompt.txt"), Path(tmp), "screen")
                    env = run.call_args.kwargs["env"]
                    expected = dict(zip(bench.EXPERIMENT_ENV, (
                        case.spec_planner, str(case.spec_explore), case.draft_vocab,
                        str(case.draft_vocab_rows), str(case.gdn_chain),
                        str(case.gdn_journal), str(case.gdn_k_only))))
                    expected["JANAS_CPU_TILES"] = tile_value
                    for name, value in expected.items():
                        self.assertEqual(env.get(name), value)
                        self.assertEqual(result["performance_environment"][name], value)
                        self.assertEqual(os.environ[name], hostile[name])
                    command = result["command"]
                    for name, value in (("--spec-planner", case.spec_planner),
                                        ("--spec-explore", str(case.spec_explore)),
                                        ("--draft-vocab", case.draft_vocab),
                                        ("--draft-vocab-rows", str(case.draft_vocab_rows))):
                        self.assertEqual(command[command.index(name) + 1], value)
                    self.assertEqual("--mtp" in command, case.speculation == "hybrid")
                    self.assertEqual("--no-mtp" in command, case.speculation == "none")
                    self.assertEqual((env["JANAS_EXPERT_BITS"], env["JANAS_HEAD_BITS"],
                                      env["JANAS_MISS_SKIP"]), ("6", "0", "0"))

    def test_controls_preserve_refinement_but_never_change_cpu_reference(self):
        args = argparse.Namespace(profiles=["cpu"], speculations=["hybrid"],
            threads=[8, 12], cache_gib=[24, 32], draft_tokens=[4, 8],
            spec_planner="fixed", spec_explore=100, draft_vocab="adaptive",
            draft_vocab_rows=8192, gdn_chain=1, gdn_journal=1, gdn_k_only=1,
            panels="experimental")
        cases = bench.screen_cases(args)
        self.assertEqual(len(cases), 2)
        ref, candidate = cases
        self.assertEqual((ref.speculation, ref.spec_planner, ref.spec_explore,
            ref.draft_vocab, ref.draft_vocab_rows, ref.gdn_chain, ref.gdn_journal,
            ref.gdn_k_only, ref.panels),
            ("none", "legacy", 20, "legacy", 0, 0, 0, 0, "auto"))
        self.assertEqual(candidate.speculation, "hybrid")
        names = ("spec_planner", "spec_explore", "draft_vocab", "draft_vocab_rows",
                 "gdn_chain", "gdn_journal", "gdn_k_only", "panels")
        refined = bench.refinement_cases(args, [candidate], 2)
        for case in refined:
            if case.speculation == "none":
                self.assertEqual(case, case.cpu_reference())
            else:
                self.assertEqual(tuple(getattr(case, n) for n in names),
                                 tuple(getattr(candidate, n) for n in names))
        for name in names[:4] + ("panels",):
            self.assertNotEqual(candidate.key(),
                dataclasses.replace(candidate, **{name: getattr(ref, name)}).key())
        self.assertNotEqual(ref.key(), dataclasses.replace(ref, gdn_chain=1).key())
        self.assertNotEqual(candidate.key(), dataclasses.replace(candidate, gdn_k_only=0).key())
        self.assertNotEqual(candidate.key(),
            dataclasses.replace(candidate, gdn_journal=0, gdn_k_only=0).key())

    def test_invalid_controls_and_gdn_dependencies_are_rejected(self):
        base = bench.Case("cpu", "mtp", 8, 32, 4)
        invalid = [{"spec_planner": "other"}, {"spec_explore": -1},
            {"spec_explore": 101}, {"spec_explore": True}, {"spec_explore": 1.5},
            {"draft_vocab": "other"}, {"draft_vocab_rows": 1},
            {"draft_vocab_rows": False}, {"gdn_chain": True}, {"gdn_chain": 2},
            {"gdn_journal": 1}, {"gdn_chain": 1, "gdn_k_only": 1},
            {"panels": "on"}, {"speculation": "other"}]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(base, **changes)
        for explore in (0, 100):
            self.assertEqual(dataclasses.replace(base, spec_explore=explore).spec_explore, explore)
        for rows in (0, 8192, 16384, 32768):
            self.assertEqual(dataclasses.replace(base, draft_vocab_rows=rows).draft_vocab_rows, rows)

    def test_cli_accepts_hybrid_controls_and_rejects_invalid_plan(self):
        from contextlib import redirect_stdout
        import io
        options = ["bench", "--plan", "--profiles", "cpu", "--speculations", "hybrid",
            "--spec-planner", "cycle", "--spec-explore", "0", "--draft-vocab", "adaptive",
            "--draft-vocab-rows", "32768", "--gdn-chain", "1", "--gdn-journal", "1",
            "--gdn-k-only", "1", "--panels", "experimental"]
        output = io.StringIO()
        with patch.object(sys, "argv", options), redirect_stdout(output):
            self.assertEqual(bench.main(), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["candidate_cases"], 2)
        self.assertEqual(plan["cases"][1]["panels"], "experimental")
        for extra in (["--spec-explore", "101"], ["--gdn-chain", "0"]):
            with patch.object(sys, "argv", options + extra), self.assertRaises(ValueError):
                bench.main()

    def test_hexagon_work_threshold_is_explicit_recorded_and_never_inherited(self):
        args = argparse.Namespace(bench=Path("bench.exe"), model=Path("model.jns"),
            mtp=Path("mtp.jns"), seed=1, gen=64, ctx=2048, rounds=1,
            warmups=0, reserve_gib=12, timeout=30, n_prompts=1)
        response = subprocess.CompletedProcess([], 1, "", "model-free fixture")
        for value in (0, 16777216):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp, \
                 patch.dict(os.environ, {"JANAS_NPU_MIN_MACS": "99"}), \
                 patch.object(bench.subprocess, "run", return_value=response) as run:
                case = bench.Case("cpu+npu", "none", 8, 32, 4, npu_min_macs=value)
                result = bench.run_case(args, case, Path("prompt.txt"), Path(tmp), "screen")
                expected = str(value) if value else None
                self.assertEqual(run.call_args.kwargs["env"].get("JANAS_NPU_MIN_MACS"), expected)
                self.assertEqual(result["performance_environment"]["JANAS_NPU_MIN_MACS"], expected)
                self.assertEqual(result["case"]["npu_min_macs"], value)

    def test_case_cache_exists_before_native_process_and_uses_forward_slashes(self):
        args = argparse.Namespace(bench=Path("bench.exe"), model=Path("model.jns"),
            mtp=Path("mtp.jns"), seed=1, gen=64, ctx=2048, rounds=1,
            warmups=0, reserve_gib=12, timeout=30, n_prompts=1)
        case = bench.Case("cpu", "none", 12, 24, 4)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            native_path = directory / "cache" / case.key()
            self.assertFalse(native_path.exists())
            if os.name == "nt":
                self.assertIn("\\", str(native_path))

            def check_before_launch(*command, **kwargs):
                actual = kwargs["env"]["XDG_CACHE_HOME"]
                self.assertEqual(actual, native_path.as_posix())
                self.assertNotIn("\\", actual)
                self.assertTrue(Path(actual).is_dir())
                return subprocess.CompletedProcess(command, 1, "", "model-free fixture")

            with patch.object(bench.subprocess, "run", side_effect=check_before_launch) as run:
                bench.run_case(args, case, Path("prompt.txt"), directory, "screen")
                run.assert_called_once()

    def test_no_speculation_ceiling_does_not_duplicate_baseline(self):
        args = argparse.Namespace(profiles=["cpu"], speculations=["none", "mtp"],
            threads=[8], cache_gib=[32], draft_tokens=[8], prefetch="auto")
        cases = bench.screen_cases(args)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].draft_tokens, 4)
        self.assertEqual(cases[1].draft_tokens, 8)

    def test_old_cases_default_to_auto_and_policy_changes_identity(self):
        old = bench.Case("cpu", "none", 12, 24, 4, 0, "cpu", "cpu", "cpu")
        legacy = dataclasses.asdict(old)
        legacy.pop("prefetch")
        legacy.pop("gpu_min_tokens")
        legacy.pop("npu_min_tokens")
        legacy.pop("npu_min_macs")
        for name in ("spec_planner", "spec_explore", "draft_vocab", "draft_vocab_rows",
                     "gdn_chain", "gdn_journal", "gdn_k_only", "panels"):
            legacy.pop(name)
        self.assertEqual(bench.Case(**legacy), old)
        self.assertEqual(old.prefetch, "auto")
        self.assertEqual((old.gpu_min_tokens, old.npu_min_tokens), (0, 0))
        off = dataclasses.replace(old, prefetch="off")
        self.assertNotEqual(old.key(), off.key())
        self.assertEqual(off.cpu_reference().prefetch, "off")

    def test_manual_accelerator_thresholds_preserve_bounded_cases(self):
        args = argparse.Namespace(profiles=list(bench.PROFILES),
            speculations=["none", "lookup", "mtp", "auto"], threads=[8, 12],
            cache_gib=[24, 32], draft_tokens=[4, 8], prefetch="off",
            gpu_min_tokens=1, npu_min_tokens=8)
        cases = bench.screen_cases(args)
        self.assertEqual(len(cases), 16)
        self.assertTrue(all((c.gpu_min_tokens, c.npu_min_tokens) == (1, 8) for c in cases))
        refined = bench.refinement_cases(args, [cases[0]], 2)
        self.assertTrue(all((c.gpu_min_tokens, c.npu_min_tokens) == (1, 8) for c in refined))
        self.assertNotEqual(cases[0].key(), dataclasses.replace(cases[0], gpu_min_tokens=0).key())

    def test_screen_and_refinement_preserve_policy_without_expanding_matrix(self):
        args = argparse.Namespace(profiles=list(bench.PROFILES),
            speculations=["none", "lookup", "mtp", "auto"], threads=[4, 8, 12],
            cache_gib=[8, 24, 0], draft_tokens=[2, 4, 8], prefetch="off")
        cases = bench.screen_cases(args)
        self.assertEqual(len(cases), 16)
        self.assertTrue(all(c.prefetch == "off" for c in cases))
        args.prefetch = "auto"
        refined = bench.refinement_cases(args, [cases[0]], 2)
        self.assertTrue(all(c.prefetch == "off" for c in refined))

    def test_prefetch_settings_do_not_silently_inherit_and_are_recorded(self):
        records = [{"schema": 1, "kind": "setup", "gpu": "", "npu": ""}]
        records += [{"schema": 1, "kind": "sample", "phase": phase,
                     "output_tokens": 64, "output_hash": "same",
                     "output_tokens_per_second": 10}
                    for phase in ("fresh_process", "warm")]
        response = subprocess.CompletedProcess([], 0,
                    "\n".join(json.dumps(x) for x in records), "")
        args = argparse.Namespace(bench=Path("bench.exe"), model=Path("model.jns"),
            mtp=Path("mtp.jns"), seed=1, gen=64, ctx=2048, rounds=1,
            warmups=0, reserve_gib=12, timeout=30, n_prompts=1)
        expected = {"auto": (None, None), "off": ("0", "0"),
                    "decode-off": ("0", None), "prompt-off": (None, "0")}
        for policy, values in expected.items():
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp, \
                 patch.dict(os.environ, {"JANAS_PREFETCH": "10,2", "JANAS_PREFETCH_PREFILL": "2",
                                          "JANAS_GPU_MIN_N": "128", "JANAS_NPU_MIN_N": "128",
                                          "JANAS_KV": "16", "JANAS_ATTN": "int16",
                                          "JANAS_CPU_TILES": "1",
                                          "JANAS_API_KEY": "test-secret"}), \
                 patch.object(bench.subprocess, "run", return_value=response) as run:
                case = bench.Case("cpu", "none", 12, 24, 4, prefetch=policy)
                result = bench.run_case(args, case, Path("prompt.txt"), Path(tmp), "screen")
                env = run.call_args.kwargs["env"]
                self.assertEqual(tuple(env.get(k) for k in ("JANAS_PREFETCH", "JANAS_PREFETCH_PREFILL")), values)
                recorded = result["performance_environment"]
                self.assertEqual(tuple(recorded[k] for k in ("JANAS_PREFETCH", "JANAS_PREFETCH_PREFILL")), values)
                self.assertNotIn("JANAS_API_KEY", recorded)
                self.assertNotIn("JANAS_GPU_MIN_N", env)
                self.assertNotIn("JANAS_NPU_MIN_N", env)
                for key, value in (("JANAS_KV", "16"), ("JANAS_ATTN", "int16"),
                                   ("JANAS_CPU_TILES", "1")):
                    self.assertNotIn(key, env)
                    self.assertIsNone(recorded[key])
                    self.assertEqual(os.environ[key], value)
                self.assertEqual(result["status"], "ok")
                forced = dataclasses.replace(case, gpu_min_tokens=1, npu_min_tokens=8)
                forced_result = bench.run_case(args, forced, Path("prompt.txt"), Path(tmp), "screen")
                forced_env = run.call_args.kwargs["env"]
                self.assertEqual((forced_env["JANAS_GPU_MIN_N"], forced_env["JANAS_NPU_MIN_N"]), ("1", "8"))
                self.assertEqual(forced_result["performance_environment"]["JANAS_NPU_MIN_N"], "8")
                self.assertEqual(os.environ["JANAS_KV"], "16")
                self.assertEqual(os.environ["JANAS_ATTN"], "int16")
                self.assertNotIn("JANAS_CPU_TILES", forced_env)
                self.assertIsNone(forced_result["performance_environment"]["JANAS_CPU_TILES"])
                self.assertEqual(os.environ["JANAS_CPU_TILES"], "1")

    def test_preparing_weights_is_not_execution(self):
        samples = [{"phase": "warm", "accelerator_counters": {
            "npu_prepared_plans": 12, "npu_weight_bytes": 1 << 28,
            "gpu_product_submissions": 4}}]
        self.assertFalse(bench.execution_evidence(samples, True, True)["verified"])

    def test_every_requested_device_needs_completed_work_each_sample(self):
        samples = [{"phase": phase, "accelerator_counters": {
            "npu_products": 7, "gpu_products": 4}}
                   for phase in ("fresh_process", "warm")]
        self.assertTrue(bench.execution_evidence(samples, True, True)["verified"])
        samples[1]["accelerator_counters"]["npu_products"] = 0
        self.assertFalse(bench.execution_evidence(samples, True, True)["verified"])

    def test_exact_output_without_execution_cannot_recommend_accelerator(self):
        cpu = bench.Case("cpu", "none", 12, 0, 4)
        gpu = dataclasses.replace(cpu, profile="cpu+gpu", prompt_on="gpu",
                                  drafts_on="gpu", reply_on="gpu")
        rows = [row(cpu), row(gpu, rate=100)]
        bench.annotate(rows)
        self.assertFalse(rows[1]["recommendation_eligible"])
        rows[1]["accelerator_execution_verified"] = True
        bench.annotate(rows)
        self.assertTrue(rows[1]["recommendation_eligible"])

    def test_nonrepeatable_cpu_reference_cannot_approve_accelerator(self):
        cpu = bench.Case("cpu", "none", 8, 32, 4)
        gpu = dataclasses.replace(cpu, profile="cpu+gpu", prompt_on="gpu",
                                  drafts_on="gpu", reply_on="gpu")
        rows = [row(cpu), row(gpu, rate=100)]
        rows[0]["repeatable_output"] = False
        rows[1].update(repeatable_output=True, accelerator_execution_verified=True)
        bench.annotate(rows)
        self.assertIsNone(rows[1]["matches_cpu_reference"])
        self.assertFalse(rows[1]["recommendation_eligible"])
        self.assertTrue(rows[1]["discovery_eligible"])
        self.assertEqual(rows[1]["quality_validation"], "unverified_no_cpu_reference")

    def test_screen_is_bounded_and_baselines_precede_variants(self):
        args = argparse.Namespace(profiles=list(bench.PROFILES),
            speculations=["none", "lookup", "mtp", "auto"], threads=[4, 8, 12],
            cache_gib=[8, 24, 0], draft_tokens=[2, 4, 8])
        cases = bench.screen_cases(args)
        self.assertEqual(len(cases), 16)
        for index, case in enumerate(cases):
            self.assertLessEqual(cases.index(case.cpu_reference()), index)

    def test_fast_output_change_cannot_become_recommendation(self):
        cpu = bench.Case("cpu", "none", 12, 0, 4)
        mtp = dataclasses.replace(cpu, speculation="mtp")
        rows = [row(cpu), row(mtp, rate=100, signature=["different", 64])]
        self.assertEqual(bench.best(rows, 1, matching_only=True), [cpu])
        self.assertEqual(bench.best(rows, 1), [cpu])
        self.assertFalse(rows[1]["matches_cpu_reference"])

    def test_accelerator_output_difference_remains_discovery_candidate(self):
        cpu = bench.Case("cpu", "none", 12, 0, 4)
        gpu = dataclasses.replace(cpu, profile="cpu+gpu", prompt_on="gpu",
                                  drafts_on="gpu", reply_on="gpu")
        rows = [row(cpu), row(gpu, rate=100, signature=["different", 64])]
        self.assertEqual(bench.best(rows, 1), [gpu])
        self.assertEqual(bench.best(rows, 1, matching_only=True), [cpu])
        self.assertEqual(rows[1]["quality_validation"], "unverified_accelerator_output_difference")

    def test_incomplete_prompt_matrix_cannot_win(self):
        cpu = bench.Case("cpu", "none", 12, 0, 4)
        gpu = dataclasses.replace(cpu, profile="cpu+gpu", prompt_on="gpu",
                                  drafts_on="gpu", reply_on="gpu")
        rows = [row(dataclasses.replace(cpu, prompt=i), prompts=[0, 1]) for i in range(2)]
        rows.append(row(gpu, rate=100, prompts=[0, 1]))
        self.assertEqual(bench.best(rows, 1), [cpu])

    def test_ranking_aggregates_same_prompts(self):
        cpu = bench.Case("cpu", "none", 12, 0, 4)
        mtp = dataclasses.replace(cpu, speculation="mtp")
        rows = [row(dataclasses.replace(cpu, prompt=i), rate=10, prompts=[0, 1]) for i in range(2)]
        rows += [row(dataclasses.replace(mtp, prompt=i), rate=r, prompts=[0, 1])
                 for i, r in enumerate([40, 4])]
        ranked = bench.configurations(rows)
        self.assertAlmostEqual(ranked[0]["warm_geomean_tokens_per_second"], 160 ** 0.5)
        self.assertEqual(bench.best(rows, 1), [mtp])

    def test_old_cli_is_rejected(self):
        response = subprocess.CompletedProcess([], 2, "", "usage: janas-bench --gen")
        with patch.object(bench.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Rebuild"):
                bench.validate_cli(Path("old.exe"))

    def test_cli_flag_prefix_cannot_satisfy_new_control(self):
        flags = bench.REQUIRED_FLAGS - {"--draft-vocab"}
        response = subprocess.CompletedProcess([], 0, " ".join(sorted(flags)), "")
        with patch.object(bench.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(RuntimeError, r"missing: --draft-vocab$"):
                bench.validate_cli(Path("partial.exe"))

    def test_cpu_fallback_is_not_reported_as_gpu(self):
        case = bench.Case("cpu+gpu", "none", 12, 0, 4,
                          prompt_on="gpu", drafts_on="gpu", reply_on="gpu")
        records = [{"schema": 1, "kind": "setup", "gpu": "", "npu": ""}]
        records += [{"schema": 1, "kind": "sample", "phase": phase,
                     "output_tokens": 64, "output_hash": "same",
                     "output_tokens_per_second": 100}
                    for phase in ("fresh_process", "warm")]
        response = subprocess.CompletedProcess([], 0,
                    "\n".join(json.dumps(x) for x in records), "")
        args = argparse.Namespace(bench=Path("bench.exe"), model=Path("model.jns"),
            mtp=Path("mtp.jns"), seed=1, gen=64, ctx=2048, rounds=1,
            warmups=0, reserve_gib=12, timeout=30, n_prompts=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench.subprocess, "run", return_value=response):
            result = bench.run_case(args, case, Path("prompt.txt"), Path(tmp), "screen")
        self.assertEqual(result["status"], "failed")
        self.assertIn("unavailable", result["error"])

    def test_accelerator_repeat_variation_retains_speed_without_execution_claim(self):
        case = bench.Case("cpu+gpu", "none", 12, 0, 4,
                          prompt_on="gpu", drafts_on="gpu", reply_on="gpu")
        records = [{"schema": 1, "kind": "setup", "gpu": "Adreno", "npu": ""}]
        records += [{"schema": 1, "kind": "sample", "phase": phase,
                     "output_tokens": 64, "output_hash": value,
                     "output_tokens_per_second": 100}
                    for phase, value in (("fresh_process", "first"), ("warm", "second"))]
        response = subprocess.CompletedProcess([], 0,
                    "\n".join(json.dumps(x) for x in records), "")
        args = argparse.Namespace(bench=Path("bench.exe"), model=Path("model.jns"),
            mtp=Path("mtp.jns"), seed=1, gen=64, ctx=2048, rounds=1,
            warmups=0, reserve_gib=12, timeout=30, n_prompts=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench.subprocess, "run", return_value=response):
            result = bench.run_case(args, case, Path("prompt.txt"), Path(tmp), "screen")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["repeatable_output"])
        self.assertFalse(result["accelerator_execution_verified"])
        self.assertEqual(result["warm_median_tokens_per_second"], 100)


if __name__ == "__main__":
    unittest.main()

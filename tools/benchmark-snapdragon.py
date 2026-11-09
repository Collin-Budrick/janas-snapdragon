#!/usr/bin/env python3
"""Sequential, native-Windows Janas benchmarks with fixed chat workloads.

Run --plan anywhere to inspect the bounded screen/refinement design. Actual
runs require native ARM64 Windows Python and a rebuilt janas-bench.exe.
Fresh-process results do not claim that Windows's file cache or the SSD is
cold. Warm samples retain model/expert caches but recreate the chat history.
Experimental planner, draft-head, GDN and panel controls are explicit case
settings; defaults retain the established runtime policy and CPU reference.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "cpu": ("cpu", "cpu", "cpu"),
    "cpu+gpu": ("gpu", "gpu", "gpu"),
    "cpu+npu": ("npu", "cpu", "cpu"),
    "cpu+gpu+npu": ("npu+gpu", "gpu", "gpu"),
}
PROMPTS = [
    ("explanation", "Explain how a language model can use RAM, SSD storage, a CPU, "
     "a GPU and an NPU together. Describe the tradeoffs and give a concrete example. "
     "Write clear paragraphs without assuming a specific benchmark result."),
    ("code", "Write a Python function that groups a list of event dictionaries by "
     "their UTC date. Handle missing timestamps explicitly. Explain the design and "
     "show two small examples with different dates."),
    ("planning", "Plan a small community garden with herbs, vegetables and flowers. "
     "Explain the layout, watering schedule and how volunteers can share the work. "
     "Make the recommendations concrete and practical."),
]
REQUIRED_FLAGS = {
    "--speculation", "--draft-tokens", "--prompt-file", "--seed", "--threads",
    "--warmups", "--json", "--prompt-on", "--drafts-on", "--reply-on",
    "--spec-planner", "--spec-explore", "--draft-vocab", "--draft-vocab-rows",
}
PREFETCH_POLICIES = {"auto": {}, "off": {"JANAS_PREFETCH": "0", "JANAS_PREFETCH_PREFILL": "0"},
    "decode-off": {"JANAS_PREFETCH": "0"}, "prompt-off": {"JANAS_PREFETCH_PREFILL": "0"}}
PANEL_POLICIES = {"auto": {}, "off": {"JANAS_CPU_TILES": "0"},
                  "experimental": {"JANAS_CPU_TILES": "1"}}
EXPERIMENT_ENV = ("JANAS_SPEC_PLANNER", "JANAS_SPEC_EXPLORE",
    "JANAS_DRAFT_VOCAB_POLICY", "JANAS_DRAFT_VOCAB_ROWS",
    "JANAS_HYBRID_POLICY", "JANAS_LOOKUP_DRAFTS", "JANAS_MTP_DRAFTS", "JANAS_HYBRID_PROBE_PERCENT",
    "JANAS_GDN_CHAIN", "JANAS_GDN_JOURNAL", "JANAS_GDN_K_ONLY",
    "JANAS_S2_EXACT", "JANAS_S2_BRANCHES", "JANAS_S2_MAX_NODES", "JANAS_S2_MIN_GAIN_PERCENT")
RECORDED_PERFORMANCE_ENV = (
    "JANAS_GPU", "JANAS_NPU", "JANAS_GPU_DEVICE", "JANAS_GPU_SG", "JANAS_TUNE",
    "JANAS_TUNE_THREADS", "JANAS_WARM", "JANAS_MISS_SKIP", "JANAS_EXPERT_BITS",
    "JANAS_HEAD_BITS", "JANAS_KERNELS", "JANAS_EXPERTS", "JANAS_BLOCK_THREADS",
    "JANAS_KV", "JANAS_ATTN", "JANAS_CPU_TILES",
    "JANAS_PREFETCH", "JANAS_PREFETCH_PREFILL", "JANAS_URING", "JANAS_URING_BATCH",
    "JANAS_URING_DEEP", "JANAS_NPU_MIN_N", "JANAS_NPU_MIN_MACS", "JANAS_NPU_MAX_MB", "JANAS_NPU_SPIN",
    "JANAS_NPU_A8", "JANAS_NPU_LAYERS", "JANAS_GPU_ALWAYS", "JANAS_NPU_ALWAYS",
    "JANAS_GPU_TOKEN", "JANAS_GPU_WHOLE", "JANAS_GPU_FAST_PROMPT",
    "JANAS_GPU_EXPERT_SHARE", "JANAS_GPU_NV", "JANAS_GPU_MIN_N", "JANAS_GPU_ALLOC",
    *EXPERIMENT_ENV)


@dataclasses.dataclass(frozen=True)
class Case:
    profile: str
    speculation: str
    threads: int
    cache_gib: float
    draft_tokens: int
    prompt: int = 0
    prompt_on: str = "cpu"
    drafts_on: str = "cpu"
    reply_on: str = "cpu"
    prefetch: str = "auto"
    gpu_min_tokens: int = 0
    npu_min_tokens: int = 0
    npu_min_macs: int = 0
    spec_planner: str = "legacy"
    spec_explore: int = 20
    draft_vocab: str = "legacy"
    draft_vocab_rows: int = 0
    hybrid_policy: str = "legacy"
    lookup_drafts: int = 8
    mtp_drafts: int = 4
    hybrid_probe_percent: int = 2
    s2_exact: str = "off"
    s2_branches: int = 2
    s2_max_nodes: int = 8
    s2_min_gain_percent: int = 5
    gdn_chain: int = 0
    gdn_journal: int = 0
    gdn_k_only: int = 0
    panels: str = "auto"

    def __post_init__(self) -> None:
        if self.speculation not in ("none", "lookup", "mtp", "auto", "hybrid"):
            raise ValueError("Invalid speculation mode")
        if self.spec_planner not in ("legacy", "cycle", "fixed"):
            raise ValueError("Invalid speculation planner")
        if type(self.spec_explore) is not int or not 0 <= self.spec_explore <= 100:
            raise ValueError("Speculation exploration must be an integer in 0..100")
        if self.hybrid_policy not in ("legacy", "cost"):
            raise ValueError("Invalid hybrid policy")
        for value, low, high in ((self.lookup_drafts, 1, 16), (self.mtp_drafts, 1, 16),
                                 (self.hybrid_probe_percent, 0, 100)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid hybrid source ceiling/probe percentage")
        if self.s2_exact not in ("off", "on"):
            raise ValueError("Invalid S2 exact mode")
        for value, low, high in ((self.s2_branches, 1, 2), (self.s2_max_nodes, 1, 32), (self.s2_min_gain_percent, 0, 100)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid S2 bound")
        if self.s2_exact == "on" and self.speculation not in ("mtp", "auto", "hybrid"):
            raise ValueError("S2 exact requires MTP speculation")
        if self.draft_vocab not in ("legacy", "adaptive"):
            raise ValueError("Invalid draft vocabulary policy")
        if type(self.draft_vocab_rows) is not int or self.draft_vocab_rows not in (0, 8192, 16384, 32768):
            raise ValueError("Draft vocabulary rows must be 0, 8192, 16384 or 32768")
        if any(type(x) is not int or x not in (0, 1)
               for x in (self.gdn_chain, self.gdn_journal, self.gdn_k_only)):
            raise ValueError("GDN controls must be integer 0 or 1")
        if self.gdn_journal and not self.gdn_chain:
            raise ValueError("GDN journal requires GDN chain")
        if self.gdn_k_only and not self.gdn_journal:
            raise ValueError("GDN K-only requires GDN journal")
        if self.panels not in PANEL_POLICIES:
            raise ValueError("Invalid CPU panel policy")

    def key(self) -> str:
        data = json.dumps(dataclasses.asdict(self), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def cpu_reference(self) -> Case:
        return dataclasses.replace(self, profile="cpu", speculation="none",
                                   draft_tokens=4, prompt_on="cpu",
                                   drafts_on="cpu", reply_on="cpu",
                                   spec_planner="legacy", spec_explore=20,
                                   draft_vocab="legacy", draft_vocab_rows=0,
                                   hybrid_policy="legacy", lookup_drafts=8, mtp_drafts=4, hybrid_probe_percent=2,
                                    s2_exact="off", s2_branches=2, s2_max_nodes=8, s2_min_gain_percent=5,
                                   gdn_chain=0, gdn_journal=0, gdn_k_only=0,
                                   panels="auto")


def csv_values(value: str, convert: Any = str) -> list[Any]:
    return list(dict.fromkeys(convert(x.strip()) for x in value.split(",") if x.strip()))


def unique(cases: list[Case]) -> list[Case]:
    return list(dict.fromkeys(cases))


def screen_cases(args: argparse.Namespace) -> list[Case]:
    cases = []
    for profile, spec in itertools.product(args.profiles, args.speculations):
        p, d, r = PROFILES[profile]
        ceiling = 4 if spec == "none" or 4 in args.draft_tokens else args.draft_tokens[0]
        cases.append(Case(profile, spec, max(args.threads), args.cache_gib[-1],
                          ceiling, prompt_on=p, drafts_on=d, reply_on=r,
                          prefetch=getattr(args, "prefetch", "auto"),
                          gpu_min_tokens=getattr(args, "gpu_min_tokens", 0),
                          npu_min_tokens=getattr(args, "npu_min_tokens", 0),
                           npu_min_macs=getattr(args, "npu_min_macs", 0),
                           **{name: getattr(args, name, field.default)
                              for name, field in Case.__dataclass_fields__.items()
                              if name in ("spec_planner", "spec_explore", "draft_vocab",
                                          "draft_vocab_rows", "hybrid_policy", "lookup_drafts", "mtp_drafts", "hybrid_probe_percent", "s2_exact", "s2_branches", "s2_max_nodes", "s2_min_gain_percent", "gdn_chain", "gdn_journal",
                                          "gdn_k_only", "panels")}))
    return with_references(cases)


def with_references(cases: list[Case]) -> list[Case]:
    # Baselines precede variants even when the caller requests only accelerators.
    return unique([x for c in cases for x in (c.cpu_reference(), c)])


def refinement_cases(args: argparse.Namespace, winners: list[Case], n_prompts: int) -> list[Case]:
    cases: list[Case] = []
    for winner in winners:
        changes = [winner]
        changes += [dataclasses.replace(winner, threads=n) for n in args.threads]
        changes += [dataclasses.replace(winner, cache_gib=n) for n in args.cache_gib]
        if winner.speculation != "none":
            changes += [dataclasses.replace(winner, draft_tokens=n) for n in args.draft_tokens]
        choices = ["cpu"]
        if "gpu" in winner.profile:
            choices.append("gpu")
        if "npu" in winner.profile:
            choices.append("npu")
        if "gpu" in winner.profile and "npu" in winner.profile:
            choices.append("npu+gpu")
        # One factor at a time, rather than a large Cartesian product.
        for phase in ("prompt_on", "drafts_on", "reply_on"):
            changes += [dataclasses.replace(winner, **{phase: v}) for v in choices]
        for c in unique(changes):
            cases.extend(dataclasses.replace(c, prompt=i) for i in range(n_prompts))
    return with_references(cases)


def identity(path: Path) -> dict[str, Any]:
    st = path.stat()
    with path.open("rb") as f:
        header = hashlib.sha256(f.read(1 << 20)).hexdigest()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "first_mib_sha256": header}


def unchanged(path: Path, frozen: dict[str, Any]) -> None:
    st = path.stat()
    if (st.st_size, st.st_mtime_ns) != (frozen["size"], frozen["mtime_ns"]):
        raise RuntimeError(f"Model changed during the benchmark: {path}")


def validate_cli(bench: Path) -> str:
    r = subprocess.run([str(bench), "--help"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=30)
    help_text = r.stdout + r.stderr
    missing = sorted(x for x in REQUIRED_FLAGS
                     if not re.search(r"(?<![\w-])" + re.escape(x) + r"(?![\w-])", help_text))
    if missing:
        raise RuntimeError("Rebuild janas-bench; required CLI flags missing: " + ", ".join(missing))
    return help_text


def execution_evidence(samples: list[dict[str, Any]], gpu: bool, npu: bool) -> dict[str, Any]:
    """Completed inference work, excluding plan creation and device discovery."""
    evidence = []
    for sample in samples:
        counters = sample.get("accelerator_counters", {})
        gpu_ran = any(counters.get(k, 0) > 0 for k in ("gpu_products", "gpu_whole_passes"))
        npu_ran = any(counters.get(k, 0) > 0 for k in ("npu_products", "npu_ffns", "npu_layer_passes"))
        evidence.append({"phase": sample.get("phase"), "index": sample.get("index"),
                         "gpu": gpu_ran, "npu": npu_ran})
    return {"verified": bool(samples) and all((not gpu or x["gpu"]) and
                                             (not npu or x["npu"]) for x in evidence),
            "scope": "completed work per sample; does not attribute work to generation vs prompt",
            "samples": evidence}


def run_case(args: argparse.Namespace, case: Case, prompt_path: Path,
             directory: Path, stage: str) -> dict[str, Any]:
    key = case.key()
    cache_root = directory / "cache" / key
    cache_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Remove inherited performance choices while retaining SDK/runtime paths.
    for name in ("JANAS_KERNELS", "JANAS_EXPERTS", "JANAS_GPU_ALWAYS", "JANAS_NPU_ALWAYS",
                 "JANAS_BLOCK_THREADS", "JANAS_GPU_TOKEN", "JANAS_GPU_WHOLE",
                 "JANAS_GPU_FAST_PROMPT", "JANAS_GPU_EXPERT_SHARE",
                 "JANAS_GPU_NV", "JANAS_GPU_MIN_N", "JANAS_NPU_MIN_N", "JANAS_NPU_MIN_MACS", "JANAS_PREFETCH",
                 "JANAS_PREFETCH_PREFILL", "JANAS_KV", "JANAS_ATTN", "JANAS_CPU_TILES",
                 *EXPERIMENT_ENV):
        env.pop(name, None)
    env.update(PREFETCH_POLICIES[case.prefetch])
    env.update(PANEL_POLICIES[case.panels])
    env.update({"JANAS_SPEC_PLANNER": case.spec_planner,
                "JANAS_SPEC_EXPLORE": str(case.spec_explore),
                "JANAS_DRAFT_VOCAB_POLICY": case.draft_vocab,
                "JANAS_DRAFT_VOCAB_ROWS": str(case.draft_vocab_rows),
                "JANAS_HYBRID_POLICY": case.hybrid_policy,
                "JANAS_LOOKUP_DRAFTS": str(case.lookup_drafts),
                "JANAS_MTP_DRAFTS": str(case.mtp_drafts),
                "JANAS_HYBRID_PROBE_PERCENT": str(case.hybrid_probe_percent),
                "JANAS_S2_EXACT": case.s2_exact, "JANAS_S2_BRANCHES": str(case.s2_branches),
                "JANAS_S2_MAX_NODES": str(case.s2_max_nodes), "JANAS_S2_MIN_GAIN_PERCENT": str(case.s2_min_gain_percent),
                "JANAS_GDN_CHAIN": str(case.gdn_chain),
                "JANAS_GDN_JOURNAL": str(case.gdn_journal),
                "JANAS_GDN_K_ONLY": str(case.gdn_k_only)})
    for name, value in (("JANAS_GPU_MIN_N", case.gpu_min_tokens),
                        ("JANAS_NPU_MIN_N", case.npu_min_tokens)):
        if type(value) is not int or not 0 <= value <= 256:
            raise ValueError("Minimum accelerator tokens must be integers in 0..256")
        if value:
            env[name] = str(value)
    if type(case.npu_min_macs) is not int or not 0 <= case.npu_min_macs <= (1 << 63) - 1:
        raise ValueError("Minimum Hexagon MACs must be an integer in 0..2^63-1")
    if case.npu_min_macs:
        env["JANAS_NPU_MIN_MACS"] = str(case.npu_min_macs)
    gpu = "gpu" in case.profile
    npu = "npu" in case.profile
    env.update({"JANAS_GPU": str(int(gpu)), "JANAS_NPU": str(int(npu)),
                "JANAS_GPU_DEVICE": "0", "JANAS_GPU_SG": "64",
                "JANAS_TUNE": "0", "JANAS_TUNE_THREADS": "0", "JANAS_WARM": "0",
                "JANAS_MISS_SKIP": "0", "JANAS_EXPERT_BITS": "6", "JANAS_HEAD_BITS": "0",
                "XDG_CACHE_HOME": cache_root.as_posix()})
    command = [str(args.bench), str(args.model), "--speculation", case.speculation,
               "--draft-tokens", str(case.draft_tokens), "--seed", str(args.seed),
               "--threads", str(case.threads), "--prompt-file", str(prompt_path),
               "--gen", str(args.gen), "--ctx", str(args.ctx), "--rounds", str(args.rounds),
               "--warmups", str(args.warmups), "--cache", str(case.cache_gib),
               "--reserve", str(args.reserve_gib), "--prompt-on", case.prompt_on,
               "--drafts-on", case.drafts_on, "--reply-on", case.reply_on,
               "--spec-planner", case.spec_planner, "--spec-explore", str(case.spec_explore),
               "--draft-vocab", case.draft_vocab, "--draft-vocab-rows", str(case.draft_vocab_rows),
               "--hybrid-policy", case.hybrid_policy, "--lookup-drafts", str(case.lookup_drafts),
               "--mtp-drafts", str(case.mtp_drafts), "--hybrid-probe-percent", str(case.hybrid_probe_percent), "--json"]
    if case.s2_exact == "on":
        command += ["--s2-exact", "on", "--s2-branches", str(case.s2_branches),
                    "--s2-max-nodes", str(case.s2_max_nodes),
                    "--s2-min-gain-percent", str(case.s2_min_gain_percent)]
    if case.speculation in ("mtp", "auto", "hybrid"):
        command += ["--mtp", str(args.mtp)]
    else:
        command.append("--no-mtp")
    t0 = time.perf_counter()
    result: dict[str, Any] = {"stage": stage, "key": key, "case": dataclasses.asdict(case),
                              "command": command, "records": [], "status": "failed",
                              "performance_environment": {name: env.get(name) for name in RECORDED_PERFORMANCE_ENV},
                              "comparison_prompt_ids": [0] if stage == "screen" else list(range(args.n_prompts))}
    try:
        r = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
        result["exit_code"] = r.returncode
        (directory / f"{stage}-{key}.stdout.log").write_text(r.stdout, encoding="utf-8")
        (directory / f"{stage}-{key}.stderr.log").write_text(r.stderr, encoding="utf-8")
        for line in r.stdout.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("schema") == 1:
                result["records"].append(value)
        setup = [x for x in result["records"] if x.get("kind") == "setup"]
        samples = [x for x in result["records"] if x.get("kind") == "sample"]
        expected = 1 + args.warmups + args.rounds
        if r.returncode:
            result["error"] = r.stderr[-4000:]
        elif len(setup) != 1 or len(samples) != expected:
            result["error"] = f"Incomplete benchmark output: {len(samples)}/{expected} samples"
        elif (gpu and not setup[0].get("gpu")) or (npu and not setup[0].get("npu")):
            result["error"] = "Requested accelerator is unavailable; CPU fallback is not that hardware case"
        elif any(x.get("output_tokens", 0) <= 0 for x in samples):
            result["error"] = "A sample produced no output tokens"
        else:
            warm = [x for x in samples if x["phase"] == "warm"]
            result["status"] = "ok"
            result["warm_median_tokens_per_second"] = statistics.median(
                x["output_tokens_per_second"] for x in warm)
            result["output_signature"] = [warm[0]["output_hash"], warm[0]["output_tokens"]]
            result["repeatable_output"] = len({(x["output_hash"], x["output_tokens"])
                                                for x in samples}) == 1
            evidence = execution_evidence(samples, gpu, npu)
            result["execution_evidence"] = evidence
            result["accelerator_execution_verified"] = evidence["verified"]
    except subprocess.TimeoutExpired as exc:
        result["error"] = f"Case exceeded {args.timeout}s"
        for name, text in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            if text:
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="replace")
                (directory / f"{stage}-{key}.{name}.log").write_text(text, encoding="utf-8")
    result["process_wall_seconds"] = time.perf_counter() - t0
    return result


def annotate(rows: list[dict[str, Any]]) -> None:
    refs = {Case(**r["case"]).key(): r["output_signature"] for r in rows
            if r.get("status") == "ok" and r["case"]["profile"] == "cpu"
            and r["case"]["speculation"] == "none"
            and r.get("repeatable_output", True) is True}
    for row in rows:
        if row.get("status") != "ok":
            continue
        ref = refs.get(Case(**row["case"]).cpu_reference().key())
        row["matches_cpu_reference"] = None if ref is None else row["output_signature"] == ref
        exact = row["matches_cpu_reference"] is True and row.get("repeatable_output", True)
        cpu = row["case"]["profile"] == "cpu"
        row["quality_validation"] = ("exact_cpu_match" if exact else
            "unverified_no_cpu_reference" if ref is None else
            "cpu_output_mismatch" if cpu else "unverified_accelerator_output_difference")
        row["recommendation_eligible"] = exact and (cpu or row.get("accelerator_execution_verified") is True)
        # Floating-point accelerator output differences need separate quality
        # validation; they remain candidates for subsequent discovery runs.
        row["discovery_eligible"] = exact or not cpu


def best(rows: list[dict[str, Any]], n: int, matching_only: bool = False) -> list[Case]:
    ranked = configurations(rows)
    eligible = "recommendation_eligible" if matching_only else "discovery_eligible"
    return [Case(**r["case"]) for r in ranked if r[eligible]][:n]


def configurations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare hardware/settings across the same complete set of prompts."""
    annotate(rows)
    prompts: set[int] = set()
    groups: dict[Case, dict[int, dict[str, Any]]] = {}
    for row in rows:
        case = Case(**row["case"])
        prompts.update(row.get("comparison_prompt_ids", [case.prompt]))
        groups.setdefault(dataclasses.replace(case, prompt=0), {})[case.prompt] = row
    output = []
    for case, group in groups.items():
        good = [r for r in group.values() if r.get("status") == "ok"]
        if {r["case"]["prompt"] for r in good} != prompts:
            continue
        rates = [r["warm_median_tokens_per_second"] for r in good]
        if not rates or any(not math.isfinite(x) or x <= 0 for x in rates):
            continue
        output.append({"stage": "refine" if any(r.get("stage") == "refine" for r in good) else "screen",
                       "case": dataclasses.asdict(case),
                       "prompt_count": len(good),
                       "warm_geomean_tokens_per_second": math.exp(statistics.mean(math.log(x) for x in rates)),
                       "matches_cpu_reference": all(r.get("matches_cpu_reference") is True for r in good),
                       "quality_validation": "exact_cpu_match" if all(r["recommendation_eligible"] for r in good)
                                             else "quality_unverified",
                       "recommendation_eligible": all(r["recommendation_eligible"] for r in good),
                       "discovery_eligible": all(r["discovery_eligible"] for r in good)})
    return sorted(output, key=lambda r: r["warm_geomean_tokens_per_second"], reverse=True)


def save_summary(directory: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    annotate(rows)
    ranked = sorted((r for r in rows if r.get("status") == "ok"),
                    key=lambda r: r["warm_median_tokens_per_second"], reverse=True)
    summary = {"measured_cases": len(rows), "successful_cases": len(ranked),
               "os_file_cache": "uncontrolled; fresh-process is not a cold-SSD claim",
               "accelerator_execution": "successful backend inference counters per sample; unavailable on older binaries; phase attribution requires a controlled phase-only run",
               "recommendations_require_matching_cpu_output": True,
               "quality_note": "Accelerator output differences are quality-unverified; they do not establish quality loss. Validate fixed-input logits/perplexity before recommending them.",
               "best_discovery_cases": [dataclasses.asdict(x) for x in best(rows, args.top)],
               "best_matching_cases": [dataclasses.asdict(x) for x in best(rows, args.top, matching_only=True)],
               "ranked_complete_configurations": configurations(rows),
               "ranked": ranked}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("screen", "refine", "both"), default="screen")
    p.add_argument("--bench", type=Path, default=ROOT / "bin/aarch64-windows/janas-bench.exe")
    p.add_argument("--model", type=Path, default=ROOT / "models/qwen3-next-80b-a3b-q4km.jns")
    p.add_argument("--mtp", type=Path, default=ROOT / "models/qwen3-next-80b-a3b-mtp-q4k.jns")
    p.add_argument("--output", type=Path)
    p.add_argument("--previous", type=Path, help="results.jsonl for refinement")
    p.add_argument("--prompt-file", type=Path, action="append", default=[])
    p.add_argument("--profiles", default=",".join(PROFILES))
    p.add_argument("--speculations", default="none,lookup,mtp,auto")
    p.add_argument("--threads", default="4,8,12")
    p.add_argument("--cache-gib", default="8,24,0", help="last value is used for screening; 0 is auto")
    p.add_argument("--draft-tokens", default="2,4,8")
    p.add_argument("--spec-planner", choices=("legacy", "cycle", "fixed"), default="legacy")
    p.add_argument("--s2-exact", choices=("off", "on"), default="off")
    p.add_argument("--s2-branches", type=int, choices=range(1, 3), default=2)
    p.add_argument("--s2-max-nodes", type=int, choices=range(1, 33), default=8)
    p.add_argument("--s2-min-gain-percent", type=int, choices=range(101), default=5)
    p.add_argument("--hybrid-policy", choices=("legacy", "cost"), default="legacy")
    p.add_argument("--lookup-drafts", type=int, choices=range(1, 17), default=8)
    p.add_argument("--mtp-drafts", type=int, choices=range(1, 17), default=4)
    p.add_argument("--hybrid-probe-percent", type=int, choices=range(0, 101), default=2)
    p.add_argument("--spec-explore", type=int, default=20, help="planner exploration percent 0..100")
    p.add_argument("--draft-vocab", choices=("legacy", "adaptive"), default="legacy")
    p.add_argument("--draft-vocab-rows", type=int, choices=(0, 8192, 16384, 32768), default=0)
    for name in ("chain", "journal", "k-only"):
        p.add_argument("--gdn-" + name, type=int, choices=(0, 1), default=0)
    p.add_argument("--panels", choices=tuple(PANEL_POLICIES), default="auto",
                   help="CPU tiles: existing auto policy, disabled, or experimental prepared panels")
    p.add_argument("--prefetch", choices=tuple(PREFETCH_POLICIES), default="auto",
                   help="screen policy; refinement retains each winning case's policy (default auto)")
    p.add_argument("--gpu-min-tokens", type=int, default=0,
                   help="manual screen threshold 0..256; 0 clears the override; refinement retains it")
    p.add_argument("--npu-min-tokens", type=int, default=0,
                   help="manual screen threshold 0..256; 0 clears the override; refinement retains it")
    p.add_argument("--npu-min-macs", type=int, default=0,
                   help="minimum rows*columns*batch for Hexagon; 0 clears the optional work filter")
    p.add_argument("--reserve-gib", type=float, default=12)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--cooldown", type=float, default=5)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--max-cases", type=int, default=32)
    p.add_argument("--plan", action="store_true", help="print cases without loading a model")
    return p


def main() -> int:
    args = parser().parse_args()
    args.profiles = csv_values(args.profiles)
    args.speculations = csv_values(args.speculations)
    args.threads = csv_values(args.threads, int)
    args.cache_gib = csv_values(args.cache_gib, float)
    args.draft_tokens = csv_values(args.draft_tokens, int)
    if (not args.profiles or any(x not in PROFILES for x in args.profiles)
        or not args.speculations or any(x not in ("none", "lookup", "mtp", "auto", "hybrid") for x in args.speculations)
        or not args.threads or any(x < 1 or x > 256 for x in args.threads)
        or not args.cache_gib or any(not math.isfinite(x) or x < 0 for x in args.cache_gib)
        or not args.draft_tokens or any(x < 1 or x > 16 for x in args.draft_tokens)
        or not 0 <= args.gpu_min_tokens <= 256 or not 0 <= args.npu_min_tokens <= 256
        or not 0 <= args.npu_min_macs <= (1 << 63) - 1
        or not 0 <= args.spec_explore <= 100
        or (args.gdn_journal and not args.gdn_chain) or (args.gdn_k_only and not args.gdn_journal)
        or args.gen < 1 or args.gen > (1 << 31) - 1 or args.ctx <= args.gen
        or args.rounds < 1 or args.rounds > 1000 or args.warmups < 0 or args.warmups > 1000
        or args.top < 1 or args.max_cases < 1 or args.seed < 1 or args.seed > (1 << 63) - 1
        or not math.isfinite(args.reserve_gib) or args.reserve_gib < 0
        or not math.isfinite(args.cooldown) or args.cooldown < 0
        or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("Invalid benchmark settings")
    prompts = [(p.stem, p.read_text(encoding="utf-8")) for p in args.prompt_file] or PROMPTS
    args.n_prompts = len(prompts)
    previous = []
    if args.previous:
        previous = [json.loads(x) for x in args.previous.read_text(encoding="utf-8").splitlines() if x.strip()]
    cases = screen_cases(args) if args.stage != "refine" else refinement_cases(args, best(previous, args.top), len(prompts))
    if args.stage == "refine" and not cases:
        raise ValueError("Refinement needs --previous results with successful complete configurations")
    if args.plan:
        print(json.dumps({"stage": args.stage, "sequential": True,
                          "screen_prompt": prompts[0][0], "refinement_prompts": [x[0] for x in prompts],
                          "cases": [dataclasses.asdict(x) for x in cases[:args.max_cases]],
                          "candidate_cases": len(cases), "maximum_cases": args.max_cases,
                          "fresh_process_os_cache": "uncontrolled",
                          "refinement": "one-factor changes on top discovery cases, retaining quality-unverified accelerators; no full Cartesian product"}, indent=2))
        return 0
    if os.name != "nt" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Run with native ARM64 Windows Python; --plan is available on any host")
    for name in ("bench", "model", "mtp"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.bench.is_file() or not args.model.is_file():
        raise FileNotFoundError("The native benchmark executable and complete model must exist")
    if any(x in ("mtp", "auto", "hybrid") for x in args.speculations) and not args.mtp.is_file():
        raise FileNotFoundError("The requested MTP/auto/hybrid cases require the MTP sidecar")
    help_text = validate_cli(args.bench)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))) / "Janas/benchmarks"
    directory = (args.output or base / stamp).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    frozen = identity(args.model)
    mtp_frozen = identity(args.mtp) if args.mtp.is_file() else None
    prompt_paths = []
    manifest_prompts = []
    for i, (name, text) in enumerate(prompts):
        encoded = text.encode("utf-8")
        path = directory / f"prompt-{i}.txt"
        path.write_bytes(encoded)
        prompt_paths.append(path)
        manifest_prompts.append({"name": name, "path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()})
    manifest = {"created": stamp, "python": sys.version, "machine": platform.machine(),
                "model": frozen, "mtp": mtp_frozen, "prompts": manifest_prompts,
                "bench_sha256": hashlib.sha256(args.bench.read_bytes()).hexdigest(),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model_hash_note": "first MiB hash plus fixed size/mtime; full-file hash omitted to avoid reading the full model before measuring",
                "quality": "high; original expert/head precision; no expert skipping",
                "performance_environment_by_case": {},
                "fresh_process": "no previous chat tokens or expected continuation supplied"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (directory / "cli-help.txt").write_text(help_text, encoding="utf-8")
    rows: list[dict[str, Any]] = []
    done: set[Case] = set()
    queue = list(cases)
    screen_done = False
    with (directory / "results.jsonl").open("w", encoding="utf-8") as output:
        while queue and len(rows) < args.max_cases:
            case = queue.pop(0)
            if case in done:
                continue
            done.add(case)
            unchanged(args.model, frozen)
            if mtp_frozen:
                unchanged(args.mtp, mtp_frozen)
            stage = "refine" if args.stage == "refine" or screen_done else "screen"
            print(f"[{len(rows) + 1}/{args.max_cases}] {stage}: {case}", flush=True)
            row = run_case(args, case, prompt_paths[case.prompt], directory, stage)
            unchanged(args.model, frozen)
            if mtp_frozen:
                unchanged(args.mtp, mtp_frozen)
            rows.append(row)
            manifest["performance_environment_by_case"][row["key"]] = row["performance_environment"]
            (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
            output.write(json.dumps(row) + "\n")
            output.flush()
            save_summary(directory, rows, args)
            if not queue and args.stage == "both" and not screen_done:
                screen_done = True
                queue = refinement_cases(args, best(rows, args.top), len(prompts))
            if queue and args.cooldown:
                time.sleep(args.cooldown)
    print(f"Results: {directory}", flush=True)
    return 0 if any(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"benchmark-snapdragon: {exc}", file=sys.stderr)
        raise SystemExit(2)

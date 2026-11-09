#!/usr/bin/env python3
"""Launch Janas with an explicitly selected Snapdragon profile.

Native ARM64 Windows Python is required to run an app. --dry-run checks a
profile and its files without starting an app. On other hosts, paths must
use that host's syntax or be project-relative; Windows paths are not mapped.
This tool never chooses a benchmark winner or claims accelerator execution.
Inherited JANAS_KV and JANAS_ATTN overrides are cleared to retain the model's
architecture defaults; HIGH preserves weight precision, not FP32 cache storage.
Inherited CPU tiling overrides are cleared before case.panels selects its policy.

Profile schema 1: model and optional mtp identities contain path, size and
first_mib_sha256 (mtime_ns is optional). case contains profile (cpu, cpu+gpu,
cpu+npu, cpu+gpu+npu), speculation (none, lookup, mtp, hybrid, auto), threads,
cache_gib, draft_tokens, prompt_on, drafts_on and reply_on. Optional case.prefetch
is auto (default), off, decode-off or prompt-off; inherited prefetch overrides
are cleared. Optional case.gpu_min_tokens and case.npu_min_tokens are manual
thresholds in 0..256; default 0 clears inherited overrides and uses the runtime
policy. Optional case.npu_min_macs is 0..2^63-1 (default 0); a positive value
filters Hexagon matrices by rows*columns*batch before preparing device weights.
Optional case.spec_planner is legacy/cycle/fixed (default legacy), spec_explore
is an integer in 0..100 (default 20), draft_vocab is legacy/adaptive (default
legacy), and draft_vocab_rows is 0/8192/16384/32768 (default 0). Optional
hybrid_policy is legacy/cost (default legacy); lookup_drafts and mtp_drafts
are independent 1..16 ceilings (defaults 8 and 4), clipped by draft_tokens.
hybrid_probe_percent is 0..100 (default 2). MTP and hybrid
modes require a frozen mtp identity. Optional integer gdn_chain, gdn_journal
and gdn_k_only are 0/1 (default 0); journal requires chain, and k_only requires
journal. Optional panels is auto/off/experimental (default auto), selecting
an unset/0/1 JANAS_CPU_TILES override. Controlled settings replace inherited
overrides and appear in the dry-run environment report. settings optionally
contains ctx, reserve_gib, seed, temperature, max_reply and preload. NPU
profiles require runtime.hexagon_root and may name host_bin and an
adsp_library_path list; defaults are <hexagon_root>/bin and /htp. Paths are
absolute or relative to the Janas project. Evidence is preserved in the
profile for review; model identity is checked at every launch.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"cpu": (False, False), "cpu+gpu": (True, False),
            "cpu+npu": (False, True), "cpu+gpu+npu": (True, True)}
DEVICES = {"cpu", "gpu", "npu", "npu+gpu"}
MODES = {"none", "lookup", "mtp", "hybrid", "auto"}
SPEC_PLANNERS = {"legacy", "cycle", "fixed"}
HYBRID_POLICIES = {"legacy", "cost"}
DRAFT_VOCAB_POLICIES = {"legacy", "adaptive"}
DRAFT_VOCAB_ROWS = {0, 8192, 16384, 32768}
PANEL_POLICIES = {"auto": None, "off": "0", "experimental": "1"}
PREFETCH_POLICIES = {"auto": {}, "off": {"JANAS_PREFETCH": "0", "JANAS_PREFETCH_PREFILL": "0"},
    "decode-off": {"JANAS_PREFETCH": "0"}, "prompt-off": {"JANAS_PREFETCH_PREFILL": "0"}}
PERFORMANCE_ENV = ("JANAS_KERNELS", "JANAS_EXPERTS", "JANAS_GPU_ALWAYS",
    "JANAS_NPU_ALWAYS", "JANAS_BLOCK_THREADS", "JANAS_GPU_TOKEN",
    "JANAS_GPU_WHOLE", "JANAS_GPU_FAST_PROMPT", "JANAS_GPU_EXPERT_SHARE",
    "JANAS_GPU_NV", "JANAS_GPU_MIN_N", "JANAS_NPU_MIN_N", "JANAS_NPU_MIN_MACS", "JANAS_PREFETCH", "JANAS_PREFETCH_PREFILL",
    "JANAS_KV", "JANAS_ATTN", "JANAS_CPU_TILES", "JANAS_SPEC_PLANNER",
    "JANAS_SPEC_EXPLORE", "JANAS_DRAFT_VOCAB_POLICY", "JANAS_DRAFT_VOCAB_ROWS",
    "JANAS_HYBRID_POLICY", "JANAS_LOOKUP_DRAFTS", "JANAS_MTP_DRAFTS", "JANAS_HYBRID_PROBE_PERCENT", "JANAS_S2_EXACT", "JANAS_S2_BRANCHES",
    "JANAS_S2_MAX_NODES", "JANAS_S2_MIN_GAIN_PERCENT",
    "JANAS_GDN_CHAIN", "JANAS_GDN_JOURNAL", "JANAS_GDN_K_ONLY")
PROFILE_ENV = {"JANAS_GPU", "JANAS_NPU", "JANAS_GPU_DEVICE", "JANAS_GPU_SG",
    "JANAS_TUNE", "JANAS_TUNE_THREADS", "JANAS_WARM", "JANAS_MISS_SKIP",
    "JANAS_EXPERT_BITS", "JANAS_HEAD_BITS", "JANAS_HEXAGON_ROOT", "ADSP_LIBRARY_PATH",
    "JANAS_PREFETCH", "JANAS_PREFETCH_PREFILL", "JANAS_GPU_MIN_N", "JANAS_NPU_MIN_N", "JANAS_NPU_MIN_MACS",
    "JANAS_KV", "JANAS_ATTN", "JANAS_CPU_TILES", "JANAS_SPEC_PLANNER",
    "JANAS_SPEC_EXPLORE", "JANAS_DRAFT_VOCAB_POLICY", "JANAS_DRAFT_VOCAB_ROWS",
    "JANAS_HYBRID_POLICY", "JANAS_LOOKUP_DRAFTS", "JANAS_MTP_DRAFTS", "JANAS_HYBRID_PROBE_PERCENT", "JANAS_S2_EXACT", "JANAS_S2_BRANCHES",
    "JANAS_S2_MAX_NODES", "JANAS_S2_MIN_GAIN_PERCENT",
    "JANAS_GDN_CHAIN", "JANAS_GDN_JOURNAL", "JANAS_GDN_K_ONLY"}


def number(value: Any, name: str, low: float, high: float, integer: bool = False) -> Any:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not low <= value <= high
        or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'} in {low}..{high}")
    return value


def project_path(value: Any, name: str, root: Path) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{name} must be a path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def model_identity(value: Any, name: str, root: Path) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a frozen model identity")
    path = project_path(value.get("path"), f"{name}.path", root)
    size = number(value.get("size"), f"{name}.size", 1, (1 << 63) - 1, True)
    expected = value.get("first_mib_sha256")
    if (not isinstance(expected, str) or len(expected) != 64
        or any(c not in "0123456789abcdef" for c in expected.lower())):
        raise ValueError(f"{name}.first_mib_sha256 must contain 64 hexadecimal characters")
    st = path.stat()
    if st.st_size != size:
        raise ValueError(f"{name} size differs from the selected profile: {path}")
    if "mtime_ns" in value:
        number(value["mtime_ns"], f"{name}.mtime_ns", 0, (1 << 63) - 1, True)
        if st.st_mtime_ns != value["mtime_ns"]:
            raise ValueError(f"{name} modification time differs from the selected profile: {path}")
    with path.open("rb") as f:
        actual = hashlib.sha256(f.read(1 << 20)).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"{name} header hash differs from the selected profile: {path}")
    return path


def load_profile(path: Path, root: Path = ROOT) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(profile, dict) or type(profile.get("schema")) is not int or profile["schema"] != 1:
        raise ValueError("A selected profile with schema 1 is required")
    model = model_identity(profile.get("model"), "model", root)
    mtp = model_identity(profile["mtp"], "mtp", root) if profile.get("mtp") is not None else None
    case = profile.get("case")
    if (not isinstance(case, dict) or not isinstance(case.get("profile"), str)
        or case["profile"] not in PROFILES or not isinstance(case.get("speculation"), str)
        or case["speculation"] not in MODES):
        raise ValueError("case must select a known hardware profile and speculation mode")
    case = case.copy()
    for key, default, choices in (("spec_planner", "legacy", SPEC_PLANNERS),
                                  ("draft_vocab", "legacy", DRAFT_VOCAB_POLICIES),
                                  ("hybrid_policy", "legacy", HYBRID_POLICIES),
                                   ("s2_exact", "off", {"off", "on"}),
                                  ("panels", "auto", PANEL_POLICIES)):
        case.setdefault(key, default)
        if not isinstance(case[key], str) or case[key] not in choices:
            raise ValueError(f"case.{key} must be one of: " + ", ".join(sorted(choices)))
    for key, default, low, high in (("lookup_drafts", 8, 1, 16),
                                      ("mtp_drafts", 4, 1, 16),
                                      ("hybrid_probe_percent", 2, 0, 100),
                                       ("s2_branches", 2, 1, 2), ("s2_max_nodes", 8, 1, 32),
                                       ("s2_min_gain_percent", 5, 0, 100)):
        case.setdefault(key, default)
        number(case[key], f"case.{key}", low, high, True)
    if case["s2_exact"] == "on" and case["speculation"] not in ("mtp", "auto", "hybrid"):
        raise ValueError("S2 exact requires MTP speculation")
    case.setdefault("spec_explore", 20)
    number(case["spec_explore"], "case.spec_explore", 0, 100, True)
    case.setdefault("draft_vocab_rows", 0)
    number(case["draft_vocab_rows"], "case.draft_vocab_rows", 0, 32768, True)
    if case["draft_vocab_rows"] not in DRAFT_VOCAB_ROWS:
        raise ValueError("case.draft_vocab_rows must be 0, 8192, 16384 or 32768")
    for key in ("gdn_chain", "gdn_journal", "gdn_k_only"):
        case.setdefault(key, 0)
        number(case[key], f"case.{key}", 0, 1, True)
    if case["gdn_journal"] and not case["gdn_chain"]:
        raise ValueError("case.gdn_journal requires case.gdn_chain=1")
    if case["gdn_k_only"] and not case["gdn_journal"]:
        raise ValueError("case.gdn_k_only requires case.gdn_journal=1")
    case.setdefault("prefetch", "auto")
    if not isinstance(case["prefetch"], str) or case["prefetch"] not in PREFETCH_POLICIES:
        raise ValueError("case.prefetch must be auto, off, decode-off or prompt-off")
    for key in ("gpu_min_tokens", "npu_min_tokens"):
        case.setdefault(key, 0)
        number(case[key], f"case.{key}", 0, 256, True)
    case.setdefault("npu_min_macs", 0)
    number(case["npu_min_macs"], "case.npu_min_macs", 0, (1 << 63) - 1, True)
    number(case.get("threads"), "case.threads", 1, 256, True)
    number(case.get("cache_gib"), "case.cache_gib", 0, 1 << 20)
    number(case.get("draft_tokens"), "case.draft_tokens", 1, 16, True)
    gpu, npu = PROFILES[case["profile"]]
    for phase in ("prompt_on", "drafts_on", "reply_on"):
        device = case.get(phase)
        if (not isinstance(device, str) or device not in DEVICES
            or ("gpu" in device and not gpu) or ("npu" in device and not npu)):
            raise ValueError(f"case.{phase} is incompatible with the hardware profile")
    if case["speculation"] in ("mtp", "hybrid") and mtp is None:
        raise ValueError("MTP and hybrid speculation require a frozen mtp identity")
    settings = profile.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    settings = {"ctx": 2048, "reserve_gib": 12, "seed": 20260926,
                "temperature": 0, "max_reply": 0, "preload": False, **settings}
    number(settings["ctx"], "settings.ctx", 1, (1 << 32) - 1, True)
    number(settings["reserve_gib"], "settings.reserve_gib", 0, 1 << 20)
    number(settings["seed"], "settings.seed", 1, (1 << 63) - 1, True)
    number(settings["temperature"], "settings.temperature", 0, 100)
    number(settings["max_reply"], "settings.max_reply", 0, (1 << 31) - 1, True)
    if type(settings["preload"]) is not bool:
        raise ValueError("settings.preload must be true or false")
    runtime = profile.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    runtime = runtime.copy()
    if npu:
        hx = project_path(runtime.get("hexagon_root"), "runtime.hexagon_root", root)
        host = project_path(runtime.get("host_bin", str(hx / "bin")), "runtime.host_bin", root)
        adsp = runtime.get("adsp_library_path", [str(hx / "htp")])
        if not isinstance(adsp, list) or not adsp:
            raise ValueError("runtime.adsp_library_path must be a nonempty list")
        adsp = [project_path(x, "runtime.adsp_library_path", root) for x in adsp]
        for directory in [hx, host, *adsp]:
            if not directory.is_dir():
                raise ValueError(f"Runtime directory is missing: {directory}")
        dll = hx if (hx / "ggml-base.dll").is_file() else hx / "bin"
        if not all((dll / name).is_file() for name in ("ggml-base.dll", "ggml-hexagon.dll")):
            raise ValueError("Hexagon runtime needs matching ggml-base.dll and ggml-hexagon.dll")
        runtime.update(hexagon_root=hx, host_bin=host, adsp_library_path=adsp)
    return {"source": profile, "model": model, "mtp": mtp, "case": case,
            "settings": settings, "runtime": runtime}


def launch_environment(profile: dict[str, Any], inherited: dict[str, str]) -> dict[str, str]:
    env = inherited.copy()
    for key in PERFORMANCE_ENV:
        env.pop(key, None)
    case = profile["case"]
    for key, name, default in (
            ("spec_planner", "JANAS_SPEC_PLANNER", "legacy"),
            ("spec_explore", "JANAS_SPEC_EXPLORE", 20),
            ("draft_vocab", "JANAS_DRAFT_VOCAB_POLICY", "legacy"),
            ("draft_vocab_rows", "JANAS_DRAFT_VOCAB_ROWS", 0),
            ("hybrid_policy", "JANAS_HYBRID_POLICY", "legacy"),
            ("lookup_drafts", "JANAS_LOOKUP_DRAFTS", 8),
            ("mtp_drafts", "JANAS_MTP_DRAFTS", 4),
            ("hybrid_probe_percent", "JANAS_HYBRID_PROBE_PERCENT", 2),
            ("s2_exact", "JANAS_S2_EXACT", "off"), ("s2_branches", "JANAS_S2_BRANCHES", 2),
            ("s2_max_nodes", "JANAS_S2_MAX_NODES", 8), ("s2_min_gain_percent", "JANAS_S2_MIN_GAIN_PERCENT", 5),
            ("gdn_chain", "JANAS_GDN_CHAIN", 0),
            ("gdn_journal", "JANAS_GDN_JOURNAL", 0),
            ("gdn_k_only", "JANAS_GDN_K_ONLY", 0)):
        env[name] = str(case.get(key, default))
    panel = PANEL_POLICIES[case.get("panels", "auto")]
    if panel is not None:
        env["JANAS_CPU_TILES"] = panel
    env.update(PREFETCH_POLICIES[profile["case"].get("prefetch", "auto")])
    for key, name in (("gpu_min_tokens", "JANAS_GPU_MIN_N"),
                      ("npu_min_tokens", "JANAS_NPU_MIN_N"),
                      ("npu_min_macs", "JANAS_NPU_MIN_MACS")):
        value = profile["case"].get(key, 0)
        if value:
            env[name] = str(value)
    gpu, npu = PROFILES[profile["case"]["profile"]]
    settings = profile["settings"]
    env.update(JANAS_GPU=str(int(gpu)), JANAS_NPU=str(int(npu)),
        JANAS_GPU_DEVICE="0", JANAS_GPU_SG="64", JANAS_TUNE="0",
        JANAS_TUNE_THREADS="0", JANAS_WARM=str(int(settings["preload"])),
        JANAS_MISS_SKIP="0", JANAS_EXPERT_BITS="6", JANAS_HEAD_BITS="0")
    if npu:
        runtime = profile["runtime"]
        env["JANAS_HEXAGON_ROOT"] = str(runtime["hexagon_root"])
        env["ADSP_LIBRARY_PATH"] = ";".join(str(x) for x in runtime["adsp_library_path"])
        env["PATH"] = str(runtime["host_bin"]) + os.pathsep + env.get("PATH", "")
    return env


def launch_command(profile: dict[str, Any], app: str, executable: Path,
                   host: str, port: int, test_page: bool) -> list[str]:
    c, s = profile["case"], profile["settings"]
    command = [str(executable), str(profile["model"]), "--quality", "high",
        "--mode", "max", "--cache", str(c["cache_gib"]), "--reserve", str(s["reserve_gib"]),
        "--ctx", str(s["ctx"]), "--speculation", c["speculation"],
        "--draft-tokens", str(c["draft_tokens"]),
        "--spec-planner", c.get("spec_planner", "legacy"),
        "--spec-explore", str(c.get("spec_explore", 20)),
        "--draft-vocab", c.get("draft_vocab", "legacy"),
        "--draft-vocab-rows", str(c.get("draft_vocab_rows", 0)),
        "--hybrid-policy", c.get("hybrid_policy", "legacy"),
        "--lookup-drafts", str(c.get("lookup_drafts", 8)),
        "--mtp-drafts", str(c.get("mtp_drafts", 4)),
        "--hybrid-probe-percent", str(c.get("hybrid_probe_percent", 2)),
        "--prompt-on", c["prompt_on"],
        "--drafts-on", c["drafts_on"], "--reply-on", c["reply_on"], "--think", "off"]
    if c.get("s2_exact", "off") == "on":
        command += ["--s2-exact", "on", "--s2-branches", str(c.get("s2_branches", 2)),
                    "--s2-max-nodes", str(c.get("s2_max_nodes", 8)),
                    "--s2-min-gain-percent", str(c.get("s2_min_gain_percent", 5))]
    if profile["mtp"] is not None and c["speculation"] in ("mtp", "hybrid", "auto"):
        command += ["--mtp", str(profile["mtp"])]
    else:
        command.append("--no-mtp")
    if app == "chat":
        command += ["--no-tune", "--bits", "6", "--head", "6", "--temp",
                    str(s["temperature"]), "--seed", str(s["seed"]), "--stats"]
        if not s["preload"]:
            command.append("--no-preload")
        if s["max_reply"]:
            command += ["--max", str(s["max_reply"])]
    else:
        command += ["--host", host, "--port", str(port)]
        if test_page:
            command.append("--test")
    return command


def validate_cli(executable: Path, env: dict[str, str]) -> None:
    result = subprocess.run([str(executable), "--help"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30, env=env)
    help_text = result.stdout + result.stderr
    required = ("--speculation", "--draft-tokens", "--quality", "--prompt-on",
                "--drafts-on", "--reply-on", "--spec-planner", "--spec-explore",
                "--draft-vocab", "--draft-vocab-rows")
    if env.get("JANAS_S2_EXACT") == "on":
        required += ("--s2-exact", "--s2-branches", "--s2-max-nodes", "--s2-min-gain-percent")
    missing = [x for x in required
               if not re.search(r"(?<![\w-])" + re.escape(x) + r"(?![\w-])", help_text)]
    if missing:
        raise ValueError("Rebuild the selected app; missing CLI options: " + ", ".join(missing))


def set_affinity(threads: int) -> tuple[Any, int]:
    """Restrict this launcher before CreateProcess; the child inherits the mask."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    handle = kernel.GetCurrentProcess()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
        raise OSError(ctypes.get_last_error(), "Cannot read process affinity")
    cpus = [1 << i for i in range(8 * ctypes.sizeof(ctypes.c_size_t)) if process.value & (1 << i)]
    if threads > len(cpus):
        raise ValueError(f"Profile requests {threads} CPUs, but only {len(cpus)} are allowed")
    if not kernel.SetProcessAffinityMask(handle, sum(cpus[:threads])):
        raise OSError(ctypes.get_last_error(), "Cannot apply selected process affinity")
    return kernel, process.value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", required=True, type=Path, help="explicit selected profile JSON")
    p.add_argument("--app", choices=("chat", "server"), default="chat")
    p.add_argument("--exe", type=Path, help="override the app executable")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--test", action="store_true", help="server: enable the local /test chat page")
    p.add_argument("--dry-run", action="store_true", help="check files and show the launch without running an app")
    return p


def main() -> int:
    args = parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be 1..65535")
    profile = load_profile(args.profile)
    executable = (args.exe or ROOT / f"bin/aarch64-windows/janas-{args.app}.exe").resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"App executable is missing: {executable}")
    command = launch_command(profile, args.app, executable, args.host, args.port, args.test)
    env = launch_environment(profile, os.environ)
    if args.dry_run:
        print(json.dumps({"profile": str(args.profile.resolve()), "command": command,
            "affinity_threads": profile["case"]["threads"], "affinity_scope": "whole_process",
            "environment": {k: v for k, v in env.items() if k in PROFILE_ENV},
            "runtime_path_prefix": str(profile["runtime"].get("host_bin", "")),
            "identity_check": "size, first MiB SHA256 and optional modification time; not a full-file hash",
            "server_sampling": "request parameters; settings temperature/seed apply to chat only",
            "accelerator_execution_verified": False}, indent=2))
        return 0
    if os.name != "nt" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise RuntimeError("Run with native ARM64 Windows Python; --dry-run is available on any host")
    kernel, previous = set_affinity(profile["case"]["threads"])
    try:
        validate_cli(executable, env)
        print(f"Selected profile: {args.profile.resolve()} ({args.app})", flush=True)
        return subprocess.call(command, cwd=ROOT, env=env)
    finally:
        kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), previous)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"run-snapdragon: {exc}", file=sys.stderr)
        raise SystemExit(2)

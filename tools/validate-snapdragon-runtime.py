"""Native Windows functional gates. Default --plan launches no executable.
Execution requires explicit --execute after coordinating exclusive device use.
Every execution uses a fresh output directory and runs one child at a time.
No throughput or cold-cache claims are inferred from these checks.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "lib" / "validation" / "runtime-not-executed"
BIN = ROOT / "bin" / "aarch64-windows"
MODEL = ROOT / "models" / "qwen3-next-80b-a3b-q4km.jns"
MTP = ROOT / "models" / "qwen3-next-80b-a3b-mtp-q4k.jns"
QREF = ROOT / "models" / "cpu64.qref"
CODE = ROOT / "tools" / "benchmark-prompts" / "qwen3-next-code.tokens.json"
RECEIPT = ROOT / "tools" / "snapdragon-runtime-models.json"
BASELINE_QUALITY_EXE = None
VERIFY_MODEL_HASHES = False
SEMANTIC_PROBE_PERCENT = 2
ATTENTION_CALIBRATION = None
EXPECTED_DIGEST = "c70a79fcb00b7e3d"
EXPECTED_QREF_SHA = "52b0f576b85e9df0d2ba24380ff974f836a0f467fdcb46bbf19a747535883390"
EXPECTED_INPUTS = {
    "reference32.int32": "5799d32b7782dea2fcb07fa4a9a3a72177e89ed2319171eb58d832fe42d8a83b",
    "reference32.f32": "562144a8813208685ef1bb57ebd618cabdb58249e5ddaa01bad7c56077b60d16",
    "session-code160.int32": "28591f32dd80b77235b409ee5ed70dea865df21818021c0711a0d0ae79399f61",
}
MODEL_PINS = {
    "main": (48411922432, "fa3255c73106391e9cb5a97efd30f77a928fb4518d40060f795a8cff6656daed"),
    "mtp": (937992192, "5c9bc4cb6f739193fac5e4daa30c922f73255acf41443ecfc0123ef084584559"),
}
BATCHES = (1, 3, 5, 8, 9, 17)
EVAL_PAIRS = ((5, 1), (9, 4), (17, 8))
SESSION_CASES = (
    ("legacy-mtp4", "legacy", "mtp", 4, "legacy", 0),
    ("cycle-mtp4", "cycle", "mtp", 4, "legacy", 0),
    ("hybrid-cycle8", "cycle", "hybrid", 8, "legacy", 0),
    ("hybrid-cycle8-adaptive", "cycle", "hybrid", 8, "adaptive", 0),
    ("fixed-mtp16-adaptive", "fixed", "mtp", 16, "adaptive", 16384),
    ("hybrid-cost8", "legacy", "hybrid", 8, "legacy", 0),
    ("hybrid-cost8-probe", "legacy", "hybrid", 8, "legacy", 0),
)
VOCAB = 151936
HARDWARE_HELPERS = {"test_npu_hexagon_check.exe", "test_hx_probe.exe"}


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def environment(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("JANAS_")}
    env.update(JANAS_GPU="0", JANAS_NPU="0", JANAS_TUNE="0", JANAS_TUNE_THREADS="0",
               JANAS_WARM="0", JANAS_EXPERT_BITS="6", JANAS_HEAD_BITS="0",
               JANAS_MISS_SKIP="0", JANAS_PREFETCH="0", JANAS_PREFETCH_PREFILL="0",
               JANAS_CPU_TILES="1", JANAS_GDN_CHAIN="1", JANAS_GDN_JOURNAL="1",
               JANAS_GDN_K_ONLY="1")
    if extra:
        env.update(extra)
    return env


def prepare_inputs():
    OUT.mkdir(parents=True, exist_ok=True)
    if sha(QREF) != EXPECTED_QREF_SHA:
        raise ValueError("original CPU QREF SHA-256 mismatch")
    data = QREF.read_bytes()
    if data[:8] != b"JANASQ01":
        raise ValueError("qref magic mismatch")
    one, nv, steps, context = struct.unpack_from("<4I", data, 8)
    payload = 64 + (steps + 1) * 4
    end = payload + steps * nv * 4
    if (one, nv, steps, context) != (0x3f800000, VOCAB, 64, 1024):
        raise ValueError("qref geometry/context mismatch")
    if len(data) != end + 16 or data[end:end + 8] != b"QREFDONE":
        raise ValueError("qref length/footer mismatch")
    if f"{struct.unpack_from('<Q', data, end + 8)[0]:016x}" != EXPECTED_DIGEST:
        raise ValueError("qref digest differs from locked CPU reference")
    tokens = data[64:64 + 32 * 4]
    logits = data[payload:payload + 32 * nv * 4]
    if CODE.suffix.lower() == ".json":
        corpus = json.loads(CODE.read_text(encoding="utf-8"))
        ids = corpus.get("tokens")
        if not isinstance(ids, list) or len(ids) < 160 or any(type(t) is not int for t in ids):
            raise ValueError("frozen token JSON has no valid 160-token prefix")
        code = struct.pack("<" + "i" * len(ids), *ids)
        if corpus.get("int32_le_sha256") and sha_bytes(code) != corpus["int32_le_sha256"]:
            raise ValueError("frozen token JSON payload checksum mismatch")
    else:
        code = CODE.read_bytes()
    if len(code) < 160 * 4 or len(code) % 4:
        raise ValueError("frozen code token input is incomplete")
    session = code[:160 * 4]
    for part in (tokens, session):
        if any(t < 0 or t >= nv for t in struct.unpack('<' + 'i' * (len(part) // 4), part)):
            raise ValueError("token outside vocabulary")
    for name, part in (("reference32.int32", tokens), ("reference32.f32", logits),
                       ("session-code160.int32", session)):
        if hashlib.sha256(part).hexdigest() != EXPECTED_INPUTS[name]:
            raise ValueError("locked extracted input differs: " + name)
        (OUT / name).write_bytes(part)
    manifest = {"source_qref": str(QREF), "source_qref_sha256": sha(QREF),
                "locked_logit_fnv1a64": EXPECTED_DIGEST,
                "qref_header_bytes": 64, "qref_tokens": 65,
                "logits_payload_offset": payload, "vocab": nv,
                "eval_positions": 32, "session_tokens": 160,
                "inputs": {name: sha(OUT / name) for name in
                           ("reference32.int32", "reference32.f32", "session-code160.int32")}}
    save(OUT / "inputs.json", manifest)
    return manifest


def attention_input(path):
    if path is None:
        return None
    path = path.resolve()
    data = path.read_bytes()
    if path.name != "attn.txt" or not data or len(data) > 1 << 20:
        raise ValueError("Attention calibration requires a nonempty attn.txt of at most 1 MiB")
    for line in data.decode("utf-8").splitlines():
        parts = line.split()
        if (len(parts) != 3 or len(parts[0].encode("utf-8")) > 399 or "|" not in parts[0] or
            any(not v.isascii() or not v.isdecimal() or int(v) > 0xffffffff for v in parts[1:])):
            raise ValueError("Invalid legacy attention calibration row")
    return {"source_path": str(path), "bytes": len(data), "sha256": sha_bytes(data),
            "payload_hex": data.hex()}


def seed_attention_calibration(cache, frozen):
    if frozen is None:
        return None
    if attention_input(Path(frozen["source_path"])) != frozen:
        raise RuntimeError("Original attention calibration changed after validation was planned")
    target = cache / "janas" / "attn.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(bytes.fromhex(frozen["payload_hex"]))
    return {"source_path": frozen["source_path"], "private_path": str(target),
            "sha256": frozen["sha256"], "bytes": frozen["bytes"], "original_mutated": False}


def run(name, argv, env, timeout=600, affinity=None, expected_exit=(0,)):
    argv = [str(x) for x in argv]
    if not Path(argv[0]).is_absolute():
        raise ValueError("argv[0] must be an absolute Windows executable path")
    env = dict(env)
    cache = OUT / "cache" / name
    cache.mkdir(parents=True, exist_ok=False)
    env["XDG_CACHE_HOME"] = cache.as_posix()
    attention_seed = seed_attention_calibration(cache, ATTENTION_CALIBRATION)
    result = {"name": name, "command": argv, "attention_calibration_seed": attention_seed, "timeout_seconds": timeout,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "binary_sha256": sha(argv[0]),
              "janas_environment": {k: v for k, v in env.items() if k.upper().startswith("JANAS_")},
              "process_affinity_mask": affinity, "expected_exit_codes": list(expected_exit),
              "XDG_CACHE_HOME": env["XDG_CACHE_HOME"],
              "stdout": str(OUT / (name + ".stdout.log")),
              "stderr": str(OUT / (name + ".stderr.log")),
              "scope": "functional validation; throughput not measured"}
    old_affinity = None
    kernel = None
    handle = None
    if affinity is not None:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        old_affinity = process_mask.value
        if affinity & ~system_mask.value or not kernel.SetProcessAffinityMask(handle, affinity):
            raise ctypes.WinError(ctypes.get_last_error())
    start = time.monotonic()
    try:
        with open(result["stdout"], "wb") as stdout, open(result["stderr"], "wb") as stderr:
            p = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                 stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            result["pid"] = p.pid
            try:
                result["exit_code"] = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                subprocess.run([str(Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe"),
                                "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=30)
                if p.poll() is None:
                    p.kill()
                result["exit_code"] = p.wait(timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        result["exit_code"] = None
        result["process_error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if old_affinity is not None:
            if not kernel.SetProcessAffinityMask(handle, old_affinity):
                result["affinity_restore_error"] = ctypes.get_last_error()
    result["wall_seconds"] = time.monotonic() - start
    result["passed"] = (result.get("exit_code") in expected_exit and
                        not result.get("timed_out", False) and
                        not result.get("affinity_restore_error"))
    save(OUT / (name + ".result.json"), result)
    print(f"{name}: exit={result.get('exit_code')} timeout={result.get('timed_out', False)}", flush=True)
    return result


def units():
    cases = sorted(p.resolve() for p in BIN.glob("test*.exe") if p.name.lower() not in HARDWARE_HELPERS)
    if not cases:
        raise RuntimeError("no native Windows unit executables found")
    results = []
    for p in cases:
        r = run("unit-" + p.stem, [p], environment())
        text = read_log(r)
        errors = []
        if not re.search(re.escape(p.stem) + r"[^\n]*(?:\bok\b|\bpassed\b|\bskipped\b)", text, re.I):
            errors.append("missing named unit success/explicit skip marker")
        if re.search(r"(?im)^\s*(?:FAIL(?:ED)?\b|AddressSanitizer:|runtime error:)", text):
            errors.append("unit emitted failure/sanitizer marker")
        r.update(validation_errors=errors, explicit_skip=bool(re.search(r"(?i)\bskipped\b", text)))
        r["passed"] = r["passed"] and not errors
        save(OUT / (r["name"] + ".result.json"), r)
        results.append(r)
    summary = {"stage": "units", "count": len(results),
               "passed": sum(r["passed"] for r in results), "results": results,
               "excluded_hardware_helpers": sorted(HARDWARE_HELPERS)}
    save(OUT / "units.json", summary)
    return all(r["passed"] for r in results)


def read_log(result):
    return "\n".join(Path(result[key]).read_text(encoding="utf-8", errors="replace")
                     for key in ("stdout", "stderr"))


def finite_json(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_json(v) for v in value)
    return True


def quality():
    results = []
    for batch in BATCHES:
        r = run(f"quality64-batch{batch}", [BIN / "quality_check.exe", MODEL,
                "--profile", "cpu", "--steps", "64", "--batch", str(batch),
                "--context", "1024", "--cache-mib", "24576", "--threads", "8",
                "--reference", QREF], environment())
        records = []
        parse_errors = []
        for line in Path(r["stdout"]).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if isinstance(v, dict) and "finite_logits" in v:
                if finite_json(v):
                    records.append(v)
                else:
                    parse_errors.append("quality record contains a nonfinite JSON number")
        errors = list(parse_errors)
        if len(records) != 1:
            errors.append("expected exactly one quality metrics record")
        else:
            m = records[0]
            checks = {"finite_logits": True, "hardware_execution_verified": True,
                      "positions": 64, "batch": batch,
                      "context": 1024, "cache_mib": 24576, "vocab": VOCAB, "threads": 8,
                      "profile": "cpu", "profile_effective": "cpu",
                      "expert_bits": 6, "min_quality": 2, "tuning": False,
                      "expert_skip": False, "head_requantized": False,
                      "gpu_selected": 0, "npu_selected": 0,
                      "comparison": "compared_to_cpu", "logit_fnv1a64": EXPECTED_DIGEST,
                      "top1_agreement": 1, "nll_delta": 0, "logit_max_abs": 0,
                      "logit_mae": 0, "logit_rmse": 0, "mean_kl_ref_to_candidate": 0}
            for key, want in checks.items():
                if key not in m or m[key] != want:
                    errors.append(f"{key}: expected {want!r}, got {m.get(key)!r}")
            for key in ("nll", "perplexity", "reference_nll", "reference_perplexity"):
                if not isinstance(m.get(key), (int, float)) or not math.isfinite(m[key]):
                    errors.append(f"nonfinite/missing {key}")
            counters = m.get("accelerator_counters", {})
            for key in ("npu_products", "npu_ffns", "npu_layer_passes",
                        "gpu_product_submissions", "gpu_products", "gpu_whole_passes"):
                if counters.get(key) != 0:
                    errors.append(f"CPU profile has missing/nonzero {key}")
        r.update(metrics=records, validation_errors=errors)
        r["passed"] = r["passed"] and not errors
        save(OUT / (r["name"] + ".result.json"), r)
        results.append(r)
        save(OUT / "quality.json", {"stage": "quality", "results": results, "passed": all(x["passed"] for x in results)})
        if not r["passed"]:
            return False
    return True


def evaluation():
    env = environment({"JANAS_ATTN": "float", "JANAS_HEAD_BITS": "6"})
    probe = run("eval-cli-capabilities", [BIN / "llm_eval.exe"], env, expected_exit=(2,))
    help_text = read_log(probe)
    missing = [flag for flag in ("block=", "reject=", "strict=") if flag not in help_text]
    probe["validation_errors"] = ["missing eval capability: " + flag for flag in missing]
    probe["passed"] = probe["passed"] and not missing
    save(OUT / (probe["name"] + ".result.json"), probe)
    if not probe["passed"]:
        return False
    results = []
    for block, reject in EVAL_PAIRS:
        r = run(f"eval32-block{block}-reject{reject}", [BIN / "llm_eval.exe", MODEL,
                OUT / "reference32.int32", OUT / "reference32.f32", "24576",
                "0,1,2,3,4,5,6,7", "8,9,10,11", f"block={block}", f"reject={reject}", "strict=1"], env)
        text = read_log(r)
        required = [f"blocks of {block}: 32/32 positions bit-identical to token by token, max |diff| 0;",
                    f"blocks of {block} keeping {block - reject}:",
                    "32/32 input positions covered",
                    "top-1 agreement 32/32, max |diff| 0.000, mean |diff| 0.0000"]
        errors = ["missing success marker: " + x for x in required if x not in text]
        r.update(validation_errors=errors, block=block, rejected_suffix=reject,
                 kept_prefix=block - reject,
                 comparison_note="eval context4096 vs original QREF1024; strict bytes required without tolerance")
        r["passed"] = r["passed"] and not errors
        save(OUT / (r["name"] + ".result.json"), r)
        results.append(r)
        save(OUT / "eval.json", {"stage": "eval", "capability_probe": probe, "results": results,
                                  "passed": all(x["passed"] for x in results)})
        if not r["passed"]:
            return False
    return True


def session_controls(name):
    if not name.startswith("hybrid-cost"):
        return []
    return ["hybrid_policy=cost", "lookup_drafts=8", "mtp_drafts=4",
            "hybrid_probe_percent=" + ("100" if name.endswith("-probe") else "2")]


def session():
    results = []
    for name, planner, mode, drafts, vocab, rows in SESSION_CASES:
        r = run("session-" + name, [BIN / "llm_session.exe", MODEL,
                OUT / "session-code160.int32", "mtp=" + str(MTP), "planner=" + planner,
                "speculation=" + mode, f"drafts={drafts}", "draft_vocab=" + vocab, f"rows={rows}"] + session_controls(name),
                environment({"JANAS_ATTN": "float"}), affinity=0xff)
        text = read_log(r)
        expected = ["greedy: turns vs all at once", "greedy + speculation: turn 1",
                    "greedy + speculation: turn 2", "sampling, speculation off vs on: turn 1",
                    "sampling, speculation off vs on: turn 2"]
        errors = ["missing successful comparison: " + value for value in expected
                  if not re.search(re.escape(value) + r"\s+identical", text)]
        if "llm_session: ok" not in text:
            errors.append("missing session success marker")
        if not re.search(r"sampling differs from greedy\s+yes", text):
            errors.append("sampling did not demonstrably differ from greedy")
        settings = re.search(r"^settings: (.+)$", text, re.M)
        if not settings or any(x not in settings.group(1) for x in
                (f"planner={planner}", f"speculation={mode}", f"drafts={drafts}",
                 f"draft_vocab={vocab}", f"rows={rows}", "quality=HIGH", "expert_bits=6", "head_bits=0")):
            errors.append("missing/mismatched resolved high-quality settings")
        records = []
        for line in text.splitlines():
            if line.startswith("runtime:"):
                fields = dict((k, int(v)) for k, v in re.findall(r"([a-z_]+)=(\d+)", line))
                fields["verified"] = line.endswith(" verified")
                records.append(fields)
        selected = [v for v in records if v.get("mode") == (4 if mode == "hybrid" else 2)]
        want_planner = {"legacy": 0, "cycle": 1, "fixed": 2}[planner]
        if len(records) != 4 or len(selected) != 2 or not all(v["verified"] for v in records):
            errors.append("expected four verified runtime records, including two selected-mode passes")
        for v in selected:
            if v.get("planner") != want_planner or v.get("cycles", 0) <= 0:
                errors.append("selected planner was not actually exercised")
            if mode == "hybrid" and v.get("lookup_attempts", 0) <= 0:
                errors.append("hybrid did not consult the history index")
        hybrid_records = []
        if name.startswith("hybrid-cost"):
            for line in text.splitlines():
                if line.startswith("hybrid:"):
                    fields = dict((k, int(v)) for k, v in re.findall(r"([a-z_]+)=(\d+)", line))
                    fields["verified"] = line.endswith(" verified")
                    hybrid_records.append(fields)
            if len(hybrid_records) != 2 or not all(v.get("verified") and v.get("decisions", 0) > 0 and
                                                   v.get("mtp", 0) > 0 for v in hybrid_records):
                errors.append("cost-aware hybrid selection/MTP fallback not proven")
        mtp_exercised = bool(selected) and all(v.get("mtp_drafts", 0) > 0 and
                                               v.get("head_calls", 0) > 0 for v in selected)
        adaptive_verified = vocab == "adaptive" and mtp_exercised and all(
            v.get("repacks", 0) > 0 and v.get("head_rows", 0) > 0 for v in selected)
        if mode == "mtp" and not mtp_exercised:
            errors.append("MTP proposals were not actually verified in both greedy and sampled runs")
        if name == "fixed-mtp16-adaptive":
            if not adaptive_verified:
                errors.append("fixed adaptive-head execution unverified")
            if not all(v.get("max_width", 0) == 17 for v in selected):
                errors.append("17-row target verification not proven by runtime counters")
        r.update(validation_errors=errors, runtime_records=records, hybrid_records=hybrid_records,
                 mtp_execution_verified=mtp_exercised, adaptive_execution_verified=adaptive_verified,
                 adaptive_note="not exercised or currently a legacy comparison epoch" if vocab == "adaptive" and not adaptive_verified else None)
        r["passed"] = r["passed"] and not errors
        save(OUT / (r["name"] + ".result.json"), r)
        results.append(r)
        save(OUT / "sessions.json", {"stage": "session", "results": results,
                                      "passed": all(x["passed"] for x in results)})
        if not r["passed"]:
            return False
    return True


def semantic_hybrid():
    name = "semantic-hybrid-copy-switch"
    r = run(name, [BIN / "llm_hybrid.exe", MODEL, MTP,
                  "probe=" + str(SEMANTIC_PROBE_PERCENT)],
            environment(), timeout=1800, affinity=0xff)
    records, errors = [], []
    for line in Path(r["stdout"]).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
            if not finite_json(value):
                raise ValueError("nonfinite semantic record")
            records.append(value)
        except ValueError as error:
            errors.append(str(error))
    samples = [v for v in records if v.get("kind") == "semantic"]
    summaries = [v for v in records if v.get("kind") == "summary"]
    if len(samples) != 36 or any(v.get("identity_verified") is not True for v in samples):
        errors.append("expected 36 token-and-text-identical multi-turn semantic samples")
    for sampler in ("greedy", "seeded"):
        for mode in ("plain", "mtp4", "hybrid-cost8"):
            chosen = [v for v in samples if v.get("sampler") == sampler and v.get("mode") == mode]
            if len(chosen) != 6 or [v.get("phase") for v in chosen] != ["copy"] * 4 + ["switch", "return"]:
                errors.append("semantic sampler/source/turn coverage missing")
    if len(summaries) != 1 or summaries[0].get("identity_verified") is not True:
        errors.append("missing successful semantic identity summary")
    elif summaries[0].get("lookup_exercised") is not True or summaries[0].get("fallback_exercised") is not True:
        errors.append("copy lookup and post-switch MTP fallback were not both exercised")
    if any(v.get("input_threads") != 8 or v.get("output_threads") != 8 for v in samples):
        errors.append("semantic compute settings differ from CPU8")
    r.update(validation_errors=errors, semantic_records=records,
             probe_percent=SEMANTIC_PROBE_PERCENT,
             performance_boundary="functional scenario; source order is not matched ABBA",
             forced_probe_functional_only=SEMANTIC_PROBE_PERCENT != 2)
    r["passed"] = r["passed"] and not errors
    save(OUT / "semantic-hybrid.json", r)
    return r["passed"]


def model_metadata():
    records = []
    for role, path in (("main", MODEL), ("mtp", MTP)):
        size, pinned_sha = MODEL_PINS[role]
        actual = path.stat().st_size
        if actual != size:
            raise ValueError("model size differs from pinned input: " + path.name)
        actual_sha = sha(path) if VERIFY_MODEL_HASHES else None
        if actual_sha is not None and actual_sha != pinned_sha:
            raise ValueError("model SHA-256 differs from receipt: " + path.name)
        records.append({"path": str(path), "size": actual, "receipt_sha256": pinned_sha,
                        "sha256_recomputed_by_runner": VERIFY_MODEL_HASHES, "actual_sha256": actual_sha})
    return records


def plan(stage):
    cases = []
    if stage in ("all", "units"):
        cases += [{"stage": "units", "command": [str(p)], "timeout_seconds": 600}
                  for p in sorted(BIN.glob("test*.exe")) if p.name.lower() not in HARDWARE_HELPERS]
    if stage in ("all", "quality"):
        cases += [{"stage": "quality", "batch": b, "positions": 64, "timeout_seconds": 600,
                   "command": [str(BIN / "quality_check.exe"), str(MODEL), "--profile", "cpu",
                       "--steps", "64", "--batch", str(b), "--context", "1024", "--cache-mib", "24576",
                       "--threads", "8", "--reference", str(QREF)]} for b in BATCHES]
    if stage in ("all", "eval"):
        cases += [{"stage": "eval", "block": b, "reject": r, "strict": True, "timeout_seconds": 600,
                   "environment_overrides": {"JANAS_ATTN": "float", "JANAS_HEAD_BITS": "6"},
                   "command": [str(BIN / "llm_eval.exe"), str(MODEL), str(OUT / "reference32.int32"),
                       str(OUT / "reference32.f32"), "24576", "0,1,2,3,4,5,6,7", "8,9,10,11",
                       f"block={b}", f"reject={r}", "strict=1"]}
                  for b, r in EVAL_PAIRS]
    if stage in ("all", "session"):
        cases += [{"stage": "session", "name": n, "planner": p, "speculation": m,
                   "drafts": d, "draft_vocab": v, "rows": r, "timeout_seconds": 600,
                   "environment_overrides": {"JANAS_ATTN": "float"}, "process_affinity_mask": 255,
                   "command": [str(BIN / "llm_session.exe"), str(MODEL), str(OUT / "session-code160.int32"),
                       "mtp=" + str(MTP), "planner=" + p, "speculation=" + m, f"drafts={d}",
                       "draft_vocab=" + v, f"rows={r}"] + session_controls(n)}
                  for n, p, m, d, v, r in SESSION_CASES]
    if stage in ("all", "hybrid"):
        cases.append({"stage": "hybrid", "name": "semantic-hybrid-copy-switch",
            "probe_percent": SEMANTIC_PROBE_PERCENT, "timeout_seconds": 1800,
            "process_affinity_mask": 255,
            "command": [str(BIN / "llm_hybrid.exe"), str(MODEL), str(MTP),
                        "probe=" + str(SEMANTIC_PROBE_PERCENT)]})
    return {"mode": "plan", "launches_executables": False, "output_directory": str(OUT),
            "sequential": True, "stage": stage, "cases": cases,
            "requested_janas_environment": {k: v for k, v in environment().items() if k.startswith("JANAS_")},
            "cleared_inherited_controls": "all JANAS_* before applying listed controls",
            "locked_reference_digest": EXPECTED_DIGEST, "qref_sha256": EXPECTED_QREF_SHA,
            "execution_requires": "native Windows ARM64 Python --execute after coordinating exclusive device use",
            "reference": str(QREF), "frozen_tokens": str(CODE), "receipt": str(RECEIPT),
            "reference_generation": "--create-reference reconstructs the locked CPU reference; its digest and SHA must match the receipt",
            "scope": "functional quality and correctness only; no performance claims"}


def load_receipt(path):
    def invalid(value):
        raise ValueError("nonfinite receipt value: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate receipt key: " + key)
            result[key] = value
        return result
    receipt = json.loads(Path(path).read_text(encoding="utf-8"),
                         parse_constant=invalid, object_pairs_hook=unique)
    if not isinstance(receipt, dict) or type(receipt.get("schema")) is not int or receipt["schema"] != 1:
        raise ValueError("unknown runtime receipt schema")
    if not isinstance(receipt.get("models"), dict):
        raise ValueError("runtime receipt needs model identities")
    for key in ("main", "mtp"):
        item = receipt["models"].get(key)
        if (not isinstance(item, dict) or not isinstance(item.get("filename"), str) or
            Path(item["filename"]).name != item["filename"] or
            type(item.get("bytes")) is not int or item["bytes"] <= 0 or
            not isinstance(item.get("sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("invalid model receipt: " + key)
    reference = receipt.get("quality_reference")
    if (not isinstance(reference, dict) or
        not isinstance(reference.get("sha256"), str) or
        not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) or
        not isinstance(reference.get("logit_fnv1a64"), str) or
        not re.fullmatch(r"[0-9a-f]{16}", reference["logit_fnv1a64"])):
        raise ValueError("invalid reference receipt")
    if (reference.get("vocab"), reference.get("positions"), reference.get("context")) != (151936, 64, 1024):
        raise ValueError("this protocol requires the 64-position Qwen3-Next reference")
    return receipt


def create_reference():
    if QREF.exists():
        raise ValueError("--create-reference will not overwrite an existing reference")
    QREF.parent.mkdir(parents=True, exist_ok=True)
    part = QREF.with_name(QREF.name + "." + OUT.name + ".part")
    if part.exists():
        raise ValueError("reference temporary path already exists")
    exe = BASELINE_QUALITY_EXE or BIN / "quality_check.exe"
    r = run("create-reference", [exe, MODEL, "--profile", "cpu", "--steps", "64",
            "--batch", "1", "--context", "1024", "--cache-mib", "24576", "--threads", "8",
            "--write-reference", part], environment({"JANAS_CPU_TILES": "0", "JANAS_GDN_CHAIN": "0",
                "JANAS_GDN_JOURNAL": "0", "JANAS_GDN_K_ONLY": "0"}))
    errors = []
    records = []
    for line in read_log(r).splitlines():
        try: value = json.loads(line)
        except ValueError: continue
        if isinstance(value, dict) and "finite_logits" in value and finite_json(value): records.append(value)
    if len(records) != 1 or any(records[0].get(k) != v for k, v in {
            "profile": "cpu", "profile_effective": "cpu", "finite_logits": True,
            "positions": 64, "batch": 1, "context": 1024, "threads": 8,
            "reference_written": True, "logit_fnv1a64": EXPECTED_DIGEST,
            "expert_bits": 6, "expert_skip": False, "head_requantized": False}.items()):
        errors.append("baseline quality output does not match the locked reference protocol")
    if not part.is_file() or sha(part) != EXPECTED_QREF_SHA:
        errors.append("generated reference SHA-256 differs from receipt")
    r.update(validation_errors=errors, reference=str(QREF), temporary_reference=str(part))
    r["passed"] = r["passed"] and not errors
    save(OUT / (r["name"] + ".result.json"), r)
    if r["passed"]:
        # Native Windows rename refuses to overwrite an existing destination.
        part.rename(QREF)
    return r["passed"]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("all", "units", "quality", "eval", "session", "hybrid"), default="all")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="default; list checks without launching anything")
    mode.add_argument("--execute", action="store_true", help="explicitly run sequential native Windows checks")
    result.add_argument("--out", "--output", dest="out", type=Path, help="NEW output directory; existing paths rejected")
    result.add_argument("--bin-dir", type=Path, default=ROOT / "bin/aarch64-windows")
    result.add_argument("--model", type=Path, help="main JNS file; default from model receipt under models/")
    result.add_argument("--mtp", type=Path, help="MTP JNS file; default from model receipt under models/")
    result.add_argument("--reference", type=Path, default=ROOT / "models/cpu64.qref")
    result.add_argument("--token-input", type=Path, default=ROOT / "tools/benchmark-prompts/qwen3-next-code.tokens.json")
    result.add_argument("--model-receipt", type=Path, default=ROOT / "tools/snapdragon-runtime-models.json")
    result.add_argument("--verify-model-sha256", action="store_true", help="stream and verify both large model files before checks")
    result.add_argument("--attention-calibration-file", type=Path,
                        help="optional frozen original attn.txt, copied identically into every child private cache")
    result.add_argument("--semantic-probe-percent", type=int, choices=range(101), default=2,
                        metavar="0..100", help="hybrid semantic check; default production 2, other values functional only")
    result.add_argument("--create-reference", action="store_true", help="create a missing CPU64 reference and require its locked digest/SHA")
    result.add_argument("--baseline-quality-exe", "--baseline-exe", dest="baseline_quality_exe", type=Path,
                        help="optional baseline quality_check executable for --create-reference")
    return result


def main():
    global OUT, BIN, MODEL, MTP, QREF, CODE, RECEIPT, EXPECTED_DIGEST, EXPECTED_QREF_SHA
    global MODEL_PINS, BASELINE_QUALITY_EXE, VERIFY_MODEL_HASHES, SEMANTIC_PROBE_PERCENT, ATTENTION_CALIBRATION
    cli = parser()
    args = cli.parse_args()
    RECEIPT = args.model_receipt.resolve()
    receipt = load_receipt(RECEIPT)
    BIN = args.bin_dir.resolve()
    MODEL = (args.model or ROOT / "models" / receipt["models"]["main"]["filename"]).resolve()
    MTP = (args.mtp or ROOT / "models" / receipt["models"]["mtp"]["filename"]).resolve()
    MODEL_PINS = {"main": (receipt["models"]["main"]["bytes"], receipt["models"]["main"]["sha256"]),
                  "mtp": (receipt["models"]["mtp"]["bytes"], receipt["models"]["mtp"]["sha256"])}
    QREF, CODE = args.reference.resolve(), args.token_input.resolve()
    EXPECTED_DIGEST = receipt["quality_reference"]["logit_fnv1a64"]
    EXPECTED_QREF_SHA = receipt["quality_reference"]["sha256"]
    BASELINE_QUALITY_EXE = args.baseline_quality_exe.resolve() if args.baseline_quality_exe else None
    VERIFY_MODEL_HASHES = args.verify_model_sha256
    SEMANTIC_PROBE_PERCENT = args.semantic_probe_percent
    ATTENTION_CALIBRATION = attention_input(args.attention_calibration_file)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    OUT = args.out.resolve() if args.out else ROOT / "lib/validation" / ("snapdragon-runtime-" + stamp + "-" + str(os.getpid()))
    if args.create_reference and args.stage == "units":
        cli.error("reference generation is a model check, incompatible with --stage units")
    planned = plan(args.stage)
    planned["create_reference"] = args.create_reference
    if args.create_reference:
        part = QREF.with_name(QREF.name + "." + OUT.name + ".part")
        planned["reference_creation_command"] = [str(BASELINE_QUALITY_EXE or BIN / "quality_check.exe"),
            str(MODEL), "--profile", "cpu", "--steps", "64", "--batch", "1", "--context", "1024",
            "--cache-mib", "24576", "--threads", "8", "--write-reference", str(part)]
        planned["reference_environment_overrides"] = dict.fromkeys(
            ("JANAS_CPU_TILES", "JANAS_GDN_CHAIN", "JANAS_GDN_JOURNAL", "JANAS_GDN_K_ONLY"), "0")
    planned["baseline_quality_exe"] = str(BASELINE_QUALITY_EXE or BIN / "quality_check.exe")
    planned["model_hash_verification_requested"] = args.verify_model_sha256
    if not args.execute:
        print(json.dumps(planned, indent=2, allow_nan=False))
        return 0
    if os.name != "nt":
        cli.error("use native Windows ARM64 Python; plans work on every platform")
    OUT.mkdir(parents=True, exist_ok=False)
    lock = OUT / "runner.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("ascii")); os.close(fd)
    summary = {"scope": "functional; sequential one-model-at-a-time", "stages": [],
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "runner_sha256": sha(__file__), "native_python": sys.executable,
               "receipt": receipt, "receipt_sha256": sha(RECEIPT),
               "pid": os.getpid(), "output_directory": str(OUT),
               "removed_inherited_janas_names": sorted(k for k in os.environ if k.upper().startswith("JANAS_")),
               "plan": planned}
    try:
        if args.stage != "units":
            summary["models"] = model_metadata()
            if args.create_reference:
                if not create_reference():
                    summary["passed"] = False
                    return 1
            if not QREF.is_file():
                raise ValueError("reference missing; supply --reference or explicitly use --create-reference")
            summary["inputs"] = prepare_inputs()
        stages = [("units", units), ("quality", quality), ("eval", evaluation), ("session", session), ("hybrid", semantic_hybrid)]
        for name, fn in stages:
            if args.stage not in ("all", name): continue
            passed = fn()
            summary["stages"].append({"name": name, "passed": passed})
            save(OUT / "summary.json", summary)
            if not passed:
                summary["passed"] = False
                return 1
        summary["passed"] = True
        return 0
    except BaseException as error:
        summary["passed"] = False
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if ATTENTION_CALIBRATION is not None:
            if attention_input(Path(ATTENTION_CALIBRATION["source_path"])) != ATTENTION_CALIBRATION:
                raise RuntimeError("Original attention calibration changed during validation")
            summary["attention_calibration_original_unchanged"] = True
        summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(OUT / "summary.json", summary)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

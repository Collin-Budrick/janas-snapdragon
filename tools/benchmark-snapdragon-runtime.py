"""Plan or run reproducible native Windows ARM64 CPU runtime comparisons.

Modes: screen (one fresh + two warm per case) or paired (ABBA, five warm
samples per binary). Default --plan launches nothing. Supply a baseline
executable or manifest; --execute also requires a new --output directory.

ABBA per workload gives five warm samples per binary, split 3+2 across fresh
processes: three workloads, twelve processes, forty-two samples. Raw JSON,
logs, wall times, TTFT and resource counters are retained. Fresh process does
not imply cold Windows/SSD cache. No physical bandwidth is inferred.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PINS_FILE = ROOT / "tools/snapdragon-runtime-models.json"
PROMPTS = {
    "code": (ROOT / "tools/benchmark-prompts/repetitive-code.txt", 276,
             "638c1b5dae4fa586cd58a81f5229bd966a4430eb223746f7981bbf5c12d69faf", "aad9221895bb6fe0"),
    "prose": (ROOT / "tools/benchmark-prompts/prose.txt", 63,
              "650e8dca566860dd4964384a7aea46a37be395e9ce7d1e5d58b534b3ac2704aa", "ba1412ff1b0e9c79"),
    "long": (ROOT / "tools/benchmark-prompts/longer-code.txt", 1304,
             "8f6be1b71a3a7f9fa789309ffd09588a837567f00d5f7fdcbc312baba7e6b79c", "aad9221895bb6fe0"),
}
DEFAULT_CONTROL = dict(speculation="mtp", draft_tokens=4, spec_planner="legacy",
    spec_explore=20, draft_vocab="legacy", draft_vocab_rows=0, gdn_chain=0,
    gdn_journal=0, gdn_k_only=0, panels="auto")
CASES = {
    "baseline": dict(DEFAULT_CONTROL),
    "legacy": dict(DEFAULT_CONTROL),
    "tiles": dict(DEFAULT_CONTROL, panels="experimental"),
    "gdn-journal": dict(DEFAULT_CONTROL, gdn_chain=1, gdn_journal=1),
    "gdn-k-only": dict(DEFAULT_CONTROL, gdn_chain=1, gdn_journal=1, gdn_k_only=1),
    "cycle-mtp4": dict(DEFAULT_CONTROL, spec_planner="cycle"),
    "hybrid4": dict(DEFAULT_CONTROL, speculation="hybrid", spec_planner="cycle"),
    "hybrid8": dict(DEFAULT_CONTROL, speculation="hybrid", spec_planner="cycle", draft_tokens=8),
    "hybrid8-adaptive": dict(DEFAULT_CONTROL, speculation="hybrid", spec_planner="cycle",
                            draft_tokens=8, draft_vocab="adaptive", draft_vocab_rows=0),
}
COMMON_ENV = {"JANAS_GPU": "0", "JANAS_NPU": "0", "JANAS_TUNE": "0",
    "JANAS_TUNE_THREADS": "0", "JANAS_WARM": "0", "JANAS_MISS_SKIP": "0",
    "JANAS_EXPERT_BITS": "6", "JANAS_HEAD_BITS": "0", "JANAS_PREFETCH": "0",
    "JANAS_PREFETCH_PREFILL": "0"}



def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def valid_control(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_CONTROL):
        raise ValueError("Candidate controls must be an object containing known control fields only")
    c = dict(DEFAULT_CONTROL, **value)
    if c["speculation"] not in ("none", "lookup", "mtp", "auto", "hybrid"):
        raise ValueError("Invalid speculation mode")
    if c["spec_planner"] not in ("legacy", "cycle", "fixed"):
        raise ValueError("Invalid planner")
    for key, low, high in (("draft_tokens", 1, 16), ("spec_explore", 0, 100)):
        if type(c[key]) is not int or not low <= c[key] <= high:
            raise ValueError("Invalid integer " + key)
    if c["draft_vocab"] not in ("legacy", "adaptive") or type(c["draft_vocab_rows"]) is not int or c["draft_vocab_rows"] not in (0, 8192, 16384, 32768):
        raise ValueError("Invalid draft vocabulary policy/budget")
    if any(type(c[k]) is not int or c[k] not in (0, 1) for k in ("gdn_chain", "gdn_journal", "gdn_k_only")):
        raise ValueError("GDN values must be integer 0/1")
    if (c["gdn_journal"] and not c["gdn_chain"]) or (c["gdn_k_only"] and not c["gdn_journal"]):
        raise ValueError("GDN journal requires chain; K-only requires journal")
    if c["panels"] not in ("auto", "off", "experimental"):
        raise ValueError("Invalid panels setting")
    return c


def performance_controls(c):
    controls = dict(COMMON_ENV, JANAS_SPEC_PLANNER=c["spec_planner"],
        JANAS_SPEC_EXPLORE=str(c["spec_explore"]), JANAS_DRAFT_VOCAB_POLICY=c["draft_vocab"],
        JANAS_DRAFT_VOCAB_ROWS=str(c["draft_vocab_rows"]), JANAS_GDN_CHAIN=str(c["gdn_chain"]),
        JANAS_GDN_JOURNAL=str(c["gdn_journal"]), JANAS_GDN_K_ONLY=str(c["gdn_k_only"]))
    if c["panels"] != "auto":
        controls["JANAS_CPU_TILES"] = "1" if c["panels"] == "experimental" else "0"
    return controls


def environment(cache, control, inherited=None):
    inherited = os.environ if inherited is None else inherited
    env = {k: v for k, v in inherited.items() if not k.upper().startswith("JANAS_")}
    controls = performance_controls(control)
    env.update(controls, XDG_CACHE_HOME=cache.as_posix())
    record = {"removed_inherited_key_names": sorted(k for k in inherited if k.upper().startswith("JANAS_")),
              "controls": controls, "cleared": {k: None for k in ("JANAS_KV", "JANAS_ATTN", "JANAS_KERNELS")},
              "JANAS_CPU_TILES": controls.get("JANAS_CPU_TILES"), "XDG_CACHE_HOME": cache.as_posix()}
    return env, record


def source_identity():
    files = [ROOT / "build.sh"]
    for folder in ("src", "include"):
        files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix.lower() in (".c", ".h", ".inc", ".s")]
    files += [Path(__file__).resolve(), PINS_FILE]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files))}
    return {"sha256": digest(hashes), "files": hashes,
            "scope": "source snapshot, not proof the candidate binary was built from it"}



def strict_json(text):
    def invalid(value):
        raise ValueError("nonfinite JSON number: " + value)
    def floating(value):
        number = float(value)
        return number if math.isfinite(number) else invalid(value)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(text, parse_constant=invalid, parse_float=floating,
                      object_pairs_hook=unique)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def hex_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_pins(path):
    pins = read_json(path)
    if not isinstance(pins, dict) or type(pins.get("schema")) is not int or pins["schema"] != 1:
        raise ValueError("Model pin receipt needs schema 1")
    if not isinstance(pins.get("models"), dict):
        raise ValueError("Model pin receipt needs a models object")
    for role in ("main", "mtp"):
        model = pins.get("models", {}).get(role, {})
        if (not isinstance(model, dict) or not isinstance(model.get("filename"), str) or
            Path(model["filename"]).name != model["filename"] or
            type(model.get("bytes")) is not int or model["bytes"] <= 0 or
            not hex_sha(model.get("sha256"))):
            raise ValueError("Invalid pinned model identity: " + role)
    if (not isinstance(pins.get("published_baseline"), dict) or
        not hex_sha(pins["published_baseline"].get("sha256"))):
        raise ValueError("Invalid published baseline pin")
    return pins


def resolve_receipt_path(value, manifest):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    # The public compact schema is manifest-relative. Legacy saved-binary
    # manifests used repository-relative `saved` paths; accept those too.
    options = (manifest.parent / path, ROOT / path)
    return next((p.resolve() for p in options if p.is_file()), options[0].resolve())


def baseline_input(args, pins):
    receipt = None
    records = []
    document = {}
    if args.baseline_manifest:
        manifest = args.baseline_manifest.resolve()
        document = read_json(manifest)
        if not isinstance(document, dict):
            raise ValueError("Baseline manifest must be an object")
        records = document.get("binaries")
        if records is None:
            records = [document.get("baseline", document)]
        if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
            raise ValueError("Invalid baseline manifest records")
        receipt = {"path": str(manifest), "sha256": sha(manifest)}
    if args.baseline_exe:
        binary = args.baseline_exe.resolve()
    else:
        choices = [r for r in records if "janas-bench" in
                   Path(r.get("saved", r.get("path", ""))).name.lower()]
        if not choices and len(records) == 1:
            choices = records
        if len(choices) != 1:
            raise ValueError("Baseline manifest needs one benchmark binary, or supply --baseline-exe")
        value = choices[0].get("saved", choices[0].get("path"))
        if not isinstance(value, str) or not value:
            raise ValueError("Baseline manifest needs a nonempty path")
        binary = resolve_receipt_path(value, args.baseline_manifest.resolve())
    actual = sha(binary)
    matching = [r for r in records if r.get("sha256") == actual]
    if receipt and not matching:
        raise ValueError("Baseline executable SHA does not match its manifest")
    if matching and any("bytes" in r and r["bytes"] != binary.stat().st_size for r in matching):
        raise ValueError("Baseline executable size does not match its manifest")
    published = pins["published_baseline"]["sha256"]
    value = {"path": str(binary), "sha256": actual, "bytes": binary.stat().st_size,
             "source_commit": document.get("source_commit"),
             "published_reference_sha256": published,
             "matches_published_reference": actual == published,
             "identity_source": "validated manifest" if receipt else "explicit executable override; hashed locally"}
    return value, receipt



def prepare_inputs(args):
    pins_path = args.model_pins.resolve()
    pins = load_pins(pins_path)
    baseline, baseline_receipt = baseline_input(args, pins)
    candidate = args.bench_exe.resolve()
    candidate_sha = sha(candidate)
    if args.execute and candidate_sha == baseline["sha256"]:
        raise ValueError("Candidate equals selected baseline; choose distinct builds")
    receipt_path = args.model_receipt.resolve() if args.model_receipt else None
    receipt = read_json(receipt_path) if receipt_path else None
    if receipt is not None and (not isinstance(receipt, dict) or
        receipt.get("complete") is not True or not isinstance(receipt.get("files"), dict)):
        raise ValueError("Model download receipt must record complete verification")
    full_verify = args.execute and args.verify_model_sha256
    method = ("computed_streaming_sha256" if full_verify else
              "supplied_completed_receipt" if receipt is not None else "unverified_plan_only")
    resumed = None
    if full_verify and args.resume and args.output:
        proof_path = args.output.resolve() / "model-verification.json"
        if proof_path.is_file():
            resumed = read_json(proof_path)
            if (not isinstance(resumed, dict) or resumed.get("complete") is not True or
                resumed.get("method") != method or not isinstance(resumed.get("files"), dict)):
                raise ValueError("Resume model-verification attestation is invalid")
    models, proofs = [], {}
    for role, path in (("main", args.model), ("mtp", args.mtp)):
        pin = pins["models"][role]
        path = path.resolve()
        st = path.stat()
        if st.st_size != pin["bytes"]:
            raise ValueError("Pinned model size mismatch: " + role)
        with path.open("rb") as stream:
            header = hashlib.sha256(stream.read(1 << 20)).hexdigest()
        actual = None
        if receipt is not None:
            proof = receipt["files"].get(pin["filename"], {})
            if (not isinstance(proof, dict) or proof.get("sha256_verified") is not True or
                proof.get("sha256") != pin["sha256"]):
                raise ValueError("Model download SHA receipt mismatch: " + role)
            actual = pin["sha256"]
        if full_verify:
            if resumed is not None:
                proof = resumed["files"].get(pin["filename"], {})
                if (not isinstance(proof, dict) or proof.get("sha256_verified") is not True or
                    proof.get("sha256") != pin["sha256"] or proof.get("bytes") != st.st_size or
                    proof.get("mtime_ns") != st.st_mtime_ns or proof.get("first_mib_sha256") != header):
                    raise ValueError("Resume model identity changed: " + role)
                actual = pin["sha256"]
            else:
                print("Verifying pinned model SHA-256 before timing: " + role, file=sys.stderr, flush=True)
                actual = sha(path)
                if actual != pin["sha256"]:
                    raise ValueError("Computed model SHA mismatch: " + role)
        proofs[pin["filename"]] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "first_mib_sha256": header, "sha256": actual, "sha256_verified": actual is not None}
        models.append({"role": role, "path": str(path), "bytes": st.st_size,
                       "mtime_ns": st.st_mtime_ns, "first_mib_sha256": header,
                       "expected_full_sha256": pin["sha256"],
                       "full_sha256_previously_verified": actual,
                       "verification_method": method})
    prompts = {}
    for name, (path, tokens, expected_sha, output_hash) in PROMPTS.items():
        if sha(path) != expected_sha:
            raise ValueError("Frozen public prompt changed: " + name)
        prompts[name] = {"path": str(path), "bytes": path.stat().st_size,
                         "sha256": expected_sha, "input_tokens": tokens,
                         "output_tokens": 128, "output_hash": output_hash}
    verification = {"method": method, "complete": method != "unverified_plan_only", "files": proofs}
    return {"baseline": baseline,
            "candidate": {"path": str(candidate), "sha256": candidate_sha},
            "models": models, "prompts": prompts, "source": source_identity(),
            "model_verification": verification,
            "model_receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)} if receipt_path else None,
            "pins_receipt": {"path": str(pins_path), "sha256": sha(pins_path)},
            "baseline_receipt": baseline_receipt,
            "model_identity_boundary": (
                "Full pinned payload SHA streamed once before timing; resume reuses its recorded attestation only with unchanged size/mtime/header."
                if full_verify else "Supplied receipt attests prior full SHA verification; current size, mtime and first MiB checked. No payload rescan."
                if receipt is not None else "Plan only: expected pins and current size/header are not payload verification. Execute requires a completed receipt or --verify-model-sha256.")}


def run_spec(inputs, workload, case, control, label, rounds, sequence):
    control = valid_control(control)
    binary = inputs["baseline" if label == "A" else "candidate"]
    model, mtp = (m["path"] for m in inputs["models"])
    command = [binary["path"], model, "--speculation", control["speculation"],
        "--draft-tokens", str(control["draft_tokens"]), "--seed", "20260926", "--threads", "8",
        "--prompt-file", inputs["prompts"][workload]["path"], "--gen", "128", "--ctx", "2048",
        "--rounds", str(rounds), "--warmups", "0", "--cache", "32", "--reserve", "12",
        "--prompt-on", "cpu", "--drafts-on", "cpu", "--reply-on", "cpu", "--json"]
    command += ["--mtp", mtp] if control["speculation"] in ("mtp", "auto", "hybrid") else ["--no-mtp"]
    if label == "B":
        command += ["--spec-planner", control["spec_planner"], "--spec-explore", str(control["spec_explore"]),
                    "--draft-vocab", control["draft_vocab"], "--draft-vocab-rows", str(control["draft_vocab_rows"])]
    value = {"id": f"{sequence:02d}-{workload}-{label}-{case}", "workload": workload, "case": case,
             "binary": label, "binary_sha256": binary["sha256"], "warm_rounds": rounds,
             "control": control, "performance_controls": performance_controls(control), "command": command,
             "expected": inputs["prompts"][workload]}
    return dict(value, run_sha256=digest(value))


def make_manifest(args, protocol, runs, inputs):
    contract = {"schema": 1, "protocol": protocol, "inputs": inputs, "plan": runs,
                "settings": {"compute_threads": 8, "parent_and_child_affinity": "0xff", "cache_gib": 32,
                    "reserve_gib": 12, "quality": "HIGH", "expert_bits": 6, "head_bits": 0,
                    "sampler": "greedy", "seed": 20260926, "context_tokens": 2048, "gen": 128,
                    "prefetch": "off", "gpu": False, "npu": False,
                    "kv_and_attention": "architecture defaults; inherited overrides cleared"},
                "sampling": {"new_chat_each_sample": True, "prompt_kv_reuse": False,
                    "same_process_private_cache_reuse": "warm rounds only; fresh cache directory per process",
                    "os_file_cache": "uncontrolled across sequential processes; fresh does not mean cold SSD",
                    "phase_profiling": False, "physical_ssd_bandwidth_measured": False},
                "cooldown_seconds": args.cooldown, "timeout_seconds": args.timeout}
    return {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python": sys.version, "machine_architecture": platform.machine(),
            "contract_sha256": digest(contract), "contract": contract,
            "planned_processes": len(runs), "planned_samples": sum(1 + r["warm_rounds"] for r in runs),
            "execution_note": "default plan launches no executable; --execute explicitly runs sequential native Windows children",
            "quality_boundary": "Frozen decoded-output hashes/counts must agree. Fixed-input logit/state validation is coordinated separately.",
            "evidence_boundary": "Controls do not prove an optimization executed. Raw runtime counters retained; OS cache uncontrolled; logical expert reads are not physical SSD traffic or memory bandwidth."}



def strict_records(path):
    return [strict_json(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("{")]


def validate_records(records, run):
    errors = []
    if any(not isinstance(r, dict) or type(r.get("schema")) is not int or r["schema"] != 1 for r in records):
        return ["wrong JSON record schema"]
    setup = [r for r in records if r.get("kind") == "setup"]
    samples = [r for r in records if r.get("kind") == "sample"]
    c = run["control"]
    if len(setup) != 1 or len(samples) != 1 + run["warm_rounds"]:
        errors.append("wrong setup/sample count")
    if setup:
        expected = {"quality": "high", "requested_threads": 8, "seed": 20260926,
            "context_tokens": 2048, "generation_limit": 128, "draft_tokens_cap": c["draft_tokens"],
            "speculation": c["speculation"], "speculation_resolved": "mtp" if c["speculation"] == "auto" else c["speculation"],
            "cache_bytes_requested": 32 << 30, "reserve_bytes": 12 << 30, "gpu": "", "npu": ""}
        if run["binary"] == "B":
            expected.update({k: c[k] for k in ("spec_planner", "spec_explore", "draft_vocab", "draft_vocab_rows")})
        for key, value in expected.items():
            if setup[0].get(key) != value:
                errors.append(f"setup {key} differs from protocol")
        if setup[0].get("profile_phases_enabled"):
            errors.append("phase profiling enabled during timing")
    phases = ["fresh_process"] + ["warm"] * run["warm_rounds"]
    if [r.get("phase") for r in samples] != phases:
        errors.append("wrong sample phase sequence")
    for index, s in enumerate(samples):
        expected = run["expected"]
        for key in ("input_tokens", "output_tokens", "output_hash"):
            if s.get(key) != expected[key]:
                errors.append("sample " + key + " mismatch")
        if s.get("index") != index or s.get("cached_tokens") != 0 or s.get("prompt_tokens") != expected["input_tokens"]:
            errors.append("sample index/prompt length/KV reuse mismatch")
        if s.get("input_threads") != 8 or s.get("output_threads") != 8:
            errors.append("sample compute thread count mismatch")
        resolved = "mtp" if c["speculation"] == "auto" else c["speculation"]
        if s.get("speculation") != c["speculation"] or s.get("speculation_resolved") != resolved:
            errors.append("sample speculation mode mismatch")
        if s.get("input_tokens", 2049) + s.get("output_tokens", 0) > 2048:
            errors.append("context exceeded")
        for key in ("output_tokens_per_second", "output_seconds", "total_seconds", "first_token_seconds"):
            value = s.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append("invalid sample timing " + key)
        value = s.get("input_seconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            errors.append("invalid sample timing input_seconds")
        counters = s.get("accelerator_counters", {})
        if not isinstance(counters, dict):
            errors.append("invalid accelerator counters")
            counters = {}
        if any(counters.get(k, 0) for k in ("npu_products", "npu_ffns", "npu_layer_passes", "gpu_products", "gpu_whole_passes")):
            errors.append("unexpected accelerator execution")
        if s.get("gpu_policy_used", 0) or s.get("npu_policy_used", 0):
            errors.append("unexpected accelerator policy")
        if run["binary"] == "B" and not isinstance(s.get("speculation_metrics"), dict):
            errors.append("candidate speculation metrics missing")
    return sorted(set(errors))



def verify_inputs(inputs):
    for key in ("baseline", "candidate"):
        if sha(inputs[key]["path"]) != inputs[key]["sha256"]:
            raise RuntimeError(key + " executable changed")
    for model in inputs["models"]:
        st = Path(model["path"]).stat()
        if (st.st_size, st.st_mtime_ns) != (model["bytes"], model["mtime_ns"]):
            raise RuntimeError("Model changed after protocol was frozen")
    for prompt in inputs["prompts"].values():
        if sha(prompt["path"]) != prompt["sha256"]:
            raise RuntimeError("Prompt changed after protocol was frozen")
    for key in ("model_receipt", "pins_receipt", "baseline_receipt"):
        receipt = inputs.get(key)
        if receipt and sha(receipt["path"]) != receipt["sha256"]:
            raise RuntimeError(key + " changed after protocol was frozen")
    if source_identity() != inputs["source"]:
        raise RuntimeError("Source/runner snapshot changed after protocol was frozen")


def child_run(run, folder, timeout):
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k.GetProcessAffinityMask.restype = k.SetProcessAffinityMask.restype = wintypes.BOOL
    parent = k.GetCurrentProcess()
    old, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k.GetProcessAffinityMask(parent, ctypes.byref(old), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    if system.value & 255 != 255:
        raise RuntimeError("CPUs0..7 unavailable; this protocol requires exact0xff")
    cache = folder / "cache"
    cache.mkdir(parents=True, exist_ok=False)
    env, controlled = environment(cache, run["control"])
    record = dict(run, environment=controlled, parent_original_affinity=hex(old.value),
                  system_affinity=hex(system.value), status="running")
    write_json(folder / "command.json", record)
    started = time.perf_counter()
    timed_out = False
    with (folder / "stdout.log").open("w", encoding="utf-8") as out, (folder / "stderr.log").open("w", encoding="utf-8") as err:
        if not k.SetProcessAffinityMask(parent, 255):
            raise ctypes.WinError(ctypes.get_last_error())
        process = None
        try:
            process = subprocess.Popen(run["command"], cwd=ROOT, env=env, stdout=out, stderr=err)
        finally:
            if not k.SetProcessAffinityMask(parent, old.value):
                if process is not None:
                    process.kill(); process.wait()
                raise ctypes.WinError(ctypes.get_last_error())
        child, child_system = ctypes.c_size_t(), ctypes.c_size_t()
        try:
            if not k.GetProcessAffinityMask(wintypes.HANDLE(int(process._handle)), ctypes.byref(child), ctypes.byref(child_system)):
                raise ctypes.WinError(ctypes.get_last_error())
            if child.value != 255:
                raise RuntimeError("Child did not inherit0xff affinity")
            record["child_affinity"] = hex(child.value)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill(); process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill(); process.wait()
            raise
    record.update(exit_code=process.returncode, timeout=timed_out,
                  process_wall_seconds=time.perf_counter() - started,
                  stdout_sha256=sha(folder / "stdout.log"), stderr_sha256=sha(folder / "stderr.log"))
    try:
        record["records"] = strict_records(folder / "stdout.log")
        errors = validate_records(record["records"], run)
    except (ValueError, TypeError, UnicodeError) as exc:
        record["records"] = []
        errors = ["invalid benchmark JSON: " + str(exc)]
    if process.returncode or timed_out:
        errors.append("child failed or timed out")
    record.update(errors=sorted(set(errors)), status="invalid" if errors else "ok")
    write_json(folder / "result.json", record)
    return record


def summary(manifest, records):
    plan = manifest["contract"]["plan"]
    result = {"planned_processes": len(plan), "recorded_processes": len(records),
              "complete": len(records) == len(plan), "all_runs_valid": all(r["status"] == "ok" for r in records),
              "groups": [], "physical_bandwidth_measured": False}
    keys = list(dict.fromkeys((r["workload"], r["case"], r["binary"]) for r in plan))
    for workload, case, binary in keys:
        runs = [r for r in records if (r["workload"], r["case"], r["binary"]) == (workload, case, binary)]
        samples = [s for r in runs if r["status"] == "ok" for s in r["records"] if s.get("kind") == "sample"]
        warm = [s for s in samples if s.get("phase") == "warm"]
        def finite_positive(value):
            return type(value) in (int, float) and math.isfinite(value) and value > 0
        rates = [s["output_tokens_per_second"] for s in warm
                 if finite_positive(s.get("output_tokens_per_second"))]
        ttfts = [s["first_token_seconds"] for s in warm if finite_positive(s.get("first_token_seconds"))]
        result["groups"].append({"workload": workload, "case": case, "binary": binary,
            "processes": len(runs), "samples": len(samples), "warm_samples": len(warm),
            "median_warm_tokens_per_second": statistics.median(rates) if rates else None,
            "median_warm_ttft_seconds": statistics.median(ttfts) if ttfts else None,
            "warm_tokens_per_second": rates, "warm_records": warm,
            "fresh_records": [s for s in samples if s.get("phase") == "fresh_process"],
            "process_wall_seconds": [r["process_wall_seconds"] for r in runs],
            "setups": [s for r in runs for s in r["records"] if s.get("kind") == "setup"]})
    result["comparisons"] = []
    for group in result["groups"]:
        if group["binary"] != "B":
            continue
        base = next((g for g in result["groups"] if g["workload"] == group["workload"] and g["binary"] == "A"), None)
        wanted = [r for r in plan if r["workload"] == group["workload"] and
                  (r["binary"] == "A" or r["case"] == group["case"])]
        actual = [r for r in records if r["id"] in {p["id"] for p in wanted}]
        eligible = len(actual) == len(wanted) and all(r["status"] == "ok" for r in actual)
        old = base["median_warm_tokens_per_second"] if base else None
        new = group["median_warm_tokens_per_second"]
        result["comparisons"].append({"workload": group["workload"], "candidate_case": group["case"],
            "complete_and_output_verified": eligible,
            "candidate_to_baseline_median_rate_ratio": new / old if eligible and old and new else None,
            "scope": "warm output-tail rate ratio; full wall and TTFT retained separately; screen is discovery only"})
    return result


def completed_run(folder, run):
    path = folder / "result.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if any(value.get(k) != run[k] for k in run):
        raise RuntimeError("Resume run/control identity changed: " + run["id"])
    attempt = folder / value["attempt"]
    for stream in ("stdout", "stderr"):
        if sha(attempt / (stream + ".log")) != value[stream + "_sha256"]:
            raise RuntimeError("Resume log changed: " + run["id"])
    parsed = strict_records(attempt / "stdout.log")
    if parsed != value["records"] or validate_records(parsed, run) or value.get("status") != "ok" or value.get("exit_code") != 0 or value.get("timeout"):
        raise RuntimeError("Resume refuses invalid/inconsistent completed run: " + run["id"])
    if value.get("child_affinity") != "0xff" or value.get("environment", {}).get("controls") != run["performance_controls"]:
        raise RuntimeError("Resume affinity/environment mismatch: " + run["id"])
    return value


def native_host():
    return os.name == "nt" and platform.machine().lower() in ("arm64", "aarch64")


def execute(manifest, args):
    if not native_host():
        raise RuntimeError("--execute requires native ARM64 Windows Python")
    contract = manifest["contract"]
    if not contract["inputs"].get("model_verification", {}).get("complete"):
        raise RuntimeError("Execution refuses unverified model payloads")
    if contract["inputs"]["candidate"]["sha256"] == contract["inputs"]["baseline"]["sha256"]:
        raise RuntimeError("Candidate equals selected baseline; build first")
    directory = args.output.resolve()
    if args.resume:
        old = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if old.get("contract_sha256") != digest(old.get("contract")) or old["contract_sha256"] != manifest["contract_sha256"]:
            raise RuntimeError("Resume rejected changed inputs, runner, configuration or run order")
        manifest = old
        if read_json(directory / "model-verification.json") != contract["inputs"]["model_verification"]:
            raise RuntimeError("Resume model-verification artifact changed")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "model-verification.json", contract["inputs"]["model_verification"])
    verify_inputs(contract["inputs"])
    records = []
    missing = False
    for run in contract["plan"]:
        row = completed_run(directory / run["id"], run)
        if row is None:
            missing = True
        elif missing:
            raise RuntimeError("Resume requires completed runs to be an ordered prefix")
        else:
            records.append(row)
    for run in contract["plan"][len(records):]:
        verify_inputs(contract["inputs"])
        folder = directory / run["id"]
        folder.mkdir(exist_ok=True)
        number = 1
        while (folder / f"attempt-{number:03d}").exists():
            number += 1
        attempt = folder / f"attempt-{number:03d}"
        attempt.mkdir()
        # An interrupted attempt is preserved, and the retry gets a fresh private cache.
        row = child_run(run, attempt, args.timeout)
        row["attempt"] = attempt.name
        write_json(folder / "result.json", row)
        records.append(row)
        write_json(directory / "results.json", records)
        write_json(directory / "summary.json", summary(manifest, records))
        print(json.dumps({"id": run["id"], "status": row["status"], "errors": row["errors"]}), flush=True)
        if row["status"] != "ok":
            return 1
        if len(records) < len(contract["plan"]):
            time.sleep(args.cooldown)
    write_json(directory / "results.json", records)
    write_json(directory / "summary.json", summary(manifest, records))
    return 0



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("screen", "paired"), nargs="?", default="paired")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="default; launches no child executable")
    action.add_argument("--execute", action="store_true", help="explicitly run sequential native Windows ARM64 children")
    p.add_argument("--baseline-exe", type=Path, help="explicit control executable; its SHA is recorded")
    p.add_argument("--baseline-manifest", type=Path, help="bind the control executable to a SHA/source receipt")
    p.add_argument("--bench-exe", type=Path, default=ROOT / "bin/aarch64-windows/janas-bench.exe")
    p.add_argument("--model", type=Path, default=ROOT / "models/qwen3-next-80b-a3b-q4km.jns")
    p.add_argument("--mtp", type=Path, default=ROOT / "models/qwen3-next-80b-a3b-mtp-q4k.jns")
    p.add_argument("--model-pins", type=Path, default=PINS_FILE)
    p.add_argument("--model-receipt", type=Path, help="optional completed download receipt with per-file sha256_verified and sha256")
    p.add_argument("--verify-model-sha256", action="store_true", help="execute: stream pinned model hashes once before timing; plan never full-hashes models")
    p.add_argument("--resume", action="store_true", help="validate inputs and reuse an ordered completed prefix")
    p.add_argument("--workloads", help="ordered subset of code,prose,long; screen defaults code,prose")
    p.add_argument("--output", type=Path, help="new artifact directory; required for --execute")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--cooldown", type=float, default=5)
    p.add_argument("--cases", default=",".join(CASES), help="screen: distinct cases in execution order; baseline added if missing")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--candidate-case", choices=[k for k in CASES if k != "baseline"], default="legacy")
    selection.add_argument("--candidate-controls", type=Path, help="paired: explicit validated control object")
    return p



def check_args(args):
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.cooldown) or args.cooldown < 0:
        raise ValueError("Invalid timeout/cooldown")
    if not args.baseline_exe and not args.baseline_manifest:
        raise ValueError("Supply --baseline-exe or --baseline-manifest")
    if args.execute and args.output is None:
        raise ValueError("--execute requires --output")
    if args.execute and not args.model_receipt and not args.verify_model_sha256:
        raise ValueError("--execute needs --model-receipt or explicit --verify-model-sha256")
    if args.resume and (not args.execute or args.output is None):
        raise ValueError("--resume requires --execute and --output")
    names = (args.workloads or ("code,prose" if args.mode == "screen" else "code,prose,long")).split(",")
    if not names or len(set(names)) != len(names) or any(n not in PROMPTS for n in names):
        raise ValueError("Workloads must be a distinct subset of code,prose,long")
    if args.mode == "screen" and args.candidate_controls:
        raise ValueError("--candidate-controls is for paired mode")
    return names



def build_plan(args, inputs):
    workloads = check_args(args)
    runs = []
    if args.mode == "screen":
        cases = args.cases.split(",")
        if not cases or len(set(cases)) != len(cases) or any(c not in CASES for c in cases):
            raise ValueError("Cases must be a distinct ordered subset of " + ",".join(CASES))
        if "baseline" not in cases:
            cases.insert(0, "baseline")
        for workload in workloads:
            for case in cases:
                runs.append(run_spec(inputs, workload, case, CASES[case],
                                     "A" if case == "baseline" else "B", 2, len(runs)))
        protocol = "screen-fresh-plus-2warm"
    else:
        control = valid_control(read_json(args.candidate_controls)) if args.candidate_controls else valid_control(CASES[args.candidate_case])
        name = "custom" if args.candidate_controls else args.candidate_case
        for workload in workloads:
            for index, label in enumerate("ABBA"):
                rounds = 3 if index in (0, 1) else 2
                runs.append(run_spec(inputs, workload, "baseline" if label == "A" else name,
                                     DEFAULT_CONTROL if label == "A" else control,
                                     label, rounds, len(runs)))
        protocol = "paired-ABBA-5warm-per-binary"
    return make_manifest(args, protocol, runs, inputs)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        check_args(args)
        if args.execute and not native_host():
            raise RuntimeError("--execute requires native ARM64 Windows Python")
        inputs = prepare_inputs(args)
        manifest = build_plan(args, inputs)
        if not args.execute:
            print(json.dumps(manifest, indent=2, allow_nan=False))
            return 0
        return execute(manifest, args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

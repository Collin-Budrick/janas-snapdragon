"""Native Windows functional gates. Default --plan launches no executable.
Execution requires explicit --execute after coordinating exclusive device use.
Every execution uses a fresh output directory. Optional --fast reuses model
loads and runs independent model-free units concurrently; model jobs stay serial.
No throughput or cold-cache claims are inferred from these checks.
"""
from __future__ import annotations
import argparse
import ctypes
import contextlib
import concurrent.futures
import platform
import shutil
import threading
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
S2_CONTROLS = {"s2_exact": "off", "s2_branches": 2, "s2_max_nodes": 8, "s2_min_gain_percent": 5}
ATTENTION_CALIBRATION = None
MOJO_LIBRARY = None
MOJO_MODULE = None
FAST = False
UNIT_WORKERS = 4
SELECT_CHECKS = set()
INCREMENTAL_CACHE = None
CACHE_SOURCE_IDENTITY = None
CACHE_STATS = {"hits": 0, "misses": 0, "stored": 0, "rejected": 0}
CACHE_LOCK = threading.Lock()
FILE_IDENTITIES = {}
CACHE_PROTOCOL = "native-functional-proof-v1"
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
OPTIONAL_MODULE_UNITS = {"test_mojo.exe"}


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
    if S2_CONTROLS["s2_exact"] == "on":
        env.update({"JANAS_" + k.upper(): str(v) for k, v in S2_CONTROLS.items()})
    if MOJO_LIBRARY is not None:
        env.update(JANAS_MOJO="1", JANAS_MOJO_LIBRARY=str(MOJO_LIBRARY))
    if extra:
        env.update(extra)
    return env


def mojo_input(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError("--mojo-library requires an absolute path to an existing module file")
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def mojo_identity(env):
    # Small optional modules are hashed afresh even if size/times are unchanged.
    # Cache identities must never reuse a proof across different module bytes.
    return mojo_input(env.get("JANAS_MOJO_LIBRARY")) if env.get("JANAS_MOJO") == "1" else None


def unit_executables():
    excluded = HARDWARE_HELPERS | (OPTIONAL_MODULE_UNITS if MOJO_LIBRARY is None else set())
    if MOJO_LIBRARY is not None and not (BIN / "test_mojo.exe").is_file():
        raise ValueError("--mojo-library requires the built test_mojo.exe unit executable")
    return sorted(p.resolve() for p in BIN.glob("test*.exe") if p.name.lower() not in excluded)


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


def raw_model_environment(extra=None):
    result=environment(extra)
    for key in tuple(result):
        if key.startswith("JANAS_S2_"): result.pop(key)
    return result


def selected(name):
    return not SELECT_CHECKS or name in SELECT_CHECKS


def source_identity():
    # Linked dependency bits are bound by the executable hash. Track project
    # sources and fixture inputs without repeatedly hashing third-party trees.
    files = [Path(__file__).resolve(), ROOT / "build.sh", ROOT / "tools/build-mojo.sh"]
    for folder in ("src", "include", "tests"):
        files += [p for p in (ROOT / folder).rglob("*") if p.is_file() and
                  "third_party" not in p.parts and p.suffix.lower() in
                  (".c", ".h", ".inc", ".s", ".py", ".json", ".txt", ".bin", ".mojo")]
    values = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(files)) if p.is_file()}
    return {"sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
            "files": values, "scope": "project sources/fixtures; linked dependencies bound by executable SHA"}


def file_identity(path, large=False):
    path = Path(path).resolve()
    st = path.stat()
    fingerprint = (str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns, large)
    if fingerprint not in FILE_IDENTITIES:
        item = {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                "ctime_ns": st.st_ctime_ns}
        if large:
            with path.open("rb") as stream:
                item["first_mib_sha256"] = sha_bytes(stream.read(1 << 20))
            item["payload_boundary"] = "pinned receipt plus unchanged size/time/header; no new full payload scan"
        else:
            item["sha256"] = sha(path)
        FILE_IDENTITIES[fingerprint] = item
    return FILE_IDENTITIES[fingerprint]


def case_identity(name, argv, env, affinity=None):
    global CACHE_SOURCE_IDENTITY
    if CACHE_SOURCE_IDENTITY is None:
        CACHE_SOURCE_IDENTITY = source_identity()
    def normalize(value):
        value = str(value)
        prefix = str(OUT) + os.sep
        head, separator, tail = value.partition("=")
        if separator and tail.startswith(prefix):
            return head + "=@output/" + tail[len(prefix):].replace(os.sep,"/")
        return "@output/" + value[len(prefix):].replace(os.sep, "/") if value.startswith(prefix) else value
    model_free = name.startswith("unit-")
    identity = {"protocol": CACHE_PROTOCOL, "name": name,
        "executable": file_identity(argv[0]), "arguments": [normalize(x) for x in argv[1:]],
        "environment": {k: v for k, v in env.items() if k.upper().startswith("JANAS_")},
        "affinity": affinity, "unit_cpu8_parent": bool(FAST and model_free),
        "platform": {"os": os.name, "machine": platform.machine(),
                     "processor": os.environ.get("PROCESSOR_IDENTIFIER", ""), "version":platform.version()},
        "source": CACHE_SOURCE_IDENTITY,
        "attention": ATTENTION_CALIBRATION,
        "model_free": model_free}
    module = mojo_identity(env)
    if module is not None:
        identity["mojo_module"] = module
    identity["project_libraries"] = {p.name: file_identity(p) for pattern in ("libjanas*.dll", "janas*.dll")
        for p in Path(argv[0]).parent.glob(pattern)}
    identity["argument_files"] = {}
    for index,value in enumerate(argv[1:]):
        candidate = str(value).split("=",1)[-1]
        path = Path(candidate)
        if path.is_file() and path.resolve() not in (MODEL.resolve(), MTP.resolve()):
            item = file_identity(path)
            identity["argument_files"][str(index)] = {"logical_path": normalize(path),
                "bytes": item["bytes"], "sha256": item["sha256"]}
    if not model_free:
        identity["models"] = {k: dict(file_identity(p, large=True), pinned_bytes=MODEL_PINS[k][0],
                                      pinned_sha256=MODEL_PINS[k][1]) for k,p in (("main", MODEL), ("mtp", MTP))}
        identity["reference"] = file_identity(QREF)
        identity["tokens"] = file_identity(CODE)
        identity["model_receipt"] = file_identity(RECEIPT)
        identity["frozen_inputs"] = {p.name: dict(file_identity(p)) for p in
            (OUT / "reference32.int32", OUT / "reference32.f32", OUT / "session-code160.int32") if p.is_file()}
        # Output-directory paths are not semantic input identities.
        for item in identity["frozen_inputs"].values():
            item.pop("path", None); item.pop("mtime_ns", None); item.pop("ctime_ns", None)
    return identity


def identity_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def cache_load(name, argv, env, affinity=None):
    if INCREMENTAL_CACHE is None:
        return None
    identity = case_identity(name, argv, env, affinity)
    key = identity_key(identity)
    folder = INCREMENTAL_CACHE / key
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema") != 1 or manifest.get("validated_pass_only") is not True or
            manifest.get("identity") != identity or manifest.get("key") != key):
            raise ValueError("cache identity mismatch")
        if set(manifest.get("artifacts",{})) != {"stdout.log","stderr.log","record.json"}:
            raise ValueError("cache proof artifact set incomplete")
        if any(sha(folder / filename) != digest for filename,digest in manifest["artifacts"].items()):
            raise ValueError("cache proof checksum mismatch")
        value = json.loads((folder / "record.json").read_text(encoding="utf-8"))
        if value.get("passed") is not True or value.get("validation_errors") or not finite_json(value):
            raise ValueError("cache entry is not a validated finite passing proof")
        original = {k: value.get(k) for k in ("command", "started_utc", "wall_seconds", "stdout", "stderr")}
        for stream in ("stdout", "stderr"):
            destination = OUT / (name + "." + stream + ".log")
            shutil.copyfile(folder / (stream + ".log"), destination)
            value[stream] = str(destination)
        value.update(proof_reused=True, performed_new_execution=False,
                     requested_command=[str(x) for x in argv], cache_key=key,
                     cache_identity=identity, original_proof=original,
                     source_proof_receipt=str(folder / "manifest.json"),
                     identity_checked_utc=datetime.now(timezone.utc).isoformat(),
                     scope="explicit inherited functional proof; not a fresh execution or timing sample")
        with CACHE_LOCK: CACHE_STATS["hits"] += 1
        save(OUT / (name + ".result.json"), value)
        return value
    except FileNotFoundError:
        with CACHE_LOCK: CACHE_STATS["misses"] += 1
        return None
    except (ValueError, KeyError, TypeError, OSError):
        with CACHE_LOCK:
            CACHE_STATS["misses"] += 1; CACHE_STATS["rejected"] += 1
        return None


def save_validated(result, allow_inherited=False):
    save(OUT / (result["name"] + ".result.json"), result)
    if INCREMENTAL_CACHE is None:
        return
    identity = result.get("cache_identity")
    if identity is not None and not result.get("passed"):
        (INCREMENTAL_CACHE / identity_key(identity) / "manifest.json").unlink(missing_ok=True)
        return
    if not result.get("passed") or (result.get("proof_reused") and not allow_inherited):
        return
    if identity is None:
        return
    key = identity_key(identity)
    folder = INCREMENTAL_CACHE / key
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for stream in ("stdout", "stderr"):
        path = folder / (stream + ".log")
        shutil.copyfile(result[stream], path); artifacts[path.name] = sha(path)
    save(folder / "record.json", result); artifacts["record.json"] = sha(folder / "record.json")
    save(folder / "manifest.json", {"schema": 1, "key": key, "identity": identity,
        "artifacts": artifacts, "validated_pass_only": True})
    with CACHE_LOCK: CACHE_STATS["stored"] += 1


@contextlib.contextmanager
def cpu8_parent():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    handle = kernel.GetCurrentProcess()
    old, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(old), ctypes.byref(system)) or (0xff & ~old.value):
        raise RuntimeError("Fast units require CPU mask0xff within the current allowed process mask")
    if not kernel.SetProcessAffinityMask(handle, 0xff):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not kernel.SetProcessAffinityMask(handle, old.value):
            raise RuntimeError("Could not restore parent affinity after parallel model-free units")


def run(name, argv, env, timeout=600, affinity=None, expected_exit=(0,), allow_cache=True):
    argv = [str(x) for x in argv]
    if not Path(argv[0]).is_absolute():
        raise ValueError("argv[0] must be an absolute Windows executable path")
    env = dict(env)
    if allow_cache:
        cached = cache_load(name, argv, env, affinity)
        if cached is not None:
            return cached
    identity = case_identity(name, argv, env, affinity) if INCREMENTAL_CACHE is not None and allow_cache else None
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
    module = mojo_identity(env)
    if module is not None:
        result["mojo_module"] = module
    result["cache_identity"] = identity
    result["performed_new_execution"] = True
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
    if module is not None:
        try:
            unchanged = mojo_identity(env) == module
        except (OSError, ValueError):
            unchanged = False
        result["mojo_module_unchanged"] = unchanged
        result["passed"] = result["passed"] and unchanged
    save(OUT / (name + ".result.json"), result)
    print(f"{name}: exit={result.get('exit_code')} timeout={result.get('timed_out', False)}", flush=True)
    return result


def validate_unit_result(r,path):
    text = read_log(r)
    errors = []
    explicit_skip = bool(re.search(r"(?i)\bskipped\b", text))
    if path.name.lower() == "test_mojo.exe":
        # A field named "skipped" is not a skip when its JSON value is false.
        # Genuine skip text outside the result record still rejects this
        # explicitly requested module matrix, even beside a passing record.
        non_json = "\n".join(line for line in text.splitlines()
                             if not line.lstrip().startswith("{"))
        explicit_skip = bool(re.search(r"(?i)\bskip(?:ped)?\b", non_json))
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.lstrip().startswith("{")]
            explicit_skip = explicit_skip or any(isinstance(row, dict) and
                row.get("skipped") is True for row in rows)
            value = rows[0] if len(rows) == 1 else {}
            if (explicit_skip or value.get("test") != "mojo" or value.get("skipped") is not False or
                    type(value.get("cases")) is not int or value["cases"] < 1341 or
                    type(value.get("compared_words")) is not int or value["compared_words"] < 144570 or
                    type(value.get("failures")) is not int or value["failures"] != 0):
                raise ValueError("incomplete or skipped Mojo unit matrix")
        except (ValueError, TypeError, AttributeError):
            errors.append("missing complete passing Mojo unit JSON matrix")
    elif not re.search(re.escape(path.stem) + r"[^\n]*(?:\bok\b|\bpassed\b|\bskipped\b)", text, re.I):
        errors.append("missing named unit success/explicit skip marker")
    if re.search(r"(?im)^\s*(?:FAIL(?:ED)?\b|AddressSanitizer:|runtime error:)", text):
        errors.append("unit emitted failure/sanitizer marker")
    r.update(validation_errors=errors, explicit_skip=explicit_skip,
             parent_cpu_mask=255 if FAST else None,
             executor=r.get("executor","inherited_proof") if r.get("proof_reused") else "bounded_parallel_model_free" if FAST else "serial")
    r["passed"] = r["passed"] and not errors
    save_validated(r)
    return r


def unit_case(path):
    return validate_unit_result(run("unit-"+path.stem,[path],environment()),path)


def units():
    cases = unit_executables()
    if not cases and (not SELECT_CHECKS or any(n.startswith("unit-") for n in SELECT_CHECKS)):
        raise RuntimeError("no native Windows unit executables found")
    cases = [p for p in cases if selected("unit-" + p.stem)]
    started = time.monotonic()
    if FAST and cases:
        # One process-affinity change on the calling thread, never racing
        # per-child parent-mask mutations from worker threads.
        if INCREMENTAL_CACHE is not None:
            case_identity("unit-" + cases[0].stem, [cases[0]], environment())
        with cpu8_parent(), concurrent.futures.ThreadPoolExecutor(max_workers=UNIT_WORKERS) as pool:
            results = list(pool.map(unit_case, cases))
    else:
        results = [unit_case(p) for p in cases]
    summary = {"stage": "units", "count": len(results), "passed": sum(r["passed"] for r in results),
        "results": results, "executor": "parallel_model_free" if FAST else "serial",
        "workers": UNIT_WORKERS if FAST else 1, "cpu_mask": 255 if FAST else None,
        "elapsed_seconds": time.monotonic() - started, "excluded_hardware_helpers": sorted(HARDWARE_HELPERS),
        "excluded_optional_modules": sorted(OPTIONAL_MODULE_UNITS) if MOJO_LIBRARY is None else []}
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


def quality_argv(batch):
    return [BIN / "quality_check.exe", MODEL, "--profile", "cpu", "--steps", "64", "--batch", str(batch),
            "--context", "1024", "--cache-mib", "24576", "--threads", "8", "--reference", QREF]


def validate_quality_result(r, batch, matrix=None):
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
    errors = list(parse_errors) + list(r.get("driver_validation_errors", []))
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
    if matrix is not None and records:
        m = records[0]
        checks = {"completed": True, "exit_code": 0, "matrix_index": matrix[0],
            "matrix_count": matrix[1], "max_batch_capacity": matrix[2], "model_load_count": 1,
            "functional_only": True, "sequence_restart": "forward_position_zero",
            "actual_batches": [min(batch, 64 - i) for i in range(0, 64, batch)]}
        for key,want in checks.items():
            if m.get(key) != want: errors.append("matrix " + key + " differs from complete-width protocol")
    r.update(metrics=records, validation_errors=errors)
    r["passed"] = r["passed"] and not errors
    save_validated(r)
    return r


def quality():
    if FAST: return quality_fast()
    results = []
    for batch in BATCHES:
        if not selected(f"quality64-batch{batch}"): continue
        r = validate_quality_result(run(f"quality64-batch{batch}", quality_argv(batch), raw_model_environment()), batch)
        results.append(r)
        save(OUT / "quality.json", {"stage": "quality", "results": results, "executor": "serial", "passed": all(x["passed"] for x in results)})
        if not r["passed"]: return False
    return True


def quality_fast():
    started = time.monotonic()
    batches = [b for b in BATCHES if selected(f"quality64-batch{b}")]
    if not batches:
        save(OUT/"quality.json", {"stage":"quality","executor":"single_model_matrix","results":[],"passed":True})
        return True
    argv = quality_argv(batches[0]); at = argv.index("--batch")
    argv[at:at+2] = ["--batches", ",".join(map(str,batches))]
    driver = run("quality64-matrix", argv, raw_model_environment(), timeout=600*len(batches))
    records = []
    for line in Path(driver["stdout"]).read_text(encoding="utf-8",errors="replace").splitlines():
        try: value = json.loads(line)
        except ValueError: continue
        if isinstance(value,dict) and "batch" in value: records.append(value)
    ids = [v.get("batch") for v in records]
    protocol = [] if ids == batches else ["matrix missing/duplicate/unrequested/reordered batch records"]
    results=[]
    for index,batch in enumerate(batches):
        name=f"quality64-batch{batch}"; rows=[r for r in records if r.get("batch")==batch]
        stdout=OUT/(name+".stdout.log");stderr=OUT/(name+".stderr.log")
        stdout.write_text("\n".join(json.dumps(v) for v in rows)+"\n",encoding="utf-8")
        shutil.copyfile(driver["stderr"],stderr)
        m=rows[0] if len(rows)==1 else {}
        result=dict(driver,name=name,stdout=str(stdout),stderr=str(stderr),
            logical_command=[str(x) for x in quality_argv(batch)],
            driver_exit_code=driver.get("exit_code"),driver_wall_seconds=driver["wall_seconds"],
            exit_code=m.get("exit_code"),wall_seconds=m.get("width_seconds"),
            driver_validation_errors=list(protocol),cache_identity=None)
        result["passed"] = (m.get("completed") is True and m.get("exit_code")==0 and
            not driver.get("timed_out") and not driver.get("affinity_restore_error") and not protocol)
        results.append(validate_quality_result(result,batch,(index,len(batches),max(batches))))
    passed=driver.get("exit_code")==0 and all(r["passed"] for r in results)
    driver.update(passed=passed,matrix_validated=passed,
                  validation_errors=[] if passed else ["one or more complete matrix checks failed"])
    save_validated(driver)  # Cache the whole validated matrix, never a partial pass.
    save(OUT/"quality.json",{"stage":"quality","executor":"single_model_matrix","results":results,
        "passed":passed,"elapsed_seconds":time.monotonic()-started,"driver":driver,
        "logical_checks":len(results),"model_processes":0 if driver.get("proof_reused") else 1})
    return passed


def validate_eval_result(r,block,reject):
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
    save_validated(r)
    return r


def evaluation():
    env = raw_model_environment({"JANAS_ATTN": "float", "JANAS_HEAD_BITS": "6"})
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
        if not selected(f"eval32-block{block}-reject{reject}"): continue
        r = run(f"eval32-block{block}-reject{reject}", [BIN / "llm_eval.exe", MODEL,
                OUT / "reference32.int32", OUT / "reference32.f32", "24576",
                "0,1,2,3,4,5,6,7", "8,9,10,11", f"block={block}", f"reject={reject}", "strict=1"], env)
        r=validate_eval_result(r,block,reject)
        results.append(r)
        save(OUT / "eval.json", {"stage": "eval", "capability_probe": probe, "results": results,
                                  "passed": all(x["passed"] for x in results)})
        if not r["passed"]:
            return False
    return True


def session_s2_controls(name):
    controls = dict(S2_CONTROLS)
    # Keep the established 17-row stress proof. The production eight-node
    # ceiling intentionally prevents it, so this one explicit stress case
    # raises its test-only budget instead of weakening the coverage assertion.
    if name == "fixed-mtp16-adaptive" and controls["s2_exact"] == "on":
        controls["s2_max_nodes"] = 32
    return controls


def session_controls(name):
    controls = []
    if name.startswith("hybrid-cost"):
        controls += ["hybrid_policy=cost", "lookup_drafts=8", "mtp_drafts=4",
                     "hybrid_probe_percent=" + ("100" if name.endswith("-probe") else "2")]
    if S2_CONTROLS["s2_exact"] == "on":
        controls += [k + "=" + str(v) for k, v in session_s2_controls(name).items()]
    return controls


def session_argv(case):
    name,planner,mode,drafts,vocab,rows=case
    return [BIN / "llm_session.exe", MODEL, OUT / "session-code160.int32", "mtp=" + str(MTP),
        "planner="+planner,"speculation="+mode,f"drafts={drafts}","draft_vocab="+vocab,f"rows={rows}"] + session_controls(name)


def validate_session_result(r, case):
    name,planner,mode,drafts,vocab,rows=case
    text = read_log(r)
    expected = ["greedy: turns vs all at once", "greedy + speculation: turn 1",
                "greedy + speculation: turn 2", "sampling, speculation off vs on: turn 1",
                "sampling, speculation off vs on: turn 2"]
    errors = ["missing successful comparison: " + value for value in expected
              if not re.search(re.escape(value) + r"\s+identical", text)]
    errors += list(r.get("driver_validation_errors", []))
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
    expected_records = 2 if r.get("reference_reused") else 4
    if len(records) != expected_records or len(selected) != 2 or not all(v["verified"] for v in records):
        errors.append("expected verified runtime records including two selected-mode passes")
    if r.get("reference_reused"):
        plain = r.get("shared_reference_runtime", [])
        if len(plain) != 2 or any(v.get("mode") != 0 or not v.get("verified") for v in plain):
            errors.append("shared plain reference runtime evidence missing")
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
    r.update(s2_controls_effective=session_s2_controls(name),
             stress_override="S2 total-node budget32 retains fixed16/17-row target coverage" if name == "fixed-mtp16-adaptive" and S2_CONTROLS["s2_exact"] == "on" else None,
             validation_errors=errors, runtime_records=records, hybrid_records=hybrid_records,
             mtp_execution_verified=mtp_exercised, adaptive_execution_verified=adaptive_verified,
             adaptive_note="not exercised or currently a legacy comparison epoch" if vocab == "adaptive" and not adaptive_verified else None)
    r["passed"] = r["passed"] and not errors
    save_validated(r)
    return r


def session():
    if FAST: return session_fast()
    results=[]
    for case in SESSION_CASES:
        name=case[0]
        if not selected("session-"+name): continue
        r=validate_session_result(run("session-"+name,session_argv(case),environment({"JANAS_ATTN":"float"}),affinity=0xff),case)
        results.append(r)
        save(OUT/"sessions.json",{"stage":"session","executor":"serial","results":results,
            "passed":all(x["passed"] for x in results)})
        if not r["passed"]: return False
    return True


def session_manifest_content(cases):
    return "\n".join("\t".join([case[0]] + [str(a) for a in session_argv(case)[4:]])
                      for case in cases) + "\n"


def session_fast():
    started=time.monotonic();cases=[c for c in SESSION_CASES if selected("session-"+c[0])]
    manifest=OUT/"session-cases.tsv"
    manifest.write_text(session_manifest_content(cases),encoding="utf-8")
    if not cases:
        save(OUT/"sessions.json",{"stage":"session","executor":"single_model_matrix","results":[],"passed":True})
        return True
    argv=[BIN/"llm_session.exe",MODEL,OUT/"session-code160.int32","mtp="+str(MTP),"cases="+str(manifest)]
    driver=run("session-matrix",argv,environment({"JANAS_ATTN":"float"}),timeout=600*len(cases),affinity=0xff)
    references=[];outcomes=[];sections={};reference_lines=[];current=None
    for line in Path(driver["stdout"]).read_text(encoding="utf-8",errors="replace").splitlines():
        if line.startswith("session-reference: "):
            try: references.append(json.loads(line.split(": ",1)[1]))
            except ValueError: references.append({})
        elif line.startswith("session-case-begin: "):
            current=line.split(": ",1)[1];sections.setdefault(current,[])
        elif line.startswith("session-case: "):
            try: outcomes.append(json.loads(line.split(": ",1)[1]))
            except ValueError: outcomes.append({})
            current=None
        elif current is not None: sections[current].append(line)
        else: reference_lines.append(line)
    reference_runtime=[]
    for line in reference_lines:
        if line.startswith("runtime:"):
            fields={k:int(v) for k,v in re.findall(r"([a-z_]+)=(\d+)",line)}
            fields["verified"]=line.endswith(" verified");reference_runtime.append(fields)
    names=[c[0] for c in cases];protocol=[]
    if [r.get("name") for r in outcomes]!=names:protocol.append("session matrix missing/duplicate/unrequested/reordered case records")
    reference=references[0] if len(references)==1 else {}
    if (len(references)!=1 or reference.get("version")!=1 or reference.get("valid") is not True or
        reference.get("model_runs")!=3 or reference.get("functional_only") is not True or
        not re.fullmatch(r"[0-9a-f]{16}",str(reference.get("content_key","")))):
        protocol.append("shared immutable three-run reference proof missing/invalid")
    results=[]
    for index,case in enumerate(cases):
        name=case[0];matches=[r for r in outcomes if r.get("name")==name];m=matches[0] if len(matches)==1 else {}
        errors=list(protocol)
        wanted={"index":index,"exit_code":0,"checks":5,"functional_only":True,"reference_reused":True,
                "actual_model_runs":2,"boundary_forward_calls":1,"content_key":reference.get("content_key")}
        for key,want in wanted.items():
            if m.get(key)!=want:errors.append("session matrix "+key+" differs from independent-case protocol")
        text="\n".join(sections.get(name,[]))+"\n"
        if not re.search(r"(?m)^boundary: head=[0-9a-f]{16} mtp_kv=[0-9a-f]{16} logits=[0-9a-f]{16} identical$",text):
            errors.append("32-token bit-identical case-boundary oracle missing")
        stdout=OUT/("session-"+name+".stdout.log");stderr=OUT/("session-"+name+".stderr.log")
        stdout.write_text(text,encoding="utf-8");shutil.copyfile(driver["stderr"],stderr)
        result=dict(driver,name="session-"+name,stdout=str(stdout),stderr=str(stderr),
            logical_command=[str(x) for x in session_argv(case)],case_manifest_sha256=sha(manifest),
            shared_reference=reference,reference_reused=m.get("reference_reused"),
            shared_reference_runtime=reference_runtime,
            reference_identity_binding=driver.get("cache_identity") or case_identity("session-matrix",argv,environment({"JANAS_ATTN":"float"}),0xff),
            driver_exit_code=driver.get("exit_code"),driver_wall_seconds=driver["wall_seconds"],
            exit_code=m.get("exit_code"),wall_seconds=m.get("wall_seconds"),driver_validation_errors=errors,
            matrix_case=m,cache_identity=None)
        result["passed"]=(m.get("exit_code")==0 and not driver.get("timed_out") and
                          not driver.get("affinity_restore_error") and not errors)
        results.append(validate_session_result(result,case))
    passed=driver.get("exit_code")==0 and all(r["passed"] for r in results)
    driver.update(passed=passed,matrix_validated=passed,
                  validation_errors=[] if passed else ["one or more complete matrix checks failed"])
    save_validated(driver)
    save(OUT/"sessions.json",{"stage":"session","executor":"single_model_reference_matrix",
        "results":results,"passed":passed,"driver":driver,"elapsed_seconds":time.monotonic()-started,
        "logical_checks":len(cases)*5,"case_manifest_sha256":sha(manifest),"shared_reference":reference,
        "actual_model_runs":0 if driver.get("proof_reused") else 3+2*len(cases),
        "original_proof_model_runs":3+2*len(cases),"model_processes":0 if driver.get("proof_reused") else 1})
    return passed


def validate_semantic_result(r):
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
    if S2_CONTROLS["s2_exact"] == "on":
        mtp_samples = [v for v in samples if v.get("mode") == "mtp4"]
        if not mtp_samples or sum(v.get("s2_cycles", 0) for v in mtp_samples) == 0:
            errors.append("S2 MTP generation was not exercised")
        if sum(v.get("s2_maintenance_fast_calls", 0) for v in mtp_samples) == 0:
            errors.append("S2 maintenance path was not exercised")
    r.update(s2_controls=dict(S2_CONTROLS), validation_errors=errors, semantic_records=records,
             probe_percent=SEMANTIC_PROBE_PERCENT,
             performance_boundary="functional scenario; source order is not matched ABBA",
             forced_probe_functional_only=SEMANTIC_PROBE_PERCENT != 2)
    r["passed"] = r["passed"] and not errors
    save(OUT / "semantic-hybrid.json", r)
    save_validated(r)
    return r


def semantic_hybrid():
    if not selected("semantic-hybrid-copy-switch"): return True
    name = "semantic-hybrid-copy-switch"
    r = run(name, [BIN / "llm_hybrid.exe", MODEL, MTP,
                  "probe=" + str(SEMANTIC_PROBE_PERCENT)],
            environment(), timeout=1800, affinity=0xff)
    return validate_semantic_result(r)["passed"]


def import_model_attestation(path, proof_started):
    if path is None: raise ValueError("model proof import requires --import-model-receipt")
    attestation=json.loads(Path(path).read_text(encoding="utf-8"))
    if attestation.get("complete") is not True: raise ValueError("model payload attestation incomplete")
    for role,model in (("main",MODEL),("mtp",MTP)):
        record=attestation.get("files",{}).get(model.name,{})
        stat=model.stat();size,digest=MODEL_PINS[role]
        if (record.get("sha256_verified") is not True or record.get("sha256")!=digest or
            record.get("bytes")!=size or stat.st_size!=size or record.get("mtime_ns")!=stat.st_mtime_ns):
            raise ValueError("model attestation does not match current unchanged payload: "+role)
        verified=datetime.fromisoformat(record["verified_utc"].replace("Z","+00:00"))
        started=datetime.fromisoformat(proof_started.replace("Z","+00:00"))
        if started<verified:raise ValueError("source proof predates completed model checksum attestation")
    return {"path":str(path),"sha256":sha(path),"complete":True,
            "boundary":"prior full SHA plus unchanged size/mtime/current header; no new payload rescan"}


def import_raw_proofs(folder, planned, model_attestation=None):
    folder=Path(folder).resolve();summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    if summary.get("passed") is not True:raise ValueError("only complete passing validation directories may be imported")
    wanted={c["check_id"]:c for c in planned["cases"]}
    candidates=[]
    for name in ("units.json","quality.json","eval.json","sessions.json"):
        path=folder/name
        if path.is_file():candidates+=json.loads(path.read_text(encoding="utf-8")).get("results",[])
    path=folder/"semantic-hybrid.json"
    if path.is_file():candidates.append(json.loads(path.read_text(encoding="utf-8")))
    results=[]
    def normalized(argv,output):
        prefix=str(output)+os.sep;values=[]
        for arg in map(str,argv):
            head,sep,tail=arg.partition("=")
            if sep and tail.startswith(prefix):arg=head+"=@output/"+tail[len(prefix):].replace(os.sep,"/")
            elif arg.startswith(prefix):arg="@output/"+arg[len(prefix):].replace(os.sep,"/")
            values.append(arg)
        return values
    for old in candidates:
        name=old.get("name");spec=wanted.get(name)
        if spec is None:continue
        try:
            if old.get("passed") is not True or old.get("validation_errors"):raise ValueError("source case is not a validated pass")
            argv=spec["command"];env=environment(spec.get("environment_overrides"));stage=spec["stage"]
            if stage in ("quality","eval"):env=raw_model_environment(spec.get("environment_overrides"))
            if sha(argv[0])!=old.get("binary_sha256"):raise ValueError("executable SHA changed")
            if stage=="units" and FAST and old.get("parent_cpu_mask")!=255 and old.get("process_affinity_mask")!=255:
                raise ValueError("historical unit proof does not attest the fast CPU8 process mask")
            if normalized(old["command"],folder)!=normalized(argv,OUT):raise ValueError("case arguments changed")
            wanted_env={k:v for k,v in env.items() if k.upper().startswith("JANAS_")}
            if old.get("janas_environment")!=wanted_env:raise ValueError("effective environment changed")
            module=mojo_identity(env)
            if module is not None and old.get("mojo_module")!=module:
                raise ValueError("historical proof lacks the unchanged Mojo module identity")
            if stage!="units":
                bound=import_model_attestation(model_attestation,old["started_utc"])
                inputs=summary.get("inputs",{})
                if inputs.get("source_qref_sha256")!=EXPECTED_QREF_SHA or inputs.get("inputs")!=EXPECTED_INPUTS:
                    raise ValueError("frozen reference/token identity differs")
            else:bound=None
            seed=old.get("attention_calibration_seed")
            if ATTENTION_CALIBRATION is not None:
                if not seed or seed.get("sha256")!=ATTENTION_CALIBRATION["sha256"]:
                    raise ValueError("attention seed changed")
                private=Path(old["XDG_CACHE_HOME"])/"janas"/"attn.txt"
                if not private.is_file() or sha(private)!=ATTENTION_CALIBRATION["sha256"]:
                    raise ValueError("original private attention policy missing/retimed")
            elif seed is not None:raise ValueError("attention policy changed")
            r=dict(old,proof_reused=True,performed_new_execution=False,
                   cache_identity=case_identity(name,argv,env,spec.get("process_affinity_mask")),
                   imported_raw_proof={"directory":str(folder),"summary_sha256":sha(folder/"summary.json"),
                     "model_attestation":bound,"revalidated_by_current_checker":True},
                   scope="explicit imported historical raw proof, revalidated against current artifact/input identities")
            for stream in ("stdout","stderr"):
                path=Path(old[stream]).resolve()
                if not path.is_file() or folder not in path.parents:raise ValueError("source log missing/outside proof directory")
                destination=OUT/(name+"."+stream+".log");shutil.copyfile(path,destination);r[stream]=str(destination)
            if stage=="units":r=validate_unit_result(r,Path(argv[0]))
            elif stage=="quality":r=validate_quality_result(r,spec["batch"])
            elif stage=="eval":r=validate_eval_result(r,spec["block"],spec["reject"])
            elif stage=="session":r=validate_session_result(r,next(c for c in SESSION_CASES if "session-"+c[0]==name))
            else:r=validate_semantic_result(r)
            if not r["passed"]:raise ValueError("current strict checker rejected historical output")
            save_validated(r,allow_inherited=True)
            results.append({"name":name,"imported":True,"cache_key":identity_key(r["cache_identity"])})
        except (ValueError,KeyError,OSError) as error:
            results.append({"name":name,"imported":False,"reason":str(error)})
    return {"directory":str(folder),"results":results,"executables_launched":False,
            "scope":"explicit inherited proof import; no fresh inference or timing claim"}


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
                  for p in unit_executables()]
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
                    "s2_controls_effective": session_s2_controls(n),
                    "stress_override": "S2 total-node budget32 retains fixed16/17-row target coverage" if n == "fixed-mtp16-adaptive" and S2_CONTROLS["s2_exact"] == "on" else None,
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
    for c in cases:
        stage_name=c["stage"]
        c["check_id"] = ("unit-" + Path(c["command"][0]).stem if stage_name=="units" else
            f"quality64-batch{c['batch']}" if stage_name=="quality" else
            f"eval32-block{c['block']}-reject{c['reject']}" if stage_name=="eval" else
            "session-"+c["name"] if stage_name=="session" else "semantic-hybrid-copy-switch")
    full_ids=[c["check_id"] for c in cases]
    unknown=SELECT_CHECKS-set(full_ids)
    if unknown: raise ValueError("unknown/not-in-stage selected checks: "+", ".join(sorted(unknown)))
    cases=[c for c in cases if selected(c["check_id"])]
    drivers=[]
    if FAST:
        for stage_name in ("units","quality","eval","session","hybrid"):
            chosen=[c for c in cases if c["stage"]==stage_name]
            if not chosen: continue
            if stage_name=="quality":
                batches=[c["batch"] for c in chosen];argv=quality_argv(batches[0]);at=argv.index("--batch")
                argv[at:at+2]=["--batches",",".join(map(str,batches))]
                drivers.append({"stage":stage_name,"command":[str(x) for x in argv],"logical_checks":[c["check_id"] for c in chosen]})
            elif stage_name=="session":
                ids={c["name"] for c in chosen};ordered=[c for c in SESSION_CASES if c[0] in ids]
                content=session_manifest_content(ordered)
                drivers.append({"stage":stage_name,"command":[str(BIN/"llm_session.exe"),str(MODEL),str(OUT/"session-code160.int32"),"mtp="+str(MTP),"cases="+str(OUT/"session-cases.tsv")],
                    "logical_checks":[c["check_id"] for c in chosen],"case_manifest_content":content,
                    "case_manifest_sha256":sha_bytes(content.encode()),"reference_model_runs":3,
                    "candidate_model_runs":2*len(chosen),"comparisons_per_case":5})
            else: drivers += [{"stage":stage_name,"command":c["command"],"logical_checks":[c["check_id"]]} for c in chosen]
    return {"mode": "plan", "launches_executables": False, "output_directory": str(OUT),
            "sequential": not FAST, "models_sequential": True, "stage": stage, "cases": cases,
             "fast": FAST, "unit_workers": UNIT_WORKERS if FAST else 1, "unit_cpu_mask": 255 if FAST else None,
             "driver_invocations": drivers, "full_requested_coverage": not SELECT_CHECKS,
             "selected_checks": sorted(SELECT_CHECKS), "omitted_checks": [i for i in full_ids if not selected(i)],
             "excluded_hardware_helpers": sorted(HARDWARE_HELPERS),
             "excluded_optional_modules": sorted(OPTIONAL_MODULE_UNITS) if MOJO_LIBRARY is None else [],
             "mojo_module": mojo_identity(environment()),
             "incremental_cache": str(INCREMENTAL_CACHE) if INCREMENTAL_CACHE is not None else None,
             "cache_boundary": "explicit validated inherited proofs; EXE/DLL+argv+input content+model receipt/stat/header+source identities; no cache is accessed during plan",
            "requested_janas_environment": {k: v for k, v in environment().items() if k.startswith("JANAS_")},
            "cleared_inherited_controls": "all JANAS_* before applying listed controls",
            "locked_reference_digest": EXPECTED_DIGEST, "qref_sha256": EXPECTED_QREF_SHA,
            "execution_requires": "native Windows ARM64 Python --execute after coordinating exclusive device use",
            "reference": str(QREF), "frozen_tokens": str(CODE), "receipt": str(RECEIPT),
            "reference_generation": "--create-reference reconstructs the locked CPU reference; its digest and SHA must match the receipt",
            "scope": "functional quality and correctness only; no performance claims",
            "s2_controls": dict(S2_CONTROLS)}


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
    result.add_argument("--fast", action="store_true", help="same logical coverage: reuse model loads/references; bounded parallel model-free units")
    result.add_argument("--unit-workers", type=int, choices=range(1,9), default=4, help="--fast model-free workers sharing CPU mask0xff (models stay serial)")
    result.add_argument("--select-check", action="append", default=[], help="explicit check IDs, comma-separated; marks subset scope; default requests all checks")
    result.add_argument("--import-proofs", type=Path, action="append", default=[], help="explicit completed passing raw-proof directories to revalidate/import into the requested cache")
    result.add_argument("--import-model-receipt", type=Path, help="completed model-checksum receipt required when importing model-based historical proofs")
    result.add_argument("--incremental-cache", type=Path, help="explicit validated-pass proof cache; inherited proofs are labelled and identities rechecked")
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
    result.add_argument("--mojo-library", type=Path,
                        help="explicit absolute optional Mojo module; enables it for all checks and includes test_mojo.exe")
    result.add_argument("--s2-exact", choices=("off", "on"), default="off")
    result.add_argument("--s2-branches", type=int, choices=range(1, 3), default=2)
    result.add_argument("--s2-max-nodes", type=int, choices=range(1, 33), default=8)
    result.add_argument("--s2-min-gain-percent", type=int, choices=range(101), default=5)
    result.add_argument("--semantic-probe-percent", type=int, choices=range(101), default=2,
                        metavar="0..100", help="hybrid semantic check; default production 2, other values functional only")
    result.add_argument("--create-reference", action="store_true", help="create a missing CPU64 reference and require its locked digest/SHA")
    result.add_argument("--baseline-quality-exe", "--baseline-exe", dest="baseline_quality_exe", type=Path,
                        help="optional baseline quality_check executable for --create-reference")
    return result


def main():
    global OUT, BIN, MODEL, MTP, QREF, CODE, RECEIPT, EXPECTED_DIGEST, EXPECTED_QREF_SHA
    global MODEL_PINS, BASELINE_QUALITY_EXE, VERIFY_MODEL_HASHES, SEMANTIC_PROBE_PERCENT, ATTENTION_CALIBRATION, S2_CONTROLS
    global FAST, UNIT_WORKERS, SELECT_CHECKS, INCREMENTAL_CACHE, CACHE_SOURCE_IDENTITY, CACHE_STATS
    global MOJO_LIBRARY, MOJO_MODULE
    cli = parser()
    args = cli.parse_args()
    if args.import_proofs and not args.incremental_cache: cli.error("--import-proofs requires --incremental-cache")
    try:
        MOJO_MODULE = mojo_input(args.mojo_library)
    except (OSError, ValueError) as error:
        cli.error(str(error))
    MOJO_LIBRARY = Path(MOJO_MODULE["path"]) if MOJO_MODULE is not None else None
    FAST, UNIT_WORKERS = args.fast, args.unit_workers
    SELECT_CHECKS = {v.strip() for group in args.select_check for v in group.split(",") if v.strip()}
    INCREMENTAL_CACHE = args.incremental_cache.resolve() if args.incremental_cache else None
    CACHE_SOURCE_IDENTITY = None
    CACHE_STATS = {"hits":0,"misses":0,"stored":0,"rejected":0}
    FILE_IDENTITIES.clear()
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
    S2_CONTROLS = {k: getattr(args, k) for k in ("s2_exact", "s2_branches", "s2_max_nodes", "s2_min_gain_percent")}
    ATTENTION_CALIBRATION = attention_input(args.attention_calibration_file)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    OUT = args.out.resolve() if args.out else ROOT / "lib/validation" / ("snapdragon-runtime-" + stamp + "-" + str(os.getpid()))
    if args.create_reference and args.stage == "units":
        cli.error("reference generation is a model check, incompatible with --stage units")
    try:
        planned = plan(args.stage)
    except ValueError as error:
        cli.error(str(error))
    planned["import_proofs"] = [str(p) for p in args.import_proofs]
    planned["import_model_receipt"] = str(args.import_model_receipt) if args.import_model_receipt else None
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
    if INCREMENTAL_CACHE is not None and (INCREMENTAL_CACHE == OUT or OUT in INCREMENTAL_CACHE.parents):
        cli.error("incremental cache must be outside the new output directory")
    OUT.mkdir(parents=True, exist_ok=False)
    if INCREMENTAL_CACHE is not None:
        INCREMENTAL_CACHE.mkdir(parents=True,exist_ok=True)
        CACHE_SOURCE_IDENTITY = source_identity()
    lock = OUT / "runner.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("ascii")); os.close(fd)
    execution_started=time.monotonic()
    summary = {"scope": "functional; model jobs serial; explicit inherited cache proofs labelled", "stages": [],
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "runner_sha256": sha(__file__), "native_python": sys.executable,
               "receipt": receipt, "receipt_sha256": sha(RECEIPT),
               "pid": os.getpid(), "output_directory": str(OUT),
               "removed_inherited_janas_names": sorted(k for k in os.environ if k.upper().startswith("JANAS_")),
               "plan": planned}
    try:
        if any(c["stage"] != "units" for c in planned["cases"]):
            summary["models"] = model_metadata()
            if args.create_reference:
                if not create_reference():
                    summary["passed"] = False
                    return 1
            if not QREF.is_file():
                raise ValueError("reference missing; supply --reference or explicitly use --create-reference")
            summary["inputs"] = prepare_inputs()
        if args.import_proofs:
            summary["imported_proofs"]=[import_raw_proofs(p,planned,args.import_model_receipt) for p in args.import_proofs]
            save(OUT/"proof-imports.json",summary["imported_proofs"])
        stages = [("units", units), ("quality", quality), ("eval", evaluation), ("session", session), ("hybrid", semantic_hybrid)]
        for name, fn in stages:
            if args.stage not in ("all", name): continue
            if SELECT_CHECKS and not any(c["stage"] == name for c in planned["cases"]): continue
            started=time.monotonic()
            passed = fn()
            summary["stages"].append({"name": name, "passed": passed,"wall_seconds":time.monotonic()-started})
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
        cleanup_error = None
        if MOJO_MODULE is not None:
            try:
                unchanged = mojo_input(MOJO_LIBRARY) == MOJO_MODULE
            except (OSError, ValueError):
                unchanged = False
            summary["mojo_module_unchanged"] = unchanged
            if not unchanged:
                cleanup_error = RuntimeError("Mojo module changed during validation")
        if ATTENTION_CALIBRATION is not None:
            try:
                preserved = attention_input(Path(ATTENTION_CALIBRATION["source_path"])) == ATTENTION_CALIBRATION
            except (OSError,ValueError): preserved = False
            summary["attention_calibration_original_unchanged"] = preserved
            if not preserved: cleanup_error = RuntimeError("Original attention calibration changed during validation")
        summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
        summary["wall_seconds"] = time.monotonic()-execution_started
        summary["fast"] = FAST
        summary["full_requested_coverage"] = not SELECT_CHECKS
        summary["cache"] = dict(CACHE_STATS, enabled=INCREMENTAL_CACHE is not None,
            directory=str(INCREMENTAL_CACHE) if INCREMENTAL_CACHE is not None else None,
            boundary="cached passes are inherited proofs, never fresh runtime measurements")
        if CACHE_SOURCE_IDENTITY is not None:
            summary["source_identity"] = CACHE_SOURCE_IDENTITY
            summary["source_unchanged"] = source_identity() == CACHE_SOURCE_IDENTITY
            if not summary["source_unchanged"]:
                summary["source_change_error"] = "project/validator source changed during validation"
                cleanup_error = RuntimeError("Project/validator source changed during validation")
        if cleanup_error is not None:
            summary["passed"] = False
            summary["cleanup_error"] = {"type":type(cleanup_error).__name__,"message":str(cleanup_error)}
        save(OUT / "summary.json", summary)
        lock.unlink(missing_ok=True)
        if cleanup_error is not None: raise cleanup_error


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build the optional, runtime-free Mojo leaf module. No installation or download.
set -euo pipefail
if ! command -v python3 >/dev/null 2>&1; then
    echo 'build-mojo.sh: python3 is required for artifact auditing.' >&2
    exit 2
fi
exec python3 - "$(cd "$(dirname "$0")" && pwd)" "$@" <<'PY'
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = Path(sys.argv[1])
ROOT = SCRIPT_DIR.parent
VERSION = "1.2.0"
WINDOWS_CLANG_VERSION = "23.1.5"
EXPORTS = ("janas_mojo_abi_v1", "janas_mojo_panel_rows_v1")


def fail(message):
    raise RuntimeError(message)


def run(args):
    result = subprocess.run([str(x) for x in args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        fail("command failed: " + repr([str(x) for x in args]) + "\n" +
             result.stdout + result.stderr)
    return result.stdout, result.stderr


def local_path(value):
    if re.match(r"^[A-Za-z]:[\\/]", value):
        if not shutil.which("wslpath"):
            fail("a Windows tool path requires WSL wslpath: " + value)
        return Path(run(["wslpath", "-u", value])[0].strip())
    return Path(value).expanduser()


def tool(explicit, names, label, hint=None):
    if explicit:
        path = local_path(explicit)
        if path.is_file():
            return path.resolve()
        found = shutil.which(explicit)
        if found:
            return Path(found).resolve()
        fail(label + " was configured but not found: " + explicit)
    if hint:
        directory = local_path(hint)
        for name in names:
            path = directory / name
            if path.is_file():
                return path.resolve()
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    fail(label + " is missing; configure it explicitly. No tools are installed automatically.")


def native_path(executable, path):
    path = str(Path(path).resolve())
    if str(executable).lower().endswith(".exe") and os.name != "nt":
        if not shutil.which("wslpath"):
            fail("a native Windows executable requires WSL wslpath")
        return run(["wslpath", "-w", path])[0].strip()
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compiler_identity(mojo, version):
    identity = {"path": str(mojo), "version": version,
                "required_release": VERSION, "launcher_sha256": sha256(mojo)}
    # PyPI installs a Python entry point beside the environment's interpreter.
    # Identify the native compiler too, rather than hashing only that wrapper.
    candidates = list(mojo.parent.parent.glob("lib/python*/site-packages/modular/bin/mojo"))
    if len(candidates) == 1:
        identity["binary_path"] = str(candidates[0])
        identity["binary_sha256"] = sha256(candidates[0])
    else:
        identity["binary_path"] = str(mojo)
        identity["binary_sha256"] = sha256(mojo)
    return identity


def build(args, mojo, compiler_version, source, output, target):
    directory = output / ("aarch64-" + target)
    directory.mkdir(parents=True, exist_ok=True)
    # Stage on the output filesystem so os.replace never crosses filesystems.
    with tempfile.TemporaryDirectory(prefix=".mojo-build-", dir=directory) as temp:
        stage = Path(temp)
        triple = "aarch64-unknown-linux-gnu" if target == "linux" else "aarch64-pc-windows-msvc"
        artifact = "libjanas_mojo.so" if target == "linux" else "janas_mojo.dll"
        obj = stage / ("quant_panel.o" if target == "linux" else "quant_panel.obj")
        assembly = stage / "quant_panel.s"
        flags = ["--target-triple", triple, "--target-cpu", "oryon-1",
                 "--fp-mode", "contract=off", "-O3"]
        commands = []
        compiler_logs = []
        source_hash = sha256(source)
        backend = {"kind": "mojo-native"}
        extra_files = []
        if target == "linux":
            for kind, path in (("object", obj), ("asm", assembly)):
                command = [str(mojo), "build", str(source), *flags, "--emit", kind, "-o", str(path)]
                stdout, stderr = run(command)
                commands.append(command)
                compiler_logs.append(stdout + stderr)
        else:
            hint = os.environ.get("LLVM_MINGW_BIN")
            linker = tool(args.lld_link, ("lld-link", "lld-link.exe"), "LLD_LINK", hint)
            readobj = tool(args.llvm_readobj, ("llvm-readobj", "llvm-readobj.exe"),
                           "LLVM_READOBJ", hint)
            # Mojo's experimental COFF emitter omits Windows unwind metadata.
            # Ask LLVM to generate it from the original Mojo IR, never by
            # inventing unwind records for an already compiled prologue.
            clang = tool(args.windows_clang, ("clang", "clang.exe"),
                         "WINDOWS_CLANG", str(linker.parent))
            clang_version = run([clang, "--version"])[0].splitlines()[0]
            if not re.search(r"\bclang version " + re.escape(WINDOWS_CLANG_VERSION) +
                             r"(?:\s|$)", clang_version):
                fail("verified Windows Clang " + WINDOWS_CLANG_VERSION +
                     " is required for Mojo IR/unwind generation, found: " + clang_version)
            raw_ir = stage / "quant_panel.ll"
            unwind_ir = stage / "quant_panel_unwind.ll"
            command = [str(mojo), "build", str(source), *flags,
                       "--emit", "llvm", "-o", str(raw_ir)]
            stdout, stderr = run(command)
            commands.append(command)
            compiler_logs.append(stdout + stderr)
            raw = raw_ir.read_text(encoding="utf-8")
            rewritten, count = re.subn(r"^(define [^\n]+?) \{",
                                      lambda match: match.group(1) + " uwtable(async) {",
                                      raw, flags=re.M)
            definitions = sum(line.startswith("define ") for line in raw.splitlines())
            if not count or count != definitions or rewritten.replace(" uwtable(async) {", " {") != raw:
                fail("could not add only unwind attributes to every Mojo IR definition")
            unwind_ir.write_text(rewritten, encoding="utf-8")
            backend_flags = ["--target=aarch64-pc-windows-msvc", "-mcpu=oryon-1", "-O3",
                             "-ffp-contract=off", "-funwind-tables"]
            for emission, path in (("-c", obj), ("-S", assembly)):
                command = [str(clang), *backend_flags, emission,
                           native_path(clang, unwind_ir), "-o", native_path(clang, path)]
                stdout, stderr = run(command)
                commands.append(command)
                compiler_logs.append(stdout + stderr)
            extra_files.extend((raw_ir.name, unwind_ir.name))
            backend = {"kind": "mojo-llvm-ir-with-windows-unwind", "path": str(clang),
                       "version": clang_version, "required_release": WINDOWS_CLANG_VERSION,
                       "binary_sha256": sha256(clang), "flags": backend_flags,
                       "ir_change": "uwtable(async) function attributes only",
                       "definitions_updated": count}
        symbol_audit = ""
        imports_audit = ""
        exports_audit = ""
        object_unwind_audit = ""
        linked_unwind_audit = ""
        if target == "linux":
            cc = tool(args.cc, ("clang", "cc"), "Linux C linker")
            readelf = tool(args.readelf, ("readelf",), "readelf")
            nm = tool(args.nm, ("nm",), "nm")
            command = [str(cc), "-shared", "-nostdlib", "-Wl,-z,defs",
                       "-Wl,--no-undefined", "-Wl,-soname,libjanas_mojo.so",
                       str(obj), "-o", str(stage / artifact)]
            run(command)
            commands.append(command)
            symbol_audit = run([nm, "--undefined-only", str(obj)])[0]
            imports_audit = run([readelf, "-d", str(stage / artifact)])[0]
            exports_audit = run([nm, "-D", "--defined-only", str(stage / artifact)])[0]
            undefined = run([nm, "-D", "--undefined-only", str(stage / artifact)])[0]
            if symbol_audit.strip() or undefined.strip() or "(NEEDED)" in imports_audit:
                fail("Linux module has external/runtime dependencies:\n" +
                     symbol_audit + undefined + imports_audit)
            found_exports = {line.split()[-1] for line in exports_audit.splitlines() if line.split()}
            linker_version = run([cc, "--version"])[0].splitlines()[0]
        else:
            command = [str(linker), "/dll", "/noentry", "/machine:arm64", "/Brepro",
                       "/out:" + native_path(linker, stage / artifact),
                       *("/export:" + name for name in EXPORTS), native_path(linker, obj)]
            run(command)
            commands.append(command)
            symbol_audit = run([readobj, "--symbols", native_path(readobj, obj)])[0]
            imports_audit = run([readobj, "--coff-imports", native_path(readobj, stage / artifact)])[0]
            exports_audit = run([readobj, "--coff-exports", native_path(readobj, stage / artifact)])[0]
            object_unwind_audit = run([readobj, "--sections", "--unwind", native_path(readobj, obj)])[0]
            linked_unwind_audit = run([readobj, "--unwind", native_path(readobj, stage / artifact)])[0]
            for symbol in re.findall(r"Symbol \{(.*?)\n\s*\}", symbol_audit, re.S):
                if "IMAGE_SYM_UNDEFINED" in symbol:
                    fail("Windows object has an undefined symbol:\n" + symbol)
            if re.search(r"\b(?:Import|DelayImport)\s*\{", imports_audit):
                fail("Windows module imports a runtime/system library:\n" + imports_audit)
            if ("Name: .pdata" not in object_unwind_audit or "Name: .xdata" not in object_unwind_audit or
                not re.search(r"Function: janas_mojo_panel_rows_v1\b", object_unwind_audit) or
                "RuntimeFunction {" not in linked_unwind_audit or "FunctionLength:" not in linked_unwind_audit):
                fail("Windows panel kernel lacks decoded object/linked unwind information:\n" +
                     object_unwind_audit + linked_unwind_audit)
            found_exports = set(re.findall(r"^\s*Name: (\S+)", exports_audit, re.M))
            linker_version = run([linker, "--version"])[0].strip()
        if not set(EXPORTS).issubset(found_exports):
            fail("required C ABI exports are missing:\n" + exports_audit)
        if sha256(source) != source_hash:
            fail("source changed during compilation; old published artifacts were retained")
        audit = {"undefined_object_symbols": symbol_audit,
                 "dynamic_imports": imports_audit, "exports": exports_audit,
                 "object_unwind_information": object_unwind_audit,
                 "linked_unwind_information": linked_unwind_audit,
                 "windows_unwind_verified": target == "windows",
                 "runtime_free": True, "runtime_initialization_required": False}
        (stage / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (stage / "compiler.log").write_text("\n".join(compiler_logs), encoding="utf-8")
        # Snapshot metadata is complete before publishing any output artifact.
        files = {name: sha256(stage / name) for name in
                 (artifact, obj.name, assembly.name, "audit.json", "compiler.log", *extra_files)}
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "compiler": compiler_identity(mojo, compiler_version),
            "codegen_backend": backend,
            "source": {"path": str(source), "sha256": source_hash},
            "target": {"triple": triple, "cpu": "oryon-1", "architecture": "aarch64",
                       "required_features": ["neon", "dotprod", "i8mm"]},
            "flags": flags, "commands": commands, "linker_version": linker_version,
            "exports": list(EXPORTS), "abi_magic": "0x4a4d000100000001",
            "runtime_free": True, "files_sha256": files,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in (*files, "manifest.json"):
            os.replace(stage / name, directory / name)
        # lld's optional import library has no role in the runtime-loaded ABI.
        print(str(directory / artifact))
        print("  sha256=" + files[artifact] + " (runtime-free; manifest.json audited)")


parser = argparse.ArgumentParser(
    prog="tools/build-mojo.sh",
    description="Build Janas's optional Mojo 1.2.0 ARM64 leaf module, without downloading tools.",
    epilog="Install Mojo separately per https://mojolang.org/install/. See docs/snapdragon-mojo.md. "
           "Windows builds also require Clang 23.1.5 for generated unwind metadata; set "
           "WINDOWS_CLANG, LLD_LINK and LLVM_READOBJ, or LLVM_MINGW_BIN. "
           "No Janas build/profile defaults are changed. Only use on NEON+DOTPROD+I8MM CPUs.")
parser.add_argument("--platform", choices=("linux", "windows", "all"), default="linux")
parser.add_argument("--mojo", default=os.environ.get("MOJO"), help="Mojo executable (MOJO or PATH)")
parser.add_argument("--source", type=Path, default=ROOT / "src/mojo/quant_panel.mojo")
parser.add_argument("--output-root", type=Path, default=ROOT / "lib/mojo")
parser.add_argument("--cc", default=os.environ.get("CC"), help="Linux linker compiler")
parser.add_argument("--readelf", default=os.environ.get("READELF"))
parser.add_argument("--nm", default=os.environ.get("NM"))
parser.add_argument("--lld-link", default=os.environ.get("LLD_LINK"))
parser.add_argument("--llvm-readobj", default=os.environ.get("LLVM_READOBJ"))
parser.add_argument("--windows-clang", default=os.environ.get("WINDOWS_CLANG"))
args = parser.parse_args(sys.argv[2:])
try:
    if platform.machine().lower() not in ("aarch64", "arm64"):
        fail("run this build on ARM64 Linux/WSL; x64 runtime/ABI builds are not validated")
    source = args.source.expanduser().resolve()
    if not source.is_file():
        fail("Mojo source is missing: " + str(source))
    mojo = tool(args.mojo, ("mojo",), "MOJO")
    compiler_version = run([mojo, "--version"])[0].strip()
    if not re.search(r"\bMojo " + re.escape(VERSION) + r"(?:\s|$)", compiler_version):
        fail("verified Mojo " + VERSION + " is required, found: " + compiler_version)
    output = args.output_root.expanduser().resolve()
    for target in (("linux", "windows") if args.platform == "all" else (args.platform,)):
        build(args, mojo, compiler_version, source, output, target)
except (RuntimeError, OSError) as error:
    print("build-mojo.sh: " + str(error), file=sys.stderr)
    sys.exit(2)
PY

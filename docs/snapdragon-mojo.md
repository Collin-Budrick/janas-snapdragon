# Incremental Mojo modules in the Snapdragon fork

The first Mojo module replaces quantized CPU panel calculations inside Janas. The model,
SSD/RAM expert cache, MTP scheduler, sampling, recurrent state and public API remain owned
by Janas. This starts incremental migration toward Mojo; it does not establish that a
complete rewrite improves throughput.

## Module contract

[quant_panel.mojo](../src/mojo/quant_panel.mojo) implements Q4_K, raw Q6_K and Q6_K_P at
existing plane levels 1, 2 and 3. Its raw export covers 1–9 vectors; runtime dispatch
selects Mojo for 2–9 and keeps specialized single-token C execution. Two output rows share
decoded weights across token pairs using I8MM `SMMLA`; an odd token uses `SDOT` in the
same traversal. An odd output row and unsupported shapes use established C.

[mojo.h](../src/llm/mojo.h) defines the private, versioned C ABI. Addresses are explicit
`uint64_t` words, permitting null planes without violating Mojo's non-null pointer
contract. No C structs cross the boundary. The loader checks ABI magic and ARM64
DOTPROD/I8MM capabilities; C asserts block sizes, field offsets and quantization IDs.
Invalid shapes, unsupported CPUs, reference-kernel mode, or an unavailable module fall
back.

The leaf borrows immutable source weights and existing Q8_K activation panels. It
allocates no persistent weights, starts no scheduler and requires no Modular runtime
initialization. Existing C workers execute it in the same process; native Windows
execution uses no Windows-to-WSL calls. Activation-panel storage keeps the existing **8
MiB per-group cap** and is released after workers finish. When only Mojo requests panels,
all-single-vector groups skip their preparation.

Integer corrections, ascending block order and independent Q4_K accumulation chains match
C. Explicit FMA operations remain explicit; scale multiplication and final subtraction
remain separate. Build with `contract=off`. This module has no recurrent state and changes
no FP16 state boundary or snapshot format.

## Build and delivery

Use ARM64 Linux/WSL with **Mojo 1.2.0** installed separately. The build script requires
Python 3 for orchestration and auditing, plus ordinary linker tools. Its compiler pin
rejects unqualified versions. See the official
[installation](https://mojolang.org/install/), [platform
requirements](https://mojolang.org/docs/requirements/) and [C ABI/cross-compilation
documentation](https://mojolang.org/docs/tools/compilation/).

From the project root, with paths adjusted for the installed tools:

```bash
export MOJO=/absolute/path/to/mojo
./tools/build-mojo.sh --platform linux

# Windows tools can be native .exe paths accessible through WSL.
export WINDOWS_CLANG=/absolute/path/to/clang.exe
export LLD_LINK=/absolute/path/to/lld-link.exe
export LLVM_READOBJ=/absolute/path/to/llvm-readobj.exe
./tools/build-mojo.sh --platform windows
```

Linux produces `lib/mojo/aarch64-linux/libjanas_mojo.so`. The experimental Windows route
produces `lib/mojo/aarch64-windows/janas_mojo.dll`; it also requires the validated **Clang
23.1.5** backend. Mojo currently lacks official native Windows support. This project
validates a narrow AOT leaf, not a general Windows Mojo runtime.

Mojo's direct COFF emitter omits unwind metadata for this stack-using kernel. The Windows
build therefore emits Mojo LLVM IR, adds only `uwtable(async)` function attributes, and
lets Clang generate the object and its real unwind records. It audits decoded
`.pdata`/`.xdata` in both object and DLL. It never fabricates unwind bytes or changes IR
arithmetic/instructions. The original and attributed IR are retained for inspection. Both
platforms reject undefined symbols and runtime imports before publishing the module.

[build-mojo.sh](../tools/build-mojo.sh) records source and artifact SHA-256,
compiler/backend identities, commands, target, flags, assembly and symbol audits in each
output directory's `manifest.json`/`audit.json`. The verified target is `oryon-1`; audited
instructions require NEON, DOTPROD and I8MM. Requalify new compiler versions or additional
CPU features. Mojo uses `-O3` with `--fp-mode contract=off`; the existing Janas release C
flags remain the baseline. Generated modules and compiler installations are not committed
to the fork.

## Select the module explicitly

The low-level controls are `JANAS_MOJO=1` and an **absolute** `JANAS_MOJO_LIBRARY` path.
Loading occurs once and the handle lives until process exit. Configure these before
starting Janas; changing the path does not reload code. Missing modules preserve C
fallback and expose a status.

The native Windows launcher resolves the project-relative module path in the opt-in [Mojo
profile](../profiles/snapdragon-xelite-mojo.json):

```powershell
python tools/run-snapdragon.py --profile profiles/snapdragon-xelite-mojo.json --app chat --dry-run
python tools/run-snapdragon.py --profile profiles/snapdragon-xelite-mojo.json --app chat
```

Use native ARM64 Windows Python for an actual launch. The profile retains CPU8, a 32 GiB
expert cache, a 12 GiB reserve, MTP4 and the promoted DeltaNet chain/journal/K-only
settings. The established CPU profile remains the default until matched end-to-end
promotion gates pass. `JANAS_CPU_TILES=1` additionally forces experimental C panels,
useful for isolating C-versus-Mojo kernel cost; `JANAS_CPU_TILES=0` prevents panel
dispatch. Report those controls with results.

## Verify and compare

[test_mojo.c](../tests/test_mojo.c) compares every output word and guards against both
scalar references and current C panels. It exercises formats, widths, strides, FP16 edge
scales, FPCR modes, refusals and pool/tail fallback. The same fixtures run on native
Windows ARM64 and Linux. Absent modules skip ordinary builds; an explicitly requested test
module must load successfully.

After a release build, a Linux example is:

```bash
export JANAS_MOJO_LIBRARY="$PWD/lib/mojo/aarch64-linux/libjanas_mojo.so"
JANAS_CPU_TILES=1 bin/aarch64-linux/test_mojo
JANAS_CPU_TILES=1 bin/aarch64-linux/bench_mojo 512 2048 8 5 8 0 all > lib/mojo/microbench.jsonl
```

For native Windows use the corresponding `.exe` files and DLL path. The microbenchmark
retains first-shape calls and warm ABBA samples. Raw-kernel measurements isolate math;
group measurements include panel allocation, packing, the C pool, dispatch, counters and
freeing. It checks results outside clocks. First-shape calls follow correctness tests and
are not cold-process inference samples. Keep identical panel policy on both arms.

Require at least **5% native median group improvement** before promoting a kernel shape.
Full-profile promotion additionally requires at least 5% geometric-mean throughput
improvement across frozen code/prose/long workloads, no workload regression above 3%, and
no warm first-token regression above the greater of 10 ms or 5%. Freeze
executable/module/model/prompt hashes and initial MTP calibration; retain fresh samples
and five warm samples per workload. Full 80B comparison evidence is recorded separately in
the [Mojo benchmark report](benchmarks/snapdragon80b-20261109-mojo.md).

## Continue the migration by module

Move stateless projection, activation, normalization and routing kernels behind the same
buffer-oriented boundary first. Convert recurrent kernels only after proving every
state/output word, rounding boundary and rejected-prefix recovery. Then migrate state
ownership, scheduling, expert-cache/I/O and application layers with session tests and
comparable measurements at each boundary. Retire the corresponding C module only after its
Mojo replacement qualifies on the supported platforms; retain a reference implementation
during validation.

MAX graph execution is separate. This delivery uses Mojo leaf kernels and Janas's
scheduler/cache, introducing no MAX graph runtime. Modular's [ARM kernel
source](https://github.com/modular/modular/blob/main/max/kernels/src/linalg/arch/cpu/neon_intrinsics.mojo)
and [custom-op interface](https://max.modular.com/develop/build-custom-ops/) are
implementation references. Neither a common quantization label nor a language port proves
exact arithmetic or a speed gain. The 40 tok/s objective remains subject to matched
measurements.

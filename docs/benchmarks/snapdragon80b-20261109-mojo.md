# First modular Mojo ARM64 kernels

The existing fork now supports an optional Mojo module for exact batched Q4_K,
raw Q6_K and Q6_K_P execution. It runs inside the native Windows ARM64 process
or ARM64 Linux process through a versioned C ABI. The single-token path,
thread pool, SSD/RAM expert cache, MTP weights, router, sampler and FP16 recurrent
state remain in Janas. This is the first module of the gradual migration.

## Matched current-profile comparison

Both arms use the original Qwen3-Next-80B-A3B Q4_K_M model/MTP sidecar, CPU8,
32 GiB expert cache, 12 GiB reserve, legacy MTP4 and promoted DeltaNet
chain/journal/K-only settings. The baseline is the actual executable frozen
before this work, not an older pre-promotion build. There are 42 retained samples
per comparison: ABBA across code, prose and the longer prompt, five warm samples
per arm/workload plus fresh-process samples. Prompts and calibration are frozen;
each sample resets conversation/target seed, retains controller learning, and
requires zero cached prompt tokens.

| Workload | Current C MTP4 tok/s | Optimized Mojo tok/s | Rate change | Warm first token C / Mojo (s) |
|---|---:|---:|---:|---:|
| code | 23.90 | 23.90 | -0.01% | 7.885 / 7.894 |
| prose | 17.65 | 18.12 | +2.69% | 1.773 / 1.722 |
| long | 17.74 | 19.23 | +8.36% | 46.217 / 45.108 |

The optimized window's geometric-mean generation-rate change is **+3.62%**.
The module did not meet the **5%** whole-profile promotion threshold, so the C
profile remains the stable default. All workload rate floors and warm first-token
latency gates passed.
The initial module's separate matched window measured **+2.58%**, below the gate.
Do not compare absolute rates across those two windows as a matched initial-versus-
optimized experiment. All retained output hashes/counts match their frozen
references. Raw prompt/total/first-token timings and actual module dispatch
counts are included in the [full results](snapdragon80b-20261109-mojo.json).

## Kernel experiment and exactness

The initial Q4_K port was slower than the equivalent C prepared tile. Replacing
scalar minimum-correction work with widening SIMD pair sums and exact integer
reductions removed that gap while retaining block order and explicit float FMA
chains. At five vectors, the correction version measured **19.9%** and **9.9%**
higher throughput than the equivalent C tile for the 512x2048 and 2048x512
packing/pool-inclusive cases. These are component results, not whole-model gains.
A second scale-unpacking candidate passed exactness but had mixed pooled results
and was not selected. Q6 formats generally remain close to or below C in these
pooled measurements. [All micro samples](snapdragon80b-20261109-mojo-micro.json)
include first-shape calls and ten warm samples per arm/shape/mode.

The final module passes 1,989 fixtures / 178,914 float words across formats,
strides, guards, odd tails and 32 FPCR combinations. Another 90 singleton checks
verify unchanged C results and zero panel preparation/module execution. Final
Linux release/ASAN/UBSAN and native Windows tests pass. Six real-model frozen-
logit widths (1,3,5,8,9,17) at 64 positions match every reference logit, and greedy/
seeded multi-turn MTP/hybrid-fallback checks match plain decoding. These hybrid
sessions did not select lookup; they do not establish actual lookup/MTP source
switching. The [validation receipt](snapdragon80b-20261109-mojo-validation.json)
retains commands, identities and exact evidence boundaries. Sanitizers instrument
the C bridge/harness; they do not instrument the Mojo machine code.

## Build, deployment and limits

The [Mojo guide](../snapdragon-mojo.md) provides build and launch commands for
`profiles/snapdragon-xelite-mojo.json`. Mojo1.2.0 compiles the leaf without an
external runtime. The Windows pipeline adds only unwind attributes to emitted
LLVM IR and uses Clang23.1.5 to produce native ARM64 COFF with genuine unwind
records; Windows recognizes them. Both libraries have zero runtime imports.
Native Windows Mojo tooling remains unsupported upstream; this is a tested AOT
leaf-module pipeline, not qualification of a complete native Mojo runtime.

There is no MAX graph runtime or WSL IPC in inference. Numerical tests do not
qualify every model/context/device, and five warm samples do not control CPU
clocks, thermals or OS file-cache state. Fresh process does not mean cold SSD.
The 40 tok/s objective has not been reached. The existing C profile remains the
stable default; the Mojo profile is opt-in while individual shapes continue
qualification.

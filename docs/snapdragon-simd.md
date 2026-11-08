# Snapdragon CPU SIMD changes

These changes accelerate routing, the Gated DeltaNet recurrence, and selected
expert products on ARM64. They preserve the existing model weights, expert
selection and numerical contracts. The target remains the original
Qwen3-Next-80B-A3B-Instruct Q4_K_M model with its Q4_K MTP sidecar, HIGH quality,
all original expert bits and no expert skipping. Its recurrent state remains
FP16; intermediate recurrence arithmetic remains FP32.

The comparison baseline is source commit
`5cb445776c33aa7eaf596235f5cd2225ea385ff5`. Before changing the kernels, the
native benchmark, server and quality-test executables were copied into
`lib/validation/simd-20261108/` and identified in `baseline-binaries.json`.
The saved benchmark's SHA-256 is
`a73070dd31de59fb5e333f08d1867c61ca308cd8da00c84e8a6f7766d133ed02`.
These saved executables are the immutable A side of the paired comparison.
The recorded executable identities matter because their version strings also
identify a working tree containing local changes.

The [paired results](benchmarks/snapdragon80b-20261108-simd.json) preserve all
10 process cases and 36 samples. Warm generation medians increased from 9.60
to 18.62 tok/s on code, 8.38 to 15.64 on prose, and 9.06 to 14.93 after a
1,304-token prompt. Output hashes and counts match in every sample. The earlier
[80B measurements](benchmarks/snapdragon80b-20261108-results.json) describe the
preceding implementation and are a separate experiment.

## Router products: four experts per vector

The existing loader converts a router's F32 or BF16 weights to row-major FP16
in place. For an original F32 tensor, that canonical FP16 copy occupies only
the first half of its resident allocation. The new ARM64 path borrows the
second half for a four-row interleave. For 48 main routers of 512 experts by
2048 columns, allocating separate FP16 packed copies would consume 96 MiB;
this implementation reuses the existing allocations instead. Eligible MTP
routers use the same rule.

Each packed group stores, for every input column, the FP16 weight from four
successive experts. The four NEON lanes are independent expert outputs:

```text
acc[lane] = 0
for i = 0 .. d_model - 1:
    acc[lane] = acc[lane] + float(weight[expert + lane, i]) * input[i]
```

Each lane follows the original left-to-right FP32 accumulation. Multiplication
and addition remain separate, with `-ffp-contract=off` in the build; there is
no horizontal reduction that would reorder a router's sum. Packing copies
FP16 bits and adds no precision conversion beyond the loader's existing one.

Integration requires an original F32 router, the layer's expert count equal
to the model's expert count, a count divisible by four, matching tensor
dimensions, sufficient resident space and a canonical copy at the expected
tensor address. Other geometries and source formats retain the existing row
path. Packing rejects nonfinite weights before writing the destination, so
those routers retain their previous numerical behavior. The canonical
row-major first half remains intact for GPU consumers; the packed pointer
borrows model-owned storage and needs no separate allocation or free.

The product helper also falls back to its scalar implementation for FPCR
half-flush, alternate-half or non-nearest rounding modes. It does not change
the caller's floating-point environment. See
[router.c](../src/llm/router.c),
[model_load.c](../src/llm/model_load.c) and
[model_ffn.c](../src/llm/model_ffn.c).

## DeltaNet: four independent state rows

The half-state recurrence stores `M[j][i]` in row-major order. For one row,
the reference performs these operations in order:

```text
r[i] = float(M[j][i]) * decay
sk   = ordered_sum_i(r[i] * k[i])
d    = (v[j] - sk) * beta
u[i] = r[i] + d * k[i]
sq   = ordered_sum_i(u[i] * q[i])
M[j][i] = round_to_fp16(u[i])
output[j] = sq * scale
```

The NEON implementation places four independent rows in four FP32 lanes.
Temporary four-by-four transposes make these lanes available without changing
the persistent state layout. Both sums retain the reference's order over
`i`; multiply-adds are not fused. Output uses the FP32 updated values, and
each state element is rounded to FP16 once when stored, as before.

The vector path serves supported dimensions up to 256 when the dimension is
divisible by four. Remaining rows and other supported dimensions use the
reference path. Hardware half conversions are enabled only under the tested
FPCR half/rounding configuration. Nonfinite conversion groups use the software
converters to preserve payload behavior; incompatible FPCR modes select the
reference recurrence. The state is not promoted to FP32 and its storage
precision is not reduced. See [deltanet.c](../src/llm/deltanet.c) and
[deltanet.h](../src/llm/deltanet.h).

`janas_gdn_kernel_name()` describes the separate FP32-state entry point;
its `ref` label on ARM does not describe the optimized FP16-state path used
by this model. The half-state tests and full-model comparisons validate that
path directly.

## Expert products: bounded I8MM selection

The ARM64 I8MM implementation tiles pairs of Q6_K_P expert rows across several
activation vectors. It reconstructs the same quantized values from the bit
planes, performs integer products, and retains the reference scaling and
accumulation results. Weight-row, plane, activation and output strides are
handled separately; remaining rows use the established kernels.

Automatic selection requires ARM DOTPROD and I8MM support, the Q6_K_P format,
all six weight bits, two 256-value blocks per row (`nb=2`, or 512 columns),
and exactly four or eight activation vectors. Other widths, vector counts,
lower-bit prefixes and quantization formats keep their existing paths.
In particular, an odd-vector tile and tail can be slower than the existing
multi-vector kernel, which already shares weight unpacking.

On ARM64, `JANAS_CPU_TILES` controls this selection:

| Setting | Behavior |
| --- | --- |
| Unset | Automatic selection for the bounded cases above. |
| `0` | Disable the new I8MM tiles and retain the established row kernels. |
| `1` | Experimental selection for other supported Q6_K_P shapes and prefixes; also select the experimental fused one-vector full-six-bit SDOT path. Hardware capability checks still apply. |

Automatic selection means leaving the variable absent, rather than setting
it to the string `auto`. Experimental selection is not the profile default.
Both [run-snapdragon.py](../tools/run-snapdragon.py) and
[benchmark-snapdragon.py](../tools/benchmark-snapdragon.py) clear inherited
`JANAS_CPU_TILES`, as they do for KV and attention overrides. The benchmark
records the cleared value as JSON `null`, and the parent environment remains
unchanged. No new case-schema field or CLI knob is required. See
[quant.c](../src/llm/quant.c) and [quant_neon.h](../src/llm/quant_neon.h).

## Numerical and model validation

The focused tests check numerical equality and dispatch separately:

- [test_router.c](../tests/test_router.c) checks bit patterns against the
  scalar router, representative and tail geometries, adjacent resident
  halves, overlap/size rejection, guard bytes, all 65,536 half patterns and
  relevant FPCR fallbacks.
- [test_deltanet.c](../tests/test_deltanet.c) compares every half-state word
  and FP32 output bit across repeated steps, split row ranges, tails, signed
  zeros, subnormals and nonfinite values. Test-enabled conversion checks cover
  all half patterns, both sides of rounding midpoints and FP32 edge patterns.
  ARM tests also exercise the FPCR fallback modes. Existing x86 FMA kernels
  have their own numerical contract; these exact ARM checks do not redefine it.
- [test_quant.c](../tests/test_quant.c) checks packed and padded Q6_K_P
  storage, unaligned weights and planes, independent strides, odd row/vector
  tails, signed scale extremes, half-scale edges and all three prefix levels.
  It compares the complete output allocation by bit pattern, including guards
  and untouched padding, and verifies whether automatic, forced or disabled
  dispatch actually used tiles.

Model validation then uses the original weights with controlled CPU-only
settings. The hardened [llm_eval.c](../tests/llm_eval.c) fails on any block or
rejection-replay mismatch, including computed rows subsequently rejected,
and on nonfinite logits, failed model calls, incomplete reference reads or
dump I/O errors. `block=B reject=R` checks the first 256 input tokens, or all
tokens when fewer are supplied, and reports that coverage explicitly.
Rejection requires `0 <= R < B` and a replay-capable block size. External
reference comparisons remain numerical by default; `strict=1` requires their
logit bits to agree too. The saved baseline evaluator predates these enforcing
checks, so its exit status alone is not an exactness gate.

[llm_session.c](../tests/llm_session.c) retains its greedy and seed-42 sampled
comparisons across turns and speculation. It now selects available compute
CPUs from the detected layout and valid I/O pins from the CPU split. When
there is no usable pinned I/O set, a positive unpinned worker count is used,
with four workers as the fallback. Original expert bits, original head
precision, HIGH quality and no expert skipping are explicit. This avoids
the previous assumption that CPUs 12 through 19 exist on a 12-core X Elite.

## Paired performance protocol

Numerical gates precede timing. The prepared native Windows runner is
`lib/validation/simd-20261108/run_paired_cpu.py`; it prints a plan by default
and launches benchmarks only with `--execute`. Its bounded protocol is:

1. Use saved baseline A and candidate B, verifying executable identities.
   Clear inherited `JANAS_*` choices and use CPU8, a 32 GiB expert-cache
   request, MTP cap 4, prefetch off, context 2048, greedy seed 20260926 and
   128 output tokens. The default architecture KV representation and FP16
   recurrent state are retained.
2. Run code and prose in ABBA process order. Split five measured warm
   rounds per binary across three plus two rounds in fresh processes. Each
   process also records its fresh sample. Run the longer code prompt once
   per binary with one fresh and three warm samples, checking that input
   plus output fits the context.
3. Verify the native child inherits the same eight-CPU affinity budget.
   Use a fresh conversation for every sample, five-second gaps between
   processes, and no concurrent model work, builds or payload downloads.
   Warm rounds reuse that process's expert cache; OS file-cache state is
   uncontrolled across processes.
4. Preserve raw logs, commands, controlled environments, all sample timings,
   output hashes/counts, draft counters, cache counters and resident-memory
   observations. A mismatched or nonrepeatable output signature makes a
   comparison ineligible. Strict logit and state checks remain separate from
   the text-hash gate.
5. Compare generation rates per workload, retaining fresh results and input
   time separately. Diagnostic phase profiles and kernel-only microbenchmarks
   guide investigation; the paired model runs establish end-to-end benefit.
   Logical expert reads do not measure physical SSD throughput, and peak RSS
   is a process-lifetime value.

The paired report identifies executable hashes, quality gates, all samples
and measurement limits. It uses matched trials rather than the separate
historical 16.41 tok/s measurement as the comparison control. All 51 native
Windows unit programs and 52 Linux sanitizer-suite programs exited
successfully; optional platform checks can skip. Full-model checks reproduce
the saved 64-position logits exactly at batches 1 and 8, with strict
rejection/replay and seeded speculation/session checks also passing.
## Server inference thread placement

The server opens its main model on one thread and runs inference on another.
The pool executes logical worker zero on the inference caller, while its
background workers are individually pinned. Windows pthreads do not inherit
the opener's individual affinity, so the server now saves the main model
opener's actual CPU/group mask and applies it once at engine-thread entry.
This preserves the selected CPU rather than assuming a CPU number. The
optional embedding model does not replace the saved main-model mask.

The paired performance results above come from the benchmark executable and
remain separate from server API verification. Server checks verify the thread
placement, arithmetic response and the same 128-token code output signature.
They do not establish a separate server speedup or attribute all observed API
timing variation to thread placement.

The [server request regression test](../tests/test_server_request.c) widens a
new thread's incoming mask before invoking the real engine entry point. It
checks both saved-affinity and unavailable-affinity cases, then restores the
test opener's original mask. This check passed on native Windows ARM64 and
Linux ASan; removing only the entry restore in scratch copies caused the
expected failure on both systems. No model is loaded by this test.

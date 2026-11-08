# Snapdragon Windows ARM64 fork

The first target is a Dell XPS 13 9345 with a Snapdragon X Elite X1E-80-100,
64 GB RAM, a 2 TB NVMe SSD, Adreno X1-85 and Hexagon v73. Native CPU inference
works with the original Qwen3-Next-80B-A3B Q4_K_M JNS weights. Optional Vulkan
and Hexagon tensor backends keep Janas's expert selection and SSD/RAM cache.

The branch integrates upstream main and the ARM/NEON and Windows ARM64 work
from PRs 25, 32 and 33. The allocator and MTP split-cache fixes proposed in
PRs 31 and 34 are already covered by current upstream main.

The test weights come from
[the upstream model repository](https://huggingface.co/prabanta-dev/Qwen3-Next-80B-A3B-Instruct-JNS/tree/2ee57d760d376b590f10342617709632087e4c19),
pinned at `2ee57d760d376b590f10342617709632087e4c19`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `qwen3-next-80b-a3b-q4km.jns` | 48,411,922,432 | `fa3255c73106391e9cb5a97efd30f77a928fb4518d40060f795a8cff6656daed` |
| `qwen3-next-80b-a3b-mtp-q4k.jns` | 937,992,192 | `5c9bc4cb6f739193fac5e4daa30c922f73255acf41443ecfc0123ef084584559` |

Both complete files were size-checked, SHA-256 verified and passed `jns_check`.
Weights and local runtime binaries stay outside Git.

## Measured settings and starting the server

The selected profile uses eight CPU cores, a 32 GiB main expert cache, the
separate MTP cache, a four-token draft ceiling and demand reads from the SSD.
The 48.4 GB model stays on disk; only the needed experts enter the RAM cache.
The MTP planner adjusts its actual draft length. On the repeated code workload,
this profile generated **16.41 tokens/s**, the median of three warm samples.
The decoded output hash and 128-token count matched the CPU reference in each
sample.

| Code workload, 128 generated tokens | Warm tokens/s | Samples |
|---|---:|---:|
| CPU before Q6_K_P NEON fix, no drafts, 8 cores | 6.86 | 1 |
| CPU with Q6_K_P NEON, no drafts, 8 cores | 12.33 | 1 |
| CPU + MTP, ceiling 4, 8 cores, 32 GiB cache | **16.41** | **3, median** |
| CPU + MTP, ceiling 8, 12 cores, 32 GiB cache | 12.66 | 1 |
| CPU + MTP, ceiling 8, 8 cores, 8 GiB cache | 8.17 | 1 |
| CPU + lookup, ceiling 8, 8 cores, 32 GiB cache | 12.46 | 1 |
| CPU + MTP, ceiling 4, automatic prefetch | 16.51 | 1 |
| CPU + Hexagon draft products, MTP ceiling 8 | 12.52 | 1 |
| Native llama.cpp CPU, 8 cores, F16 K/V | 11.39 | 2, median |

These are workload-specific generation measurements, not end-to-end request
rates. General prose reached 12.64 tokens/s without drafts and 12.65 with MTP;
its drafts helped much less. Prefetch did not demonstrate a material gain.
Eight cores beat twelve in the Janas trials. The native llama.cpp 12-core trial
was stopped after 878.9 seconds without completion and receives no speed result.
The llama.cpp comparison uses the matching official Q4_K_M GGUF, the same
276-token code input and 128 generated tokens, with different cache precision
and backend arithmetic. It is a baseline, not proof of identical engine output.
Prompt time varied substantially in those llama.cpp runs and is not ranked here.

Successful NPU operation counters verified real Hexagon execution in the
draft-only configuration; it was slower than the selected CPU/MTP profile.
GPU chat trials fell back to the CPU after a startup self-check failure and
are not counted as GPU acceleration. Numerical accelerator probes are discussed
below. The [machine-readable report](benchmarks/snapdragon80b-20261108-results.json)
contains individual samples, identities, cache statistics and validation scope.

From the checkout in a native Windows ARM64 terminal:

```powershell
python tools/run-snapdragon.py --profile profiles/snapdragon-xelite-cpu.json `
  --app server --port 8089 --test
```

Open `http://127.0.0.1:8089/test` for the served chat page, or use the
OpenAI-compatible API at `http://127.0.0.1:8089/v1`. The local deployment was
checked through the API: arithmetic returned `391` for `17 * 23`, and the
code request reproduced the benchmark's decoded 128-token output. HTTP delivery
of the chat page was checked; browser interaction and document retrieval were
not part of this validation. No automatic startup service is installed.

For terminal chat, replace `--app server --port 8089 --test` with `--app chat`.
Use `--dry-run` to inspect the command and identity checks first. The launcher
checks model size and the first MiB hash; the recorded full-file hashes were
verified at download time rather than reread on every launch. The default
context is 2,048 tokens, and the profiles do not impose the benchmark's
128-token reply limit.

Two additional profiles are available:

- `profiles/snapdragon-xelite-auto-cache.json` sizes the main RAM cache from
  available memory with a 12 GiB reserve. One warm code sample reached 16.71
  tokens/s; it is useful when other applications need RAM.
- `profiles/snapdragon-xelite-npu.json` reproduces the experimental, verified
  Hexagon draft configuration. It needs the matching signed runtime described
  in [the Hexagon guide](snapdragon-hexagon.md).

## Memory and speculation

Experts occupy a bounded RAM cache and load from the JNS file on misses.
`--cache` requests GiB; zero lets Janas size the cache from available memory
and `--reserve`. An explicit size overrides that automatic calculation.
The MTP sidecar has its own expert cache, reported separately. Process RSS
also includes resident tensors, contexts, accelerator buffers and other
allocations, so cache capacity alone is not a total memory limit.

For this architecture, Janas defaults to custom INT8 attention K/V with one
FP32 scale per token/head and FP16 recurrent state. HIGH preserves the selected
expert precision; it does not change those cache formats. The llama.cpp
comparison uses F16 K/V as a conservative higher-precision baseline. Its cache
representation and arithmetic are therefore not identical to Janas's.
The harness and launcher clear inherited `JANAS_KV` and `JANAS_ATTN` overrides
so the saved profiles retain these architecture defaults.

The benchmark exposes four speculation modes:

- `none`: ordinary target-model generation.
- `lookup`: draft from matching token sequences in the current conversation.
- `mtp`: require the matching MTP head and verify its drafts with the target.
- `auto`: retain Janas's automatic choice.

Lookup currently scans conversation history; it is not a persistent external
ngram corpus. `--draft-tokens` sets the ceiling from 1 to 16. The planner may
propose fewer drafts. Drafted and accepted counts show whether a workload
actually exercises speculation. The supplied repetitive-code prompt is useful
alongside general prose; it contains input code rather than a saved completion.

## Fixed-quality benchmark

Use native Windows ARM64 Python after a release build. The GPU setup helper
`tools/setup-snapdragon-gpu.sh` supplies a pinned glslang and Vulkan headers
for the WSL cross-build. See [the Hexagon instructions](snapdragon-hexagon.md)
for the matching optional runtime, headers and signing prerequisites.

```powershell
python tools/benchmark-snapdragon.py --plan
python tools/benchmark-snapdragon.py --stage screen `
  --profiles cpu --speculations none,lookup,mtp `
  --threads 8 --cache-gib 32 --gen 128 --rounds 1 --max-cases 3 `
  --prompt-file tools/benchmark-prompts/repetitive-code.txt
```

The harness uses the same prompt, greedy sampling and seed across modes,
HIGH quality, original expert/head bits and no missing-expert skipping.
It records the model identities, executable hash, exact commands, fresh-process
and warm samples, output hashes, cache activity, draft acceptance and accelerator
completion counters. Each warm sample starts a new conversation while retaining
the model's RAM cache. Windows's file cache remains uncontrolled; a new process
does not establish a cold SSD.

The bounded refinement stage varies threads, cache size, draft ceiling and
device assignment by phase. Compare sustained generated tokens per second
separately from prompt processing, first-token latency and model load time.
Logical expert-file bytes are reported separately from physical SSD traffic,
which is not measured by these counters. Peak RSS covers the process lifetime.

An accelerator is verified only when its successful inference counters advance;
creating plans or opening a device is insufficient. Counters cover the whole
sample. Use a controlled configuration with one accelerated phase when attributing
work to prompt processing or generation.

Manual `--gpu-min-tokens` and `--npu-min-tokens` thresholds permit controlled
small-batch trials. Their default zero retains runtime policy. The optional
Hexagon-only `--npu-min-macs` threshold filters matrices by
`rows * columns * batch` before device-plan creation. It can keep tiny products
on the CPU to avoid dispatch overhead; zero preserves the normal behavior.
Each selected value is recorded in the case and reproduced by the launcher.

## Numerical validation and daily use

`npu_hexagon_check --packed` explicitly exercises hardware matrices, shared
weights, rebatching and repeated provider creation. `gpu_check` compares GPU
products with the CPU. These checks are separate from the optional-backend unit
suite. On this Adreno driver the strict GPU comparison reports floating-point
differences, so a successful GPU launch alone does not establish numerical parity.

`quality_check` teacher-forces the same tokens through the main model and
compares finite logits, NLL/perplexity, top-1 agreement and KL with a streamed
CPU reference. It uses the original weights and all selected experts. This is
a numerical probe, not a broad model-quality evaluation. Differences in generated
text remain discovery results until evaluated; the harness does not automatically
recommend a faster, changed-output run.

The [recorded 64-position probe](benchmarks/snapdragon80b-20261108-quality.json)
found identical CPU logits at batches 1 and 8. Actual accelerator completions
were verified for the following comparisons:

| Device | Batch | NLL delta from CPU | Top-1 agreement | Perplexity |
|---|---:|---:|---:|---:|
| CPU reference | 1 / 8 | 0 | 100% | 29.7654 |
| Hexagon | 1 | +0.01142 | 100% | 30.1073 |
| Hexagon | 8 | +0.01959 | 96.875% | 30.3543 |
| Adreno | 1 | +0.03058 | 96.875% | 30.6897 |
| Adreno | 8 | +0.02840 | 98.4375% | 30.6229 |

The Q6_K_P NEON optimization also reproduced the pre-optimization CPU logits
exactly at all 64 positions, at batches 1 and 8, with the original plane-based
weights. [Its validation record](benchmarks/snapdragon80b-20261108-neon-validation.json)
includes the probe and executable identities. Native quantization tests cover
multiple widths, vector counts, padded strides and scale edge cases.

These small fixed-input measurements do not establish broad quality parity.
Accelerated profiles remain experimental; retain the CPU reference when
evaluating speed or changed continuations. Hexagon's small packed kernels can
quantize activations internally even when the original weight blocks are kept.

The GPU numerical probes used the direct model API. Fresh isolated chat trials
also encountered a failed GPU startup self-check and then used CPU fallback.
Creation now rejects that broken state with an explicit reason. The numerical
probe results alone do not establish GPU chat acceleration on this driver.

`tools/run-snapdragon.py` launches chat or server from an explicitly selected
profile. It verifies model identities, pins the CPU budget before process creation,
sets runtime paths and passes the measured cache, phase and speculation choices.
Use `--help` for its profile schema and `--dry-run` to inspect a launch. No winning
profile is assumed merely because a device is present.

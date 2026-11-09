# Exact S2 runtime evaluation

The exact S² adaptation remains opt-in. It did not pass the default-profile promotion gates in this 42-sample matched ABBA trial.

| Workload | Frozen MTP4 tok/s | S2 tok/s | Change | TTFT gate |
|---|---:|---:|---:|---|
| code | 23.650 | 22.753 | -3.79% | Pass |
| prose | 19.079 | 19.174 | +0.50% | Pass |
| long | 17.922 | 19.232 | +7.31% | Pass |

Geometric-mean throughput change: +1.24%. Strict promotion gates: failed.

MTP4 remains the default. The aggregate gain is below 5%, and code exceeds the 3% regression allowance. All output identities and first-token latency gates passed. The [opt-in profile](../../profiles/snapdragon-xelite-s2-exact.json) retains CPU8, a 32 GiB expert cache, a 12 GiB reserve, GDN111 and four MTP drafts.

The implementation adds exact MTP maintenance/last-row work and a bounded two-branch root fork with staged chain verification. Target routing, model weights, quantization, FP16 boundaries and seeded target sampling remain unchanged. Native MTP KV stays distinct from target KV.

Branch choices use measured complete costs and compatible fresh/cached correction observations. Eager preparation is paid on every parent cycle; cached head scoring alone is not assumed profitable. Unknown evidence retains the fallback with bounded independent probes.

Across the 15 warm candidate replies, eight alternate branches were built through bounded probes. One cached descendant was accepted in prose; none was consumed in code or the longer prompt. The measured longer-prompt gain cannot be attributed to branch reuse alone. Pooled MTP maintenance time fell by about 13–19% across workloads, while target verification still dominated generation time. All slow samples are retained, and the code regression is counted against promotion.

This is a bounded root-fork adaptation around Janas's native MTP, rather than the complete published S² self-drafter. Immutable embedding/output weights were already shared. Native MTP has its own learned layer and KV slot; aliasing target KV would be incorrect. Approximate target expert gating and rerouting remain excluded.

Current native checks cover six frozen-logit widths, seven session configurations with 35 transcript comparisons, 67 unit programs and 36 copying/topic-switch transcripts. A single-model MTP proof compared every relevant KV/hidden/logit word over widths 1–17 and 153 accepted-prefix cases. The separate instrumented branch proof constructed 53 native alternates, consumed and accepted one cached descendant, and matched complete greedy/seeded transcripts across session changes. Its scheduling is functional evidence, not a production timing sample. The validation report also contains sanitizer, portable x64, 120-test tooling and public API activation receipts.

The validation loop now supports `--fast --incremental-cache <cache-directory>`. For identical binaries, sources and inputs, the measured repeat of six quality cases, seven session configurations and 67 unit checks took **9.09 seconds**, versus **301.55 seconds** for fresh execution: **33.16× faster through explicitly labelled proof reuse**. No model or unit executable ran again. Binary, source, model, token, reference, calibration, platform, environment or effective-control changes invalidate the relevant entries; failed matrices are never inherited as passing.

Fresh validation has not demonstrated a 3× speedup. The fresh matrix timings were 74.09 seconds for quality, 200.29 for sessions and 27.17 for units. Quality was slower than the earlier six-process 58.90-second result. The session matrix reduces model runs from 35 to 17 while retaining all five comparisons per case, and checks restored state against a full-logit oracle at each boundary. These functional timings are separate from inference throughput; the matched benchmark remains serial with all 42 samples.

Neither 40 nor 50 tok/s was demonstrated. The measurements cover this Windows ARM64 device and tested text workloads, with uncontrolled thermal, clock and operating-system file-cache effects.

Raw results: [snapdragon80b-20261109-s2-exact.json](snapdragon80b-20261109-s2-exact.json). Validation: [snapdragon80b-20261109-s2-exact-validation.json](snapdragon80b-20261109-s2-exact-validation.json).

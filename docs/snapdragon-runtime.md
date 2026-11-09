# Exact chain execution and runtime experiments

These changes optimize the existing model's execution and speculative proposal
pipeline. They keep the current target weights, quantization, expert selection
and full target vocabulary. They add no pruning, target-expert substitution,
new draft weights or speculative trees. The reference remains the fork's
existing CPU arithmetic, including FP16 recurrent-state rounding after every
token; mathematical equivalence alone is insufficient.

The new paths are experimental. Ordinary `auto` speculation, the legacy
planner and the legacy draft vocabulary remain the defaults. New activation
panels and the DeltaNet chain/journal require explicit selection. Existing
qualified ARM kernel choices remain available in automatic mode. Unit tests
establish local correctness; whole-model quality and timing gates determine
whether any new setting should become a recommended default.

The implementation builds on work Janas already did: layer-major block
verification, grouping tokens by expert, a recurrent base/tip pair, and replay
from logged projected inputs. Its existing 32-row state tile already stays in
one core's cache across a block. The chain adds a four-row transposed FP32
scratch representation across tokens; the journal removes repeated recovery
dot products. It does not introduce another full-state snapshot per candidate.

The native chat, fair-chat benchmark and server accept these controls:

| Option | Values and behavior |
|---|---|
| `--speculation` | `none`, `lookup`, `mtp`, `hybrid`, `auto`; `mtp` and `hybrid` require an open MTP head. |
| `--draft-tokens` | Ceiling of 1–16 proposed tokens; the default ceiling is 8 with MTP and 4 otherwise. |
| `--spec-planner` | `legacy` by default, `cycle` for measured cycle utility, or `fixed` for controlled depth experiments. Selecting `hybrid` or adaptive draft vocabulary with `legacy` resolves to `cycle`; `fixed` remains fixed. |
| `--spec-explore` | 0–100; the cycle controller's default is 20. |
| `--draft-vocab` | `legacy` by default, or experimental `adaptive`. |
| `--draft-vocab-rows` | `0` for automatic selection, or `8192`, `16384`, `32768`; bounded by the existing draft-head allocation. |

For example, append the following to an existing working native command with
its current model and MTP paths:

```text
--speculation hybrid --spec-planner cycle --spec-explore 20 --draft-tokens 8 --draft-vocab adaptive --draft-vocab-rows 0
```

The four runtime options also have environment equivalents:
`JANAS_SPEC_PLANNER`, `JANAS_SPEC_EXPLORE`, `JANAS_DRAFT_VOCAB_POLICY` and
`JANAS_DRAFT_VOCAB_ROWS`. Invalid values are rejected. Set them before creating
the chat. CLI options set these process-local values; the versioned C setter
can configure a chat explicitly.

The cycle controller chooses a draft depth from zero through the available
ceiling using delivered tokens per measured cycle time. It includes proposal
preparation, verification, sampling and required MTP maintenance rather than
optimizing acceptance alone. It first collects at least three observations per
available action, uses a 5% switching margin, and then makes exploratory choices
at the configured rate. Its random stream is separate from the target sampler.
Observations distinguish source, warm state and draft-vocabulary policy. A
small context based only on past expert-group reuse backs off to aggregate
observations when it has insufficient samples; it does not probe or alter
future target routing to choose a draft depth.

Hybrid mode checks canonical token history before computing an MTP proposal.
The session-local index searches suffix lengths eight down to one, selects the
latest exact earlier occurrence with a known continuation, and proposes at most
16 tokens. It indexes confirmed history, never rejected draft tokens. It is
allocated lazily, owns at most 8 MiB including metadata, and retains a bounded
tail for longer histories. The current layout uses 6,422,608 bytes for 32,768
history tokens. Truncation, replacement and context sliding resynchronize the
index. Lookup hits still pass through ordinary target verification; accepted
inputs still receive required MTP-state maintenance.

Filters and nonzero presence/frequency penalties constrain the new speculative
modes to a one-row target pass. Guard counters make that fallback visible.
Multi-token filtered or penalized speculation is outside this implementation.

Adaptive vocabulary selection changes only MTP proposals. It chooses among
the legacy draft rows and 8,192/16,384/32,768-row candidates, using special tokens,
high target-logit tokens, recent canonical-history tokens and legacy coverage.
Selected weight rows are copied byte-for-byte from the original source into
the existing draft-head allocation. No row is requantized. Metadata is extra;
a second full output-weight allocation is not created.

A vocabulary epoch normally lasts at least 16 matching observed cycles; reply,
history or configuration changes can require a new epoch. Feedback carries its
actual epoch/policy/row tag so delayed results do not advance a different active
epoch. Legacy epochs provide a comparison and can win the cost decision. The
target output head continues to score the full vocabulary. Repacking marks the
shared draft-head allocation dirty for device caches: that head stays on the CPU
for the rest of the model lifetime, including after restoring legacy rows, so a
cached GPU/NPU plan cannot read stale packed weights. Unsupported geometry or
allocation failures restore the legacy draft path.

The ARM I8MM activation panels repack the existing Q8_K activation bytes once
per product group and share identical input panels among compatible products.
They preserve original Q4_K, Q6_K and Q6_K_P weight bytes and the established
integer/reduction arithmetic. A product group owns at most 8 MiB of panel storage
including metadata; unsupported shapes, tails or exhausted capacity use existing
kernels. Current panel geometry covers up to nine input vectors.

| `JANAS_CPU_TILES` | ARM behavior |
|---|---|
| Unset | Existing automatic selection. New panel shapes have no automatic allowlist entries yet. Previously qualified Q6_K_P cases remain eligible. |
| `0` | Disable the ARM tile path for comparison. |
| `1` | Request experimental tiles/panels where CPU features and geometry permit. |

The DeltaNet switches are independent of the planner and default to disabled:

| Variable | Dependency and effect |
|---|---|
| `JANAS_GDN_CHAIN=1` | Qualified ARM CPU path; up to 17 new verification inputs, with up to 17 pending replay inputs. Four independent rows remain in FP32 scratch across tokens. |
| `JANAS_GDN_JOURNAL=1` | Requires the chain. Lazily retain exact FP32 per-row delta values in token-major order, at most 16 MiB per model. |
| `JANAS_GDN_K_ONLY=1` | Requires a valid recovery journal. During accepted-prefix replay, compute/normalize K while advancing every convolution channel's persistent window; omit unused replay Q/V arithmetic. New tokens compute all channels. |

Set an exact `1` to enable each GDN switch; unset or `0` keeps it off. For
example, in PowerShell:

```powershell
$env:JANAS_GDN_CHAIN = '1'
$env:JANAS_GDN_JOURNAL = '1'
$env:JANAS_GDN_K_ONLY = '1'
```

Every token performs the same ordered FP32 reductions with separate multiply
and add operations. Output is formed before that token's half-state rounding;
the rounded half values are expanded again before the next token. The base is
saved after the accepted **input** prefix: A accepted draft tokens consume A+1
inputs, including the known root. The tip holds the new verification endpoint.
There is no approximate low-rank sum or delayed half rounding.

The journal stores delta values, not intermediate full state matrices. Recovery
still applies decay, the recorded delta update and FP16 rounding sequentially
for every accepted input. It skips the repeated state–K dot and discarded output
dot. The old row's entire prefix is consumed before new deltas overwrite the
same allocation. The current 36-layer, 32-value-head, dimension-128 geometry
needs 9,437,184 bytes for 16 inputs or 10,027,008 bytes for 17, below the 16 MiB
cap. Larger unsupported/budget-exceeding cases retain projected-input replay.

Journal validity includes the recurrent base/log position and buffer identity,
FPCR mode and complete successful-pass publication. Reset, restore, shift,
failed passes and device-produced state invalidate it. A split device
continuation cannot publish a partial journal. Every active compute worker's
FPCR is checked before K-only work is omitted. Unsupported modes/geometry use
the established path; exceptional chain arithmetic recomputes through the
scalar reference. Existing snapshots and their serialized version/payload are
unchanged. The journal is derived scratch and is never serialized; restored
snapshots recover through the existing projected-input log.

The public additions in [llm.h](../include/janas/llm.h) use separate versioned
structures, leaving the previous parameter/statistics layouts intact:

```c
struct janas_llm_speculation_config cfg;
janas_llm_speculation_config_default(&cfg);
cfg.mode = JANAS_LLM_SPEC_HYBRID;
cfg.max_drafts = 8;
cfg.planner = JANAS_LLM_SPEC_PLANNER_CYCLE;
cfg.explore_percent = 20;
cfg.draft_vocab_policy = JANAS_LLM_DRAFT_VOCAB_ADAPTIVE;
cfg.draft_vocab_rows = 0;
int32_t rc = janas_llm_chat_set_speculation_config(chat, &cfg);

struct janas_llm_speculation_metrics metrics = {
    .size = sizeof(metrics),
    .version = JANAS_LLM_SPEC_METRICS_VERSION
};
rc = janas_llm_chat_speculation_metrics(chat, &metrics);
```

Check each return code. Configuration version 1 requires the complete struct;
invalid configuration leaves the chat unchanged. The metrics getter accepts a
supported version-1 prefix size. The older
`janas_llm_chat_speculation(chat, mode, max_drafts)` remains available.

Fair-chat benchmark JSON adds `speculation_metrics`. OpenAI chat/completion
usage adds `usage.janas.speculation`, with `reply_scope: first_choice`, alongside
the existing usage fields. The engine captures that job-owned snapshot with the
first choice's statistics before a later job runs. HTTP serialization never
queries live chat counters. Cycle/source counters cover active decode with
incoming preparation; expert and GDN counters/timers cover the current reply/request, including
prompt processing and the main model's MTP head or separate MTP sidecar. GDN token counters count recurrent-layer inputs; `scalar_replay` denotes
recomputation from the projected-input log and may use NEON. Recap work before
the reply baseline is excluded. Journal bytes, pending preparation,
buffered tokens and current head policy/epoch are gauges. Nested timing fields
overlap; they must not be added as independent wall times. Expert counts are
per-layer/per-pass logical work, not physical SSD activity. Source order is
`plain`, `lookup`, `mtp`, `draft`.

CPU dispatch counters cover CPU-selected synchronous groups, including NPU
residual/failure work that returns to the CPU. GPU-managed CPU fallback is
outside their scope. Prepared row counts come from actual kernel returns;
allocated panel views alone do not establish execution. Prepared dispatch
includes single-token/odd-token SDOT work and is not an I8MM instruction count.
Row-vector counts
multiply logical matrix rows by the number of activation vectors. Panel bytes
are summed allocated workspace, while peak bytes are a model-lifetime gauge.
Panel preparation time includes allocation and packing, excludes release, and
is nested inside the existing forward/draft/verification wall times.

Focused tests cover chain widths 1–17 and accepted-prefix boundaries; bitwise
state/output comparisons, half conversion edges, FPCR fallbacks and nonfinite
payloads; aliased journal reuse; full/K-only convolution replay with repeated
rejection; budget/lifecycle and unchanged snapshot payloads; and metrics JSON
ownership, numeric precision and compatibility. The GDN and server JSON tests
pass on native Windows ARM64 and Linux ARM64, with Linux address/undefined
behavior/leak sanitizer checks. These tests do not establish a whole-model
speed improvement or broad Snapdragon/GPU/NPU qualification.

The [reproduction tools](../tools/snapdragon-runtime-tools.md) provide
plan-first native quality/rejection/session gates and matched component/screen/ABBA
protocols, with configurable frozen baseline executables and model receipts.

## Release validation and measurements

The final CPU runtime was checked against the frozen 80B reference and compared with the previous executable in matched ABBA trials. Each workload has five warm samples per binary; all fresh samples and component timings are retained in the [raw report](benchmarks/snapdragon80b-20261108-runtime.json).

| Workload | Previous tok/s | Candidate tok/s | Rate change | Warm first token, previous → candidate |
|---|---:|---:|---:|---:|
| Code | 16.14 | 23.32 | +44.4% | 9.150s → 7.908s |
| Prose | 16.31 | 18.90 | +15.9% | 1.984s → 1.711s |
| Longer prompt (1,304 tokens) | 17.01 | 17.68 | +3.9% | 50.914s → 49.242s |

The geometric-mean rate change is **+20.3%**.
The CPU profile passed the promotion gates and now enables exact DeltaNet chain, delta journal and K-only recovery. CPU8, the 32 GiB expert cache and 12 GiB reserve remain unchanged.

The planner and draft vocabulary remain legacy by default. The hybrid/adaptive screening runs regressed prose and were not selected for the promotion trial. The explicit `--fair-retain-controller` benchmark mode measures learned costs across samples while clearing conversation state and resetting the target sampler; the historical screen created a new controller for every sample.

New activation-panel shapes remain opt-in. Fixed-width specialization removed compiler-generated hot-loop helper calls. Three flat component cases passed the measured gain/drift gate; variable production expert-group topology has not passed a promotion gate. Existing automatic Q6_K_P choices are preserved. Model-level rates above use those established automatic kernels, not blanket panel activation.

Validation includes 61 native Windows ARM64 unit programs; six 64-position full-vocabulary frozen-logit checks at widths 1/3/5/8/9/17 with zero error; three real-model rejection cases; five multi-turn greedy/seeded configurations and three default-attention configurations. Synthetic tests compare every recurrent state/output word, all 16 partial snapshot prefixes, FPCR changes and journal refusal. Linux sanitizer and emulated Windows x64 checks are separate from native model proof. All 75 Python tooling tests pass on Linux and Windows. See the [correctness receipts](benchmarks/snapdragon80b-20261108-runtime-correctness.json) and [component source bundle](benchmarks/snapdragon80b-20261108-runtime-component-sources.json).

**40 tok/s remains unachieved in this balanced trial; 50 tok/s is also unachieved.** These are one-machine measurements, not forecasts. Fresh process does not mean cold SSD cache, physical memory bandwidth was not measured, and no full-context, sustained-conversation, browser, RAG, GPU or NPU qualification is claimed.

Implementation references: [Arm I8MM](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction),
[S²-MoE](https://github.com/angerybob/S2-MoE), and
[FastMTP](https://arxiv.org/abs/2509.18362). The runtime borrows exact integer
tilings and proposal-cost ideas; it does not import approximate target gating,
trained heads or shared-context replacements. Published speedups are not local
incremental performance measurements.

## Cost-aware hybrid selection

`profiles/snapdragon-xelite-hybrid.json` opts into `--speculation hybrid --hybrid-policy cost --draft-tokens 8 --lookup-drafts 8 --mtp-drafts 4 --hybrid-probe-percent 2`. The global draft limit remains authoritative. The calibrated legacy MTP planner remains the fallback, including confidence stopping and calibration updates. `--hybrid-policy legacy` retains the earlier hybrid cycle controller. The promoted CPU profile changes only after matched performance gates pass.

The selector considers conversation-only exact ngram matches before constructing an MTP draft. Successful lookup avoids MTP head and chain scoring while retaining MTP state maintenance. It compares useful delivered output per complete cycle cost, with suffix reliability classes 1, 2–3 and 4–8, settling/warm separation and vocabulary compatibility. Three compatible observations and a 5% estimated utility margin are required for ordinary lookup selection. Two percent of eligible cycles probe independently of target sampling; initial probes use one draft for short suffixes or two for suffixes of at least four. Probe ceilings increase at most twofold after three observations and at least two-thirds full-prefix acceptance. First-draft rejection suspends its class for 32 cycles. No exhaustive depth sweep runs.

The separate size/version-tagged `janas_llm_hybrid_config` and `janas_llm_hybrid_metrics` APIs leave the existing public speculation configuration, metrics, chat parameters and statistics layouts unchanged. CLI and profiles expose policy, source ceilings and probe percentage. Benchmark JSON includes offered matches, source decisions/reasons, evidence, predicted utilities, accepted/delivered tokens and causal costs.

Recovery remains fused with subsequent DeltaNet convolution and recurrent traversal. Its attribution uses **measured pool wall time apportioned by replay/new row counts**, explicitly reported as `fused_gdn_row_share_estimate`; it is not a separately timed recovery phase. Attribution transfers that estimate once to the rejected proposal causing it and removes the same amount from the next cycle. End-to-end wall time remains measured directly. Selection is predictive and can incur occasional exploratory losses.

Benchmark comparisons must freeze the current promoted MTP4 executable and GDN111 controls, rather than the historical GDN000 baseline. `tools/benchmark-snapdragon-runtime.py` accepts `--baseline-controls` and `--calibration-file`. Each process receives the same frozen calibration in a private cache; neither arm trains the next arm. Both retain controller state while resetting conversation tokens/target RNG between samples and require zero cached prompt tokens. Fresh-process and five warm samples per arm/workload are retained in matched ABBA trials. Multi-turn copying and topic switches are separate activity/correctness controls, not extra workloads added to improve the promotion aggregate.

### Attention consistency found during hybrid validation

The previous AUTO attention selector could choose different query-score precision for the same token depending on verification width. A reproducible 1,024-token boundary case cached `UINT32_MAX` (always floating point) for single tokens and `1024` for batches; after four identical copying replies, ordinary MTP4 diverged from plain decoding at a topic switch. The target weights, KV quantization and DeltaNet rounding were unchanged.

AUTO now follows the **single-token precision policy at each absolute token position** for every width. A batch crossing that threshold is split only within attention scoring, preserving query/output row offsets and custom visibility bounds. Fixed `float` and `int16` settings are unchanged. Existing two-column calibration files remain readable. This corrects target verification rather than choosing a new approximation. Regression coverage includes differing calibration columns, threshold crossings and widths 1–17. Matched trials also freeze the original profile's `attn.txt` in both private caches; runtime timing calibration must not silently change the numerical reference between arms.

### Matched hybrid evaluation (8 November 2026)

[Raw benchmark and source decisions](benchmarks/snapdragon80b-20261108-hybrid-cost.json) and [correctness/attention regression proof](benchmarks/snapdragon80b-20261108-hybrid-cost-validation.json) retain the frozen executable, calibration, controls and all samples.

| Workload | Frozen current MTP4 | Cost hybrid8/MTP4 | Change |
|---|---:|---:|---:|
| Code | 20.34 tok/s | 19.83 tok/s | -2.5% |
| Prose | 16.61 tok/s | 14.32 tok/s | -13.8% |
| Longer prompt | 10.30 tok/s | 14.28 tok/s | +38.7% |

The geometric mean improved 5.2%, but prose failed the 3% regression floor, so **the CPU profile retains MTP4**. Hybrid remains opt-in. These primary warm replies selected lookup only for bounded exploration; the apparent aggregate gain is not evidence of normal lookup exploitation. Long-workload timings varied substantially (warm generation 7.48–16.75 tok/s); all slow samples are retained. The machine used Balanced power on battery, declining from 56% to 32% over the primary trials.

The separate conversation check demonstrated normal lookup use: 696 accepted lookup tokens across copying/return phases, then zero lookup selections and 63 MTP selections across the two unrelated topic switches. Every output token and text byte matched plain decoding under greedy and seeded sampling. Its ordered diagnostic timings are not a promotion benchmark.

A same-binary prose ABBA check with lookup probes disabled measured MTP4 at 11.00 tok/s and cost hybrid at 10.59 tok/s. It did not eliminate fallback timing/draft-depth differences; no deterministic arithmetic discrepancy was found. Timing-based legacy draft-depth decisions and system variability remain performance limitations. This diagnostic does not replace the frozen-current-profile comparison. Neither 40 nor 50 tok/s was demonstrated.

### Prepared MTP verification and calibration

Commit `844ffb6` corrects two shared MTP planner behaviors used by ordinary MTP and cost hybrid fallback. Chain construction still estimates the cost of additional draft steps. Once the chain exists, verification compares every available prefix using the same actual incoming preparation cost plus its predicted target-pass cost. Choosing a shorter prefix cannot recover head or chain work already performed.

A prepared first MTP draft that the planner declines now receives an exact calibration label from the target's already sampled first row. A matching non-stop token is a success; a mismatch or stop token is a failure. Stateful filters and nonzero presence/frequency penalties exclude this update, and lookup proposals cannot supply MTP labels from stale confidence. Later unexecuted draft positions remain unlabelled. This adds no target rows, head scoring or sampler/RNG calls. Drafted/accepted counters continue to count only proposals actually verified.

The diagnostic trace replay changed 21 positive-depth decisions under the preparation-cost correction, with no depth-zero decisions rescued. Six declined first drafts in confidence bin four supplied four correct labels; a sequential replay incorporating those labels changed two subsequent choices. Labels in other bins also reduced confidence. These mechanism checks do not explain the entire earlier 13.8% prose regression or establish a throughput gain. Diagnostic timings span a power-state transition; matched benchmark and promotion results are reported separately.


<!-- hybrid-source-pricing:start -->

## Hybrid source pricing follow-up

Hybrid remains opt-in; the default CPU profile uses MTP4. The latest comparison uses the same current executable for both modes. Earlier trials and failed gates remain available. See the [readable report](benchmarks/snapdragon80b-20261108-hybrid-prose-fix.md), [raw trials](benchmarks/snapdragon80b-20261108-hybrid-prose-fix.json) and [validation](benchmarks/snapdragon80b-20261108-hybrid-prose-fix-validation.json).

<!-- hybrid-source-pricing:end -->


<!-- fast-validation:start -->

## Faster functional validation

`tools/validate-snapdragon-runtime.py --fast` retains the requested logical coverage. Quality validation loads one CPU model for widths **1, 3, 5, 8, 9 and 17**. Session validation loads one model for all **seven configurations**, reuses three immutable plain-decoding reference runs, and retains all **35 greedy/seeded comparisons** plus a bit-identical 32-token boundary check per case. Independent model-free unit executables run with bounded workers inside CPU mask `0xff`; model jobs stay serial.

The fixed16 session stress case explicitly uses an S2 node budget of **32** when S2 is enabled, preserving the required 17-row target verification. Other cases retain the configured eight-node budget. The override appears in the plan and result; it does not change the runtime profile.

Use native Windows ARM64 Python from the repository root. Planning remains the default and launches no executable:

```powershell
python tools/validate-snapdragon-runtime.py --fast --stage session --s2-exact on --plan
```

An explicit cache accelerates repeated checks of unchanged artifacts. Use a new output directory for every invocation:

```powershell
python tools/validate-snapdragon-runtime.py --fast --stage session --execute --s2-exact on --s2-branches 2 --s2-max-nodes 8 --s2-min-gain-percent 5 --reference lib/validation/quality80b-20261108/cpu64.qref --incremental-cache lib/validation/proof-cache --out lib/validation/session-fast-001
```

Repeat with `session-fast-002` to reuse eligible proofs. Omit `--incremental-cache` to force new executions. Cache keys bind executable and project DLL hashes, arguments and generated case-file contents, reference/tokens/attention policy, model receipt and file fingerprints, effective controls, platform and source identities. Only validated passes are stored; a failed matrix cannot populate a passing matrix entry. Reused results retain their original execution/timings and are explicitly marked **inherited proofs**, with zero new executions where applicable.

`--select-check session-fixed-mtp16-adaptive` selects targeted work and explicitly marks the result as a subset. `--import-proofs <passing-output-directory>` can revalidate historical raw proofs for the current artifacts; model proofs also require `--import-model-receipt <completed-checksum-receipt>`. Changed identities and incomplete/failed source directories do not qualify; stored cache artifacts also require matching checksums. Large model payloads bind the pinned receipt plus unchanged file metadata/header checks. Importing older model proofs additionally requires the completed checksum attestation; reuse does not claim a new full payload scan.

Measured on 9 November 2026 with identical artifacts and inputs:

| Functional stage | Fresh fast execution | Same-artifact proof reuse |
|---|---:|---:|
| Six quality widths | 74.09 s | 3.82 s |
| Seven session configurations | 200.29 s | 3.69 s |
| 67 model-free units | 27.17 s | 1.58 s |
| **Total** | **301.55 s** | **9.09 s** |

The measured repeat-run speedup is **33.16×**, retaining 80 logical cases and 35 session comparisons; the cached run launched no new executable or model process. **A 3× fresh-validation speedup was not established**: the fresh quality stage was slower than the earlier 58.90-second separate-process run. These are functional validation timings, not inference throughput gains. The matched inference benchmark remains serial ABBA with all 42 samples and unchanged promotion gates.

Raw local receipts are retained under `lib/validation/s2-exact-20261109/testing-speed/summary.json`, with fresh and cached stage receipts in their separate output directories.

<!-- fast-validation:end -->

## Exact S²-MoE scheduling adaptation (9 November 2026)

The opt-in [S² profile](../profiles/snapdragon-xelite-s2-exact.json) keeps MTP4 and adds `--s2-exact on --s2-branches 2 --s2-max-nodes 8 --s2-min-gain-percent 5`. It skips unused native-MTP FFN/normalization work, builds a bounded root sibling when cost evidence or a bounded probe permits, and stages a cached continuation through the unchanged target verifier after the target samples that sibling. Canonical MTP maintenance still runs. Target routing, weights, quantization, FP16 state boundaries and seeded sampling stay unchanged.

This adapts the exact scheduling concepts to Janas's existing MTP layer. Native MTP already shares immutable embedding/output weights, but its learned layer and KV slot differ from the target; target/draft KV aliasing and approximate expert gating are excluded. The implementation supports a bounded root fork, rather than arbitrary nested draft trees.

The [matched report](benchmarks/snapdragon80b-20261109-s2-exact.md) retains 42 samples: code 23.65→22.75 tok/s (-3.79%), prose 19.08→19.17 (+0.50%), longer prompt 17.92→19.23 (+7.31%). Geometric-mean gain 1.24% misses 5%, and code exceeds the 3% regression allowance. **MTP4 remains the default; S² is opt-in.** All output identities and warm first-token gates passed. Neither 40 nor 50 tok/s was demonstrated.

Native proof includes 17 MTP widths and 153 accepted-prefix KV/hidden/logit comparisons, every frozen target-logit width, seven greedy/seeded session configurations, 67 unit programs, and 36 copy/topic-switch transcripts. A separate instrumented real-model proof executed 53 native alternate branches and accepted one cached descendant with identical complete transcripts; its scheduling is not a production performance sample. See the [raw results](benchmarks/snapdragon80b-20261109-s2-exact.json) and [validation receipts](benchmarks/snapdragon80b-20261109-s2-exact-validation.json).

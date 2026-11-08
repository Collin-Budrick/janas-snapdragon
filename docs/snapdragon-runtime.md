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

Measured results are pending the whole-model quality gate and paired balanced
workload measurements. Existing published SIMD results describe the earlier
release and do not measure these experiments.

Implementation references: [Arm I8MM](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction),
[S²-MoE](https://github.com/angerybob/S2-MoE), and
[FastMTP](https://arxiv.org/abs/2509.18362). The runtime borrows exact integer
tilings and proposal-cost ideas; it does not import approximate target gating,
trained heads or shared-context replacements. Published speedups are not local
incremental performance measurements.

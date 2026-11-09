# Hybrid source pricing evaluation

Hybrid remains opt-in.

The final comparison uses the same current executable and initial calibration for MTP4 and hybrid lookup8/MTP4. Earlier trials and failures are retained in the raw report.

| Workload | MTP4 tok/s | Hybrid tok/s | Change | Warm TTFT gate |
|---|---:|---:|---:|---|
| code | 23.621 | 23.686 | +0.27% | Pass |
| prose | 16.969 | 16.534 | -2.56% | Pass |
| long | 17.243 | 18.981 | +10.08% | Pass |

Geometric-mean throughput change: +2.46%. Strict final promotion gates: failed.

The final comparison retains a 2.56% prose slowdown. Aggregate throughput improves by 2.46%, below the required 5%; the default CPU MTP4 profile remains unchanged.

Promotion requires at least 5% geometric-mean improvement, no workload regression above 3%, and no material warm first-token regression. A passing historical comparison cannot replace a failed current-mode comparison.

## What changed

The old verifier charged for draft steps again after their preparation had already finished, and declined first drafts missed an available calibration outcome. Hybrid also allowed lookup passes and deferred lookup recovery to change the calibrated MTP timing table. These prices could shorten a useful MTP fallback chain. The regression fixtures reproduce those behaviors before the fixes.

The verifier prices already prepared drafts separately from construction. MTP cost learning excludes lookup-selected work and skips lookup-caused recovery samples. The generic hybrid estimator still observes lookup execution. The exact index uses 401,488 bytes at a 2048-token context, with the same 8 MiB ceiling.

Across the five warm samples per primary workload, lookup accepted 8 of 8 proposed drafts for code, 0 of 3 for prose, and 6 of 6 for the longer prompt. All these selections were exploratory probes; none passed the normal exploitation threshold. The longer-prompt gain therefore does not establish that ngram lookup alone produced that improvement.

The separate greedy/seeded conversation test accepted 45 lookup tokens during copying/return phases and exercised nine normal utility-selected lookup cycles on return. Unrelated topic switches selected MTP 54 times and lookup zero times. All generated tokens and text matched the plain-decoding references. These ordered functional timings are separate from the promotion comparison.

## Trial history

| Trial | Purpose | Samples | Result |
|---|---|---:|---|
| A | Initial follow-up against frozen pre-fix MTP4; no execution hold | 42 | -22.77% geometric mean; historical gate failed |
| B | Corrected binary MTP4 versus hybrid probe zero; prose diagnostic, no execution hold | 14 | -5.00% prose; not eligible for balanced promotion |
| C | One-process 17-sample shared-calibration QoS factorial; diagnostic only | 17 | Shared-calibration diagnostic; no promotion claim |
| D | Quiet execution-held comparison against the older frozen MTP4 binary | 42 | +7.28% geometric mean; historical gate passed |
| E | Quiet execution-held same-binary comparison: corrected MTP4 versus hybrid probe two | 42 | -10.07% geometric mean; historical gate failed |

The trace repricing changed 21 positive-depth choices and reversed no zero-depth declines. Six missing labels in confidence bin four included four correct proposals; earlier-only extra labels reversed two historical choices. Negative labels also mattered. These observations do not prove that one change explains the entire earlier 13.8% regression.

## Validation and limits

The old source-pricing behavior fails five targeted assertions; the corrected generator passes the same fixture on Windows ARM64, portable x64 and Linux with ASAN/UBSAN. The context-sized index also passes its current Linux ASAN/UBSAN fixture. The unchanged benchmark harness has 101-test Linux and native Windows ARM64 logs.

Current native unit, fixed-logit, rejection and greedy/seeded session receipts, semantic source activity, sanitizer/portable fixtures and the restored-server verification are in the validation JSON. Ordered semantic and QoS timings are not matched throughput evidence.

Automatic system/display idle is held equally during measured runs, with prior requests restored. Power is sampled; temperature, clocks and other scheduling effects remain uncontrolled. No sleep cause, cold-SSD bandwidth or 40–50 tok/s achievement is claimed.

Raw evidence: [snapdragon80b-20261108-hybrid-prose-fix.json](snapdragon80b-20261108-hybrid-prose-fix.json). Validation: [snapdragon80b-20261108-hybrid-prose-fix-validation.json](snapdragon80b-20261108-hybrid-prose-fix-validation.json).

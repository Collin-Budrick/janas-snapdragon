/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_S2_CONTROLLER_H
#define JANAS_S2_CONTROLLER_H
#include <stdint.h>
#define JANAS_S2_MAX_NODES 32
#define JANAS_S2_MAX_PATH 16
#define JANAS_S2_VOCABS 4
#define JANAS_S2_REUSE 4
#define JANAS_S2_STALE_CYCLES 64

enum janas_s2_reason {
    JANAS_S2_DISABLED = 0, JANAS_S2_NO_CAPACITY, JANAS_S2_NO_EVIDENCE,
    JANAS_S2_STALE, JANAS_S2_CPU_COST_UNKNOWN, JANAS_S2_COSTLY,
    JANAS_S2_BENEFICIAL, JANAS_S2_PROBE, JANAS_S2_WEAK_CONFIDENCE
};
struct janas_s2_sample {
    unsigned count, base_count, matched_comparisons, full_count;
    uint64_t last_cycle;
    double seconds, delivered, verify_seconds, draft_seconds, expert_cpu_seconds;
    double groups, assignments, eager_seconds, saved_seconds;
};
struct janas_s2_root { unsigned count, matched; };
struct janas_s2_controller {
    struct janas_s2_sample width[2][JANAS_S2_VOCABS][JANAS_S2_REUSE][18];
    struct janas_s2_sample branch[2][JANAS_S2_VOCABS][JANAS_S2_REUSE][17];
    struct janas_s2_sample fresh_after_rank2[2][JANAS_S2_VOCABS][JANAS_S2_REUSE][17];
    struct janas_s2_sample cached_after_rank2[2][JANAS_S2_VOCABS][JANAS_S2_REUSE][17];
    struct janas_s2_root root[2][JANAS_S2_VOCABS][10];
    uint64_t rng, last_probe;
};
struct janas_s2_offer {
    uint64_t cycle;
    unsigned warm, vocab, reuse, primary_nodes, available_alt_nodes;
    unsigned branches, max_nodes, min_gain_percent;
    /* Probability of the alternative root in the unmodified draft head.
       This is a draft-confidence heuristic, never a target route prediction. */
    double rank2_probability, draftstep_seconds;
};
struct janas_s2_decision {
    unsigned alt_nodes, reason, probe, evidence, matched_comparisons;
    double probability, baseline_seconds, predicted_seconds;
    double baseline_utility, predicted_utility, resident_cpu_seconds;
};
struct janas_s2_observation {
    uint64_t cycle;
    unsigned warm, vocab, reuse, width, alt_nodes, delivered, accepted_prefix;
    unsigned root_offered, rank2matched;
    double rank2_probability;
    /* Capture target phase/group deltas around MAIN forward only, before MTP.
       Full seconds include causal preparation, verification, sampling,
       maintenance and recovery exactly once. */
    double full_seconds, verify_seconds, draft_seconds, expert_cpu_seconds;
    uint64_t expert_groups, expert_assignments;
    /* Eager cost is paid for every built branch, including unmatched roots.
       continuation_kind: 0 ordinary, 1 fresh MTP after a rank-2 correction,
       2 cached continuation after the same event. width is the ACTUAL next
       target pass width (alternative root plus descendants). For these
       observational continuation samples use the parent's warm/vocab/reuse
       key. At least three fresh AND three cached observations at compatible
       width are required; these are predictions, not exact counterfactuals. */
    double eager_seconds;
    unsigned continuation_kind;
    /* Deprecated internal fields accepted as zero during integration; direct
       alleged savings are not evidence and are never used by the pruner. */
    double saved_seconds;
    unsigned matched_comparison;
};
/* Previous measured assignment occupancy, normalized by its pass width.
   0/1/2 are low/medium/high; 3 means unknown. No future route is evaluated. */
unsigned janas_s2_reuse_class(uint64_t groups, uint64_t assignments, unsigned width);
void janas_s2_init(struct janas_s2_controller *p, uint64_t seed);
struct janas_s2_decision janas_s2_choose(struct janas_s2_controller *p,
                                       const struct janas_s2_offer *o);
void janas_s2_observe(struct janas_s2_controller *p,
                       const struct janas_s2_observation *o);
#endif

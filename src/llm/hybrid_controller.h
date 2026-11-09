/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_HYBRID_CONTROLLER_H
#define JANAS_HYBRID_CONTROLLER_H
#include <stdint.h>

#define JANAS_HYBRID_MAX_K 16
#define JANAS_HYBRID_VOCABS 4
#define JANAS_HYBRID_LOOKUP 1
#define JANAS_HYBRID_MTP 2

enum janas_hybrid_reason {
    JANAS_HYBRID_NO_MATCH = 0,
    JANAS_HYBRID_NO_EVIDENCE,
    JANAS_HYBRID_MTP_CHEAPER,
    JANAS_HYBRID_LOOKUP_FASTER,
    JANAS_HYBRID_LOOKUP_PROBE,
    JANAS_HYBRID_MTP_REFRESH,
    JANAS_HYBRID_COOLDOWN,
    JANAS_HYBRID_NO_CAPACITY
};

struct janas_hybrid_offer {
    unsigned lookup_available, lookup_suffix, lookup_cap, mtp_cap;
    unsigned warm, vocab, reuse, probe_percent;
};
struct janas_hybrid_decision {
    unsigned source, depth, reason, probe, evidence;
    unsigned lookup_evidence, mtp_evidence, suffix_class;
    double lookup_utility, mtp_utility;
    double lookup_seconds, mtp_seconds;
};
/* seconds must include preparation, verification, sampling and maintenance,
   with nested recovery charged exactly once to the source causing it.
   pass_width=0 reports abandoned preparation: accepted_prefix, delivered and
   verification_seconds must then be zero; it supplies cost, not outcomes. */
struct janas_hybrid_observation {
    unsigned source, depth, accepted_prefix, delivered, pass_width;
    unsigned lookup_suffix, warm, vocab, reuse;
    double seconds, verification_seconds;
};
struct janas_hybrid_sample {
    double count, seconds, verification_seconds, delivered, full;
    double abandoned_seconds;
};
struct janas_hybrid_width {
    double count, seconds;
};
struct janas_hybrid_controller {
    struct janas_hybrid_sample lookup[2][JANAS_HYBRID_VOCABS][3][JANAS_HYBRID_MAX_K + 1];
    struct janas_hybrid_sample mtp[2][JANAS_HYBRID_VOCABS];
    struct janas_hybrid_sample mtp_reuse[2][JANAS_HYBRID_VOCABS][2];
    struct janas_hybrid_width width[2][JANAS_HYBRID_VOCABS][JANAS_HYBRID_MAX_K + 2];
    struct janas_hybrid_width width_reuse[2][JANAS_HYBRID_VOCABS][2][JANAS_HYBRID_MAX_K + 2];
    uint64_t cooldown_until[2][JANAS_HYBRID_VOCABS][3];
    uint64_t cycles, rng;
};

void janas_hybrid_init(struct janas_hybrid_controller *p, uint64_t seed);
unsigned janas_hybrid_suffix_class(unsigned suffix);
/* One call per generation cycle; only this private RNG is consumed. Invalid
   vocabulary inputs do not change controller state. No model work is issued. */
struct janas_hybrid_decision janas_hybrid_choose(
    struct janas_hybrid_controller *p, const struct janas_hybrid_offer *offer);
void janas_hybrid_observe(struct janas_hybrid_controller *p,
                          const struct janas_hybrid_observation *observation);
#endif

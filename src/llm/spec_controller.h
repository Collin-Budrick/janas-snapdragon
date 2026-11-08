/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_SPEC_CONTROLLER_H
#define JANAS_SPEC_CONTROLLER_H
#include <stdint.h>
#define JANAS_CYCLE_MAX_K 16
#define JANAS_CYCLE_SOURCES 4
#define JANAS_CYCLE_VOCABS 4
struct janas_cycle_observation {
    uint64_t count;
    double seconds, delivered;
};
struct janas_cycle_bucket {
    struct janas_cycle_observation depth[JANAS_CYCLE_MAX_K + 1];
    unsigned incumbent;
};
struct janas_cycle_controller {
    struct janas_cycle_bucket bucket[JANAS_CYCLE_SOURCES][2][JANAS_CYCLE_VOCABS];
    struct janas_cycle_bucket reuse_bucket[JANAS_CYCLE_SOURCES][2][JANAS_CYCLE_VOCABS][2];
    uint64_t rng;
};
void janas_cycle_init(struct janas_cycle_controller *p, uint64_t seed);
unsigned janas_cycle_vocab_key(unsigned policy, unsigned rows);
unsigned janas_cycle_choose(struct janas_cycle_controller *p, unsigned source,
                            unsigned warm, unsigned vocab, unsigned cap,
                            unsigned explore_percent);
void janas_cycle_observe(struct janas_cycle_controller *p, unsigned source,
                         unsigned warm, unsigned vocab, unsigned depth,
                         double seconds, unsigned delivered);
/* Past observed assignments per expert group: 0 below two, 1 at least two,
   >=2 unknown. Sparse context estimates back off to the aggregate table. */
unsigned janas_cycle_choose_reuse(struct janas_cycle_controller *p, unsigned source,
                                  unsigned warm, unsigned vocab, unsigned cap,
                                  unsigned explore_percent, unsigned reuse);
void janas_cycle_observe_reuse(struct janas_cycle_controller *p, unsigned source,
                               unsigned warm, unsigned vocab, unsigned depth,
                               double seconds, unsigned delivered, unsigned reuse);
#endif

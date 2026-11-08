/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spec_controller.h"
#include <math.h>
#include <string.h>

static uint64_t next_random(struct janas_cycle_controller *p)
{
    /* Separate from the target sampler, including when its seed is zero. */
    uint64_t x = (p->rng += UINT64_C(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
void janas_cycle_init(struct janas_cycle_controller *p, uint64_t seed)
{
    memset(p, 0, sizeof(*p));
    p->rng = seed ^ UINT64_C(0x6465636f64656379);
}
unsigned janas_cycle_vocab_key(unsigned policy, unsigned rows)
{
    return !policy ? 0 : !rows ? 2 : rows <= 8192 ? 1 : rows <= 16384 ? 2 : 3;
}
static struct janas_cycle_bucket *bucket(struct janas_cycle_controller *p,
                                         unsigned source, unsigned warm,
                                         unsigned vocab)
{
    if (source >= JANAS_CYCLE_SOURCES || vocab >= JANAS_CYCLE_VOCABS)
        return 0;
    return &p->bucket[source][!!warm][vocab];
}
static const struct janas_cycle_observation *estimate(
    const struct janas_cycle_bucket *b, const struct janas_cycle_bucket *fallback,
    unsigned depth)
{
    return b->depth[depth].count >= 3 || !fallback ? &b->depth[depth]
                                                  : &fallback->depth[depth];
}
static unsigned choose(struct janas_cycle_controller *p, struct janas_cycle_bucket *b,
                        const struct janas_cycle_bucket *fallback, unsigned cap,
                        unsigned explore_percent)
{
    if (!b) return 0;
    if (cap > JANAS_CYCLE_MAX_K) cap = JANAS_CYCLE_MAX_K;
    /* A balanced sweep, including zero, before making an empirical choice.
       A short reply/context only exposes actions that actually fit. */
    unsigned least = 0;
    for (unsigned i = 1; i <= cap; i++)
        if (estimate(b, fallback, i)->count < estimate(b, fallback, least)->count) least = i;
    if (estimate(b, fallback, least)->count < 3) return least;
    if (explore_percent > 100) explore_percent = 100;
    if (next_random(p) % 100 < explore_percent)
        return (unsigned)(next_random(p) % (cap + 1));
    unsigned best = b->incumbent <= cap ? b->incumbent : 0;
    const struct janas_cycle_observation *incumbent = estimate(b, fallback, best);
    double rate = incumbent->seconds > 0 ? incumbent->delivered / incumbent->seconds : 0;
    unsigned candidate = best;
    double candidate_rate = rate;
    for (unsigned i = 0; i <= cap; i++) {
        const struct janas_cycle_observation *o = estimate(b, fallback, i);
        double r = o->count >= 3 && o->seconds > 0 ? o->delivered / o->seconds : 0;
        if (r > candidate_rate) { candidate = i; candidate_rate = r; }
    }
    if (candidate_rate > rate * 1.05) b->incumbent = candidate;
    return b->incumbent <= cap ? b->incumbent : best;
}
unsigned janas_cycle_choose(struct janas_cycle_controller *p, unsigned source,
                            unsigned warm, unsigned vocab, unsigned cap,
                            unsigned explore_percent)
{
    return choose(p, bucket(p, source, warm, vocab), 0, cap, explore_percent);
}
void janas_cycle_observe(struct janas_cycle_controller *p, unsigned source,
                         unsigned warm, unsigned vocab, unsigned depth,
                         double seconds, unsigned delivered)
{
    struct janas_cycle_bucket *b = bucket(p, source, warm, vocab);
    if (!b || depth > JANAS_CYCLE_MAX_K || !isfinite(seconds) || seconds <= 0)
        return;
    struct janas_cycle_observation *o = &b->depth[depth];
    /* Bounded recency prevents one early cache state dominating forever. */
    if (o->count >= 64) {
        o->count /= 2; o->seconds *= 0.5; o->delivered *= 0.5;
    }
    o->count++; o->seconds += seconds; o->delivered += delivered;
}
unsigned janas_cycle_choose_reuse(struct janas_cycle_controller *p, unsigned source,
                                  unsigned warm, unsigned vocab, unsigned cap,
                                  unsigned explore_percent, unsigned reuse)
{
    struct janas_cycle_bucket *aggregate = bucket(p, source, warm, vocab);
    if (!aggregate || reuse >= 2) return choose(p, aggregate, 0, cap, explore_percent);
    return choose(p, &p->reuse_bucket[source][!!warm][vocab][reuse], aggregate,
                   cap, explore_percent);
}
void janas_cycle_observe_reuse(struct janas_cycle_controller *p, unsigned source,
                               unsigned warm, unsigned vocab, unsigned depth,
                               double seconds, unsigned delivered, unsigned reuse)
{
    janas_cycle_observe(p, source, warm, vocab, depth, seconds, delivered);
    if (!bucket(p, source, warm, vocab) || reuse >= 2 || depth > JANAS_CYCLE_MAX_K ||
        !isfinite(seconds) || seconds <= 0) return;
    struct janas_cycle_observation *o = &p->reuse_bucket[source][!!warm][vocab][reuse].depth[depth];
    if (o->count >= 64) { o->count /= 2; o->seconds *= 0.5; o->delivered *= 0.5; }
    o->count++; o->seconds += seconds; o->delivered += delivered;
}

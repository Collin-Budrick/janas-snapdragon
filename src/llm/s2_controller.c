/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "s2_controller.h"
#include <math.h>
#include <string.h>

static uint64_t random_next(struct janas_s2_controller *p)
{
    uint64_t x = (p->rng += UINT64_C(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
unsigned janas_s2_reuse_class(uint64_t groups, uint64_t assignments, unsigned width)
{
    if (!groups || !width || assignments < groups) return 3;
    double occupancy = (double)assignments / (double)groups / width;
    return occupancy < 1.0 / 3.0 ? 0 : occupancy < 2.0 / 3.0 ? 1 : 2;
}
void janas_s2_init(struct janas_s2_controller *p, uint64_t seed)
{
    if (!p) return;
    memset(p, 0, sizeof(*p));
    p->rng = seed ^ UINT64_C(0x7332657861637470);
}
static unsigned bin(double probability)
{ return probability >= 1 ? 9 : (unsigned)(probability * 10); }
static int fresh(const struct janas_s2_sample *s, uint64_t cycle)
{ return cycle >= s->last_cycle && cycle - s->last_cycle <= JANAS_S2_STALE_CYCLES; }
static void decay(struct janas_s2_sample *s)
{
    if (s->count < 64 && s->base_count < 64 && s->matched_comparisons < 64) return;
    s->count /= 2; s->base_count /= 2; s->matched_comparisons /= 2; s->full_count /= 2;
    s->seconds *= .5; s->delivered *= .5; s->verify_seconds *= .5;
    s->draft_seconds *= .5; s->expert_cpu_seconds *= .5;
    s->groups *= .5; s->assignments *= .5;
    s->eager_seconds *= .5; s->saved_seconds *= .5;
}
void janas_s2_observe(struct janas_s2_controller *p,
                      const struct janas_s2_observation *o)
{
    if (!p || !o || o->vocab >= JANAS_S2_VOCABS || o->reuse >= JANAS_S2_REUSE ||
        o->width > 17 || o->alt_nodes > JANAS_S2_MAX_PATH ||
        (o->alt_nodes && o->alt_nodes < 2) ||
        !isfinite(o->rank2_probability) || o->rank2_probability < 0 || o->rank2_probability > 1 ||
        !isfinite(o->full_seconds) || o->full_seconds < 0 ||
        !isfinite(o->verify_seconds) || o->verify_seconds < 0 ||
        !isfinite(o->draft_seconds) || o->draft_seconds < 0 ||
        !isfinite(o->expert_cpu_seconds) || o->expert_cpu_seconds < 0 ||
        !isfinite(o->eager_seconds) || o->eager_seconds < 0 ||
        !isfinite(o->saved_seconds) ||
        (o->rank2matched && !o->root_offered) ||
        (o->matched_comparison && !o->alt_nodes) || o->continuation_kind > 2) return;
    unsigned warm = !!o->warm;
    if (o->root_offered) {
        struct janas_s2_root *r = &p->root[warm][o->vocab][bin(o->rank2_probability)];
        if (r->count >= 64) { r->count /= 2; r->matched /= 2; }
        r->count++; r->matched += !!o->rank2matched;
    }
    if (o->width && o->full_seconds > 0 && o->verify_seconds > 0 &&
        o->full_seconds >= o->verify_seconds &&
        o->verify_seconds >= o->expert_cpu_seconds &&
        (!o->expert_groups || o->expert_cpu_seconds > 0)) {
        struct janas_s2_sample *s = &p->width[warm][o->vocab][o->reuse][o->width];
        decay(s); s->count++; s->last_cycle = o->cycle;
        s->verify_seconds += o->verify_seconds;
        s->expert_cpu_seconds += o->expert_cpu_seconds;
        s->groups += (double)o->expert_groups;
        s->assignments += (double)o->expert_assignments;
        /* Alternative eager work must not contaminate ordinary MTP's cost. */
        if (!o->alt_nodes && o->continuation_kind != 2) {
            s->base_count++; s->seconds += o->full_seconds;
            s->draft_seconds += o->draft_seconds; s->delivered += o->delivered;
        }
    }
    if (o->continuation_kind && o->width >= 2 && o->width <= 16 &&
        o->full_seconds >= o->verify_seconds && o->verify_seconds > 0 &&
        o->verify_seconds >= o->expert_cpu_seconds &&
        (!o->expert_groups || o->expert_cpu_seconds > 0)) {
        struct janas_s2_sample *s = o->continuation_kind == 1
            ? &p->fresh_after_rank2[warm][o->vocab][o->reuse][o->width]
            : &p->cached_after_rank2[warm][o->vocab][o->reuse][o->width];
        decay(s); s->count++; s->last_cycle = o->cycle;
        s->seconds += o->full_seconds; s->delivered += o->delivered;
        s->full_count += o->accepted_prefix >= o->width - 1;
        s->verify_seconds += o->verify_seconds;
        s->expert_cpu_seconds += o->expert_cpu_seconds;
        s->groups += (double)o->expert_groups; s->assignments += (double)o->expert_assignments;
    }
    if (o->alt_nodes && o->eager_seconds > 0) {
        struct janas_s2_sample *s = &p->branch[warm][o->vocab][o->reuse][o->alt_nodes];
        decay(s); s->last_cycle = o->cycle;
        s->count++; s->eager_seconds += o->eager_seconds;
    }
}
struct janas_s2_decision janas_s2_choose(struct janas_s2_controller *p,
                                       const struct janas_s2_offer *o)
{
    struct janas_s2_decision d;
    memset(&d, 0, sizeof(d));
    d.reason = JANAS_S2_DISABLED;
    if (!p || !o || o->branches < 2) return d;
    d.reason = JANAS_S2_NO_CAPACITY;
    if (o->vocab >= JANAS_S2_VOCABS || o->reuse >= JANAS_S2_REUSE ||
        !o->primary_nodes || o->primary_nodes > JANAS_S2_MAX_PATH ||
        o->max_nodes > JANAS_S2_MAX_NODES || o->max_nodes <= o->primary_nodes ||
        o->min_gain_percent > 100 || !isfinite(o->rank2_probability) ||
        o->rank2_probability < 0 || o->rank2_probability > 1 ||
        !isfinite(o->draftstep_seconds) || o->draftstep_seconds < 0) return d;
    unsigned cap = o->available_alt_nodes;
    if (cap > JANAS_S2_MAX_PATH) cap = JANAS_S2_MAX_PATH;
    if (cap > o->max_nodes - o->primary_nodes) cap = o->max_nodes - o->primary_nodes;
    if (cap < 2) return d;
    d.reason = JANAS_S2_WEAK_CONFIDENCE;
    if (o->rank2_probability <= 0) return d;
    unsigned warm = !!o->warm;
    const struct janas_s2_root *root = &p->root[warm][o->vocab][bin(o->rank2_probability)];
    d.probability = (root->matched + 2 * o->rank2_probability) / (root->count + 2.0);
    const struct janas_s2_sample *base = &p->width[warm][o->vocab][o->reuse][o->primary_nodes + 1];
    d.reason = JANAS_S2_NO_EVIDENCE;
    if (base->base_count < 3 || root->count < 3) return d;
    d.evidence = base->base_count;
    d.baseline_seconds = base->seconds / base->base_count;
    d.baseline_utility = base->seconds > 0 ? base->delivered / base->seconds : 0;
    d.resident_cpu_seconds = base->count ? base->expert_cpu_seconds / base->count : 0;
    if (base->groups > 0 && d.resident_cpu_seconds <= 0) {
        d.reason = JANAS_S2_CPU_COST_UNKNOWN; return d;
    }
    int baseline_fresh = fresh(base, o->cycle);
    d.reason = baseline_fresh ? JANAS_S2_NO_EVIDENCE : JANAS_S2_STALE;
    if (baseline_fresh && d.baseline_seconds > 0 && d.baseline_utility > 0) {
        double root_seconds = d.baseline_seconds;
        double best_gain = 1 + o->min_gain_percent * .01;
        for (unsigned k = 2; k <= cap; k++) {
            const struct janas_s2_sample *s = &p->branch[warm][o->vocab][o->reuse][k];
            const struct janas_s2_sample *f = &p->fresh_after_rank2[warm][o->vocab][o->reuse][k];
            const struct janas_s2_sample *c = &p->cached_after_rank2[warm][o->vocab][o->reuse][k];
            if (s->count < 3 || f->count < 3 || c->count < 3 ||
                !fresh(s, o->cycle) || !fresh(f, o->cycle) || !fresh(c, o->cycle)) continue;
            /* Comparable observed correction classes, not an exact same-state
               counterfactual. Eager build cost is charged on every root, while
               continuation benefit is weighted by calibrated root survival. */
            double probability = d.probability;
            double baseline = root_seconds + probability * f->seconds / f->count;
            double candidate = root_seconds + s->eager_seconds / s->count +
                               probability * c->seconds / c->count;
            double baseline_output = base->delivered / base->base_count + probability * f->delivered / f->count;
            double candidate_output = base->delivered / base->base_count + probability * c->delivered / c->count;
            double rate = baseline > 0 ? baseline_output / baseline : 0;
            double candidate_rate = candidate > 0 ? candidate_output / candidate : 0;
            double gain = rate > 0 ? candidate_rate / rate : 0;
            d.reason = JANAS_S2_COSTLY;
            if (gain > best_gain) {
                best_gain = gain; d.alt_nodes = k;
                d.baseline_seconds = baseline; d.baseline_utility = rate;
                d.predicted_seconds = candidate; d.predicted_utility = candidate_rate;
                d.matched_comparisons = f->count < c->count ? f->count : c->count;
            }
        }
    }
    if (d.alt_nodes) d.reason = JANAS_S2_BENEFICIAL;
    /* Independent randomness; no forced sweep or target-sampler RNG. At most
       one initial two-node probe per 64 eligible generation cycles. */
    unsigned probe_depth = 2;
    for (unsigned k = 2; k <= cap; k++) {
        const struct janas_s2_sample *c = &p->cached_after_rank2[warm][o->vocab][o->reuse][k];
        if (c->count >= 3 && c->full_count * 3 >= c->count * 2 && fresh(c, o->cycle)) {
            unsigned next = k * 2 < cap ? k * 2 : cap;
            if (next > probe_depth) probe_depth = next;
        }
    }
    if (probe_depth > d.alt_nodes && (!p->last_probe || (o->cycle >= p->last_probe && o->cycle - p->last_probe >= 64)) &&
        (random_next(p) & 63) == 0) {
        p->last_probe = o->cycle ? o->cycle : 1; d.alt_nodes = probe_depth;
        d.reason = JANAS_S2_PROBE; d.probe = 1;
        d.predicted_seconds = d.baseline_seconds + o->draftstep_seconds * (probe_depth - 1);
        d.predicted_utility = d.predicted_seconds > 0
            ? (base->delivered / base->base_count) / d.predicted_seconds : 0;
    }
    return d;
}

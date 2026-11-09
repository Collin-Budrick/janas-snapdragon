/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "hybrid_controller.h"
#include <math.h>
#include <string.h>

static unsigned limit(unsigned value, unsigned cap)
{ return value < cap ? value : cap; }

static uint64_t next_random(struct janas_hybrid_controller *p)
{
    uint64_t x = (p->rng += UINT64_C(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

void janas_hybrid_init(struct janas_hybrid_controller *p, uint64_t seed)
{
    if (!p) return;
    memset(p, 0, sizeof(*p));
    p->rng = seed ^ UINT64_C(0x687962726964636f);
}

unsigned janas_hybrid_suffix_class(unsigned suffix)
{ return suffix <= 1 ? 0 : suffix <= 3 ? 1 : 2; }

static const struct janas_hybrid_sample *mtp_sample(
    const struct janas_hybrid_controller *p, unsigned warm, unsigned vocab,
    unsigned reuse)
{
    const struct janas_hybrid_sample *base = &p->mtp[warm][vocab];
    if (reuse < 2 && p->mtp_reuse[warm][vocab][reuse].count >= 3)
        return &p->mtp_reuse[warm][vocab][reuse];
    return base;
}

static double width_seconds(const struct janas_hybrid_controller *p,
                             unsigned warm, unsigned vocab, unsigned reuse,
                             unsigned width, double fallback)
{
    const struct janas_hybrid_width *base = &p->width[warm][vocab][width];
    if (reuse < 2 && p->width_reuse[warm][vocab][reuse][width].count >= 3)
        base = &p->width_reuse[warm][vocab][reuse][width];
    return base->count >= 3 && base->seconds > 0
               ? base->seconds / base->count : fallback;
}

static double lookup_seconds(const struct janas_hybrid_controller *p,
                              const struct janas_hybrid_offer *o, unsigned k,
                              const struct janas_hybrid_sample *s)
{
    if (s->count <= 0 || s->seconds <= 0) return 0;
    double measured_verification = s->verification_seconds / s->count;
    double predicted = (s->seconds + s->abandoned_seconds) / s->count - measured_verification +
        width_seconds(p, !!o->warm, o->vocab, o->reuse, k + 1,
                      measured_verification);
    /* Width predictors share actual target work across sources, but never
       remove the observed preparation, recovery or sampling cost. */
    return predicted > 0 ? predicted : (s->seconds + s->abandoned_seconds) / s->count;
}

struct janas_hybrid_decision janas_hybrid_choose(
    struct janas_hybrid_controller *p, const struct janas_hybrid_offer *o)
{
    struct janas_hybrid_decision d;
    memset(&d, 0, sizeof(d));
    d.source = JANAS_HYBRID_MTP;
    d.reason = JANAS_HYBRID_NO_EVIDENCE;
    if (!p || !o || o->vocab >= JANAS_HYBRID_VOCABS) return d;
    d.depth = limit(o->mtp_cap, JANAS_HYBRID_MAX_K);
    unsigned cap = limit(limit(o->lookup_cap, JANAS_HYBRID_MAX_K),
                         o->lookup_available);
    unsigned warm = !!o->warm;
    unsigned cls = janas_hybrid_suffix_class(o->lookup_suffix);
    d.suffix_class = cls;
    p->cycles++;
    const struct janas_hybrid_sample *m = mtp_sample(p, warm, o->vocab, o->reuse);
    d.mtp_evidence = (unsigned)m->count;
    double mtp_seconds = m->seconds + m->abandoned_seconds;
    d.mtp_seconds = m->count > 0 ? mtp_seconds / m->count : 0;
    d.mtp_utility = mtp_seconds > 0 ? m->delivered / mtp_seconds : 0;
    if (!o->lookup_available || !o->lookup_suffix) {
        d.reason = JANAS_HYBRID_NO_MATCH;
        return d;
    }
    if (!cap) {
        d.reason = JANAS_HYBRID_NO_CAPACITY;
        return d;
    }
    if (p->cycles <= p->cooldown_until[warm][o->vocab][cls]) {
        d.reason = JANAS_HYBRID_COOLDOWN;
        return d;
    }
    const struct janas_hybrid_sample *samples = p->lookup[warm][o->vocab][cls];
    unsigned best = 0;
    double best_rate = 0;
    for (unsigned k = 1; k <= cap; k++) {
        const struct janas_hybrid_sample *s = &samples[k];
        if (s->count < 3 || s->seconds <= 0) continue;
        double seconds = lookup_seconds(p, o, k, s);
        double rate = seconds > 0 ? (s->delivered / s->count) / seconds : 0;
        if (rate > best_rate) {
            best = k; best_rate = rate;
            d.lookup_evidence = (unsigned)s->count;
            d.lookup_seconds = seconds;
        }
    }
    d.lookup_utility = best_rate;
    if (m->count >= 3 && d.mtp_utility > 0) {
        d.reason = JANAS_HYBRID_MTP_CHEAPER;
        if (best && best_rate > d.mtp_utility * 1.05) {
            d.source = JANAS_HYBRID_LOOKUP;
            d.depth = best;
            d.reason = JANAS_HYBRID_LOOKUP_FASTER;
            d.evidence = d.lookup_evidence;
        }
    }
    /* The normal action always remains MTP until it has a measured cost.
       Probes have no mandatory sweep and share one allowance with MTP refresh. */
    unsigned percent = limit(o->probe_percent, 100);
    if (m->count < 3 || !percent || next_random(p) % 100 >= percent) return d;
    d.probe = 1;
    unsigned probe_depth = limit(o->lookup_suffix >= 4 ? 2 : 1, cap);
    for (unsigned k = 1; k <= cap; k++) {
        const struct janas_hybrid_sample *s = &samples[k];
        if (s->count >= 3 && s->full * 3 >= s->count * 2) {
            unsigned next = limit(k * 2, cap);
            if (next > probe_depth) probe_depth = next;
        }
    }
    if (d.source == JANAS_HYBRID_LOOKUP &&
        (probe_depth <= best || !(next_random(p) & 1))) {
        d.source = JANAS_HYBRID_MTP;
        d.depth = limit(o->mtp_cap, JANAS_HYBRID_MAX_K);
        d.reason = JANAS_HYBRID_MTP_REFRESH;
        d.evidence = d.mtp_evidence;
    } else {
        d.source = JANAS_HYBRID_LOOKUP;
        d.depth = probe_depth;
        d.reason = JANAS_HYBRID_LOOKUP_PROBE;
        d.evidence = (unsigned)samples[probe_depth].count;
        d.lookup_evidence = d.evidence;
        d.lookup_seconds = lookup_seconds(p, o, probe_depth, &samples[probe_depth]);
        d.lookup_utility = d.lookup_seconds > 0
            ? samples[probe_depth].delivered / samples[probe_depth].count /
              d.lookup_seconds : 0;
    }
    return d;
}

static void add_sample(struct janas_hybrid_sample *s, double seconds,
                        double verification, unsigned delivered, unsigned full)
{
    if (s->count >= 64) {
        s->count *= 0.5; s->seconds *= 0.5;
        s->verification_seconds *= 0.5; s->delivered *= 0.5; s->full *= 0.5;
        s->abandoned_seconds *= 0.5;
    }
    s->count++; s->seconds += seconds; s->verification_seconds += verification;
    s->delivered += delivered; s->full += !!full;
}

static void add_width(struct janas_hybrid_width *w, double seconds)
{
    if (w->count >= 64) { w->count *= 0.5; w->seconds *= 0.5; }
    w->count++; w->seconds += seconds;
}

void janas_hybrid_observe(struct janas_hybrid_controller *p,
                          const struct janas_hybrid_observation *o)
{
    if (!p || !o || o->vocab >= JANAS_HYBRID_VOCABS ||
        (o->source != JANAS_HYBRID_LOOKUP && o->source != JANAS_HYBRID_MTP) ||
        o->depth > JANAS_HYBRID_MAX_K || o->accepted_prefix > o->depth ||
        o->pass_width > JANAS_HYBRID_MAX_K + 1 ||
        !isfinite(o->seconds) || o->seconds <= 0 ||
        !isfinite(o->verification_seconds) || o->verification_seconds < 0 ||
        (o->source == JANAS_HYBRID_LOOKUP && (!o->lookup_suffix || !o->depth)))
        return;
    unsigned warm = !!o->warm, vocab = o->vocab;
    /* A prepared proposal abandoned before target execution has no observed
       acceptance outcome. Keep its real cost without fabricating a rejected
       prefix, evidence count, width observation or cooldown. */
    if (!o->pass_width) {
        if (o->accepted_prefix || o->delivered || o->verification_seconds) return;
        if (o->source == JANAS_HYBRID_LOOKUP) {
            unsigned cls = janas_hybrid_suffix_class(o->lookup_suffix);
            p->lookup[warm][vocab][cls][o->depth].abandoned_seconds += o->seconds;
        } else {
            p->mtp[warm][vocab].abandoned_seconds += o->seconds;
            if (o->reuse < 2)
                p->mtp_reuse[warm][vocab][o->reuse].abandoned_seconds += o->seconds;
        }
        return;
    }
    double verification = o->verification_seconds < o->seconds
                              ? o->verification_seconds : o->seconds;
    if (verification > 0) {
        add_width(&p->width[warm][vocab][o->pass_width], verification);
        if (o->reuse < 2)
            add_width(&p->width_reuse[warm][vocab][o->reuse][o->pass_width], verification);
    }
    if (o->source == JANAS_HYBRID_MTP) {
        add_sample(&p->mtp[warm][vocab], o->seconds, verification, o->delivered, 0);
        if (o->reuse < 2)
            add_sample(&p->mtp_reuse[warm][vocab][o->reuse], o->seconds,
                       verification, o->delivered, 0);
        return;
    }
    unsigned cls = janas_hybrid_suffix_class(o->lookup_suffix);
    if (!o->accepted_prefix)
        p->cooldown_until[warm][vocab][cls] = p->cycles + 32;
    /* A verified longer chain also observes every shorter prefix. Only
       measured target widths can lower its cost estimate; until available,
       retain the complete longer pass cost as a conservative backoff. */
    for (unsigned k = 1; k <= o->depth; k++) {
        unsigned delivered = limit(o->delivered, limit(o->accepted_prefix, k) + 1);
        double target = width_seconds(p, warm, vocab, o->reuse, k + 1, verification);
        double seconds = o->seconds - verification + target;
        if (seconds <= 0) seconds = o->seconds;
        add_sample(&p->lookup[warm][vocab][cls][k], seconds, target,
                   delivered, o->accepted_prefix >= k);
    }
}

/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Adaptive draft-only row selection. Target weights, logits and RNG are
   never changed. Only metadata is additional; dh_w/dh_logits are reused. */
#define _GNU_SOURCE
#include "model.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model_impl.h"

struct vocab_observation { double reward, seconds; uint64_t count; };
struct janas_draft_vocab {
    int32_t *ids;
    uint8_t *seen;
    uint32_t n, cycles, requested, last_history;
    int active, dirty, trial, incumbent;
    uint64_t epoch;
    struct vocab_observation obs[4]; /* legacy, 8k, 16k, 32k */
};
struct ranked_id { float score; int32_t id; };

/* Smaller score, or larger ID at equal score, is the worse heap item. */
static int worse(struct ranked_id a, struct ranked_id b)
{ return a.score < b.score || (a.score == b.score && a.id > b.id); }
static int rank_desc(const void *aa, const void *bb)
{
    const struct ranked_id *a = aa, *b = bb;
    if (a->score != b->score) return a->score > b->score ? -1 : 1;
    return (a->id > b->id) - (a->id < b->id);
}
static uint32_t top_ids(const float *l, uint32_t n, struct ranked_id heap[2048])
{
    uint32_t len = 0;
    for (uint32_t id = 0; id < n; id++) {
        if (!isfinite(l[id])) continue;
        struct ranked_id x = {l[id], (int32_t)id};
        if (len < 2048) {
            uint32_t at = len++;
            while (at && worse(x, heap[(at - 1) / 2])) {
                heap[at] = heap[(at - 1) / 2]; at = (at - 1) / 2;
            }
            heap[at] = x;
        } else if (worse(heap[0], x)) {
            uint32_t at = 0;
            while (2 * at + 1 < len) {
                uint32_t child = 2 * at + 1;
                if (child + 1 < len && worse(heap[child + 1], heap[child])) child++;
                if (!worse(heap[child], x)) break;
                heap[at] = heap[child]; at = child;
            }
            heap[at] = x;
        }
    }
    qsort(heap, len, sizeof(*heap), rank_desc);
    return len;
}
static uint32_t budget(int trial)
{
    static const uint32_t rows[] = {0, 8192, 16384, 32768};
    return rows[trial];
}
static int trial_of(uint32_t rows)
{ return rows == 8192 ? 1 : rows == 16384 ? 2 : rows == 32768 ? 3 : -1; }
static double utility(const struct vocab_observation *o)
{ return o->count >= 3 && o->seconds > 0 ? o->reward / o->seconds : 0; }

uint32_t janas_llm_mtp_vocab_rows(const struct janas_llm_model *m)
{ return m->dh_adaptive && m->dh_adaptive->active ? m->dh_adaptive->n : m->dh_n; }
int janas_llm_mtp_vocab_policy(const struct janas_llm_model *m)
{ return m->dh_adaptive && m->dh_adaptive->active; }
int janas_llm_mtp_vocab_cpu_only(const struct janas_llm_model *m)
{ return m->dh_adaptive && m->dh_adaptive->dirty; }
int32_t janas_llm_mtp_vocab_token(const struct janas_llm_model *m, uint32_t row)
{
    if (row >= janas_llm_mtp_vocab_rows(m)) return -1;
    return janas_llm_mtp_vocab_policy(m) ? m->dh_adaptive->ids[row] : m->dh_ids[row];
}
void janas_llm_mtp_costs(const struct janas_llm_model *m, struct janas_mtp_costs *out)
{
    memset(out, 0, sizeof(*out));
    if (!m) return;
    out->maintenance_seconds = m->mtp_maintenance_s;
    out->head_seconds = m->mtp_head_s;
    out->repack_seconds = m->mtp_repack_s;
    out->head_calls = m->mtp_head_calls; out->repacks = m->mtp_repacks;
    out->vocab_fallbacks = m->mtp_vocab_fallbacks;
    out->vocab_rows = janas_llm_mtp_vocab_rows(m);
    out->vocab_policy = (uint32_t)janas_llm_mtp_vocab_policy(m);
    out->vocab_epoch = m->dh_adaptive ? m->dh_adaptive->epoch : 0;
}
static void restore_legacy(struct janas_llm_model *m)
{
    if (!m || !m->dh_adaptive || !m->dh_adaptive->active) return;
    double t = now();
    for (uint32_t i = 0; i < m->dh_n; i++)
        memcpy(m->dh_w + (size_t)i * m->dh_row,
               m->dh_src + (size_t)m->dh_ids[i] * m->dh_row, m->dh_row);
    m->dh_adaptive->active = 0;
    m->mtp_repack_s += now() - t; m->mtp_repacks++;
}
void janas_llm_mtp_vocab_legacy(struct janas_llm_model *m)
{
    restore_legacy(m);
    if (m && m->dh_adaptive) {
        m->dh_adaptive->epoch = 0;
        m->dh_adaptive->cycles = 16;
        m->dh_adaptive->requested = UINT32_MAX;
    }
}
void janas_llm_mtp_vocab_new_reply(struct janas_llm_model *m)
{
    if (m && m->dh_adaptive) {
        m->dh_adaptive->cycles = 16;
        m->dh_adaptive->last_history = 0;
    }
}
void janas_llm_mtp_vocab_free(struct janas_llm_model *m)
{
    if (!m || !m->dh_adaptive) return;
    free(m->dh_adaptive->ids); free(m->dh_adaptive->seen);
    free(m->dh_adaptive); m->dh_adaptive = NULL;
}
void janas_llm_mtp_vocab_feedback_epoch(struct janas_llm_model *m, uint64_t epoch,
                                       uint32_t policy, uint32_t rows,
                                       uint32_t drafted, uint32_t accepted,
                                       double seconds)
{
    if (!m || !m->dh_adaptive || !drafted || accepted > drafted ||
        !(seconds > 0) || !isfinite(seconds)) return;
    struct janas_draft_vocab *v = m->dh_adaptive;
    int trial = policy ? trial_of(rows) : 0;
    if (policy > 1 || trial < 0) return;
    struct vocab_observation *o = &v->obs[trial];
    o->reward = .98 * o->reward + accepted + 1.;
    o->seconds = .98 * o->seconds + seconds;
    o->count++;
    if ((epoch == UINT64_MAX || epoch == v->epoch) &&
        policy == (uint32_t)janas_llm_mtp_vocab_policy(m) &&
        (!policy || rows == janas_llm_mtp_vocab_rows(m))) v->cycles++;
}
void janas_llm_mtp_vocab_feedback(struct janas_llm_model *m, uint32_t policy,
                                 uint32_t rows, uint32_t drafted,
                                 uint32_t accepted, double seconds)
{
    janas_llm_mtp_vocab_feedback_epoch(m, UINT64_MAX, policy, rows,
                                      drafted, accepted, seconds);
}
static int choose(struct janas_draft_vocab *v, uint32_t requested, uint32_t cap)
{
    int fixed = requested ? trial_of(requested) : -1;
    if (!v->epoch) return fixed >= 0 ? fixed : 2;
    if (fixed >= 0) {
        if (v->epoch == 1 || v->epoch % 10 == 1) return 0;
        if (v->epoch % 5 == 0) return fixed;
        double u = utility(&v->obs[fixed]), old = utility(&v->obs[0]);
        return old > 0 && u < old * .95 ? 0 : fixed;
    }
    /* Initial balanced epochs: 16k, legacy, 8k, legacy, 32k. */
    static const int initial[] = {2, 0, 1, 0, 3};
    if (v->epoch < 5) {
        int x = initial[v->epoch];
        return budget(x) <= cap ? x : 0;
    }
    int best = v->incumbent;
    for (int i = 0; i < 4; i++)
        if (budget(i) <= cap && utility(&v->obs[i]) > utility(&v->obs[best]) * 1.05)
            best = i;
    v->incumbent = best;
    if (v->epoch % 5 == 0) {
        int probe = 0;
        for (int i = 1; i < 4; i++)
            if (budget(i) <= cap && v->obs[i].count < v->obs[probe].count) probe = i;
        return probe;
    }
    return best;
}
static void add_id(struct janas_draft_vocab *v, uint32_t nv, uint32_t cap, int32_t id)
{
    if (id < 0 || (uint32_t)id >= nv || v->n >= cap || (v->seen[id] & 1)) return;
    v->seen[id] |= 1; v->ids[v->n++] = id;
}
int janas_llm_mtp_vocab_prepare(struct janas_llm_model *m, int policy,
                               uint32_t rows, const float *logits,
                               const int32_t *history, uint32_t nh,
                               const int32_t *special, uint32_t ns)
{
    if (!m) return -1;
    if (!policy) { janas_llm_mtp_vocab_legacy(m); return 0; }
    if (policy != 1 || !m->mtp || m->asst || !m->dh_w || !m->dh_src ||
        !m->dh_row || !m->dh_cap || !logits || (nh && !history) ||
        (ns && !special) || (rows && trial_of(rows) < 0) || rows > m->dh_cap)
        goto fallback;
    double start = now();
    struct janas_draft_vocab *v = m->dh_adaptive;
    if (!v) {
        v = calloc(1, sizeof(*v));
        if (!v) goto fallback;
        v->ids = malloc((size_t)m->dh_cap * sizeof(*v->ids));
        v->seen = calloc(m->n_vocab, 1);
        if (!v->ids || !v->seen) {
            free(v->ids); free(v->seen); free(v); goto fallback;
        }
        m->dh_adaptive = v;
    }
    if (v->epoch && v->requested == rows && v->cycles < 16 && nh >= v->last_history)
        return 0;
    if (v->requested != rows) { v->epoch = 0; v->cycles = 0; }
    int selected = choose(v, rows, m->dh_cap);
    if (selected && budget(selected) > m->dh_cap) selected = 0;
    v->requested = rows; v->cycles = 0; v->last_history = nh; v->epoch++;
    if (!selected) {
        restore_legacy(m); v->trial = 0;
        return 0;
    }
    uint32_t target = budget(selected);
    v->n = 0; memset(v->seen, 0, m->n_vocab);
    for (uint32_t i = 0; i < ns; i++) add_id(v, m->n_vocab, target, special[i]);
    struct ranked_id heap[2048];
    uint32_t ntop = top_ids(logits, m->n_vocab, heap);
    if (!ntop) goto fallback;
    for (uint32_t i = 0; i < ntop; i++) add_id(v, m->n_vocab, target, heap[i].id);
    uint32_t recent = 0;
    for (uint32_t j = nh; j && recent < 4096; ) {
        int32_t id = history[--j];
        if (id < 0 || (uint32_t)id >= m->n_vocab || (v->seen[id] & 2)) continue;
        v->seen[id] |= 2; recent++; add_id(v, m->n_vocab, target, id);
    }
    for (uint32_t i = 0; i < m->dh_n; i++) add_id(v, m->n_vocab, target, m->dh_ids[i]);
    for (uint32_t i = 0; i < m->n_vocab && v->n < target; i++)
        add_id(v, m->n_vocab, target, (int32_t)i);
    for (uint32_t i = 0; i < v->n; i++)
        memcpy(m->dh_w + (size_t)i * m->dh_row,
               m->dh_src + (size_t)v->ids[i] * m->dh_row, m->dh_row);
    v->active = v->dirty = 1; v->trial = selected;
    m->mtp_repacks++; m->mtp_repack_s += now() - start;
    return 1;
fallback:
    janas_llm_mtp_vocab_legacy(m); m->mtp_vocab_fallbacks++;
    return -1;
}

void janas_llm_model_runtime_stats(const struct janas_llm_model *m,
                                  struct janas_llm_runtime_stats *out)
{
    memset(out, 0, sizeof(*out));
    if (!m) return;
    *out = m->runtime_stats;
    out->gdn_chain_tokens = m->gdn_chain_tokens;
    out->gdn_journal_replay_tokens = m->gdn_journal_replay_tokens;
    out->gdn_scalar_replay_tokens = m->gdn_scalar_replay_tokens;
    out->gdn_konly_replay_tokens = m->gdn_konly_replay_tokens;
    out->gdn_core_seconds = m->gdn_core_seconds;
    out->gdn_conv_seconds = m->gdn_conv_seconds;
    out->gdn_recovery_seconds = m->gdn_recovery_seconds;
    out->gdn_delta_bytes = m->gdn_delta_bytes;
}

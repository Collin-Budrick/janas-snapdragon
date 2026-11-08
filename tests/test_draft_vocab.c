/* SPDX-License-Identifier: GPL-3.0-or-later */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "llm/model.h"
#include "llm/model_impl.h"

static int failures;
#define CHECK(x) do { if (!(x)) { printf("line %d: %s\n", __LINE__, #x); failures++; } } while (0)
static int contains(const struct janas_llm_model *m, int32_t id)
{
    for (uint32_t i = 0; i < janas_llm_mtp_vocab_rows(m); i++)
        if (janas_llm_mtp_vocab_token(m, i) == id) return 1;
    return 0;
}
static void check_rows(const struct janas_llm_model *m)
{
    uint8_t *seen = calloc(m->n_vocab, 1);
    for (uint32_t i = 0; i < janas_llm_mtp_vocab_rows(m); i++) {
        int32_t id = janas_llm_mtp_vocab_token(m, i);
        CHECK(id >= 0 && (uint32_t)id < m->n_vocab);
        if (id < 0 || (uint32_t)id >= m->n_vocab) continue;
        CHECK(!seen[id]); seen[id] = 1;
        CHECK(!memcmp(m->dh_w + (size_t)i * m->dh_row,
                      m->dh_src + (size_t)id * m->dh_row, m->dh_row));
    }
    free(seen);
}
static void feedback_epoch(struct janas_llm_model *m, uint32_t accepted,
                            double seconds)
{
    uint32_t policy = (uint32_t)janas_llm_mtp_vocab_policy(m);
    uint32_t rows = janas_llm_mtp_vocab_rows(m);
    for (int i = 0; i < 16; i++)
        janas_llm_mtp_vocab_feedback(m, policy, rows, 4, accepted, seconds);
}
int main(void)
{
    struct janas_llm_model m = {0};
    m.mtp = 1; m.n_vocab = 50000; m.dh_n = 32768;
    m.dh_cap = 40960; m.dh_row = 8;
    uint64_t *source = malloc((size_t)m.n_vocab * sizeof(*source));
    float *logits = malloc((size_t)m.n_vocab * sizeof(*logits));
    m.dh_w = malloc((size_t)m.dh_cap * m.dh_row);
    m.dh_ids = malloc((size_t)m.dh_cap * sizeof(*m.dh_ids));
    m.dh_in = calloc(m.n_vocab, 1);
    if (!source || !logits || !m.dh_w || !m.dh_ids || !m.dh_in) return 2;
    m.dh_src = (const uint8_t *)source;
    for (uint32_t i = 0; i < m.n_vocab; i++) {
        source[i] = (uint64_t)i * UINT64_C(0x100000001);
        logits[i] = (float)i;
    }
    memcpy(m.dh_w, source, (size_t)m.dh_n * m.dh_row);
    for (uint32_t i = 0; i < m.dh_n; i++) { m.dh_ids[i] = (int32_t)i; m.dh_in[i] = 1; }
    int32_t history[] = {49990, 45001, 45001, 45002, -1, 50000};
    int32_t special[] = {42000, -1, 50001};
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 1);
    CHECK(janas_llm_mtp_vocab_rows(&m) == 8192);
    CHECK(janas_llm_mtp_vocab_policy(&m) == 1);
    CHECK(janas_llm_mtp_vocab_cpu_only(&m));
    CHECK(contains(&m, 42000) && contains(&m, 45001) && contains(&m, 45002));
    CHECK(contains(&m, 49999) && contains(&m, 47952));
    check_rows(&m);
    struct janas_mtp_costs before, after;
    janas_llm_mtp_costs(&m, &before);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 0);
    janas_llm_mtp_costs(&m, &after); CHECK(after.repacks == before.repacks);
    for (int i = 0; i < 16; i++) janas_llm_mtp_vocab_feedback(&m, 1, 8192, 4, 4, .1);
    /* The comparison epoch restores all legacy rows, including higher IDs. */
    m.dh_ids[m.dh_n++] = 45003; m.dh_in[45003] = 1;
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 0);
    CHECK(janas_llm_mtp_vocab_policy(&m) == 0);
    CHECK(janas_llm_mtp_vocab_rows(&m) == 32769);
    CHECK(contains(&m, 45003)); check_rows(&m);
    CHECK(janas_llm_mtp_vocab_cpu_only(&m)); /* mutable pointer stays on CPU */
    for (int i = 0; i < 16; i++) janas_llm_mtp_vocab_feedback(&m, 0, 32769, 4, 1, .5);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 1);
    check_rows(&m);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 123, logits, history, 6, special, 3) == -1);
    CHECK(janas_llm_mtp_vocab_policy(&m) == 0); check_rows(&m);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 16384, logits, history, 6, special, 3) == 1);
    CHECK(janas_llm_mtp_vocab_rows(&m) == 16384); check_rows(&m);
    /* Invalid/all-masked lagged rows cannot leave a partial repack active. */
    for (uint32_t i = 0; i < m.n_vocab; i++) logits[i] = -INFINITY;
    janas_llm_mtp_vocab_new_reply(&m);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 32768, logits, history, 6, special, 3) == -1);
    CHECK(!janas_llm_mtp_vocab_policy(&m)); check_rows(&m);
    CHECK(janas_llm_mtp_vocab_token(&m, m.dh_n) == -1);
    /* A failed refresh must not make a subsequent valid row wait for an
     * epoch of unrelated legacy feedback, even at the same requested size. */
    for (uint32_t i = 0; i < m.n_vocab; i++) logits[i] = (float)i;
    janas_llm_mtp_costs(&m, &before);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 32768, logits, history, 6, special, 3) == 1);
    CHECK(janas_llm_mtp_vocab_policy(&m) == 1 && janas_llm_mtp_vocab_rows(&m) == 32768);
    janas_llm_mtp_costs(&m, &after); CHECK(after.repacks == before.repacks + 1);
    check_rows(&m);
    /* Explicitly disabling and then re-enabling the same budget is a fresh
     * request, not a scheduled legacy-comparison epoch. */
    janas_llm_mtp_vocab_legacy(&m);
    CHECK(!janas_llm_mtp_vocab_policy(&m)); check_rows(&m);
    CHECK(janas_llm_mtp_vocab_cpu_only(&m));
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 32768, logits, history, 6, special, 3) == 1);
    CHECK(janas_llm_mtp_vocab_policy(&m) == 1 && janas_llm_mtp_vocab_rows(&m) == 32768);
    check_rows(&m);

    /* Isolate utility history. An initially slower fixed budget should use
     * legacy, but retain a charged periodic adaptive probe and recover when
     * the complete-cycle cost/acceptance improves. */
    janas_llm_mtp_vocab_legacy(&m);
    janas_llm_mtp_vocab_free(&m);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 1);
    feedback_epoch(&m, 0, 1.0); /* one produced token per second */
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 0);
    CHECK(!janas_llm_mtp_vocab_policy(&m));
    feedback_epoch(&m, 1, 1.0); /* two produced tokens per second */
    int recovered_probe = 0;
    for (int epoch = 0; epoch < 4; epoch++) {
        janas_llm_mtp_costs(&m, &before);
        int rc = janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3);
        if (janas_llm_mtp_vocab_policy(&m)) {
            CHECK(rc == 1 && janas_llm_mtp_vocab_rows(&m) == 8192);
            janas_llm_mtp_costs(&m, &after);
            CHECK(after.repacks == before.repacks + 1 &&
                  after.repack_seconds >= before.repack_seconds);
            check_rows(&m);
            recovered_probe = 1;
            break;
        }
        CHECK(rc == 0); check_rows(&m);
        feedback_epoch(&m, 1, 1.0);
    }
    CHECK(recovered_probe);
    feedback_epoch(&m, 4, .01);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 1);
    CHECK(janas_llm_mtp_vocab_policy(&m) == 1 && janas_llm_mtp_vocab_rows(&m) == 8192);
    check_rows(&m);
    /* A proposal from the preceding subset epoch may finish after the next
     * subset was packed at the same budget. It can teach budget utility,
     * but must not age the new epoch or trigger its premature refresh. */
    janas_llm_mtp_costs(&m, &before);
    CHECK(before.vocab_epoch > 1);
    for (int i = 0; i < 16; i++)
        janas_llm_mtp_vocab_feedback_epoch(&m, before.vocab_epoch - 1,
                                           1, 8192, 4, 4, .01);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 0);
    janas_llm_mtp_costs(&m, &after);
    CHECK(after.vocab_epoch == before.vocab_epoch && after.repacks == before.repacks);
    for (int i = 0; i < 16; i++)
        janas_llm_mtp_vocab_feedback_epoch(&m, before.vocab_epoch,
                                           1, 8192, 4, 4, .01);
    CHECK(janas_llm_mtp_vocab_prepare(&m, 1, 8192, logits, history, 6, special, 3) == 1);
    janas_llm_mtp_costs(&m, &after);
    CHECK(after.vocab_epoch == before.vocab_epoch + 1);
    check_rows(&m);
    for (uint32_t i = 0; i < m.n_vocab; i++)
        CHECK(source[i] == (uint64_t)i * UINT64_C(0x100000001));
    janas_llm_mtp_costs(&m, &after); CHECK(after.vocab_fallbacks >= 2);
    janas_llm_mtp_vocab_free(&m); CHECK(!m.dh_adaptive);
    free(source); free(logits); free(m.dh_w); free(m.dh_ids); free(m.dh_in);
    puts(failures ? "test_draft_vocab: FAILED" : "test_draft_vocab: ok");
    return failures != 0;
}

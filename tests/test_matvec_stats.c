/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "llm/matvec.h"
#include "llm/quant_panel.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_case(struct janas_pool *pool, size_t rows, size_t nv)
{
    size_t count = 2 * rows + 1;
    struct janas_block_q4k *w = calloc(count, sizeof(*w));
    struct janas_block_q8k *x = calloc(nv, sizeof(*x));
    float *y = calloc(count * nv, sizeof(*y));
    float *plain = calloc(count * nv, sizeof(*plain));
    assert(w && x && y && plain);
    for (size_t r = 0; r < count; r++) {
        memset(w + r, (int)(17 * r + 29), sizeof(*w));
        w[r].d = janas_fp32_to_fp16(.001f);
        w[r].dmin = janas_fp32_to_fp16(.0005f);
    }
    float source[JANAS_QK];
    for (size_t v = 0; v < nv; v++) {
        for (size_t k = 0; k < JANAS_QK; k++)
            source[k] = (float)((int)((k + v * 7) % 19) - 9) * .13f;
        janas_q8k_quantize(source, x + v, JANAS_QK);
    }
    struct janas_matvec_task tasks[3];
    size_t off = 0;
    for (size_t i = 0; i < 3; i++) {
        size_t nr = i < 2 ? rows : 1;
        tasks[i] = (struct janas_matvec_task){.type=JANAS_Q4_K, .w=w+off,
            .x=x, .y=y+off*nv, .rows=nr, .cols=JANAS_QK, .n_vec=nv};
        off += nr;
    }
    struct janas_matvec_stats stats;
    memset(&stats, 0xcc, sizeof stats);
    janas_matvec_q4k_group_stats(pool, tasks, 3, &stats);
    assert(stats.groups == 1 && stats.tasks == 3);
    assert(stats.prepared_rows + stats.established_rows == count);
    assert(stats.prepared_row_vectors + stats.established_row_vectors == count * nv);
    assert(stats.prepared_row_vectors == stats.prepared_rows * nv);
    assert(stats.established_row_vectors == stats.established_rows * nv);
    assert(isfinite(stats.panel_prepare_seconds) && stats.panel_prepare_seconds >= 0);
    if (janas_quant_panels_enabled()) {
        assert(stats.prepared_rows > 0 && stats.established_rows > 0);
        assert(stats.panel_views == 2 && stats.panel_unique == 1);
        assert(stats.panel_dedup_hits == 1 && stats.panel_bytes > 0);
        assert(stats.panel_bytes <= JANAS_Q8K_PANEL_CAP);
    } else {
        assert(!stats.prepared_rows && !stats.prepared_row_vectors);
        assert(!stats.panel_views && !stats.panel_unique && !stats.panel_dedup_hits);
        assert(!stats.panel_bytes && stats.panel_prepare_seconds == 0);
    }
    off = 0;
    for (size_t i = 0; i < 3; i++) {
        size_t nr = tasks[i].rows;
        for (size_t v = 0; v < nv; v++)
            for (size_t r = 0; r < nr; r++) {
                float expected = janas_q4k_dot_ref(w + off + r, x + v, 1);
                assert(memcmp(&expected, tasks[i].y + v * nr + r, sizeof expected) == 0);
            }
        tasks[i].y = plain + off * nv;
        off += nr;
    }
    janas_matvec_q4k_group(pool, tasks, 3);
    assert(memcmp(y, plain, count * nv * sizeof(*y)) == 0);
    memset(&stats, 0xcc, sizeof stats);
    janas_matvec_q4k_group_stats(pool, NULL, 0, &stats);
    assert(stats.groups == 1 && stats.tasks == 0);
    assert(!stats.prepared_rows && !stats.established_rows && !stats.panel_bytes);
    free(w); free(x); free(y); free(plain);
}

int main(void)
{
    struct janas_pool *pool = janas_pool_create(4, NULL);
    assert(pool);
    check_case(pool, 5, 3); /* exact odd-row fallback and shared input */
    check_case(pool, 513, 9); /* concurrent claims, odd token, many row pieces */
    janas_pool_destroy(pool);
    puts("test_matvec_stats: exact rows/vectors, dedup, reset and NULL path ok");
    return 0;
}

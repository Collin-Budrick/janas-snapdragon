/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Activation panels rearrange signed bytes; they must not quantize again,
   copy another activation representation, or change the source metadata. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llm/matvec.h"
#include "llm/quant_panel.h"

static int failures;

/* A separate GNU-linker test build may use --wrap=calloc to exercise the
   metadata allocation failure without adding a production fault hook. */
#if defined(JANAS_TEST_PANEL_WRAP_CALLOC)
static int fail_metadata_alloc;
void *__real_calloc(size_t n, size_t size);
void *__wrap_calloc(size_t n, size_t size)
{
    if (fail_metadata_alloc) {
        fail_metadata_alloc = 0;
        return NULL;
    }
    return __real_calloc(n, size);
}
#endif

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            failures++;                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fputc('\n', stderr);                                               \
        }                                                                      \
    } while (0)

enum { MAX_NB = 8, MAX_NV = 10, MAX_XS = MAX_NB + 3, GUARD = 16 };

static void fill_source(struct janas_block_q8k *x, size_t n, unsigned salt)
{
    unsigned char *bytes = (unsigned char *)x;
    for (size_t i = 0; i < n * sizeof(*x); i++)
        bytes[i] = (unsigned char)(i * 29 + i / 257 + salt);
    for (size_t b = 0; b < n; b++)
        for (size_t i = 0; i < JANAS_QK; i++) {
            int value = (int)((b * 37 + i * 13 + i / 8 + salt) & 255);
            x[b].qs[i] = (int8_t)(value < 128 ? value : value - 256);
            /* The independently quantized 32-value representation must not
               accidentally be the source of the K panel. */
            x[b].qs32[i] = (int8_t)(x[b].qs[i] ^ 0x5a);
        }
}

static int all_bytes(const void *p, size_t n, unsigned char value)
{
    const unsigned char *bytes = p;
    for (size_t i = 0; i < n; i++)
        if (bytes[i] != value)
            return 0;
    return 1;
}

static void check_layout(const struct janas_q8k_panel *p,
                         const struct janas_block_q8k *x, size_t xs,
                         size_t nb, size_t nv)
{
    CHECK(p->x == x && p->x_stride == xs && p->nb == nb && p->nv == nv &&
              p->pairs == nv / 2,
          "panel metadata: nb %zu nv %zu", nb, nv);
    if (nv < 2)
        return;
    CHECK(p->qs != NULL, "panel bytes absent: nb %zu nv %zu", nb, nv);
    if (!p->qs)
        return;
    for (size_t pair = 0; pair < nv / 2; pair++)
        for (size_t b = 0; b < nb; b++)
            for (size_t chunk = 0; chunk < JANAS_QK / 8; chunk++) {
                const int8_t *record =
                    p->qs + ((pair * nb + b) * (JANAS_QK / 8) + chunk) * 16;
                const int8_t *a = x[(2 * pair) * xs + b].qs + 8 * chunk;
                const int8_t *c = x[(2 * pair + 1) * xs + b].qs + 8 * chunk;
                CHECK(!memcmp(record, a, 8) && !memcmp(record + 8, c, 8),
                      "panel layout: nb %zu nv %zu pair %zu block %zu chunk %zu",
                      nb, nv, pair, b, chunk);
            }
}

static void test_bytes(void)
{
    static const size_t widths[] = {1, 2, 8};
    for (size_t i = 0; i < sizeof widths / sizeof *widths; i++)
        for (size_t nv = 1; nv <= 9; nv++) {
            size_t nb = widths[i];
            CHECK(janas_q8k_panel_bytes(nb, nv) == nv / 2 * nb * 512,
                  "panel size: nb %zu nv %zu", nb, nv);
        }
    CHECK(janas_q8k_panel_bytes(0, 2) == SIZE_MAX, "zero block count");
    CHECK(janas_q8k_panel_bytes(1, 0) == SIZE_MAX, "zero token count");
    CHECK(janas_q8k_panel_bytes(1, 10) == SIZE_MAX,
          "unsupported token count");
    CHECK(janas_q8k_panel_bytes(SIZE_MAX, 2) == SIZE_MAX,
          "block count overflow");
    CHECK(janas_q8k_panel_bytes(SIZE_MAX / 512 + 1, 2) == SIZE_MAX,
          "panel multiplication overflow");
    CHECK(janas_q8k_panel_bytes(2, SIZE_MAX) == SIZE_MAX,
          "token count overflow");
}

static void test_pack(void)
{
    enum { N = MAX_NV * MAX_XS, PAYLOAD = 4 * MAX_NB * 512 };
    struct janas_block_q8k *x = malloc(N * sizeof(*x));
    struct janas_block_q8k *saved = malloc(N * sizeof(*saved));
    _Alignas(16) unsigned char storage[GUARD + PAYLOAD + GUARD];
    static const size_t blocks[] = {1, 2, 8};
    CHECK(x && saved, "pack fixture allocation");
    if (!x || !saved)
        goto done;
    fill_source(x, N, 11);
    memcpy(saved, x, N * sizeof(*x));
    for (size_t i = 0; i < sizeof blocks / sizeof *blocks; i++)
        for (size_t nv = 1; nv <= 9; nv++) {
            size_t nb = blocks[i], xs = nb + 2;
            size_t bytes = janas_q8k_panel_bytes(nb, nv);
            struct janas_q8k_panel p = {0};
            memset(storage, 0xa5, sizeof storage);
            int rc = janas_q8k_panel_pack(
                &p, bytes ? storage + GUARD : NULL, bytes, x, xs, nb, nv);
            CHECK(rc == 0, "pack: nb %zu nv %zu", nb, nv);
            if (rc == 0) {
                check_layout(&p, x, xs, nb, nv);
                CHECK(p.qs == (bytes ? (const int8_t *)storage + GUARD : NULL),
                      "pack storage binding: nb %zu nv %zu", nb, nv);
            }
            CHECK(all_bytes(storage, GUARD, 0xa5) &&
                      all_bytes(storage + GUARD + bytes,
                                sizeof storage - GUARD - bytes, 0xa5),
                  "pack changed storage guards: nb %zu nv %zu", nb, nv);
            CHECK(!memcmp(x, saved, N * sizeof(*x)),
                  "pack changed source or metadata: nb %zu nv %zu", nb, nv);
        }
    {
        struct janas_q8k_panel p = {0};
        CHECK(janas_q8k_panel_pack(&p, storage + GUARD, 1024, x, 0, 2, 2) == 0,
              "default pack stride rejected");
        check_layout(&p, x, 2, 2, 2);
        CHECK(!memcmp(x, saved, N * sizeof(*x)),
              "default stride pack changed source");
    }
done:
    free(x);
    free(saved);
}

static void test_pack_invalid(void)
{
    struct janas_block_q8k x[6], saved[6];
    _Alignas(16) unsigned char storage[1024];
    struct janas_q8k_panel p;
    fill_source(x, 6, 71);
    memcpy(saved, x, sizeof x);
#define BAD_PACK(label, out, mem, cap, input, xs, nb, nv)                       \
    do {                                                                       \
        memset(&p, 0, sizeof p);                                                \
        memset(storage, 0xa5, sizeof storage);                                  \
        CHECK(janas_q8k_panel_pack(out, mem, cap, input, xs, nb, nv) == -1,      \
              "%s accepted", label);                                         \
        CHECK(all_bytes(storage, sizeof storage, 0xa5),                         \
              "%s changed destination", label);                              \
        CHECK(!memcmp(x, saved, sizeof x), "%s changed source", label);        \
    } while (0)
    BAD_PACK("short storage", &p, storage, 511, x, 1, 1, 2);
    BAD_PACK("missing storage", &p, NULL, 512, x, 1, 1, 2);
    BAD_PACK("missing source", &p, storage, 512, NULL, 1, 1, 2);
    BAD_PACK("missing result", NULL, storage, 512, x, 1, 1, 2);
    BAD_PACK("short input stride", &p, storage, 1024, x, 1, 2, 2);
    BAD_PACK("overflowing stride", &p, storage, 1024, x, SIZE_MAX, 1, 4);
    BAD_PACK("overflowing block count", &p, storage, sizeof storage, x,
             SIZE_MAX, SIZE_MAX, 2);
    BAD_PACK("zero block count", &p, storage, sizeof storage, x, 1, 0, 2);
    BAD_PACK("zero token count", &p, storage, sizeof storage, x, 1, 1, 0);
    BAD_PACK("unsupported token count", &p, storage, sizeof storage, x,
             1, 1, 10);
#undef BAD_PACK
}

static int prepared(const struct janas_q8k_panels *p, size_t i)
{
    return p->views && i < p->n && p->views[i].x != NULL;
}

static void check_freed(const struct janas_q8k_panels *p)
{
    CHECK(!p->views && !p->allocation && !p->n && !p->unique && !p->bytes,
          "free did not reset panel group");
}

static void test_group(void)
{
    enum { N = MAX_NV * MAX_XS, NT = 12, CAP = 8 * 1024 * 1024 };
    struct janas_block_q8k *x = malloc(N * sizeof(*x));
    struct janas_block_q8k *other = malloc(N * sizeof(*other));
    struct janas_block_q8k *saved = malloc(2 * N * sizeof(*saved));
    unsigned char weights[2 * MAX_NB * sizeof(struct janas_block_q6k)] = {0};
    float y[NT][MAX_NV * 5];
    struct janas_matvec_task tasks[NT];
    struct janas_q8k_panels p = {0};
    CHECK(x && other && saved, "group fixture allocation");
    if (!x || !other || !saved)
        goto done;
    fill_source(x, N, 3);
    fill_source(other, N, 149);
    memcpy(saved, x, N * sizeof(*x));
    memcpy(saved + N, other, N * sizeof(*other));
    for (size_t i = 0; i < NT; i++) {
        tasks[i] = (struct janas_matvec_task){
            .type = JANAS_Q4_K, .w = weights, .x = x, .y = y[i],
            .rows = 2, .cols = 512, .n_vec = 4,
            .x_stride = 4, .y_stride = 5};
        for (size_t j = 0; j < MAX_NV * 5; j++)
            y[i][j] = -12345.0f;
    }
    tasks[2].type = JANAS_Q6_K;
    tasks[3].type = JANAS_Q6_K_P;
    tasks[4].type = JANAS_Q8_0;
    tasks[5].x = other;
    tasks[6].x_stride = 5;
    tasks[7].cols = 2048;
    tasks[7].x_stride = 10;
    tasks[8].n_vec = 2;
    tasks[9].cols = 544; /* not a complete K block */
    tasks[10].type = JANAS_Q5_K;
    tasks[11].n_vec = 10;
    CHECK(janas_q8k_panels_create(tasks, NT, CAP, &p) == 0,
          "create panel group");
    CHECK(p.views && p.allocation && p.n == NT && p.unique == 5,
          "group state: n %zu unique %zu", p.n, p.unique);
    size_t meta = NT * (sizeof(*p.views) + sizeof(size_t));
    size_t payload = 3 * janas_q8k_panel_bytes(2, 4) +
                     janas_q8k_panel_bytes(8, 4) +
                     janas_q8k_panel_bytes(2, 2);
    CHECK(p.bytes == meta + payload && p.bytes <= CAP,
          "group cap accounting: %zu bytes", p.bytes);
    if (p.views && p.n == NT) {
        for (size_t i = 0; i < 4; i++) {
            CHECK(prepared(&p, i), "compatible K task %zu not prepared", i);
            CHECK(p.views[i].qs == p.views[0].qs,
                  "gate/up/K-format task %zu not deduplicated", i);
        }
        static const size_t distinct[] = {0, 5, 6, 7, 8};
        for (size_t i = 0; i < sizeof distinct / sizeof *distinct; i++) {
            size_t k = distinct[i];
            CHECK(prepared(&p, k), "distinct task %zu not prepared", k);
            if (prepared(&p, k))
                check_layout(&p.views[k], tasks[k].x, tasks[k].x_stride,
                             tasks[k].cols / JANAS_QK, tasks[k].n_vec);
            for (size_t j = 0; j < i; j++)
                CHECK(p.views[k].qs != p.views[distinct[j]].qs,
                      "different panel keys %zu and %zu aliased", k,
                      distinct[j]);
        }
        CHECK(!prepared(&p, 4) && !prepared(&p, 9) && !prepared(&p, 10) &&
                  !prepared(&p, 11),
              "unsupported type, width or token count was prepared");
    }
    CHECK(!memcmp(x, saved, N * sizeof(*x)) &&
              !memcmp(other, saved + N, N * sizeof(*other)),
          "group changed source or metadata");
    for (size_t i = 0; i < NT; i++)
        for (size_t j = 0; j < MAX_NV * 5; j++)
            CHECK(y[i][j] == -12345.0f, "group wrote task output");
    size_t exact_cap = p.bytes;
    janas_q8k_panels_free(&p);
    check_freed(&p);
    if (exact_cap) {
        CHECK(janas_q8k_panels_create(tasks, NT, exact_cap, &p) == 0 &&
                  p.unique == 5 && p.bytes <= exact_cap,
              "exact group cap rejected");
        janas_q8k_panels_free(&p);
        CHECK(janas_q8k_panels_create(tasks, NT, exact_cap - 1, &p) == 0 &&
                  p.bytes < exact_cap,
              "group exceeded smaller cap");
        janas_q8k_panels_free(&p);
    }
    for (size_t cap = 0; cap <= 1; cap++) {
        CHECK(janas_q8k_panels_create(tasks, NT, cap, &p) == 0 &&
                  !p.views && !p.allocation && !p.unique && p.bytes <= cap,
              "tiny cap %zu did not fall back", cap);
        janas_q8k_panels_free(&p);
    }
done:
    janas_q8k_panels_free(&p);
    check_freed(&p);
    janas_q8k_panels_free(&p);
    free(x);
    free(other);
    free(saved);
}

static void test_group_defaults_and_invalid(void)
{
    struct janas_block_q8k x[6];
    struct janas_matvec_task tasks[2] = {
        {.type = JANAS_Q4_K, .x = x, .rows = 2, .cols = 512},
        {.type = JANAS_Q6_K, .x = x, .rows = 2, .cols = 512,
         .n_vec = 1, .x_stride = 2}};
    struct janas_q8k_panels p = {0};
    fill_source(x, 6, 89);
    CHECK(janas_q8k_panels_create(tasks, 2, 4096, &p) == 0,
          "default task fields rejected");
    CHECK(p.views && p.n == 2 && p.unique == 1 && !p.allocation &&
              p.bytes == 2 * (sizeof(*p.views) + sizeof(size_t)),
          "default one-token fields did not normalize and deduplicate");
    if (p.views)
        for (size_t i = 0; i < 2; i++)
            if (prepared(&p, i)) {
                check_layout(&p.views[i], x, 2, 2, 1);
                CHECK(!p.views[i].qs, "one-token group has packed bytes");
            }
    janas_q8k_panels_free(&p);
    CHECK(janas_q8k_panels_create(NULL, 0, 4096, &p) == 0,
          "empty task group rejected");
    janas_q8k_panels_free(&p);
    CHECK(janas_q8k_panels_create(NULL, 1, 4096, &p) == -1,
          "missing task array accepted");
    CHECK(janas_q8k_panels_create(tasks, 1, 4096, NULL) == -1,
          "missing group output accepted");
    /* Reject impossible allocation sizes before walking the task array. */
    int rc = janas_q8k_panels_create(tasks, SIZE_MAX, 4096, &p);
    CHECK(rc == 0 && !p.views && !p.allocation && !p.bytes,
          "overflowing task count did not fail safely");
    janas_q8k_panels_free(&p);
    tasks[0].n_vec = 4;
    tasks[0].x_stride = SIZE_MAX;
    rc = janas_q8k_panels_create(tasks, 1, 4096, &p);
    CHECK(rc == 0 && !prepared(&p, 0),
          "overflowing input span did not fall back");
    janas_q8k_panels_free(&p);
    tasks[0].n_vec = 2;
    tasks[0].x_stride = 1;
    CHECK(janas_q8k_panels_create(tasks, 1, 4096, &p) == 0 &&
              !prepared(&p, 0),
          "short group input stride did not fall back");
    janas_q8k_panels_free(&p);
    tasks[0].x_stride = 0;
    tasks[0].cols = (JANAS_Q8K_PANEL_CAP / 512 + 1) * JANAS_QK;
    CHECK(janas_q8k_panels_create(tasks, 1, SIZE_MAX, &p) == 0 &&
              !prepared(&p, 0) && !p.allocation &&
              p.bytes <= JANAS_Q8K_PANEL_CAP,
          "caller cap did not clamp to eight MiB");
    janas_q8k_panels_free(&p);
    tasks[0].cols = tasks[1].cols = 512;
    tasks[0].rows = 1;
    tasks[1].x = NULL;
    CHECK(janas_q8k_panels_create(tasks, 2, 4096, &p) == 0 &&
              !prepared(&p, 0) && !prepared(&p, 1),
          "single row or absent input was prepared");
    janas_q8k_panels_free(&p);
    check_freed(&p);
    janas_q8k_panels_free(NULL);
}

#if defined(JANAS_TEST_PANEL_WRAP_CALLOC)
static void test_metadata_oom(void)
{
    struct janas_block_q8k x[4];
    struct janas_matvec_task task = {.type = JANAS_Q4_K, .x = x,
                                     .rows = 2, .cols = 512, .n_vec = 2};
    struct janas_q8k_panels p = {0};
    fill_source(x, 4, 113);
    fail_metadata_alloc = 1;
    CHECK(janas_q8k_panels_create(&task, 1, 4096, &p) == 0 &&
              !fail_metadata_alloc,
          "metadata OOM was not exercised");
    check_freed(&p);
    janas_q8k_panels_free(&p);
    janas_q8k_panels_free(&p);
    CHECK(janas_q8k_panels_create(&task, 1, 4096, &p) == 0 &&
              prepared(&p, 0),
          "retry after metadata OOM failed");
    janas_q8k_panels_free(&p);
    check_freed(&p);
}
#endif

int main(void)
{
    test_bytes();
    test_pack();
    test_pack_invalid();
    test_group();
    test_group_defaults_and_invalid();
#if defined(JANAS_TEST_PANEL_WRAP_CALLOC)
    test_metadata_oom();
#endif
    printf("test_quant_panel: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}

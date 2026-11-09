/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Exact modular Mojo leaf validation. No model or external corpus is used.
   Set JANAS_MOJO_LIBRARY to require a module; otherwise absence is a skip. */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/pool.h"
#include "llm/matvec.h"
#include "llm/mojo.h"
#include "llm/quant_panel.h"

typedef int32_t (*mojo_export_fn)(int32_t, int32_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

static int mojo_failures;
static size_t mojo_cases, mojo_words;
static uint32_t mojo_rng = 0x189f427bu;

#define MCHECK(cond, ...) do { if (!(cond)) { \
    mojo_failures++; fprintf(stderr, "FAIL mojo %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static uint32_t mojo_random(void)
{
    uint32_t x = mojo_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return mojo_rng = x;
}

static uint64_t mojo_hash(const void *data, size_t bytes)
{
    const uint8_t *p = data;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; i++)
        h = (h ^ p[i]) * UINT64_C(1099511628211);
    return h;
}

struct mojo_fixture {
    int type, level;
    size_t rows, nb, nv, rb, pb, xs, ys, out_count, pack_bytes;
    uint8_t *w, *p1, *p0;
    struct janas_block_q8k *x;
    int8_t *storage;
    float *a, *b, *ref;
    struct janas_q8k_panel panel;
};

static void mojo_fixture_free(struct mojo_fixture *f)
{
    free(f->w); free(f->p1); free(f->p0); free(f->x); free(f->storage);
    free(f->a); free(f->b); free(f->ref);
    memset(f, 0, sizeof *f);
}

/* trial 1 includes signed zero/subnormal/maximum half scales; trial 2
   catches an invalid merger of the two independent Q4_K FMA chains. */
static int mojo_fixture_create(struct mojo_fixture *f, int type, int level,
    size_t rows, size_t nb, size_t nv, int padded, int trial)
{
    static const uint16_t half[] = {0, 0x8000, 1, 0x03ff, 0x0400,
                                    0x3c00, 0xbc00, 0x7bff};
    memset(f, 0, sizeof *f);
    f->type = type; f->level = level; f->rows = rows;
    f->nb = nb; f->nv = nv;
    f->rb = nb * janas_qtype_block_size(type) + (padded ? 18 : 0);
    f->pb = nb * JANAS_Q6KP_PLANE + (padded ? 7 : 0);
    f->xs = nb + (padded ? 3 : 0); f->ys = rows + (padded ? 5 : 0);
    f->out_count = 8 + nv * f->ys;
    f->pack_bytes = janas_q8k_panel_bytes(nb, nv);
    f->w = malloc(rows * f->rb + 16);
    f->p1 = malloc(rows * f->pb + 16);
    f->p0 = malloc(rows * f->pb + 16);
    f->x = malloc(nv * f->xs * sizeof *f->x);
    f->storage = malloc(f->pack_bytes + 32);
    f->a = malloc(f->out_count * sizeof *f->a);
    f->b = malloc(f->out_count * sizeof *f->b);
    f->ref = malloc(f->out_count * sizeof *f->ref);
    struct janas_block_q6k *q6 = malloc(nb * sizeof *q6);
    float *xf = malloc(nb * JANAS_QK * sizeof *xf);
    if (!f->w || !f->p1 || !f->p0 || !f->x || !f->storage ||
        !f->a || !f->b || !f->ref || !q6 || !xf) {
        free(q6); free(xf); mojo_fixture_free(f); return -1;
    }
    memset(f->w, 0xa5, rows * f->rb + 16);
    memset(f->p1, 0xa5, rows * f->pb + 16);
    memset(f->p0, 0xa5, rows * f->pb + 16);
    memset(f->x, 0x5a, nv * f->xs * sizeof *f->x);
    memset(f->storage, 0x6c, f->pack_bytes + 32);
    for (size_t r = 0; r < rows; r++) {
        for (size_t i = 0; i < nb * sizeof *q6; i++)
            ((uint8_t *)q6)[i] = (uint8_t)mojo_random();
        for (size_t b = 0; b < nb; b++) {
            q6[b].d = trial ? half[(r * nb + b) % 8] :
                janas_fp32_to_fp16((float)(1 + mojo_random() % 200) / 16384.0f);
            if (trial)
                for (size_t g = 0; g < 16; g++)
                    q6[b].scales[g] = (g & 1) ? 127 : -128;
        }
        if (type == JANAS_Q4_K) {
            struct janas_block_q4k *q4 = (void *)(f->w + r * f->rb);
            for (size_t b = 0; b < nb; b++) {
                for (size_t i = 0; i < sizeof *q4; i++)
                    ((uint8_t *)(q4 + b))[i] = (uint8_t)mojo_random();
                q4[b].d = trial ? half[(r * nb + b) % 8] :
                    janas_fp32_to_fp16((float)(1 + mojo_random() % 100) / 16384.0f);
                q4[b].dmin = trial ? half[(r + 3 * b + 1) % 8] :
                    janas_fp32_to_fp16((float)(1 + mojo_random() % 100) / 16384.0f);
                if (trial == 2) {
                    memset(q4[b].qs, 0x11, sizeof q4[b].qs);
                    memset(q4[b].scales, 0xff, sizeof q4[b].scales);
                    q4[b].d = b ? 1 : 0x7bff;
                    q4[b].dmin = b ? 0 : 0x7bff;
                }
            }
        } else if (type == JANAS_Q6_K) {
            memcpy(f->w + r * f->rb, q6, nb * sizeof *q6);
        } else {
            janas_q6k_to_planes(q6, nb, (void *)(f->w + r * f->rb),
                f->p1 + r * f->pb, f->p0 + r * f->pb);
        }
    }
    for (size_t v = 0; v < nv; v++) {
        for (size_t i = 0; i < nb * JANAS_QK; i++)
            xf[i] = trial == 2 ? 0.05f + (float)((i + 13 * v) % 31) / 31 :
                (float)((int)(mojo_random() % 2049) - 1024) / 1024;
        janas_q8k_quantize(xf, f->x + v * f->xs, nb * JANAS_QK);
    }
    free(q6); free(xf);
    if (janas_q8k_panel_pack(&f->panel,
            f->pack_bytes ? f->storage + 16 : NULL, f->pack_bytes,
            f->x, f->xs, nb, nv)) {
        mojo_fixture_free(f); return -1;
    }
    for (size_t i = 0; i < f->out_count; i++)
        f->a[i] = f->b[i] = f->ref[i] = -12345.0f;
    for (size_t v = 0; v < nv; v++)
        for (size_t r = 0; r < rows; r++) {
            const void *w = f->w + r * f->rb;
            f->ref[4 + r + v * f->ys] = type == JANAS_Q4_K ?
                janas_q4k_dot_ref(w, f->x + v * f->xs, nb) :
                type == JANAS_Q6_K ? janas_q6k_dot_ref(w, f->x + v * f->xs, nb) :
                janas_q6kp_dot_ref(w, level >= 2 ? f->p1 + r * f->pb : NULL,
                    level >= 3 ? f->p0 + r * f->pb : NULL,
                    f->x + v * f->xs, nb);
        }
    return 0;
}

static size_t mojo_bad_word(const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (memcmp(a + i, b + i, sizeof *a)) return i;
    return n;
}

static void mojo_check_words(const float *a, const float *b, size_t n,
    const struct mojo_fixture *f, const char *label)
{
    size_t bad = mojo_bad_word(a, b, n);
    MCHECK(bad == n, "%s type=%d level=%d nb=%zu nv=%zu rows=%zu at=%zu",
        label, f->type, f->level, f->nb, f->nv, f->rows, bad);
    mojo_words += n;
}

static uint64_t mojo_inputs_hash(const struct mojo_fixture *f)
{
    return mojo_hash(f->w, f->rows * f->rb + 16) ^
        mojo_hash(f->p1, f->rows * f->pb + 16) ^
        (mojo_hash(f->p0, f->rows * f->pb + 16) << 1) ^
        mojo_hash(f->x, f->nv * f->xs * sizeof *f->x) ^
        mojo_hash(f->storage, f->pack_bytes + 32);
}

static int mojo_export_call(mojo_export_fn fn, struct mojo_fixture *f,
    size_t rows, float *y)
{
    return fn(f->type, f->level, (uint64_t)(uintptr_t)f->w, f->rb,
        f->level >= 2 ? (uint64_t)(uintptr_t)f->p1 : 0,
        f->level >= 3 ? (uint64_t)(uintptr_t)f->p0 : 0,
        f->pb, (uint64_t)(uintptr_t)f->x, f->xs,
        (uint64_t)(uintptr_t)f->panel.qs, f->nb, f->nv, rows,
        (uint64_t)(uintptr_t)y, f->ys);
}

static void mojo_check_fixture(mojo_export_fn fn, struct mojo_fixture *f)
{
    uint64_t before = mojo_inputs_hash(f);
    int rc = mojo_export_call(fn, f, f->rows, f->a + 4);
    MCHECK(rc == 1, "export declined type=%d nb=%zu nv=%zu", f->type, f->nb, f->nv);
    mojo_check_words(f->a, f->ref, f->out_count, f, "Mojo/reference bits and guards");
    janas_mojo_set_enabled(0);
    size_t done = janas_dot_prepared_rows(f->type, f->w, f->rb,
        f->level >= 2 ? f->p1 : NULL, f->level >= 3 ? f->p0 : NULL,
        f->pb, &f->panel, f->rows, f->b + 4, f->ys);
    MCHECK(done == f->rows, "C prepared path unavailable: done=%zu", done);
    mojo_check_words(f->a, f->b, f->out_count, f, "Mojo/current C bits and guards");
    janas_mojo_set_enabled(1);
    for (size_t i = 0; i < f->out_count; i++) f->b[i] = -12345.0f;
    done = janas_mojo_panel_rows(f->type, f->w, f->rb,
        f->level >= 2 ? f->p1 : NULL, f->level >= 3 ? f->p0 : NULL,
        f->pb, &f->panel, f->rows, f->b + 4, f->ys);
    MCHECK(done == (f->nv == 1 ? 0 : f->rows), "runtime row count %zu", done);
    if (f->nv != 1)
        mojo_check_words(f->a, f->b, f->out_count, f, "runtime/export bits");
    else
        for (size_t i = 0; i < f->out_count; i++)
            MCHECK(f->b[i] == -12345.0f, "single-vector decline wrote output");
    MCHECK(before == mojo_inputs_hash(f), "kernel changed immutable inputs");
    mojo_cases++;
}

static void mojo_test_refusal(mojo_export_fn fn)
{
    struct mojo_fixture f;
    if (mojo_fixture_create(&f, JANAS_Q4_K, 0, 4, 2, 3, 1, 0)) {
        MCHECK(0, "refusal fixture allocation"); return;
    }
    struct janas_q8k_panel bad = f.panel;
#define DECLINE(label, TYPE, W, RB, P1, P0, PB, PANEL, ROWS, Y, YS) do { \
    size_t nr = janas_mojo_panel_rows(TYPE, W, RB, P1, P0, PB, PANEL, ROWS, Y, YS); \
    MCHECK(nr == 0, "%s accepted", label); \
    for (size_t i = 0; i < f.out_count; i++) \
        MCHECK(f.a[i] == -12345.0f, "%s wrote on refusal", label); \
} while (0)
    janas_mojo_set_enabled(1);
    DECLINE("unsupported type", JANAS_Q5_K, f.w, f.rb, NULL, NULL, f.pb,
        &f.panel, 4, f.a + 4, f.ys);
    DECLINE("null weights", f.type, NULL, f.rb, NULL, NULL, f.pb,
        &f.panel, 4, f.a + 4, f.ys);
    DECLINE("null result", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &f.panel, 4, NULL, f.ys);
    DECLINE("null panel", f.type, f.w, f.rb, NULL, NULL, f.pb,
        NULL, 4, f.a + 4, f.ys);
    DECLINE("short row stride", f.type, f.w, 1, NULL, NULL, f.pb,
        &f.panel, 4, f.a + 4, f.ys);
    DECLINE("no row pairs", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &f.panel, 1, f.a + 4, f.ys);
    bad.nv = 0;
    DECLINE("zero vectors", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    bad = f.panel; bad.nv = 10;
    DECLINE("unsupported vectors", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    bad = f.panel; bad.qs = NULL;
    DECLINE("missing panel bytes", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    bad = f.panel; bad.x_stride = 1;
    DECLINE("short input stride", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    bad = f.panel; bad.nb = 0;
    DECLINE("zero blocks", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    DECLINE("short output stride", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &f.panel, 4, f.a + 4, 1);
    bad = f.panel; bad.pairs = 0;
    DECLINE("inconsistent pair count", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    bad = f.panel; bad.nb = SIZE_MAX;
    DECLINE("block multiplication overflow", f.type, f.w, SIZE_MAX, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    bad = f.panel; bad.x_stride = SIZE_MAX;
    DECLINE("input span overflow", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &bad, 4, f.a + 4, f.ys);
    DECLINE("output span overflow", f.type, f.w, f.rb, NULL, NULL, f.pb,
        &f.panel, 4, f.a + 4, SIZE_MAX);
    DECLINE("plane zero without one", JANAS_Q6_K_P, f.w, f.rb, NULL, f.p0, f.pb,
        &f.panel, 4, f.a + 4, f.ys);
    DECLINE("short plane stride", JANAS_Q6_K_P, f.w, f.rb, f.p1, NULL, 1,
        &f.panel, 4, f.a + 4, f.ys);
    /* The public leaf itself refuses odd rows rather than partially writing. */
    MCHECK(mojo_export_call(fn, &f, 3, f.a + 4) == 0, "export accepted odd rows");
    for (size_t i = 0; i < f.out_count; i++)
        MCHECK(f.a[i] == -12345.0f, "export refusal changed guard/result");
#undef DECLINE
    struct mojo_fixture invalid = f;
#define EXPORT_DECLINE(label) do { \
    MCHECK(mojo_export_call(fn, &invalid, invalid.rows, invalid.a + 4) == 0, \
        "export %s accepted", label); \
    for (size_t i = 0; i < f.out_count; i++) \
        MCHECK(f.a[i] == -12345.0f, "export %s wrote on refusal", label); \
    invalid = f; \
} while (0)
    invalid.type = JANAS_Q5_K; EXPORT_DECLINE("unsupported format");
    invalid.level = 1; EXPORT_DECLINE("incompatible prefix level");
    invalid.nv = 0; EXPORT_DECLINE("zero vectors");
    invalid.nv = 10; EXPORT_DECLINE("unsupported vectors");
    invalid.nb = 0; EXPORT_DECLINE("zero blocks");
    invalid.x = NULL; EXPORT_DECLINE("null activations");
    invalid.panel.qs = NULL; EXPORT_DECLINE("null pair panel");
    invalid.xs = 1; EXPORT_DECLINE("short input stride");
    invalid.ys = 1; EXPORT_DECLINE("short output stride");
    invalid.rb = 1; EXPORT_DECLINE("short weight stride");
#undef EXPORT_DECLINE
    mojo_fixture_free(&f);
}

static void mojo_test_group(void)
{
    const int types[] = {JANAS_Q4_K, JANAS_Q6_K, JANAS_Q6_K_P,
                          JANAS_Q6_K_P, JANAS_Q6_K_P};
    const int levels[] = {0, 0, 1, 2, 3};
    struct janas_pool *pool = janas_pool_create(2, NULL);
    MCHECK(pool != NULL, "group pool allocation");
    if (!pool) return;
    for (size_t c = 0; c < 5; c++)
        for (size_t nv = 1; nv <= 9; nv++) {
            struct mojo_fixture f;
            if (mojo_fixture_create(&f, types[c], levels[c], 7, 3, nv, 0, 0)) {
                MCHECK(0, "group fixture allocation"); continue;
            }
            struct janas_matvec_task task = {
                .type = f.type, .w = f.w,
                .plane = {f.level >= 2 ? f.p1 : NULL, f.level >= 3 ? f.p0 : NULL},
                .x = f.x, .y = f.a + 4, .rows = f.rows,
                .cols = f.nb * JANAS_QK, .n_vec = f.nv,
                .x_stride = f.xs, .y_stride = f.ys
            };
            uint64_t before = mojo_inputs_hash(&f);
            janas_mojo_set_enabled(0);
            janas_matvec_q4k_group(pool, &task, 1);
            mojo_check_words(f.a, f.ref, f.out_count, &f, "C group and odd row fallback");
            task.y = f.b + 4;
            struct janas_mojo_stats a = {0}, b = {0};
            janas_mojo_get_stats(&a);
            janas_mojo_set_enabled(1);
            janas_matvec_q4k_group(pool, &task, 1);
            janas_mojo_get_stats(&b);
            mojo_check_words(f.b, f.ref, f.out_count, &f, "Mojo group and odd row fallback");
            MCHECK(nv == 1 ? b.calls == a.calls : b.calls > a.calls,
                "group did not preserve dispatch selection: type=%d nv=%zu", f.type, nv);
            MCHECK(before == mojo_inputs_hash(&f), "group changed immutable inputs");
            mojo_cases++;
            mojo_fixture_free(&f);
        }
    janas_pool_destroy(pool);
}

#if defined(__aarch64__)
static uint64_t mojo_fpcr(void)
{
    uint64_t value;
    __asm__ volatile("mrs %0, fpcr" : "=r"(value));
    return value;
}

static void mojo_set_fpcr(uint64_t value)
{
    __asm__ volatile("msr fpcr, %0\n\tisb" : : "r"(value) : "memory");
}

static void mojo_test_fpcr(mojo_export_fn fn)
{
    const int types[] = {JANAS_Q4_K, JANAS_Q6_K, JANAS_Q6_K_P};
    uint64_t saved = mojo_fpcr();
    for (size_t c = 0; c < 3; c++)
        for (size_t nv = 1; nv <= 9; nv++) {
            struct mojo_fixture f;
            if (mojo_fixture_create(&f, types[c], c == 2 ? 3 : 0,
                    4, 3, nv, 1, 1)) {
                MCHECK(0, "FPCR fixture allocation"); continue;
            }
            for (uint64_t mode = 0; mode < 32; mode++) {
                uint64_t mask = (UINT64_C(3) << 22) | (UINT64_C(1) << 24) |
                    (UINT64_C(1) << 19) | (UINT64_C(1) << 25);
                uint64_t control = (saved & ~mask) | ((mode & 3) << 22);
                if (mode & 4) control |= UINT64_C(1) << 24;
                if (mode & 8) control |= UINT64_C(1) << 19;
                if (mode & 16) control |= UINT64_C(1) << 25;
                mojo_set_fpcr(control);
                uint64_t actual_control = mojo_fpcr();
                int rc = mojo_export_call(fn, &f, f.rows, f.a + 4);
                MCHECK(rc == 1, "FPCR export declined");
                janas_mojo_set_enabled(0);
                size_t done = janas_dot_prepared_rows(f.type, f.w, f.rb,
                    f.level >= 2 ? f.p1 : NULL, f.level >= 3 ? f.p0 : NULL,
                    f.pb, &f.panel, f.rows, f.b + 4, f.ys);
                MCHECK(done == f.rows, "FPCR C prepared declined");
                /* The established C kernel, under the same FPCR, is the
                   authority; the scalar reference is not assumed invariant. */
                mojo_check_words(f.a, f.b, f.out_count, &f, "FPCR C/Mojo words");
                MCHECK(mojo_fpcr() == actual_control, "kernel changed FPCR");
                mojo_set_fpcr(saved);
                mojo_cases++;
            }
            mojo_fixture_free(&f);
        }
    mojo_set_fpcr(saved);
}
#endif

static int mojo_test_all(mojo_export_fn fn)
{
    static const size_t blocks[] = {1, 2, 3, 12};
    static const int types[] = {JANAS_Q4_K, JANAS_Q6_K, JANAS_Q6_K_P,
                                JANAS_Q6_K_P, JANAS_Q6_K_P};
    static const int levels[] = {0, 0, 1, 2, 3};
    for (size_t c = 0; c < 5; c++)
        for (size_t b = 0; b < 4; b++)
            for (size_t nv = 1; nv <= 9; nv++)
                for (int pad = 0; pad < 2; pad++)
                    for (int trial = 0; trial < 3; trial++) {
                        struct mojo_fixture f;
                        size_t rows = 2 + 2 * ((c + b + nv) % 3);
                        if (mojo_fixture_create(&f, types[c], levels[c], rows,
                            blocks[b], nv, pad, trial)) {
                            MCHECK(0, "fixture allocation"); return 1;
                        }
                        mojo_check_fixture(fn, &f);
                        mojo_fixture_free(&f);
                        if (mojo_failures > 20) return 1;
                    }
    mojo_test_refusal(fn);
    mojo_test_group();
#if defined(__aarch64__)
    mojo_test_fpcr(fn);
#endif
    return mojo_failures != 0;
}

static mojo_export_fn mojo_test_load(void)
{
    const char *path = getenv("JANAS_MOJO_LIBRARY");
    if (!janas_mojo_available()) {
        if (path && *path)
            MCHECK(0, "required module unavailable: %s", janas_mojo_status());
        else
            fprintf(stderr, "SKIP optional Mojo module: %s\n", janas_mojo_status());
        return NULL;
    }
    if (!path || !*path) {
        MCHECK(0, "available module has no explicit test library path"); return NULL;
    }
    /* Keep this handle alive until process exit, matching the runtime loader. */
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        MCHECK(0, "test dlopen: %s", dlerror()); return NULL;
    }
    mojo_export_fn fn = NULL;
    void *sym = dlsym(handle, "janas_mojo_panel_rows_v1");
    _Static_assert(sizeof fn == sizeof sym, "function pointer size");
    memcpy(&fn, &sym, sizeof fn);
    MCHECK(fn != NULL, "missing version 1 panel export");
    return fn;
}

#ifndef JANAS_MOJO_TEST_LIBRARY
static int mojo_test_single_runtime(void)
{
    const int types[] = {JANAS_Q4_K, JANAS_Q6_K, JANAS_Q6_K_P,
                          JANAS_Q6_K_P, JANAS_Q6_K_P};
    const int levels[] = {0, 0, 1, 2, 3};
    const size_t blocks[] = {1, 3, 12}, rows[] = {1, 7, 64};
    MCHECK(getenv("JANAS_CPU_TILES") == NULL,
        "--single-runtime requires JANAS_CPU_TILES to be unset");
    MCHECK(janas_mojo_available(), "single-runtime requires a configured module: %s",
        janas_mojo_status());
    if (mojo_failures) return 1;
    struct janas_pool *pool = janas_pool_create(2, NULL);
    MCHECK(pool != NULL, "single-runtime pool allocation");
    if (!pool) return 1;
    for (size_t c = 0; c < 5; c++)
        for (size_t b = 0; b < 3; b++)
            for (size_t r = 0; r < 3; r++)
                for (size_t implicit = 0; implicit < 2; implicit++) {
                    struct mojo_fixture f;
                    if (mojo_fixture_create(&f, types[c], levels[c], rows[r],
                            blocks[b], 1, 0, 1)) {
                        MCHECK(0, "single-runtime fixture allocation"); continue;
                    }
                    struct janas_matvec_task task = {
                        .type = f.type, .w = f.w,
                        .plane = {f.level >= 2 ? f.p1 : NULL,
                                  f.level >= 3 ? f.p0 : NULL},
                        .x = f.x, .y = f.a + 4, .rows = f.rows,
                        .cols = f.nb * JANAS_QK, .n_vec = implicit ? 0 : 1,
                        .x_stride = f.xs, .y_stride = f.ys
                    };
                    uint64_t before_hash = mojo_inputs_hash(&f);
                    struct janas_matvec_stats base, selected;
                    struct janas_mojo_stats before, after;
                    janas_mojo_set_enabled(0);
                    MCHECK(!janas_quant_panels_enabled(),
                        "disabled baseline unexpectedly enabled panels");
                    janas_mojo_get_stats(&before);
                    janas_matvec_q4k_group_stats(pool, &task, 1, &base);
                    janas_mojo_get_stats(&after);
                    mojo_check_words(f.a, f.ref, f.out_count, &f,
                        "single-runtime disabled C/reference words");
                    MCHECK(base.panel_views == 0 && base.panel_unique == 0 &&
                        base.panel_bytes == 0 && base.prepared_rows == 0 &&
                        base.panel_prepare_seconds == 0.0 &&
                        base.established_rows == f.rows,
                        "disabled baseline allocated/prepared a panel");
                    MCHECK(after.calls == before.calls && after.rows == before.rows &&
                        after.row_vectors == before.row_vectors &&
                        after.refusals == before.refusals,
                        "disabled baseline changed Mojo counters");
                    for (size_t i = 0; i < f.out_count; i++)
                        f.a[i] = f.b[i] = -12345.0f;
                    /* Both an explicit M=1 task and its legacy n_vec=0
                       alias share activations within this actual group. */
                    struct janas_matvec_task tasks[2] = {task, task};
                    tasks[1].y = f.b + 4;
                    tasks[1].n_vec = implicit ? 1 : 0;
                    janas_mojo_set_enabled(1);
                    MCHECK(janas_mojo_enabled() && janas_quant_panels_enabled() &&
                        janas_quant_panels_min_vectors() == 2,
                        "single-runtime did not select Mojo-only panel policy");
                    janas_mojo_get_stats(&before);
                    janas_matvec_q4k_group_stats(pool, tasks, 2, &selected);
                    janas_mojo_get_stats(&after);
                    mojo_check_words(f.a, f.ref, f.out_count, &f,
                        "single-runtime Mojo-on first C output");
                    mojo_check_words(f.b, f.ref, f.out_count, &f,
                        "single-runtime Mojo-on second C output");
                    MCHECK(selected.panel_views == 0 && selected.panel_unique == 0 &&
                        selected.panel_bytes == 0 && selected.prepared_rows == 0 &&
                        selected.panel_prepare_seconds == 0.0 &&
                        selected.established_rows == 2 * f.rows,
                        "all-singleton group allocated/prepared a panel");
                    MCHECK(after.calls == before.calls && after.rows == before.rows &&
                        after.row_vectors == before.row_vectors &&
                        after.refusals == before.refusals,
                        "all-singleton group executed/refused Mojo work");
                    MCHECK(before_hash == mojo_inputs_hash(&f),
                        "single-runtime changed immutable inputs");
                    mojo_cases++;
                    mojo_fixture_free(&f);
                }
    janas_pool_destroy(pool);
    janas_mojo_set_enabled(-1);
    printf("{\"test\":\"mojo\",\"mode\":\"single_runtime\",\"cases\":%zu,"
        "\"compared_words\":%zu,\"failures\":%d,\"skipped\":false}\n",
        mojo_cases, mojo_words, mojo_failures);
    return mojo_failures ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && !strcmp(argv[1], "--single-runtime"))
        return mojo_test_single_runtime();
    if (argc != 1) {
        fprintf(stderr, "usage: test_mojo [--single-runtime]\n"); return 2;
    }
    mojo_export_fn fn = mojo_test_load();
    if (fn) {
        MCHECK(janas_quant_panels_enabled(), "launch with JANAS_CPU_TILES=1 for C comparison");
        if (!mojo_failures) mojo_test_all(fn);
    }
    janas_mojo_set_enabled(-1);
    printf("{\"test\":\"mojo\",\"cases\":%zu,\"compared_words\":%zu,"
           "\"failures\":%d,\"skipped\":%s}\n", mojo_cases, mojo_words,
           mojo_failures, fn ? "false" : "true");
    return mojo_failures ? 1 : 0;
}
#endif

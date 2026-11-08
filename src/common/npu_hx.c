/* SPDX-License-Identifier: GPL-3.0-or-later */
/* The common NPU matrix interface on a dynamically loaded Hexagon provider.
   GGML supplies tensor operations; Janas keeps model execution and caches. */
#include "npu.h"

#if defined(_WIN32) && defined(__aarch64__) && defined(JANAS_HEXAGON)

#include "aligned.h"
#include "npu_hexagon.h"
#include "npu_ir.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct hx_alloc {
    void *ptr;
    struct hx_alloc *next;
};

struct janas_npu {
    struct janas_hx_context *hx;
    struct hx_alloc *allocs;
    struct janas_npu_mm *products;
    int made;
    double setup_s;
};

struct janas_npu_mm {
    struct janas_npu *owner;
    struct janas_hx_plan *plan;
    struct janas_npu_mm *next;
    float *x[2], *y[2];
    int active[2], result[2];
};

static double seconds(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float half_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 31u, fraction = h & 1023u, bits;
    if (!exponent) {
        if (!fraction)
            bits = sign;
        else {
            int e = -14;
            while (!(fraction & 1024u)) {
                fraction <<= 1;
                e--;
            }
            bits = sign | (uint32_t)(e + 127) << 23 |
                   (fraction & 1023u) << 13;
        }
    } else if (exponent == 31u)
        bits = sign | 0x7f800000u | fraction << 13;
    else
        bits = sign | (exponent + 112u) << 23 | fraction << 13;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static uint16_t float_half(float f)
{
    /* The row-scaled INT8 interface promises finite FP16 weights. The
       native packed interface avoids this conversion entirely. */
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    uint32_t sign = b >> 16 & 0x8000u, exponent = b >> 23 & 255u;
    uint32_t fraction = b & 0x7fffffu;
    if (exponent == 255u)
        return (uint16_t)(sign | 0x7c00u | (fraction ? 0x200u : 0));
    int e = (int)exponent - 127 + 15;
    if (e >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10)
            return (uint16_t)sign;
        fraction |= 0x800000u;
        int shift = 14 - e;
        uint32_t v = fraction >> shift;
        uint32_t rest = fraction & ((1u << shift) - 1), half = 1u << (shift - 1);
        return (uint16_t)(sign | (v + (rest > half || (rest == half && (v & 1)))));
    }
    uint32_t v = (uint32_t)e << 10 | fraction >> 13, rest = fraction & 8191u;
    return (uint16_t)(sign | (v + (rest > 4096u || (rest == 4096u && (v & 1)))));
}

struct janas_npu *janas_npu_create(char *err, size_t err_len)
{
    struct janas_npu *u = calloc(1, sizeof *u);
    if (!u) {
        snprintf(err, err_len, "no memory for Hexagon context");
        return NULL;
    }
    u->hx = janas_hx_create(err, err_len);
    if (!u->hx) {
        free(u);
        return NULL;
    }
    return u;
}

void janas_npu_destroy(struct janas_npu *u)
{
    if (!u)
        return;
    while (u->products)
        janas_npu_mm_destroy(u->products);
    while (u->allocs)
        janas_npu_free(u, u->allocs->ptr);
    janas_hx_destroy(u->hx);
    free(u);
}

const char *janas_npu_name(const struct janas_npu *u)
{
    return u ? janas_hx_name(u->hx) : "";
}

void *janas_npu_alloc(struct janas_npu *u, size_t bytes)
{
    if (!u || !bytes)
        return NULL;
    struct hx_alloc *a = malloc(sizeof *a);
    if (!a)
        return NULL;
    a->ptr = janas_aligned_alloc(4096, bytes);
    if (!a->ptr) {
        free(a);
        return NULL;
    }
    memset(a->ptr, 0, bytes);
    a->next = u->allocs;
    u->allocs = a;
    return a->ptr;
}

void janas_npu_free(struct janas_npu *u, void *ptr)
{
    if (!u || !ptr)
        return;
    for (struct hx_alloc **a = &u->allocs; *a; a = &(*a)->next) {
        if ((*a)->ptr != ptr)
            continue;
        struct hx_alloc *old = *a;
        *a = old->next;
        janas_aligned_free(old->ptr);
        free(old);
        return;
    }
}

struct janas_npu_mm *janas_npu_mm_create(struct janas_npu *u, int m, int n,
                                          int k, int wtype, const void *w,
                                          const uint16_t *scale, float *x,
                                          float *y, char *err, size_t err_len)
{
    if (!u || m <= 0 || n <= 0 || k <= 0 || !w || !x || !y ||
        (wtype == JANAS_NPU_I8 && !scale) ||
        (size_t)n > SIZE_MAX / (size_t)k / sizeof(uint16_t)) {
        snprintf(err, err_len, "bad Hexagon product");
        return NULL;
    }
    enum janas_hx_type type;
    switch (wtype) {
    case JANAS_NPU_F16: type = JANAS_HX_F16; break;
    case JANAS_NPU_I8: type = JANAS_HX_F16; break;
    case JANAS_NPU_Q4_0: type = JANAS_HX_Q4_0; break;
    case JANAS_NPU_Q4_1: type = JANAS_HX_Q4_1; break;
    case JANAS_NPU_Q8_0: type = JANAS_HX_Q8_0; break;
    case JANAS_NPU_IQ4_NL: type = JANAS_HX_IQ4_NL; break;
    case JANAS_NPU_MXFP4: type = JANAS_HX_MXFP4; break;
    case JANAS_NPU_Q4_K: type = JANAS_HX_Q4_K; break;
    case JANAS_NPU_Q5_K: type = JANAS_HX_Q5_K; break;
    case JANAS_NPU_Q6_K: type = JANAS_HX_Q6_K; break;
    default:
        snprintf(err, err_len, "unsupported Hexagon weight type");
        return NULL;
    }
    double start = seconds();
    uint16_t *converted = NULL;
    if (wtype == JANAS_NPU_I8) {
        converted = malloc((size_t)n * k * sizeof *converted);
        if (!converted) {
            snprintf(err, err_len, "no memory for Hexagon weight conversion");
            return NULL;
        }
        const int8_t *q = w;
        for (int row = 0; row < n; row++) {
            float s = half_float(scale[row]);
            for (int col = 0; col < k; col++) {
                float value = s * q[(size_t)row * k + col];
                if (!isfinite(value) || fabsf(value) > 65504.0f) {
                    free(converted);
                    snprintf(err, err_len, "Hexagon FP16 conversion out of range");
                    return NULL;
                }
                converted[(size_t)row * k + col] = float_half(value);
            }
        }
        w = converted;
    }
    struct janas_npu_mm *p = calloc(1, sizeof *p);
    if (p)
        p->plan = janas_hx_plan_create(u->hx, type, w, (size_t)k, (size_t)n,
                                      (size_t)m);
    free(converted);
    if (!p || !p->plan) {
        snprintf(err, err_len, "%s", p ? janas_hx_error(u->hx) : "no memory");
        free(p);
        return NULL;
    }
    if (janas_hx_plan_run(p->plan, x, y)) {
        snprintf(err, err_len, "%s", janas_hx_error(u->hx));
        janas_hx_plan_destroy(p->plan);
        free(p);
        return NULL;
    }
    p->owner = u;
    p->x[0] = x;
    p->y[0] = y;
    p->next = u->products;
    u->products = p;
    u->made++;
    u->setup_s += seconds() - start;
    return p;
}

struct janas_npu_mm *janas_npu_mm_rebatch(const struct janas_npu_mm *source,
                                          int m, float *x, float *y,
                                          char *err, size_t err_len)
{
    if (!source || m <= 0 || !x || !y || source->active[0] || source->active[1]) {
        snprintf(err, err_len, "bad or in-flight Hexagon rebatch source");
        return NULL;
    }
    struct janas_npu *u = source->owner;
    double start = seconds();
    struct janas_npu_mm *p = calloc(1, sizeof *p);
    if (p)
        p->plan = janas_hx_plan_rebatch(source->plan, (size_t)m);
    if (!p || !p->plan) {
        snprintf(err, err_len, "%s", p ? janas_hx_error(u->hx) : "no memory");
        free(p);
        return NULL;
    }
    if (janas_hx_plan_run(p->plan, x, y)) {
        snprintf(err, err_len, "%s", janas_hx_error(u->hx));
        janas_hx_plan_destroy(p->plan);
        free(p);
        return NULL;
    }
    p->owner = u;
    p->x[0] = x;
    p->y[0] = y;
    p->next = u->products;
    u->products = p;
    u->made++;
    u->setup_s += seconds() - start;
    return p;
}

void janas_npu_mm_destroy(struct janas_npu_mm *p)
{
    if (!p)
        return;
    struct janas_npu_mm **link = &p->owner->products;
    while (*link && *link != p)
        link = &(*link)->next;
    if (*link)
        *link = p->next;
    janas_hx_plan_destroy(p->plan);
    free(p);
}

size_t janas_npu_mm_weight_bytes(const struct janas_npu_mm *p)
{
    struct janas_hx_stats stats = {0};
    if (p)
        janas_hx_plan_stats(p->plan, &stats);
    return stats.device_weight_bytes;
}

size_t janas_npu_weight_bytes(const struct janas_npu *u)
{
    return u ? janas_hx_weight_bytes(u->hx) : 0;
}

struct janas_npu_mm *janas_npu_graph_create(struct janas_npu *u,
                                             struct npu_ir *ir, int nargs,
                                             float *const *args, const int *rows,
                                             const int *cols, char *err,
                                             size_t err_len)
{
    (void)u, (void)nargs, (void)args, (void)rows, (void)cols;
    npu_ir_free(ir);
    snprintf(err, err_len, "Hexagon uses matrix products; this layer graph is unsupported");
    return NULL;
}

struct janas_npu_mm *janas_npu_ffn_create(struct janas_npu *u, int m, int d,
                                           int ff, int act, const int8_t *gq,
                                           const uint16_t *gs, const int8_t *uq,
                                           const uint16_t *us, const int8_t *dq,
                                           const uint16_t *ds, float *x,
                                           float *y, char *err, size_t err_len)
{
    (void)u, (void)m, (void)d, (void)ff, (void)act, (void)gq, (void)gs;
    (void)uq, (void)us, (void)dq, (void)ds, (void)x, (void)y;
    snprintf(err, err_len, "Hexagon fused FFN is unsupported; use matrix products");
    return NULL;
}

int janas_npu_mm_rebind(struct janas_npu_mm *p, int nargs, float *const *args)
{
    if (!p || nargs != 2 || !args || !args[0] || !args[1] ||
        p->active[0] || p->active[1])
        return -1;
    p->x[0] = args[0], p->y[0] = args[1];
    return 0;
}

int janas_npu_mm_bind_alt(struct janas_npu_mm *p, int nargs, float *const *args)
{
    if (!p || nargs != 2 || !args || !args[0] || !args[1] ||
        p->active[0] || p->active[1])
        return -1;
    p->x[1] = args[0], p->y[1] = args[1];
    return 0;
}

int janas_npu_mm_begin_v(struct janas_npu_mm *p, int variant)
{
    if (!p || variant < 0 || variant > 1 || p->active[variant] ||
        !p->x[variant] || !p->y[variant])
        return -1;
    /* Synchronous tensor submission first: preserving the caller's
       begin/finish ownership contract allows later queue overlap. */
    p->result[variant] = janas_hx_plan_run(p->plan, p->x[variant], p->y[variant]);
    if (p->result[variant])
        return -1;
    p->active[variant] = 1;
    return 0;
}

int janas_npu_mm_finish_v(struct janas_npu_mm *p, int variant)
{
    if (!p || variant < 0 || variant > 1 || !p->active[variant])
        return -1;
    p->active[variant] = 0;
    return p->result[variant];
}

int janas_npu_mm_begin(struct janas_npu_mm *p)
{
    return janas_npu_mm_begin_v(p, 0);
}

int janas_npu_mm_finish(struct janas_npu_mm *p)
{
    return janas_npu_mm_finish_v(p, 0);
}

int janas_npu_set_cache(struct janas_npu *u, const char *dir)
{
    /* There is no device graph compiler cache on this custom-kernel path;
       owned packed matrices stay in memory for their plan's lifetime. */
    (void)dir;
    return u ? 0 : -1;
}

int janas_npu_keep_awake(struct janas_npu *u, int on)
{
    (void)u;
    return on ? -1 : 0;
}

void janas_npu_stats(const struct janas_npu *u, int *products, int *cached,
                     double *compile_seconds)
{
    *products = u ? u->made : 0;
    *cached = 0;
    *compile_seconds = u ? u->setup_s : 0;
}

#endif

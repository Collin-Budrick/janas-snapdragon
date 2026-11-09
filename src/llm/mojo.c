/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "mojo.h"
#include "quant.h"
#include "quant_panel.h"
#include "common/sysinfo.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* The Mojo module decodes these immutable on-disk/in-memory layouts. A C
   layout change must fail the build until the leaf and ABI version agree. */
_Static_assert(sizeof(struct janas_block_q4k) == 144, "Mojo Q4_K ABI");
_Static_assert(offsetof(struct janas_block_q4k, scales) == 4 &&
                   offsetof(struct janas_block_q4k, qs) == 16, "Mojo Q4 offsets");
_Static_assert(sizeof(struct janas_block_q6k) == 210, "Mojo Q6_K ABI");
_Static_assert(offsetof(struct janas_block_q6k, scales) == 192 &&
                   offsetof(struct janas_block_q6k, d) == 208, "Mojo Q6 offsets");
_Static_assert(sizeof(struct janas_block_q6kp) == 82, "Mojo Q6_K_P ABI");
_Static_assert(offsetof(struct janas_block_q6kp, q) == 0 &&
                   offsetof(struct janas_block_q6kp, scales) == 64 &&
                   offsetof(struct janas_block_q6kp, d) == 80,
               "Mojo Q6_K_P offsets");
_Static_assert(sizeof(struct janas_block_q8k) == 628, "Mojo Q8_K ABI");
_Static_assert(offsetof(struct janas_block_q8k, d) == 0 &&
                   offsetof(struct janas_block_q8k, qs) == 4 &&
                   offsetof(struct janas_block_q8k, bsums) == 260,
               "Mojo Q8_K offsets");
_Static_assert(JANAS_Q4_K == 12 && JANAS_Q6_K == 14 && JANAS_Q6_K_P == 214,
               "Mojo quantization IDs");

static pthread_once_t module_once = PTHREAD_ONCE_INIT;
static atomic_int requested;
static int startup_request;
static janas_mojo_panel_fn_v1 panel_fn;
static void *module_handle;
static const char *module_status = "not loaded";
static atomic_uint_fast64_t calls, completed_rows, row_vectors, refusals;

__attribute__((constructor)) static void mojo_settings(void)
{
    const char *s = getenv("JANAS_MOJO");
    startup_request = s && !strcmp(s, "1");
    atomic_store_explicit(&requested, startup_request, memory_order_relaxed);
}

static void module_load(void)
{
#if defined(__aarch64__) && UINTPTR_MAX == UINT64_MAX
    if (!janas_cpu_arm_dotprod() || !janas_cpu_arm_i8mm()) {
        module_status = "CPU lacks ARM dotprod/I8MM";
        return;
    }
    const char *path = getenv("JANAS_MOJO_LIBRARY");
    if (!path || !*path) {
        module_status = "JANAS_MOJO_LIBRARY is not configured";
        return;
    }
    /* Explicit paths keep module selection reproducible in both hosts. */
#if defined(_WIN32)
    int absolute = ((path[0] >= 'A' && path[0] <= 'Z') ||
                    (path[0] >= 'a' && path[0] <= 'z')) && path[1] == ':' &&
                   (path[2] == '/' || path[2] == '\\');
    absolute |= path[0] == '\\' && path[1] == '\\';
#else
    int absolute = path[0] == '/';
#endif
    if (!absolute) {
        module_status = "JANAS_MOJO_LIBRARY must be an absolute path";
        return;
    }
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        module_status = "Mojo module could not be loaded";
        return;
    }
    janas_mojo_abi_fn_v1 abi = NULL;
    janas_mojo_panel_fn_v1 fn = NULL;
    void *symbol = dlsym(h, "janas_mojo_abi_v1");
    _Static_assert(sizeof(abi) == sizeof(symbol), "Mojo function address");
    memcpy(&abi, &symbol, sizeof abi);
    symbol = dlsym(h, "janas_mojo_panel_rows_v1");
    memcpy(&fn, &symbol, sizeof fn);
    if (!abi || !fn || abi() != JANAS_MOJO_ABI_V1) {
        module_status = "Mojo module ABI mismatch";
        dlclose(h);
        return;
    }
    module_handle = h;
    panel_fn = fn;
    module_status = "ready: ARM64 Mojo quantized panels ABI v1";
#else
    module_status = "Mojo panels require ARM64 with 64-bit addresses";
#endif
}

int janas_mojo_available(void)
{
    pthread_once(&module_once, module_load);
    return module_handle != NULL && panel_fn != NULL;
}

int janas_mojo_enabled(void)
{
    return atomic_load_explicit(&requested, memory_order_relaxed) &&
           janas_mojo_available();
}

int janas_mojo_set_enabled(int enabled)
{
    return atomic_exchange_explicit(&requested,
                                    enabled < 0 ? startup_request : !!enabled,
                                    memory_order_relaxed);
}

const char *janas_mojo_status(void)
{
    (void)janas_mojo_available();
    return module_status;
}

static int span_fits(size_t count, size_t stride, size_t last, size_t unit)
{
    size_t limit = (size_t)PTRDIFF_MAX / unit;
    return last <= limit &&
           (count <= 1 || stride <= (limit - last) / (count - 1));
}

size_t janas_mojo_panel_rows(
    int type, const void *w, size_t row_bytes, const uint8_t *p1,
    const uint8_t *p0, size_t pb, const struct janas_q8k_panel *p,
    size_t rows, float *y, size_t ys)
{
    if (!janas_mojo_enabled() || janas_kernels_ref())
        return 0;
    size_t r2 = rows / 2 * 2;
    if (!w || !y || !p || !p->x || !p->nb || p->nv < 2 || p->nv > 9 ||
        p->pairs != p->nv / 2 || !p->qs || !r2 || ys < r2 ||
        p->x_stride < p->nb ||
        (type != JANAS_Q4_K && type != JANAS_Q6_K && type != JANAS_Q6_K_P))
        return 0;
    size_t block = type == JANAS_Q4_K ? sizeof(struct janas_block_q4k) :
                   type == JANAS_Q6_K ? sizeof(struct janas_block_q6k) :
                                       sizeof(struct janas_block_q6kp);
    if (p->nb > (size_t)PTRDIFF_MAX / block || row_bytes < p->nb * block ||
        (row_bytes & 1) || !span_fits(r2, row_bytes, p->nb * block, 1) ||
        !span_fits(p->nv, p->x_stride, p->nb, sizeof(*p->x)) ||
        !span_fits(p->nv, ys, r2, sizeof(*y)) ||
        p->nb > (size_t)PTRDIFF_MAX / 512 / p->pairs ||
        ((uintptr_t)w & 1) || ((uintptr_t)p->x & 3) || ((uintptr_t)y & 3))
        return 0;
    if (type == JANAS_Q6_K_P &&
        ((p0 && !p1) ||
         ((p1 || p0) &&
          (p->nb > (size_t)PTRDIFF_MAX / JANAS_Q6KP_PLANE ||
           pb < p->nb * JANAS_Q6KP_PLANE ||
           !span_fits(r2, pb, p->nb * JANAS_Q6KP_PLANE, 1)))))
        return 0;
    int level = type == JANAS_Q6_K_P ? (p0 ? 3 : p1 ? 2 : 1) : 0;
    int32_t ok = panel_fn(
        (int32_t)type, (int32_t)level, (uint64_t)(uintptr_t)w, row_bytes,
        (uint64_t)(uintptr_t)p1, (uint64_t)(uintptr_t)p0, pb,
        (uint64_t)(uintptr_t)p->x, p->x_stride, (uint64_t)(uintptr_t)p->qs,
        p->nb, p->nv, r2, (uint64_t)(uintptr_t)y, ys);
    if (ok != 1) {
        atomic_fetch_add_explicit(&refusals, 1, memory_order_relaxed);
        return 0;
    }
    atomic_fetch_add_explicit(&calls, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&completed_rows, r2, memory_order_relaxed);
    atomic_fetch_add_explicit(&row_vectors, r2 * p->nv, memory_order_relaxed);
    return r2;
}

void janas_mojo_get_stats(struct janas_mojo_stats *out)
{
    if (out)
        *out = (struct janas_mojo_stats){
            atomic_load_explicit(&calls, memory_order_relaxed),
            atomic_load_explicit(&completed_rows, memory_order_relaxed),
            atomic_load_explicit(&row_vectors, memory_order_relaxed),
            atomic_load_explicit(&refusals, memory_order_relaxed)};
}

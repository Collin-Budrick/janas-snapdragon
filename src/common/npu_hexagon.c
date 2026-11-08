/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Optional tensor-operation adapter to GGML's MIT-licensed public C API.
   The separately supplied Hexagon runtime owns its DSP protocol and packing. */
#include "npu_hexagon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void copy_error(char *out, size_t size, const char *message)
{
    if (out && size)
        snprintf(out, size, "%s", message ? message : "");
}

#if defined(JANAS_HEXAGON) && JANAS_HEXAGON && defined(_WIN32) && \
    (defined(__aarch64__) || defined(_M_ARM64))

#include <limits.h>
#include <malloc.h>
#include <stdarg.h>
#include <wchar.h>
#include <windows.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

/* Use the matching runtime's headers to retain its exact C signatures, but
   resolve every call dynamically: enabling the adapter adds no import library. */
#define HX_FUNCTIONS(X)                                                        \
    X(ggml_init) X(ggml_free) X(ggml_tensor_overhead)                            \
    X(ggml_graph_overhead_custom) X(ggml_new_tensor_2d)                          \
    X(ggml_new_graph_custom) X(ggml_build_forward_expand) X(ggml_mul_mat)         \
    X(ggml_nbytes) X(ggml_blck_size) X(ggml_type_size) X(ggml_type_name)          \
    X(ggml_set_name) X(ggml_backend_reg_dev_count) X(ggml_backend_reg_dev_get)    \
    X(ggml_backend_dev_init) X(ggml_backend_dev_name)                            \
    X(ggml_backend_dev_supports_op) X(ggml_backend_free)                         \
    X(ggml_backend_get_max_size) X(ggml_backend_alloc_ctx_tensors)               \
    X(ggml_backend_buffer_set_usage) X(ggml_backend_buffer_get_size)             \
    X(ggml_backend_buffer_free) X(ggml_backend_tensor_set)                       \
    X(ggml_backend_tensor_get) X(ggml_backend_graph_compute)                     \
    X(ggml_backend_synchronize)

struct hx_api {
#define HX_DECLARE(name) __typeof__(&name) name;
    HX_FUNCTIONS(HX_DECLARE)
    HX_DECLARE(ggml_backend_hexagon_reg)
#undef HX_DECLARE
};

/* GGML installs a process-wide C++ terminate handler in its DLL constructor
   and does not restore it on unload. Keep these modules alive for the
   process, including when device initialization fails or is retried. */
static struct {
    SRWLOCK lock;
    HMODULE base, hexagon;
    wchar_t *base_path, *hexagon_path;
} runtime = {.lock = SRWLOCK_INIT};

struct janas_hx_context {
    HMODULE base, hexagon;
    struct hx_api api;
    ggml_backend_dev_t device;
    ggml_backend_t backend;
    struct janas_hx_plan *plans;
    size_t weight_bytes;
    double ticks_per_ms;
    char name[128];
    char error[512];
};

struct hx_weights {
    struct janas_hx_context *ctx;
    enum ggml_type type;
    size_t refs;
    size_t k, n, bytes, device_bytes;
    int charged;
    struct ggml_context *metadata_ctx;
    void *metadata;
    ggml_backend_buffer_t buffer;
    struct ggml_tensor *tensor;
};

struct janas_hx_plan {
    struct janas_hx_context *ctx;
    struct janas_hx_plan *prev, *next;
    int linked;
    struct hx_weights *weights;
    struct ggml_context *compute_ctx;
    void *compute_meta;
    ggml_backend_buffer_t compute_buffer;
    struct ggml_tensor *input, *output;
    struct ggml_cgraph *graph;
    size_t input_bytes, output_bytes;
    struct janas_hx_stats stats;
};

static void errorf(struct janas_hx_context *ctx, const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(ctx->error, sizeof(ctx->error), format, args);
    va_end(args);
}

static double now_ms(const struct janas_hx_context *ctx)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / ctx->ticks_per_ms;
}

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (b && a > SIZE_MAX / b)
        return 0;
    *out = a * b;
    return 1;
}

static wchar_t *environment(const wchar_t *name)
{
    DWORD size = GetEnvironmentVariableW(name, NULL, 0);
    if (!size || size > 32768)
        return NULL;
    wchar_t *value = malloc((size_t)size * sizeof(*value));
    if (!value)
        return NULL;
    DWORD got = GetEnvironmentVariableW(name, value, size);
    if (!got || got >= size) {
        free(value);
        return NULL;
    }
    return value;
}

static wchar_t *join_path(const wchar_t *root, const wchar_t *leaf)
{
    size_t n = wcslen(root), m = wcslen(leaf);
    if (n + m + 2 > 32768)
        return NULL;
    wchar_t *path = malloc((n + m + 2) * sizeof(*path));
    if (!path)
        return NULL;
    memcpy(path, root, n * sizeof(*path));
    if (n && root[n - 1] != L'\\' && root[n - 1] != L'/')
        path[n++] = L'\\';
    memcpy(path + n, leaf, (m + 1) * sizeof(*path));
    return path;
}

static int file_exists(const wchar_t *path)
{
    if (!path)
        return 0;
    DWORD attributes = GetFileAttributesW(path);
    return attributes != INVALID_FILE_ATTRIBUTES &&
           !(attributes & FILE_ATTRIBUTE_DIRECTORY);
}

static wchar_t *absolute_path(const wchar_t *path)
{
    DWORD size = GetFullPathNameW(path, 0, NULL, NULL);
    if (!size || size > 32768)
        return NULL;
    wchar_t *out = malloc((size_t)size * sizeof(*out));
    if (!out)
        return NULL;
    DWORD got = GetFullPathNameW(path, size, out, NULL);
    if (!got || got >= size) {
        free(out);
        return NULL;
    }
    for (wchar_t *p = out; *p; p++)
        if (*p == L'/')
            *p = L'\\';
    return out;
}

static int load_runtime(struct janas_hx_context *ctx, const wchar_t *base,
                         const wchar_t *hexagon)
{
    wchar_t *bp = absolute_path(base), *hp = absolute_path(hexagon);
    if (!bp || !hp) {
        free(bp), free(hp);
        errorf(ctx, "cannot resolve absolute Hexagon runtime paths");
        return 0;
    }
    int ok = 0;
    AcquireSRWLockExclusive(&runtime.lock);
    if (runtime.base_path && (_wcsicmp(runtime.base_path, bp) ||
                              _wcsicmp(runtime.hexagon_path, hp))) {
        errorf(ctx, "Hexagon runtime is fixed for this process; restart to change JANAS_HEXAGON_ROOT");
        goto done;
    }
    if (!runtime.base_path) {
        runtime.base_path = bp, runtime.hexagon_path = hp;
        bp = hp = NULL;
    }
    DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    if (!runtime.base)
        runtime.base = LoadLibraryExW(runtime.base_path, NULL, flags);
    if (!runtime.base) {
        errorf(ctx, "cannot load ggml-base.dll (Windows error %lu)",
               (unsigned long)GetLastError());
        goto done;
    }
    HMODULE pinned;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, runtime.base_path,
                            &pinned)) {
        /* The registry still retains its LoadLibrary reference on failure:
           never leave GGML's terminate handler pointing to unloaded code. */
        errorf(ctx, "cannot pin ggml-base.dll for process lifetime (Windows error %lu)",
               (unsigned long)GetLastError());
        goto done;
    }
    if (!runtime.hexagon)
        runtime.hexagon = LoadLibraryExW(runtime.hexagon_path, NULL, flags);
    if (!runtime.hexagon) {
        errorf(ctx, "cannot load ggml-hexagon.dll (Windows error %lu)",
               (unsigned long)GetLastError());
        goto done;
    }
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN, runtime.hexagon_path,
                            &pinned)) {
        errorf(ctx, "cannot pin ggml-hexagon.dll for process lifetime (Windows error %lu)",
               (unsigned long)GetLastError());
        goto done;
    }
    ctx->base = runtime.base;
    ctx->hexagon = runtime.hexagon;
    ok = 1;
done:
    if (!runtime.base) {
        free(runtime.base_path), free(runtime.hexagon_path);
        runtime.base_path = runtime.hexagon_path = NULL;
    }
    ReleaseSRWLockExclusive(&runtime.lock);
    free(bp), free(hp);
    return ok;
}

static int load_api(struct janas_hx_context *ctx)
{
#define HX_LOAD_FROM(module, name)                                             \
    do {                                                                      \
        FARPROC symbol = GetProcAddress(module, #name);                         \
        if (!symbol) {                                                        \
            errorf(ctx, "Hexagon runtime is missing %s", #name);               \
            return 0;                                                         \
        }                                                                     \
        _Static_assert(sizeof(ctx->api.name) == sizeof(symbol),                \
                       "Windows function pointer size");                     \
        memcpy(&ctx->api.name, &symbol, sizeof(symbol));                       \
    } while (0)
#define HX_LOAD_BASE(name) HX_LOAD_FROM(ctx->base, name);
    HX_FUNCTIONS(HX_LOAD_BASE)
    HX_LOAD_FROM(ctx->hexagon, ggml_backend_hexagon_reg);
#undef HX_LOAD_BASE
#undef HX_LOAD_FROM
    return 1;
}

struct janas_hx_context *janas_hx_create(char *error, size_t error_size)
{
    copy_error(error, error_size, "");
    struct janas_hx_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        copy_error(error, error_size, "cannot allocate Hexagon context");
        return NULL;
    }
    wchar_t *root = environment(L"JANAS_HEXAGON_ROOT");
    wchar_t *adsp = environment(L"ADSP_LIBRARY_PATH");
    wchar_t *base = NULL, *hexagon = NULL;
    if (!root) {
        errorf(ctx, "JANAS_HEXAGON_ROOT must name the matching Hexagon runtime");
        goto fail;
    }
    if (!adsp) {
        errorf(ctx, "ADSP_LIBRARY_PATH must include the runtime's htp directory");
        goto fail;
    }
    base = join_path(root, L"ggml-base.dll");
    if (!file_exists(base)) {
        free(base);
        base = join_path(root, L"bin\\ggml-base.dll");
        hexagon = join_path(root, L"bin\\ggml-hexagon.dll");
    } else {
        hexagon = join_path(root, L"ggml-hexagon.dll");
    }
    if (!file_exists(base) || !file_exists(hexagon)) {
        errorf(ctx, "matching ggml-base.dll and ggml-hexagon.dll not found in JANAS_HEXAGON_ROOT");
        goto fail;
    }
    if (!load_runtime(ctx, base, hexagon))
        goto fail;
    if (!load_api(ctx))
        goto fail;
    ggml_backend_reg_t registry = ctx->api.ggml_backend_hexagon_reg();
    if (!registry || !ctx->api.ggml_backend_reg_dev_count(registry)) {
        errorf(ctx, "Hexagon runtime found no NPU device");
        goto fail;
    }
    ctx->device = ctx->api.ggml_backend_reg_dev_get(registry, 0);
    if (!ctx->device || !(ctx->backend = ctx->api.ggml_backend_dev_init(ctx->device, NULL))) {
        errorf(ctx, "cannot initialize the Hexagon NPU device");
        goto fail;
    }
    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
        errorf(ctx, "Windows performance counter is unavailable");
        goto fail;
    }
    ctx->ticks_per_ms = (double)frequency.QuadPart / 1000.0;
    const char *name = ctx->api.ggml_backend_dev_name(ctx->device);
    snprintf(ctx->name, sizeof(ctx->name), "%s", name ? name : "Hexagon");
    free(root);
    free(adsp);
    free(base);
    free(hexagon);
    return ctx;

fail:
    copy_error(error, error_size, ctx->error);
    free(root);
    free(adsp);
    free(base);
    free(hexagon);
    janas_hx_destroy(ctx);
    return NULL;
}

void janas_hx_destroy(struct janas_hx_context *ctx)
{
    if (!ctx)
        return;
    while (ctx->plans)
        janas_hx_plan_destroy(ctx->plans);
    if (ctx->backend)
        ctx->api.ggml_backend_free(ctx->backend);
    /* The process registry retains and pins the runtime DLLs. */
    free(ctx);
}

const char *janas_hx_name(const struct janas_hx_context *ctx)
{
    return ctx ? ctx->name : "Hexagon unavailable";
}

const char *janas_hx_error(const struct janas_hx_context *ctx)
{
    return ctx ? ctx->error : "Hexagon context is unavailable";
}

size_t janas_hx_weight_bytes(const struct janas_hx_context *ctx)
{
    return ctx ? ctx->weight_bytes : 0;
}

static int weight_type(enum janas_hx_type type, enum ggml_type *ggml,
                       const char **name, size_t *block, size_t *bytes)
{
    switch (type) {
    case JANAS_HX_F32:    *ggml = GGML_TYPE_F32;    *name = "f32";    *block = 1;   *bytes = 4; break;
    case JANAS_HX_F16:    *ggml = GGML_TYPE_F16;    *name = "f16";    *block = 1;   *bytes = 2; break;
    case JANAS_HX_Q4_0:   *ggml = GGML_TYPE_Q4_0;   *name = "q4_0";   *block = 32;  *bytes = 18; break;
    case JANAS_HX_Q4_1:   *ggml = GGML_TYPE_Q4_1;   *name = "q4_1";   *block = 32;  *bytes = 20; break;
    case JANAS_HX_Q8_0:   *ggml = GGML_TYPE_Q8_0;   *name = "q8_0";   *block = 32;  *bytes = 34; break;
    case JANAS_HX_IQ4_NL: *ggml = GGML_TYPE_IQ4_NL; *name = "iq4_nl"; *block = 32;  *bytes = 18; break;
    case JANAS_HX_MXFP4:  *ggml = GGML_TYPE_MXFP4;  *name = "mxfp4";  *block = 32;  *bytes = 17; break;
    case JANAS_HX_Q4_K:   *ggml = GGML_TYPE_Q4_K;   *name = "q4_K";   *block = 256; *bytes = 144; break;
    case JANAS_HX_Q5_K:   *ggml = GGML_TYPE_Q5_K;   *name = "q5_K";   *block = 256; *bytes = 176; break;
    case JANAS_HX_Q6_K:   *ggml = GGML_TYPE_Q6_K;   *name = "q6_K";   *block = 256; *bytes = 210; break;
    default: return 0;
    }
    return 1;
}

static void weights_release(struct hx_weights *weights)
{
    if (!weights || --weights->refs)
        return;
    struct janas_hx_context *ctx = weights->ctx;
    if (weights->buffer)
        ctx->api.ggml_backend_buffer_free(weights->buffer);
    if (weights->charged)
        ctx->weight_bytes -= weights->device_bytes;
    if (weights->metadata_ctx)
        ctx->api.ggml_free(weights->metadata_ctx);
    _aligned_free(weights->metadata);
    free(weights);
}

/* Consumes one weight-owner reference, including on failure. Only X, Y and
   their graph belong to a batch plan; the weight tensor outlives every plan. */
static struct janas_hx_plan *plan_batch(struct hx_weights *weights, size_t m)
{
    struct janas_hx_context *ctx = weights->ctx;
    size_t k = weights->k, n = weights->n, count, input_bytes, output_bytes;
    if (!m || m > INT32_MAX || !mul_size(k, m, &count) ||
        !mul_size(count, sizeof(float), &input_bytes) ||
        !mul_size(n, m, &count) ||
        !mul_size(count, sizeof(float), &output_bytes) ||
        input_bytes > SIZE_MAX - output_bytes) {
        errorf(ctx, "invalid or overflowing Hexagon batch shape");
        weights_release(weights);
        return NULL;
    }
    size_t maximum = ctx->api.ggml_backend_get_max_size(ctx->backend);
    if (maximum && (weights->bytes > maximum || input_bytes > maximum ||
                    output_bytes > maximum || input_bytes > maximum - output_bytes)) {
        errorf(ctx, "Hexagon matrix exceeds a device buffer limit");
        weights_release(weights);
        return NULL;
    }
    struct janas_hx_plan *plan = calloc(1, sizeof(*plan));
    if (!plan) {
        errorf(ctx, "cannot allocate Hexagon matrix plan");
        weights_release(weights);
        return NULL;
    }
    plan->ctx = ctx;
    plan->weights = weights;
    plan->input_bytes = input_bytes;
    plan->output_bytes = output_bytes;
    plan->stats.weight_bytes = weights->bytes;
    size_t compute_meta_size = 8 * ctx->api.ggml_tensor_overhead() +
        ctx->api.ggml_graph_overhead_custom(16, false) + 1024;
    plan->compute_meta = _aligned_malloc(compute_meta_size, 64);
    if (!plan->compute_meta) {
        errorf(ctx, "cannot allocate Hexagon graph metadata");
        goto fail;
    }
    struct ggml_init_params cp = {compute_meta_size, plan->compute_meta, true};
    plan->compute_ctx = ctx->api.ggml_init(cp);
    if (!plan->compute_ctx) {
        errorf(ctx, "cannot initialize Hexagon compute context");
        goto fail;
    }
    plan->input = ctx->api.ggml_new_tensor_2d(plan->compute_ctx, GGML_TYPE_F32,
                                             (int64_t)k, (int64_t)m);
    if (!plan->input) {
        errorf(ctx, "cannot create Hexagon input tensor");
        goto fail;
    }
    plan->output = ctx->api.ggml_mul_mat(plan->compute_ctx, weights->tensor,
                                         plan->input);
    if (!plan->output || !ctx->api.ggml_backend_dev_supports_op(ctx->device,
                                                               plan->output)) {
        errorf(ctx, "Hexagon does not support %s matrix K=%zu N=%zu M=%zu",
               ctx->api.ggml_type_name(weights->type), k, n, m);
        goto fail;
    }
    ctx->api.ggml_set_name(plan->input, "janas_hx_input");
    ctx->api.ggml_set_name(plan->output, "janas_hx_output");
    plan->graph = ctx->api.ggml_new_graph_custom(plan->compute_ctx, 16, false);
    if (!plan->graph) {
        errorf(ctx, "cannot create Hexagon compute graph");
        goto fail;
    }
    ctx->api.ggml_build_forward_expand(plan->graph, plan->output);
    if (!weights->buffer) {
        weights->buffer = ctx->api.ggml_backend_alloc_ctx_tensors(
            weights->metadata_ctx, ctx->backend);
        if (!weights->buffer) {
            errorf(ctx, "cannot allocate Hexagon weight buffer");
            goto fail;
        }
        ctx->api.ggml_backend_buffer_set_usage(weights->buffer,
                                              GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        weights->device_bytes = ctx->api.ggml_backend_buffer_get_size(weights->buffer);
        if (weights->device_bytes > SIZE_MAX - ctx->weight_bytes) {
            errorf(ctx, "Hexagon total weight-buffer size overflows");
            goto fail;
        }
        ctx->weight_bytes += weights->device_bytes;
        weights->charged = 1;
    }
    plan->compute_buffer = ctx->api.ggml_backend_alloc_ctx_tensors(
        plan->compute_ctx, ctx->backend);
    if (!plan->compute_buffer) {
        errorf(ctx, "cannot allocate Hexagon compute buffer");
        goto fail;
    }
    if (ctx->api.ggml_nbytes(weights->tensor) != weights->bytes ||
        ctx->api.ggml_nbytes(plan->input) != input_bytes ||
        ctx->api.ggml_nbytes(plan->output) != output_bytes) {
        errorf(ctx, "Hexagon tensor byte sizes do not match the matrix plan");
        goto fail;
    }
    plan->stats.device_weight_bytes = weights->device_bytes;
    plan->stats.device_compute_bytes = ctx->api.ggml_backend_buffer_get_size(
        plan->compute_buffer);
    plan->next = ctx->plans;
    if (ctx->plans)
        ctx->plans->prev = plan;
    ctx->plans = plan;
    plan->linked = 1;
    return plan;

fail:
    janas_hx_plan_destroy(plan);
    return NULL;
}

struct janas_hx_plan *janas_hx_plan_create(struct janas_hx_context *ctx,
                                         enum janas_hx_type type,
                                         const void *source, size_t k,
                                         size_t n, size_t m)
{
    if (!ctx)
        return NULL;
    ctx->error[0] = 0;
    enum ggml_type ggml;
    const char *name;
    size_t block, bytes, row_bytes, weight_bytes;
    if (!source || !k || !n || !m || k > INT32_MAX || n > INT32_MAX ||
        m > INT32_MAX || !weight_type(type, &ggml, &name, &block, &bytes)) {
        errorf(ctx, "invalid Hexagon matrix dimensions, weights or type");
        return NULL;
    }
    if (k % block || !mul_size(k / block, bytes, &row_bytes) ||
        !mul_size(row_bytes, n, &weight_bytes)) {
        errorf(ctx, "Hexagon weight size overflows or violates quantization block size");
        return NULL;
    }
    const char *actual_name = ctx->api.ggml_type_name(ggml);
    if (!actual_name || _stricmp(actual_name, name) ||
        ctx->api.ggml_blck_size(ggml) != (int64_t)block ||
        ctx->api.ggml_type_size(ggml) != bytes) {
        errorf(ctx, "Hexagon runtime's %s block ABI does not match", name);
        return NULL;
    }
    struct hx_weights *weights = calloc(1, sizeof(*weights));
    if (!weights) {
        errorf(ctx, "cannot allocate Hexagon weight owner");
        return NULL;
    }
    weights->ctx = ctx;
    weights->type = ggml;
    weights->refs = 1;
    weights->k = k;
    weights->n = n;
    weights->bytes = weight_bytes;
    size_t metadata_size = 2 * ctx->api.ggml_tensor_overhead() + 1024;
    weights->metadata = _aligned_malloc(metadata_size, 64);
    if (!weights->metadata) {
        errorf(ctx, "cannot allocate Hexagon weight metadata");
        weights_release(weights);
        return NULL;
    }
    struct ggml_init_params wp = {metadata_size, weights->metadata, true};
    weights->metadata_ctx = ctx->api.ggml_init(wp);
    if (!weights->metadata_ctx ||
        !(weights->tensor = ctx->api.ggml_new_tensor_2d(weights->metadata_ctx,
                                                       ggml, (int64_t)k,
                                                       (int64_t)n))) {
        errorf(ctx, "cannot initialize Hexagon weight tensor");
        weights_release(weights);
        return NULL;
    }
    ctx->api.ggml_set_name(weights->tensor, "janas_hx_weights");
    struct janas_hx_plan *plan = plan_batch(weights, m);
    if (!plan)
        return NULL; /* plan_batch consumed the owner reference */
    double start = now_ms(ctx);
    ctx->api.ggml_backend_tensor_set(weights->tensor, source, 0, weight_bytes);
    ctx->api.ggml_backend_synchronize(ctx->backend);
    plan->stats.upload_ms = now_ms(ctx) - start;
    return plan;
}

struct janas_hx_plan *janas_hx_plan_rebatch(const struct janas_hx_plan *source,
                                           size_t m)
{
    if (!source)
        return NULL;
    struct janas_hx_context *ctx = source->ctx;
    ctx->error[0] = 0;
    if (!m || m > INT32_MAX || source->weights->refs == SIZE_MAX) {
        errorf(ctx, "invalid Hexagon batch shape or weight reference count");
        return NULL;
    }
    source->weights->refs++;
    return plan_batch(source->weights, m);
}

int janas_hx_plan_run(struct janas_hx_plan *plan, const float *input,
                      float *output)
{
    if (!plan)
        return -1;
    struct janas_hx_context *ctx = plan->ctx;
    ctx->error[0] = 0;
    if (!input || !output) {
        errorf(ctx, "Hexagon run requires input and output arrays");
        return -1;
    }
    plan->stats.input_ms = plan->stats.execute_ms = plan->stats.output_ms = 0;
    double start = now_ms(ctx);
    ctx->api.ggml_backend_tensor_set(plan->input, input, 0, plan->input_bytes);
    plan->stats.input_ms = now_ms(ctx) - start;
    start = now_ms(ctx);
    enum ggml_status status = ctx->api.ggml_backend_graph_compute(ctx->backend,
                                                                 plan->graph);
    ctx->api.ggml_backend_synchronize(ctx->backend);
    plan->stats.execute_ms = now_ms(ctx) - start;
    if (status != GGML_STATUS_SUCCESS) {
        errorf(ctx, "Hexagon graph execution failed (status %d)", (int)status);
        return -1;
    }
    start = now_ms(ctx);
    ctx->api.ggml_backend_tensor_get(plan->output, output, 0, plan->output_bytes);
    plan->stats.output_ms = now_ms(ctx) - start;
    plan->stats.runs++;
    return 0;
}

void janas_hx_plan_destroy(struct janas_hx_plan *plan)
{
    if (!plan)
        return;
    struct janas_hx_context *ctx = plan->ctx;
    ctx->api.ggml_backend_synchronize(ctx->backend);
    if (plan->compute_buffer)
        ctx->api.ggml_backend_buffer_free(plan->compute_buffer);
    if (plan->compute_ctx)
        ctx->api.ggml_free(plan->compute_ctx);
    _aligned_free(plan->compute_meta);
    weights_release(plan->weights);
    if (plan->linked) {
        if (plan->prev)
            plan->prev->next = plan->next;
        else
            ctx->plans = plan->next;
        if (plan->next)
            plan->next->prev = plan->prev;
    }
    free(plan);
}

void janas_hx_plan_stats(const struct janas_hx_plan *plan,
                         struct janas_hx_stats *stats)
{
    if (stats) {
        if (plan)
            *stats = plan->stats;
        else
            memset(stats, 0, sizeof(*stats));
    }
}

#else

static const char unavailable[] =
    "Hexagon backend requires native Windows ARM64 and JANAS_HEXAGON";

struct janas_hx_context *janas_hx_create(char *error, size_t error_size)
{
    copy_error(error, error_size, unavailable);
    return NULL;
}

void janas_hx_destroy(struct janas_hx_context *ctx) { (void)ctx; }
const char *janas_hx_name(const struct janas_hx_context *ctx)
{
    (void)ctx;
    return "Hexagon unavailable";
}
const char *janas_hx_error(const struct janas_hx_context *ctx)
{
    (void)ctx;
    return unavailable;
}
size_t janas_hx_weight_bytes(const struct janas_hx_context *ctx)
{
    (void)ctx;
    return 0;
}
struct janas_hx_plan *janas_hx_plan_create(struct janas_hx_context *ctx,
                                         enum janas_hx_type type,
                                         const void *weights, size_t k,
                                         size_t n, size_t m)
{
    (void)ctx, (void)type, (void)weights, (void)k, (void)n, (void)m;
    return NULL;
}
struct janas_hx_plan *janas_hx_plan_rebatch(const struct janas_hx_plan *source,
                                           size_t m)
{
    (void)source, (void)m;
    return NULL;
}
int janas_hx_plan_run(struct janas_hx_plan *plan, const float *input, float *output)
{
    (void)plan, (void)input, (void)output;
    return -1;
}
void janas_hx_plan_destroy(struct janas_hx_plan *plan) { (void)plan; }
void janas_hx_plan_stats(const struct janas_hx_plan *plan,
                         struct janas_hx_stats *stats)
{
    (void)plan;
    if (stats)
        memset(stats, 0, sizeof(*stats));
}

#endif

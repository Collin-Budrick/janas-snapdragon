/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_NPU_HEXAGON_H
#define JANAS_NPU_HEXAGON_H

#include <stddef.h>
#include <stdint.h>

struct janas_hx_context;
struct janas_hx_plan;

/* These identifiers are deliberately independent of ggml_type. Packed
   formats use the ordinary GGUF block layout, before backend repacking. */
enum janas_hx_type {
    JANAS_HX_F32,
    JANAS_HX_F16,
    JANAS_HX_Q4_0,
    JANAS_HX_Q4_1,
    JANAS_HX_Q8_0,
    JANAS_HX_IQ4_NL,
    JANAS_HX_MXFP4,
    JANAS_HX_Q4_K,
    JANAS_HX_Q5_K,
    JANAS_HX_Q6_K,
};

struct janas_hx_stats {
    size_t weight_bytes;
    size_t device_weight_bytes;
    size_t device_compute_bytes;
    uint64_t runs;
    double upload_ms;
    double input_ms;
    double execute_ms;
    double output_ms;
};

/* Optional Windows ARM64 backend. JANAS_HEXAGON_ROOT names the runtime root
   containing ggml-base.dll and ggml-hexagon.dll (or a bin/ directory with
   those libraries). ADSP_LIBRARY_PATH must include the corresponding htp/
   directory before creating a context. This module changes no persistent or
   process-global environment configuration. Its runtime DLLs stay loaded
   for the process (GGML installs a process-wide terminate handler); restart
   the process before changing JANAS_HEXAGON_ROOT to a different runtime.
   A context and its plans belong to one calling thread. Destroy all plans
   before destroying their context. Failure to create a context is ordinary
   accelerator unavailability and permits the caller's CPU fallback. */
struct janas_hx_context *janas_hx_create(char *error, size_t error_size);
void janas_hx_destroy(struct janas_hx_context *ctx);
const char *janas_hx_name(const struct janas_hx_context *ctx);
const char *janas_hx_error(const struct janas_hx_context *ctx);
/* Unique device weight-buffer bytes owned by this context, shared plans
   counted once. Does not include per-plan compute buffers or host metadata. */
size_t janas_hx_weight_bytes(const struct janas_hx_context *ctx);

/* W is N contiguous rows of K weights. Input and output are contiguous
   float32 arrays X[M][K] and Y[M][N], computing Y = X * W^T. Creation
   uploads an owned copy of W; its source may be released after success.
   Plans keep their weight and compute allocations for repeated runs. */
struct janas_hx_plan *janas_hx_plan_create(struct janas_hx_context *ctx,
                                         enum janas_hx_type type,
                                         const void *weights, size_t k,
                                         size_t n, size_t m);
/* Another actual batch shape using the same owned weight tensor and buffer.
   Either plan may be destroyed first; the last plan releases the weights. */
struct janas_hx_plan *janas_hx_plan_rebatch(const struct janas_hx_plan *source,
                                           size_t m);
/* Synchronous: 0 on success, -1 on failure. */
int janas_hx_plan_run(struct janas_hx_plan *plan, const float *input,
                      float *output);
void janas_hx_plan_destroy(struct janas_hx_plan *plan);
void janas_hx_plan_stats(const struct janas_hx_plan *plan,
                         struct janas_hx_stats *stats);

#endif

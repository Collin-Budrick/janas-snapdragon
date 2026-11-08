/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * deltanet.h - the Gated DeltaNet recurrence of Qwen3-Next's linear
 * attention layers, one head and one token at a time.
 *
 * The state of a head is a dim x dim matrix S (key index i, value index j),
 * stored transposed as M[j][i] = S[i][j] so that every row of M is a
 * contiguous vector over the key index. One step, as in llama.cpp's fused
 * gated_delta_net operator:
 *
 *   S     *= decay                       (decay = exp(g))
 *   d[j]   = (v[j] - sum_i S[i][j] k[i]) * beta
 *   S[i][j] += k[i] d[j]
 *   o[j]   = scale * sum_i S[i][j] q[i]
 *
 * Row j of M depends only on d[j], so a step is a single pass over the state,
 * and any range of rows can be stepped on its own (v and o then start at the
 * range's first row): a head can be split among threads.
 */
#ifndef JANAS_LLM_DELTANET_H
#define JANAS_LLM_DELTANET_H

#include <stddef.h>
#include <stdint.h>

/* One step on rows rows of the state M (dim wide, dim a multiple of 8). */
void janas_gdn_step(float *M, const float *q, const float *k, const float *v,
                    float decay, float beta, float scale, float *o, size_t dim,
                    size_t rows);

/* Portable version, for tests. Not bit-identical to the AVX2 one. */
void janas_gdn_step_ref(float *M, const float *q, const float *k,
                        const float *v, float decay, float beta, float scale,
                        float *o, size_t dim, size_t rows);

/*
 * The same on a state stored in IEEE half precision (uint16_t), for half the
 * memory traffic: each row is computed in float and rounded once, when it is
 * stored back. dim at most 256.
 */
void janas_gdn_step_h(uint16_t *M, const float *q, const float *k,
                      const float *v, float decay, float beta, float scale,
                      float *o, size_t dim, size_t rows);

/* Portable half-state reference, for exact ARM kernel/replay comparisons.
   This private header is not part of the public library ABI. */
void janas_gdn_step_h_ref(uint16_t *M, const float *q, const float *k,
                          const float *v, float decay, float beta, float scale,
                          float *o, size_t dim, size_t rows);

/* An exact, bounded chain on independent rows. Strides count floats. Inputs
   contain replay tokens first, then n new tokens; outputs/delta contain only
   the new tokens. replay_delta, when present, is the exact d computed by an
   earlier verification, so replay needs k/decay but neither q/v nor a dot.
   The state is rounded to half AFTER EACH TOKEN; output uses the unrounded
   update. M is advanced past replay, tip past replay+n (M==tip is allowed).
   delta and replay_delta may alias: a row consumes its entire old prefix
   before writing its new deltas. Other arrays must not overlap the state.
   The caller retains the legacy path if this returns -1 (nothing touched). */
#define JANAS_GDN_CHAIN_MAX 17u
struct janas_gdn_chain {
    const float *q, *k, *v, *decay, *beta;
    size_t x_stride, gate_stride;
    float *out;
    size_t out_stride;
    const float *replay_delta;
    float *delta;
    size_t delta_stride;
};
int janas_gdn_chain_h(uint16_t *M, uint16_t *tip,
                      const struct janas_gdn_chain *chain, float scale,
                      size_t dim, size_t rows, size_t replay, size_t n);
/* Production opt-in is limited to the qualified ARM arithmetic/FPCR mode;
   the chain itself has a portable reference fallback for tests. */
int janas_gdn_chain_eligible(void);
uint64_t janas_gdn_fp_mode(void);

#ifdef JANAS_DELTANET_TEST
/* Exercise the same half conversions as the ARM kernel, without arithmetic. */
int janas_gdn_test_neon_half_enabled(void);
void janas_gdn_test_half_to_float(const uint16_t *h, float *f, size_t n);
void janas_gdn_test_float_to_half(const float *f, uint16_t *h, size_t n);
#endif

/* "avx2" or "ref": the kernel janas_gdn_step uses on this machine. */
const char *janas_gdn_kernel_name(void);

#endif

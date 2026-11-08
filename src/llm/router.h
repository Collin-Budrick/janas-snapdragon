/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_LLM_ROUTER_H
#define JANAS_LLM_ROUTER_H

#include <stddef.h>
#include <stdint.h>

/* Four expert rows interleaved by input column, padded with zero rows.
   Returns zero for empty/overflowing geometry. Packing copies bits only;
   source and destination must be disjoint. Nonfinite weights are rejected
   so their scalar NaN payload/exception behavior remains unchanged. */
size_t janas_router4_bytes(size_t rows, size_t cols);
int janas_router4_pack(uint16_t *dst, size_t bytes, const uint16_t *src,
                       size_t rows, size_t cols);

/* One packed four-row group against one token. Each lane follows the
   original left-to-right FP32 multiply, then add (no fused/reordered sum). */
void janas_router4_dot(const uint16_t *packed, const float *x, size_t cols,
                       float out[4]);

#endif

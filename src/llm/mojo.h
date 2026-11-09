/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_LLM_MOJO_H
#define JANAS_LLM_MOJO_H

#include <stddef.h>
#include <stdint.h>

struct janas_q8k_panel;

/* Private leaf-module ABI. Addresses are explicit 64-bit words so nullable
   planes never acquire Mojo's non-null Pointer contract. No C structs cross
   the ABI. The module borrows buffers for the duration of this call only. */
#define JANAS_MOJO_ABI_V1 UINT64_C(0x4a4d000100000001)
typedef uint64_t (*janas_mojo_abi_fn_v1)(void);
typedef int32_t (*janas_mojo_panel_fn_v1)(
    int32_t type, int32_t level, uint64_t w, uint64_t row_bytes,
    uint64_t p1, uint64_t p0, uint64_t pb, uint64_t x, uint64_t x_stride,
    uint64_t qs, uint64_t nb, uint64_t nv, uint64_t rows, uint64_t y,
    uint64_t y_stride);

/* Optional, off by default. JANAS_MOJO=1 requests the module at the absolute
   JANAS_MOJO_LIBRARY path. Loading and ABI/CPU checks run once. The loaded
   code lives until process exit: pool workers can never outlive its handle.
   Failure leaves the established C dispatch available. */
int janas_mojo_available(void);
int janas_mojo_enabled(void);
const char *janas_mojo_status(void);
/* Internal experiment control; change between completed pool calls only.
   -1 restores the startup request; returns the previous requested value. */
int janas_mojo_set_enabled(int enabled);

/* Return even rows completed, or zero without writing output. Runtime M=1
   keeps its existing specialized C path; the raw export also tests M=1. */
size_t janas_mojo_panel_rows(
    int type, const void *w, size_t row_bytes, const uint8_t *p1,
    const uint8_t *p0, size_t pb, const struct janas_q8k_panel *panel,
    size_t rows, float *y, size_t ys);

struct janas_mojo_stats {
    uint64_t calls, rows, row_vectors, refusals;
};
void janas_mojo_get_stats(struct janas_mojo_stats *out);

#endif

/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef JANAS_LLM_QUANT_PANEL_H
#define JANAS_LLM_QUANT_PANEL_H

#include <stddef.h>
#include <stdint.h>

struct janas_block_q8k;
struct janas_matvec_task;

/* Borrowed original scales/sums, plus immutable pairs of eight qs bytes:
   qs[((pair * nb + block) * 32 + chunk8) * 16] = token0[8], token1[8].
   An odd token uses x directly. No floating-point values are recomputed. */
struct janas_q8k_panel {
    const struct janas_block_q8k *x;
    const int8_t *qs;
    size_t x_stride, nb, nv, pairs;
};

size_t janas_q8k_panel_bytes(size_t nb, size_t nv); /* SIZE_MAX: invalid */
int janas_q8k_panel_pack(struct janas_q8k_panel *out, void *storage,
                         size_t bytes, const struct janas_block_q8k *x,
                         size_t xs, size_t nb, size_t nv);

#define JANAS_Q8K_PANEL_CAP ((size_t)8 << 20)
struct janas_q8k_panels {
    struct janas_q8k_panel *views; /* x == NULL: use the established path */
    size_t n, unique, bytes;
    void *allocation;
};
/* Fail soft on allocation/size/cap limits; malformed top-level args: -1.
   bytes includes the descriptor/offset workspace and the packed storage. */
int janas_q8k_panels_create(const struct janas_matvec_task *tasks, size_t n,
                            size_t cap, struct janas_q8k_panels *out);
void janas_q8k_panels_free(struct janas_q8k_panels *panels);

#endif

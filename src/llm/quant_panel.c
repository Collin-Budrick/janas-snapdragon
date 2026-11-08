/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "quant_panel.h"
#include "matvec.h"
#include "common/aligned.h"

#include <stdlib.h>
#include <string.h>

static int span_ok(size_t xs, size_t nb, size_t nv)
{
    return xs >= nb && (nv == 1 || xs <= (SIZE_MAX - nb) / (nv - 1)) &&
           (nv - 1) * xs + nb <= SIZE_MAX / sizeof(struct janas_block_q8k);
}

size_t janas_q8k_panel_bytes(size_t nb, size_t nv)
{
    if (!nb || !nv || nv > 9 || nb > SIZE_MAX / 512 / (nv / 2 ? nv / 2 : 1))
        return SIZE_MAX;
    return (nv / 2) * nb * 512;
}

int janas_q8k_panel_pack(struct janas_q8k_panel *out, void *storage,
                         size_t bytes, const struct janas_block_q8k *x,
                         size_t xs, size_t nb, size_t nv)
{
    size_t need = janas_q8k_panel_bytes(nb, nv);
    if (!out || !x || need == SIZE_MAX || bytes < need || (need && !storage))
        return -1;
    if (!xs)
        xs = nb;
    if (!span_ok(xs, nb, nv))
        return -1;
    int8_t *dst = storage;
    for (size_t p = 0; p < nv / 2; p++)
        for (size_t b = 0; b < nb; b++)
            for (size_t k = 0; k < 32; k++) {
                int8_t *d = dst + ((p * nb + b) * 32 + k) * 16;
                memcpy(d, (x + (2 * p) * xs + b)->qs + 8 * k, 8);
                memcpy(d + 8, (x + (2 * p + 1) * xs + b)->qs + 8 * k, 8);
            }
    *out = (struct janas_q8k_panel){x, storage, xs, nb, nv, nv / 2};
    return 0;
}

static int eligible(const struct janas_matvec_task *t)
{
    size_t nv = t->n_vec ? t->n_vec : 1;
    size_t nb = t->cols / JANAS_QK, xs = t->x_stride ? t->x_stride : nb;
    return t->x && t->rows >= 2 && t->cols && t->cols % JANAS_QK == 0 &&
           nv <= 9 && span_ok(xs, nb, nv) &&
           (t->type == JANAS_Q4_K || t->type == JANAS_Q6_K ||
            t->type == JANAS_Q6_K_P);
}

void janas_q8k_panels_free(struct janas_q8k_panels *p)
{
    if (!p)
        return;
    janas_aligned_free(p->allocation);
    free(p->views);
    memset(p, 0, sizeof(*p));
}

int janas_q8k_panels_create(const struct janas_matvec_task *tasks, size_t n,
                            size_t cap, struct janas_q8k_panels *out)
{
    if (!out || (n && !tasks))
        return -1;
    memset(out, 0, sizeof(*out));
    if (cap > JANAS_Q8K_PANEL_CAP)
        cap = JANAS_Q8K_PANEL_CAP;
    size_t unit = sizeof(struct janas_q8k_panel) + sizeof(size_t);
    if (!n || n > SIZE_MAX / unit || n * unit > cap)
        return 0;
    int any = 0;
    for (size_t i = 0; i < n; i++)
        any |= eligible(tasks + i);
    if (!any)
        return 0;
    size_t meta = n * unit, used = 0;
    out->views = calloc(1, meta);
    if (!out->views)
        return 0;
    size_t *offset = (size_t *)(out->views + n);
    for (size_t i = 0; i < n; i++) {
        const struct janas_matvec_task *t = tasks + i;
        if (!eligible(t))
            continue;
        size_t nb = t->cols / JANAS_QK, nv = t->n_vec ? t->n_vec : 1;
        size_t xs = t->x_stride ? t->x_stride : nb;
        size_t bytes = janas_q8k_panel_bytes(nb, nv), j = 0;
        for (; j < i; j++) {
            const struct janas_q8k_panel *v = out->views + j;
            if (v->x == t->x && v->x_stride == xs && v->nb == nb && v->nv == nv)
                break;
        }
        if (j < i) {
            out->views[i] = out->views[j];
            offset[i] = offset[j];
        } else if (bytes != SIZE_MAX && bytes <= cap - meta - used) {
            out->views[i] =
                (struct janas_q8k_panel){t->x, NULL, xs, nb, nv, nv / 2};
            offset[i] = used;
            used += bytes;
            out->unique++;
        }
    }
    if (used) {
        out->allocation = janas_aligned_alloc(16, used);
        if (!out->allocation) {
            janas_q8k_panels_free(out);
            return 0;
        }
    }
    for (size_t i = 0; i < n; i++) {
        struct janas_q8k_panel *v = out->views + i;
        if (!v->x)
            continue;
        void *dst = used ? (char *)out->allocation + offset[i] : NULL;
        size_t j = 0;
        for (; j < i; j++)
            if (out->views[j].x && offset[j] == offset[i] &&
                out->views[j].x == v->x && out->views[j].nb == v->nb &&
                out->views[j].nv == v->nv &&
                out->views[j].x_stride == v->x_stride)
                break;
        if (j == i) {
            if (janas_q8k_panel_pack(v, dst,
                                     janas_q8k_panel_bytes(v->nb, v->nv), v->x,
                                     v->x_stride, v->nb, v->nv) != 0)
                memset(v, 0, sizeof(*v));
        } else
            v->qs = out->views[j].qs;
    }
    out->n = n;
    out->bytes = meta + used;
    return 0;
}

/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Private to quant.c, after the established NEON format helpers. */
#define NEON_PANEL __attribute__((target("+dotprod+i8mm")))

static NEON_PANEL void panel_unpack(int type, const void *row, size_t b,
                                    const uint8_t *p1, const uint8_t *p0,
                                    int8_t q[256], int sc[16], int mn[8],
                                    float *d, float *dm)
{
    if (type == JANAS_Q4_K) {
        const struct janas_block_q4k *w =
            (const struct janas_block_q4k *)row + b;
        uint8_t s[8], m[8];
        unpack_q4k(w, q);
        unpack_scales(w->scales, s, m);
        for (int g = 0; g < 8; g++)
            sc[g] = s[g], mn[g] = m[g];
        *d = neon_h2f(w->d);
        *dm = neon_h2f(w->dmin);
    } else if (type == JANAS_Q6_K) {
        const struct janas_block_q6k *w =
            (const struct janas_block_q6k *)row + b;
        uint8x16_t m4 = vdupq_n_u8(15), m2 = vdupq_n_u8(3);
        int8x16_t center = vdupq_n_s8(32);
        for (int n = 0; n < 2; n++)
            for (int h = 0; h < 2; h++) {
                const uint8_t *ql = w->ql + 64 * n + 16 * h;
                uint8x16_t a = vld1q_u8(ql), z = vld1q_u8(ql + 32);
                uint8x16_t hi = vld1q_u8(w->qh + 32 * n + 16 * h);
                int g = 8 * n + h;
#define STORE_Q6(G, LO, HI)                                                    \
    vst1q_s8(q + 16 * (G),                                                     \
             vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(LO, HI)), center))
                STORE_Q6(g, vandq_u8(a, m4), vshlq_n_u8(vandq_u8(hi, m2), 4));
                STORE_Q6(g + 2, vandq_u8(z, m4),
                         vshlq_n_u8(vandq_u8(vshrq_n_u8(hi, 2), m2), 4));
                STORE_Q6(g + 4, vshrq_n_u8(a, 4),
                         vshlq_n_u8(vandq_u8(vshrq_n_u8(hi, 4), m2), 4));
                STORE_Q6(g + 6, vshrq_n_u8(z, 4),
                         vshlq_n_u8(vshrq_n_u8(hi, 6), 4));
#undef STORE_Q6
            }
        for (int g = 0; g < 16; g++)
            sc[g] = w->scales[g];
        *d = neon_h2f(w->d);
    } else {
        const struct janas_block_q6kp *w =
            (const struct janas_block_q6kp *)row + b;
        int level = p0 ? 3 : p1 ? 2 : 1;
        for (int n = 0; n < 2; n++)
            for (int h = 0; h < 2; h++) {
                size_t at = b * JANAS_Q6KP_PLANE + 32 * n + 16 * h;
                uint8x16_t a = vld1q_u8(w->q + 32 * n + 16 * h);
                uint8x16_t z = p1 ? vld1q_u8(p1 + at) : vdupq_n_u8(0);
                uint8x16_t c = p0 ? vld1q_u8(p0 + at) : vdupq_n_u8(0);
                for (int part = 0; part < 4; part++)
                    vst1q_s8(q + 16 * (8 * n + h + 2 * part),
                             vreinterpretq_s8_u8(
                                 neon_q6kp_piece(a, z, c, 2 * part, level)));
            }
        for (int g = 0; g < 16; g++)
            sc[g] = w->scales[g];
        *d = neon_h2f(w->d) * 0.5f;
    }
}

/* Each block's decoded weights serve ALL token pairs and the odd SDOT
   column before the next block. There is no separate odd-token matrix pass.
   Temporary decoded bytes are per-block scratch, never a weight cache. */
static NEON_PANEL void panel_rows_neon(int type, const void *w,
                                       size_t row_bytes, const uint8_t *p1,
                                       const uint8_t *p0, size_t pb,
                                       const struct janas_q8k_panel *panel,
                                       size_t rows, float *y, size_t ys)
{
    size_t nv = panel->nv, pairs = panel->pairs, nb = panel->nb;
    int groups = type == JANAS_Q4_K ? 8 : 16;
    int len = type == JANAS_Q4_K ? 32 : 16;
    int level = p0 ? 3 : p1 ? 2 : 1;
    int factor = level == 3 ? 2 : level == 2 ? 8 : 32;
    int center = level == 3 ? 64 : level == 2 ? 61 : 49;
    for (size_t r = 0; r < rows; r += 2) {
        const void *wr[2] = {(const char *)w + r * row_bytes,
                             (const char *)w + (r + 1) * row_bytes};
        float32x4_t acc[4], accm[4];
        float odd[2] = {0}, oddm[2] = {0};
        for (size_t p = 0; p < pairs; p++)
            acc[p] = accm[p] = vdupq_n_f32(0.0f);
        for (size_t b = 0; b < nb; b++) {
            int8_t q[2][256];
            int sc[2][16], mn[2][8];
            float d[2], dm[2] = {0};
            for (int k = 0; k < 2; k++)
                panel_unpack(type, wr[k], b,
                             p1 ? p1 + (r + (size_t)k) * pb : NULL,
                             p0 ? p0 + (r + (size_t)k) * pb : NULL, q[k], sc[k],
                             mn[k], d + k, dm + k);
            int32x4_t sum[4];
            for (size_t p = 0; p < pairs; p++)
                sum[p] = vdupq_n_s32(0);
            int32x4_t os[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
            const struct janas_block_q8k *ox =
                panel->x + (nv - 1) * panel->x_stride + b;
            for (int g = 0; g < groups; g++) {
                int32x4_t s =
                    vcombine_s32(vdup_n_s32(sc[0][g]), vdup_n_s32(sc[1][g]));
                int32x4_t dot[4];
                for (size_t p = 0; p < pairs; p++)
                    dot[p] = vdupq_n_s32(0);
                for (int k = 0; k < len / 8; k++) {
                    int at = g * len + k * 8;
                    int8x16_t a =
                        vcombine_s8(vld1_s8(q[0] + at), vld1_s8(q[1] + at));
                    for (size_t p = 0; p < pairs; p++) {
                        const int8_t *bp =
                            panel->qs +
                            ((p * nb + b) * 32 + (size_t)at / 8) * 16;
                        dot[p] = vmmlaq_s32(dot[p], a, vld1q_s8(bp));
                    }
                }
                for (size_t p = 0; p < pairs; p++)
                    sum[p] = vmlaq_s32(sum[p], dot[p], s);
                if (nv & 1) {
                    for (int k = 0; k < 2; k++) {
                        int32x4_t od = vdupq_n_s32(0);
                        for (int i = 0; i < len; i += 16)
                            od = vdotq_s32(od, vld1q_s8(q[k] + g * len + i),
                                           vld1q_s8(ox->qs + g * len + i));
                        os[k] = vmlaq_n_s32(os[k], od, sc[k][g]);
                    }
                }
            }
            float32x4_t ds = vcombine_f32(vdup_n_f32(d[0]), vdup_n_f32(d[1]));
            float32x4_t ms = vcombine_f32(vdup_n_f32(dm[0]), vdup_n_f32(dm[1]));
            for (size_t p = 0; p < pairs; p++) {
                const struct janas_block_q8k *a =
                    panel->x + 2 * p * panel->x_stride + b;
                const struct janas_block_q8k *c = a + panel->x_stride;
                int32_t correction[4] = {0};
                if (type != JANAS_Q6_K)
                    for (int k = 0; k < 2; k++)
                        for (int g = 0; g < groups; g++) {
                            int sa = type == JANAS_Q4_K
                                         ? a->bsums[2 * g] + a->bsums[2 * g + 1]
                                         : a->bsums[g];
                            int sb = type == JANAS_Q4_K
                                         ? c->bsums[2 * g] + c->bsums[2 * g + 1]
                                         : c->bsums[g];
                            int scale =
                                type == JANAS_Q4_K ? mn[k][g] : sc[k][g];
                            correction[2 * k] += scale * sa;
                            correction[2 * k + 1] += scale * sb;
                        }
                int32x4_t cv = vld1q_s32(correction), si = sum[p];
                if (type == JANAS_Q6_K_P)
                    si = vsubq_s32(vmulq_n_s32(si, factor),
                                   vmulq_n_s32(cv, center));
                float32x2_t xd = vdup_n_f32(a->d);
                xd = vset_lane_f32(c->d, xd, 1);
                float32x4_t dx = vcombine_f32(xd, xd);
                acc[p] =
                    vfmaq_f32(acc[p], vmulq_f32(ds, dx), vcvtq_f32_s32(si));
                if (type == JANAS_Q4_K)
                    accm[p] = vfmaq_f32(accm[p], vmulq_f32(ms, dx),
                                        vcvtq_f32_s32(cv));
            }
            if (nv & 1)
                for (int k = 0; k < 2; k++) {
                    int32_t si = vaddvq_s32(os[k]), correction = 0;
                    if (type != JANAS_Q6_K)
                        for (int g = 0; g < groups; g++)
                            correction +=
                                type == JANAS_Q4_K
                                    ? mn[k][g] * (ox->bsums[2 * g] +
                                                  ox->bsums[2 * g + 1])
                                    : sc[k][g] * ox->bsums[g];
                    if (type == JANAS_Q6_K_P)
                        si = factor * si - center * correction;
                    odd[k] = fmaf(d[k] * ox->d, (float)si, odd[k]);
                    if (type == JANAS_Q4_K)
                        oddm[k] =
                            fmaf(dm[k] * ox->d, (float)correction, oddm[k]);
                }
        }
        for (size_t p = 0; p < pairs; p++) {
            float a[4];
            vst1q_f32(a,
                      type == JANAS_Q4_K ? vsubq_f32(acc[p], accm[p]) : acc[p]);
            y[r + (2 * p) * ys] = a[0];
            y[r + (2 * p + 1) * ys] = a[1];
            y[r + 1 + (2 * p) * ys] = a[2];
            y[r + 1 + (2 * p + 1) * ys] = a[3];
        }
        if (nv & 1)
            for (int k = 0; k < 2; k++)
                y[r + (size_t)k + (nv - 1) * ys] =
                    type == JANAS_Q4_K ? odd[k] - oddm[k] : odd[k];
    }
}

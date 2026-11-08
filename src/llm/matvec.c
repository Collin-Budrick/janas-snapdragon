/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * matvec.c - quantized matrix-vector product on a thread pool (see matvec.h).
 */
#include "matvec.h"
#include "quant_panel.h"
#include "common/sysinfo.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Rows are claimed in chunks: large enough that each thread streams a long
 * contiguous run (the hardware prefetcher restarts at every jump: 16-row
 * chunks reached 44 GB/s where 256-row chunks reach 52, on 6 P-cores), small
 * enough that every thread gets about four chunks on small matrices.
 *
 * The chunk is not fixed: it is a fraction of what is left. On cores of three
 * different speeds a fixed chunk leaves everyone waiting for the last one
 * claimed - with four chunks a thread that is half the tail of the run. Taking
 * 1/(2*threads) of the remaining rows keeps the first chunks long, where the
 * prefetcher earns its keep, and ends with chunks of CHUNK_MIN, where nobody
 * waits long. Chunks of different sizes still partition the rows exactly: a
 * fetch_add publishes its own size, so [old, old+chunk) is nobody else's.
 */
#define CHUNK_MIN 16
#define CHUNK_MAX 256

/* Claims the next chunk of [0, total): returns its first row, or total when
   there is nothing left, and writes the end in *end. */
static inline size_t claim(atomic_size_t *next, size_t total, size_t div,
                           size_t *end)
{
    *end = total;
    size_t seen = atomic_load_explicit(next, memory_order_relaxed);
    if (seen >= total)
        return total;
    size_t rem = total - seen, c = rem / div;
    if (c < CHUNK_MIN)
        c = CHUNK_MIN;
    else if (c > CHUNK_MAX)
        c = CHUNK_MAX;
    size_t g0 = atomic_fetch_add_explicit(next, c, memory_order_relaxed);
    if (g0 >= total)
        return total;
    *end = g0 + c < total ? g0 + c : total;
    return g0;
}

/*
 * A chunk's rows on many vectors are taken in pieces whose weights stay in
 * the level-2 cache while every group of eight vectors goes over them: the
 * kernels run the groups outside and the rows inside, so a chunk larger
 * than the cache was read from memory again for each group of eight - the
 * whole matrix once every eight tokens of a prompt. A piece is a quarter of
 * the cache (the rest for the vectors), a cache of 256 KB where the
 * machine does not say. Not by default: on the laptop's Core Ultra (2 MB of
 * L2) a chunk already fits and pieces of 128 KB cost 6% of a prompt (Qwen3-
 * 4B, 52.9 against 56.1 tokens/s, AVX2), and on a Ryzen 5 5500 (512 KB, a
 * prompt 2.2 times slower than llama.cpp's, issue #45) it is not measured
 * yet. JANAS_ROW_PIECE_KB=<n> takes pieces of n KB, -1 a quarter of the
 * L2, 0 or unset the chunk whole.
 */
static size_t row_piece_bytes(void)
{
    static _Atomic size_t bytes; /* 0: not yet known */
    size_t b = atomic_load_explicit(&bytes, memory_order_relaxed);
    if (b)
        return b;
    const char *e = getenv("JANAS_ROW_PIECE_KB");
    if (e && atoi(e) < 0) {
        size_t l2 = janas_l2_bytes();
        b = (l2 ? l2 : (size_t)256 << 10) / 4;
    } else {
        b = e && atoi(e) > 0 ? (size_t)atoi(e) << 10 : SIZE_MAX;
    }
    atomic_store_explicit(&bytes, b, memory_order_relaxed);
    return b;
}

struct matvec_job {
    const struct janas_block_q4k *w;
    const struct janas_block_q8k *x;
    float *y;
    size_t rows;
    size_t nb;  /* blocks per row */
    size_t div; /* the chunk is 1/div of the rows still to claim */
    atomic_size_t next;
};

static void matvec_worker(void *arg, int tid, int n_threads)
{
    (void)tid;
    (void)n_threads;
    struct matvec_job *job = arg;
    for (;;) {
        size_t r1;
        size_t r0 = claim(&job->next, job->rows, job->div, &r1);
        if (r0 >= job->rows)
            break;
        for (size_t r = r0; r < r1; r++)
            job->y[r] = janas_q4k_dot(job->w + r * job->nb, job->x, job->nb);
    }
}

void janas_matvec_q4k(struct janas_pool *pool, const struct janas_block_q4k *w,
                      size_t rows, size_t cols, const struct janas_block_q8k *x,
                      float *y)
{
    struct matvec_job job = {
        .w = w, .x = x, .y = y, .rows = rows, .nb = cols / JANAS_QK};
    job.div = (size_t)janas_pool_active(pool) * 2;
    atomic_init(&job.next, 0);
    janas_pool_run(pool, matvec_worker, &job);
}

struct group_partial {
    uint64_t prepared_rows, prepared_row_vectors;
    uint64_t established_rows, established_row_vectors;
    uint64_t padding[4]; /* each worker writes its own cache line */
};

struct group_job {
    const struct janas_matvec_task *tasks;
    const struct janas_q8k_panel *panels;
    size_t n;
    size_t *first_row; /* first global row of each task, n + 1 entries */
    size_t div;        /* the chunk is 1/div of the rows still to claim */
    atomic_size_t next;
    int collect;
    struct group_partial *partial;
    atomic_uint_fast64_t prepared_rows, prepared_row_vectors;
    atomic_uint_fast64_t established_rows, established_row_vectors;
};

static inline void group_count(struct group_job *job, int tid, size_t rows,
                               size_t nv, int prepared)
{
    if (!job->collect || !rows)
        return;
    if (job->partial) {
        struct group_partial *p = job->partial + tid;
        if (prepared) {
            p->prepared_rows += rows;
            p->prepared_row_vectors += (uint64_t)rows * nv;
        } else {
            p->established_rows += rows;
            p->established_row_vectors += (uint64_t)rows * nv;
        }
    } else {
        /* Pools above the bounded stack workspace still report accurately. */
        atomic_fetch_add_explicit(prepared ? &job->prepared_rows : &job->established_rows,
                                  rows, memory_order_relaxed);
        atomic_fetch_add_explicit(prepared ? &job->prepared_row_vectors : &job->established_row_vectors,
                                  (uint64_t)rows * nv, memory_order_relaxed);
    }
}

static void group_worker(void *arg, int tid, int n_threads)
{
    (void)n_threads;
    struct group_job *job = arg;
    size_t total = job->first_row[job->n];
    size_t t = 0;
    for (;;) {
        size_t g1;
        size_t g0 = claim(&job->next, total, job->div, &g1);
        if (g0 >= total)
            break;
        /* a thread's claims grow, whatever the size: the task index only
           goes forward. A claim may span tasks: each part with its task's
           sizes worked out once, and on one vector all its rows in one
           call (janas_dot_rows) */
        while (g0 < g1) {
            while (job->first_row[t + 1] <= g0)
                t++;
            const struct janas_matvec_task *k = &job->tasks[t];
            size_t r0 = g0 - job->first_row[t];
            size_t end =
                job->first_row[t + 1] < g1 ? job->first_row[t + 1] : g1;
            size_t nr = end - g0;
            /* a row may end inside a group of 256 (janas_dot_rows_n) */
            size_t nb = k->cols / JANAS_QK, tail = k->cols % JANAS_QK;
            size_t row_bytes = janas_qtype_row_bytes(k->type, k->cols);
            const char *w0 = (const char *)k->w + r0 * row_bytes;
            size_t nv = k->n_vec ? k->n_vec : 1;
            size_t xs = k->x_stride ? k->x_stride : nb + (tail != 0);
            size_t ys = k->y_stride ? k->y_stride : k->rows;
            if (job->panels && job->panels[t].x) {
                size_t pb = nb * JANAS_Q6KP_PLANE;
                const uint8_t *p1 = k->plane[0], *p0 = k->plane[1];
                size_t done = janas_dot_prepared_rows(
                    k->type, w0, row_bytes, p1 ? p1 + r0 * pb : NULL,
                    p0 ? p0 + r0 * pb : NULL, pb, job->panels + t, nr,
                    k->y + r0, ys);
                group_count(job, tid, done, nv, 1);
                r0 += done;
                nr -= done;
                w0 += done * row_bytes;
                if (!nr) {
                    g0 = end;
                    continue;
                }
            }
            group_count(job, tid, nr, nv, 0);
            if (k->type == JANAS_Q6_K_P) {
                size_t pb = nb * JANAS_Q6KP_PLANE;
                /* eight rows at a time as tiles, where they serve */
                const uint8_t *pl1 = k->plane[0], *pl0 = k->plane[1];
                size_t r8 = janas_q6k_tile_rows(
                    k->type, w0, row_bytes, pl1 ? pl1 + r0 * pb : NULL,
                    pl0 ? pl0 + r0 * pb : NULL, pb, k->x, xs, nv, nb, nr,
                    k->y + r0, ys);
                for (size_t r = r0 + r8; r < r0 + nr; r++) {
                    const void *row = (const char *)k->w + r * row_bytes;
                    const uint8_t *p1 =
                        k->plane[0] ? (const uint8_t *)k->plane[0] + r * pb
                                    : NULL;
                    const uint8_t *p0 =
                        k->plane[1] ? (const uint8_t *)k->plane[1] + r * pb
                                    : NULL;
                    janas_q6kp_dot_multi(row, p1, p0, k->x, xs, nv, nb,
                                         k->y + r, ys);
                }
            } else if (tail && nv == 1) {
                janas_dot_rows_n(k->type, w0, row_bytes, k->x, k->cols, nr,
                                 k->y + r0);
            } else if (tail) {
                janas_dot_multi_rows_n(k->type, w0, row_bytes, k->x, xs, nv,
                                       k->cols, nr, k->y + r0, ys);
            } else if (nv == 1) {
                janas_dot_rows(k->type, w0, row_bytes, k->x, nb, nr, k->y + r0);
            } else {
                size_t piece = row_piece_bytes() / row_bytes;
                if (nv <= 8 || piece < 1)
                    piece = nv <= 8 ? nr : 1;
                for (size_t i = 0; i < nr; i += piece)
                    janas_dot_multi_rows(
                        k->type, w0 + i * row_bytes, row_bytes, k->x, xs, nv,
                        nb, nr - i < piece ? nr - i : piece, k->y + r0 + i, ys);
            }
            g0 = end;
        }
    }
}

void janas_matvec_q4k_group(struct janas_pool *pool,
                            const struct janas_matvec_task *tasks, size_t n)
{
    janas_matvec_q4k_group_stats(pool, tasks, n, NULL);
}

static double group_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

void janas_matvec_q4k_group_stats(struct janas_pool *pool,
                                  const struct janas_matvec_task *tasks,
                                  size_t n, struct janas_matvec_stats *stats)
{
    if (stats) {
        memset(stats, 0, sizeof(*stats));
        stats->groups = 1;
        stats->tasks = n;
    }
    size_t first_row_small[65];
    size_t *first_row = first_row_small;
    if (n + 1 > sizeof(first_row_small) / sizeof(first_row_small[0])) {
        first_row = malloc((n + 1) * sizeof(size_t));
        if (!first_row)
            abort();
    }
    first_row[0] = 0;
    for (size_t i = 0; i < n; i++)
        first_row[i + 1] = first_row[i] + tasks[i].rows;

    struct janas_q8k_panels panels = {0};
    if (janas_quant_panels_enabled()) {
        double begin = stats ? group_now() : 0;
        janas_q8k_panels_create(tasks, n, JANAS_Q8K_PANEL_CAP, &panels);
        if (stats) {
            stats->panel_prepare_seconds = group_now() - begin;
            stats->panel_bytes = panels.bytes;
            stats->panel_unique = panels.unique;
            for (size_t i = 0; panels.views && i < n; i++)
                stats->panel_views += panels.views[i].x != NULL;
            if (stats->panel_views >= stats->panel_unique)
                stats->panel_dedup_hits = stats->panel_views - stats->panel_unique;
        }
    }
    _Alignas(64) struct group_partial partial[64];
    int active = janas_pool_active(pool);
    struct group_job job = {.tasks = tasks, .panels = panels.views,
                            .n = n, .first_row = first_row,
                            .collect = stats != NULL};
    if (stats && active <= (int)(sizeof partial / sizeof partial[0])) {
        memset(partial, 0, (size_t)active * sizeof partial[0]);
        job.partial = partial;
    }
    job.div = (size_t)active * 2;
    atomic_init(&job.next, 0);
    atomic_init(&job.prepared_rows, 0);
    atomic_init(&job.prepared_row_vectors, 0);
    atomic_init(&job.established_rows, 0);
    atomic_init(&job.established_row_vectors, 0);
    janas_pool_run(pool, group_worker, &job);
    if (stats) {
        if (job.partial) {
            for (int i = 0; i < active; i++) {
                stats->prepared_rows += partial[i].prepared_rows;
                stats->prepared_row_vectors += partial[i].prepared_row_vectors;
                stats->established_rows += partial[i].established_rows;
                stats->established_row_vectors += partial[i].established_row_vectors;
            }
        } else {
            stats->prepared_rows = atomic_load_explicit(&job.prepared_rows, memory_order_relaxed);
            stats->prepared_row_vectors = atomic_load_explicit(&job.prepared_row_vectors, memory_order_relaxed);
            stats->established_rows = atomic_load_explicit(&job.established_rows, memory_order_relaxed);
            stats->established_row_vectors = atomic_load_explicit(&job.established_row_vectors, memory_order_relaxed);
        }
    }
    janas_q8k_panels_free(&panels);
    if (first_row != first_row_small)
        free(first_row);
}

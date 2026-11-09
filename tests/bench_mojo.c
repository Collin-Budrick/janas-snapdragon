/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Matched C/Mojo ABBA microbenchmarks. The group measurement uses the real
   Janas pool dispatch and includes panel allocation, packing and free.
   Usage: JANAS_CPU_TILES=1 JANAS_MOJO_LIBRARY=/absolute/module
     bench_mojo [rows=512 cols=2048 iterations=8 rounds=5 threads=8 nv=0 kind=all]
   nv=0 covers 1..9; kind is all|q4k|q6k|q6kp1|q6kp2|q6kp3.
   JSON-lines retain every first-shape and warm measurement. First-shape
   calls follow correctness validation and are not cold-process samples.
   No model jobs. */
#define JANAS_MOJO_TEST_LIBRARY
#include "test_mojo.c"
#include <errno.h>
#include <time.h>

#include "common/sysinfo.h"

static uint64_t mojo_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static size_t mojo_arg(const char *s, size_t fallback)
{
    if (!s) return fallback;
    char *end = NULL;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    return errno || !*s || *end || n > SIZE_MAX ? SIZE_MAX : (size_t)n;
}

static int mojo_kind(const char *name, int *type, int *level)
{
    *level = 0;
    if (!strcmp(name, "q4k")) *type = JANAS_Q4_K;
    else if (!strcmp(name, "q6k")) *type = JANAS_Q6_K;
    else if (strlen(name) == 5 && !strncmp(name, "q6kp", 4) &&
             name[4] >= '1' && name[4] <= '3' && !name[5]) {
        *type = JANAS_Q6_K_P; *level = name[4] - '0';
    } else return -1;
    return 0;
}

static int mojo_measure(mojo_export_fn fn, struct mojo_fixture *f,
    struct janas_pool *pool, int group, int candidate, size_t iterations,
    size_t round, size_t position, const char *kind, const char *lifecycle)
{
    janas_mojo_set_enabled(candidate);
    struct janas_matvec_task task = {
        .type = f->type, .w = f->w,
        .plane = {f->level >= 2 ? f->p1 : NULL, f->level >= 3 ? f->p0 : NULL},
        .x = f->x, .y = f->a + 4, .rows = f->rows,
        .cols = f->nb * JANAS_QK, .n_vec = f->nv,
        .x_stride = f->xs, .y_stride = f->ys
    };
    struct janas_mojo_stats before = {0}, after = {0};
    janas_mojo_get_stats(&before);
    uint64_t begin = mojo_now_ns();
    for (size_t i = 0; i < iterations; i++) {
        if (group) {
            janas_matvec_q4k_group(pool, &task, 1);
        } else if (candidate) {
            if (mojo_export_call(fn, f, f->rows, f->a + 4) != 1) return 1;
        } else {
            if (janas_dot_prepared_rows(f->type, f->w, f->rb,
                    f->level >= 2 ? f->p1 : NULL,
                    f->level >= 3 ? f->p0 : NULL, f->pb, &f->panel,
                    f->rows, f->a + 4, f->ys) != f->rows) return 1;
        }
    }
    uint64_t elapsed = mojo_now_ns() - begin;
    janas_mojo_get_stats(&after);
    size_t bad = mojo_bad_word(f->a, f->ref, f->out_count);
    if (bad != f->out_count) {
        fprintf(stderr, "bench_mojo: output mismatch kind=%s nv=%zu mode=%s arm=%s at=%zu\n",
            kind, f->nv, group ? "group" : "kernel", candidate ? "mojo" : "c", bad);
        return 1;
    }
    printf("{\"schema\":1,\"benchmark\":\"mojo\",\"kind\":\"%s\","
        "\"rows\":%zu,\"cols\":%zu,\"vectors\":%zu,\"threads\":%d,"
        "\"mode\":\"%s\",\"arm\":\"%s\",\"c_policy\":\"prepared_tiles\",\"order\":\"ABBA\","
        "\"round\":%zu,\"position\":%zu,\"lifecycle\":\"%s\","
        "\"iterations\":%zu,\"elapsed_ns\":%llu,\"ns_per_call\":%.3f,"
        "\"output_hash\":\"%016llx\",\"mojo_calls\":%llu,"
        "\"mojo_rows\":%llu,\"mojo_row_vectors\":%llu}\n",
        kind, f->rows, f->nb * JANAS_QK, f->nv, group ? janas_pool_size(pool) : 1,
        group ? "packing_pool" : "kernel_only", candidate ? "mojo" : "c",
        round, position, lifecycle, iterations, (unsigned long long)elapsed,
        (double)elapsed / (double)iterations,
        (unsigned long long)mojo_hash(f->a, f->out_count * sizeof *f->a),
        (unsigned long long)(after.calls - before.calls),
        (unsigned long long)(after.rows - before.rows),
        (unsigned long long)(after.row_vectors - before.row_vectors));
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv)
{
    size_t rows = mojo_arg(argc > 1 ? argv[1] : NULL, 512);
    size_t cols = mojo_arg(argc > 2 ? argv[2] : NULL, 2048);
    size_t iterations = mojo_arg(argc > 3 ? argv[3] : NULL, 8);
    size_t rounds = mojo_arg(argc > 4 ? argv[4] : NULL, 5);
    size_t threads = mojo_arg(argc > 5 ? argv[5] : NULL, 8);
    size_t selected_nv = mojo_arg(argc > 6 ? argv[6] : NULL, 0);
    const char *selected_kind = argc > 7 ? argv[7] : "all";
    int requested_type, requested_level;
    if (strcmp(selected_kind, "all") &&
        mojo_kind(selected_kind, &requested_type, &requested_level)) {
        fprintf(stderr, "bench_mojo: unknown kind %s\n", selected_kind);
        return 2;
    }
    if (rows < 2 || rows > 131072 || (rows & 1) || cols < 256 || cols > 131072 ||
        cols % JANAS_QK || !iterations || iterations > 100000 || !rounds ||
        rounds > 1000 || !threads || threads > 64 || selected_nv > 9 ||
        rows > ((size_t)512 << 20) / (cols / JANAS_QK * 210)) {
        fprintf(stderr, "bench_mojo: invalid or excessive dimensions/iteration controls\n");
        return 2;
    }
    if (!janas_quant_panels_enabled()) {
        fprintf(stderr, "bench_mojo: launch with JANAS_CPU_TILES=1 on ARM64 I8MM\n");
        return 2;
    }
    mojo_export_fn fn = mojo_test_load();
    if (!fn) return 1;
    /* Independent numerical fixtures precede all clocks. */
    if (mojo_test_all(fn)) return 1;
    int cpus[JANAS_MAX_CPUS];
    struct janas_cpu_layout layout;
    int have_layout = !janas_cpu_layout(NULL, 0, &layout) && layout.n >= (int)threads;
    if (have_layout)
        for (size_t i = 0; i < threads; i++) cpus[i] = layout.cpus[i];
    struct janas_pool *pool = janas_pool_create((int)threads, have_layout ? cpus : NULL);
    if (!pool) { fprintf(stderr, "bench_mojo: pool allocation failed\n"); return 1; }
    unsigned char saved[JANAS_CPU_SAVE_BYTES];
    int affinity_saved = !janas_cpus_save(saved);
    if (have_layout) janas_pin_current_thread(cpus[0]);
    const char *kinds[] = {"q4k", "q6k", "q6kp1", "q6kp2", "q6kp3"};
    int result = 0;
    for (size_t k = 0; k < 5 && !result; k++) {
        if (strcmp(selected_kind, "all") && strcmp(selected_kind, kinds[k])) continue;
        int type, level;
        if (mojo_kind(kinds[k], &type, &level)) { result = 2; break; }
        for (size_t nv = selected_nv ? selected_nv : 1;
             nv <= (selected_nv ? selected_nv : 9) && !result; nv++) {
            struct mojo_fixture f;
            if (mojo_fixture_create(&f, type, level, rows, cols / JANAS_QK, nv, 0, 0)) {
                fprintf(stderr, "bench_mojo: fixture allocation failed\n"); result = 1; break;
            }
            for (int group = 0; group < 2 && !result; group++) {
                /* Retain one first call per arm/mode; no warmup is hidden. */
                for (int arm = 0; arm < 2 && !result; arm++)
                    result = mojo_measure(fn, &f, pool, group, arm, 1, 0,
                        (size_t)arm, kinds[k], "first_shape_call");
                for (size_t r = 0; r < rounds && !result; r++)
                    for (size_t p = 0; p < 4 && !result; p++)
                        result = mojo_measure(fn, &f, pool, group, p == 1 || p == 2,
                            iterations, r, p, kinds[k], "warm");
            }
            mojo_fixture_free(&f);
        }
    }
    if (affinity_saved) janas_cpus_restore(saved);
    janas_pool_destroy(pool);
    janas_mojo_set_enabled(-1);
    return result;
}

/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Recurring model-free coverage for matrix reference validation/statistics,
   exact metric reuse, directed rounding and native FPCR/MXCSR invalidation.
   Compile the helper's dependency-free fixture entry, without a model/runtime. */
#define QUALITY_CHECK_SELFTEST_ONLY
#define main quality_matrix_selftest_main
#include "quality_check.c"
#undef main

int main(void)
{
    int rc = quality_matrix_selftest_main();
    if (!rc) puts("test_quality_matrix: ok");
    return rc;
}

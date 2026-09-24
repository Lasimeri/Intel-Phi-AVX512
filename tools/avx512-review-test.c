/* avx512-review-test.c: the transparent path against the defects a review
 * of 2026-09-24 described, each as an ordinary AVX-512 program checked
 * lane for lane against a scalar reference compiled without AVX-512 (so
 * the reference runs natively and is never intercepted). See
 * avx512-review-test.md.
 *
 *   gcc -O2 -mavx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq \
 *       -o /tmp/avx512-review-test tools/avx512-review-test.c
 *   scripts/phi512.sh /tmp/avx512-review-test
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SENTINEL 12345.0f

static int failures;

static void report(const char *name, long bad, long n)
{
    printf("%-44s %s (%ld of %ld wrong)\n", name, bad ? "FAIL" : "ok", bad, n);
    if (bad) failures++;
}

/* 1. A split loop whose trip count leaves threads without a slice: every
 * trip count from 40 to 200 vectors, each element incremented once, and
 * the floats past the end left alone. */
__attribute__((noinline)) static void add_one(float *a, long n)
{
    const __m512 one = _mm512_set1_ps(1.0f);
    for (long i = 0; i < n; i += 16)
        _mm512_storeu_ps(a + i, _mm512_add_ps(_mm512_loadu_ps(a + i), one));
}

__attribute__((target("no-avx512f"))) static long check_add_one(void)
{
    long bad = 0;
    for (long v = 40; v <= 200; v++) {
        long n = v * 16;
        float *a = aligned_alloc(64, (size_t)(n + 64) * sizeof(float));
        for (long i = 0; i < n + 64; i++) a[i] = (float)i;
        add_one(a, n);
        for (long i = 0; i < n + 64; i++) {
            float want = i < n ? (float)i + 1.0f : (float)i;
            if (a[i] != want) bad++;
        }
        free(a);
    }
    return bad;
}

/* 2. A masked store over a large output: only the lanes where x > 0 are
 * written; the others must keep what they held. */
__attribute__((noinline)) static void store_positive(float *out, const float *x, long n)
{
    for (long i = 0; i < n; i += 16) {
        __m512 v = _mm512_load_ps(x + i);
        __mmask16 k = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ);
        _mm512_mask_store_ps(out + i, k, v);
    }
}

__attribute__((target("no-avx512f"))) static long check_store_positive(void)
{
    long n = 65536, bad = 0;
    float *x = aligned_alloc(64, (size_t)n * sizeof(float));
    float *out = aligned_alloc(64, (size_t)n * sizeof(float));
    for (long i = 0; i < n; i++) {
        x[i] = (i % 3 == 0) ? -1.0f - (float)i : (float)i + 1.0f;
        out[i] = SENTINEL;
    }
    store_positive(out, x, n);
    for (long i = 0; i < n; i++) {
        float want = x[i] > 0 ? x[i] : SENTINEL;
        if (out[i] != want) bad++;
    }
    free(x);
    free(out);
    return bad;
}

/* 3. Masked unaligned moves with a mask that is not a prefix (0xAAAA:
 * the odd lanes). x86 loads and stores lane j at address + 4j. */
__attribute__((noinline)) static void masked_unaligned(float *dst, const float *src, float *loaded)
{
    __mmask16 k = 0xAAAA;
    __m512 v = _mm512_mask_loadu_ps(_mm512_set1_ps(-7.0f), k, src);
    _mm512_storeu_ps(loaded, v);
    _mm512_mask_storeu_ps(dst, k, _mm512_loadu_ps(src));
}

__attribute__((target("no-avx512f"))) static long check_masked_unaligned(long *nlanes)
{
    float *buf = aligned_alloc(64, 256 * sizeof(float));
    float *src = buf + 1, *dst = buf + 65, loaded[16];
    long bad = 0;
    for (int i = 0; i < 256; i++) buf[i] = SENTINEL;
    for (int j = 0; j < 16; j++) src[j] = (float)(100 + j);
    masked_unaligned(dst, src, loaded);
    for (int j = 0; j < 16; j++) {
        float want_load = (j & 1) ? (float)(100 + j) : -7.0f;
        float want_store = (j & 1) ? (float)(100 + j) : SENTINEL;
        if (loaded[j] != want_load) bad++;
        if (dst[j] != want_store) bad++;
    }
    *nlanes = 32;
    free(buf);
    return bad;
}

int main(void)
{
    long lanes = 0;
    report("split loops of 40 to 200 vectors", check_add_one(), 161);
    report("masked store over 256 KiB", check_store_positive(), 65536);
    long bad = check_masked_unaligned(&lanes);
    report("masked unaligned load and store, k=0xAAAA", bad, lanes);
    printf("%s\n", failures ? "FAILED" : "all agree");
    return failures ? 1 : 0;
}

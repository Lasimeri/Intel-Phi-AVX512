/* vpu_matmul.c: whole matrix multiplies for the ggml backend, across the
 * pool. The host uploads a program's weight tensors once (UPLOAD, kept by
 * id in the card's memory), then each MUL_MAT of the model is one MATMUL
 * request: d[n][m] = a[m][k] . b[n][k]. The rows of a are split across
 * the threads; each dot product is a kernel of vpu_matmul_kernel.S (the
 * card's 16-lane fused multiply-adds, weights up-converted from float16
 * by the load itself), whose 16 partial sums are added here.
 *
 * b and d move through the window each time (n rows: the batch of
 * tokens, small); a moves once per tensor for the model's weights, or per
 * request for a tensor the host does not keep (a_id 0). See vpu_matmul.md. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include "vpu_proto.h"
#include "vpu_matmul.h"

int vpu_pull(void *dst, size_t len, uint64_t off);
int vpu_push(const void *src, size_t len, uint64_t off);
int vpu_pool_map(void (*fn)(void *arg, int slice, int nslices), void *arg, int nslices);
int vpu_pool_threads(void);
void phi_dot_f16(const void *a, const float *b, long k16, float *out);
void phi_dot_f32(const void *a, const float *b, long k16, float *out);

#define CACHE_MAX 4096
static struct { uint64_t id; void *p; size_t bytes; } g_cache[CACHE_MAX];
static int g_ncache;

/* Streaming buffers, grown as needed. */
static struct { void *p; size_t cap; } g_a, g_b, g_d;

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Memory for a tensor or a buffer: huge pages when the card has them
 * (the DMA is faster with them, vpu_worker.md), else 4 KiB pages. */
static void *big_alloc(size_t bytes)
{
    size_t huge = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void *p = mmap(NULL, huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) return p;
    size_t small = (bytes + 4095) & ~(size_t)4095;
    p = mmap(NULL, small, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

static void big_free(void *p, size_t bytes)
{
    size_t huge = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    munmap(p, huge);
}

static int grow(void *slot, size_t bytes)
{
    struct { void *p; size_t cap; } *b = slot;
    if (b->cap >= bytes) return 0;
    if (b->p) big_free(b->p, b->cap);
    b->cap = 0;
    b->p = big_alloc(bytes);
    if (!b->p) return -1;
    b->cap = bytes;
    return 0;
}

static int cache_find(uint64_t id)
{
    for (int i = 0; i < g_ncache; i++)
        if (g_cache[i].id == id) return i;
    return -1;
}

static void cache_drop(int i)
{
    big_free(g_cache[i].p, g_cache[i].bytes);
    g_cache[i] = g_cache[--g_ncache];
}

/* One float from a half (round to nearest even is not needed: exact). */
static float half_to_float(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {
            /* subnormal half: normalise */
            int sh = 0;
            while ((m & 0x400) == 0) { m <<= 1; sh++; }
            m &= 0x3ff;
            bits = s | ((uint32_t)(113 - sh) << 23) | (m << 13);
        }
    } else if (e == 31) bits = s | 0x7f800000 | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

struct job {
    const unsigned char *a;
    const unsigned char *b;
    float *d;
    uint64_t m, n, k, nb_a, nb_b;
    int f16;
};

static void rows_slice(void *arg, int slice, int nslices)
{
    struct job *j = arg;
    uint64_t i0 = j->m * (uint64_t)slice / (uint64_t)nslices, i1 = j->m * (uint64_t)(slice + 1) / (uint64_t)nslices;
    long k16 = (long)(j->k / 16);
    uint64_t tail = j->k % 16;
    float out[16] __attribute__((aligned(64)));
    for (uint64_t i = i0; i < i1; i++) {
        const unsigned char *arow = j->a + i * j->nb_a;
        for (uint64_t c = 0; c < j->n; c++) {
            const float *brow = (const float *)(j->b + c * j->nb_b);
            float sum = 0.0f;
            if (k16 > 0) {
                if (j->f16) phi_dot_f16(arow, brow, k16, out);
                else phi_dot_f32(arow, brow, k16, out);
                for (int l = 0; l < 16; l++) sum += out[l];
            }
            for (uint64_t t = 0; t < tail; t++) {
                uint64_t x = (uint64_t)k16 * 16 + t;
                float av = j->f16 ? half_to_float(((const uint16_t *)arow)[x]) : ((const float *)arow)[x];
                sum += av * brow[x];
            }
            j->d[c * j->m + i] = sum;
        }
    }
}

static size_t blocks(uint64_t bytes)
{
    return (size_t)((bytes + VPU_BLOCK - 1) & ~(uint64_t)(VPU_BLOCK - 1));
}

int vpu_matmul_run(volatile unsigned char *ctrl, uint32_t kernel, int threads, int verbose,
                   uint64_t *compute_ns, uint64_t *pull_ns, uint64_t *push_ns, int *live)
{
    struct vpu_matmul mm;
    memcpy(&mm, (const void *)(ctrl + VPU_OFF_MATMUL), sizeof mm);
    *compute_ns = *pull_ns = *push_ns = 0;
    *live = 0;

    if (kernel == VPU_K_FREE) {
        if (mm.a_id == 0) {
            while (g_ncache > 0) cache_drop(0);
        } else {
            int i = cache_find(mm.a_id);
            if (i >= 0) cache_drop(i);
        }
        return VPU_OK;
    }
    if (kernel == VPU_K_UPLOAD) {
        if (mm.a_id == 0 || mm.bytes == 0 || mm.a_off % VPU_BLOCK != 0) return VPU_E_REQUEST;
        int i = cache_find(mm.a_id);
        if (i >= 0) cache_drop(i);
        if (g_ncache >= CACHE_MAX) return VPU_E_ALLOC;
        size_t len = blocks(mm.bytes);
        void *p = big_alloc(len);
        if (!p) return VPU_E_ALLOC;
        uint64_t t0 = now_ns();
        if (vpu_pull(p, len, mm.a_off) != 0) { big_free(p, len); return VPU_E_PULL; }
        *pull_ns = now_ns() - t0;
        g_cache[g_ncache].id = mm.a_id;
        g_cache[g_ncache].p = p;
        g_cache[g_ncache].bytes = len;
        g_ncache++;
        if (verbose) { printf("matmul: upload id %llu, %llu bytes\n", (unsigned long long)mm.a_id, (unsigned long long)mm.bytes); fflush(stdout); }
        return VPU_OK;
    }
    /* MATMUL */
    if (mm.m == 0 || mm.n == 0 || mm.k == 0 || mm.b_off % VPU_BLOCK != 0 || mm.d_off % VPU_BLOCK != 0) return VPU_E_REQUEST;
    if (mm.a_type != VPU_MM_F32 && mm.a_type != VPU_MM_F16) return VPU_E_REQUEST;
    const unsigned char *a;
    uint64_t t0 = now_ns();
    if (mm.a_id != 0) {
        int i = cache_find(mm.a_id);
        if (i < 0) return VPU_E_REQUEST;
        if (mm.m * mm.nb_a > g_cache[i].bytes) return VPU_E_REQUEST;
        a = g_cache[i].p;
    } else {
        if (mm.a_off % VPU_BLOCK != 0) return VPU_E_REQUEST;
        size_t len = blocks(mm.m * mm.nb_a);
        if (grow(&g_a, len) != 0) return VPU_E_ALLOC;
        if (vpu_pull(g_a.p, len, mm.a_off) != 0) return VPU_E_PULL;
        a = g_a.p;
    }
    size_t blen = blocks(mm.n * mm.nb_b), dlen = blocks(mm.n * mm.m * 4);
    if (grow(&g_b, blen) != 0 || grow(&g_d, dlen) != 0) return VPU_E_ALLOC;
    if (vpu_pull(g_b.p, blen, mm.b_off) != 0) return VPU_E_PULL;
    *pull_ns = now_ns() - t0;

    struct job j = { a, g_b.p, g_d.p, mm.m, mm.n, mm.k, mm.nb_a, mm.nb_b, mm.a_type == VPU_MM_F16 };
    int max = vpu_pool_threads() + 1;
    if (threads < 1) threads = 1;
    if (threads > max) threads = max;
    if ((uint64_t)threads > mm.m) threads = (int)mm.m;
    uint64_t c0 = now_ns();
    *live = vpu_pool_map(rows_slice, &j, threads);
    *compute_ns = now_ns() - c0;

    uint64_t p0 = now_ns();
    if (vpu_push(g_d.p, dlen, mm.d_off) != 0) return VPU_E_PUSH;
    *push_ns = now_ns() - p0;
    if (verbose) {
        printf("matmul: %llux%llu . %llux%llu (%s%s) on %d threads: pull %.3f compute %.3f push %.3f ms\n",
               (unsigned long long)mm.m, (unsigned long long)mm.k, (unsigned long long)mm.n, (unsigned long long)mm.k,
               mm.a_type == VPU_MM_F16 ? "f16" : "f32", mm.a_id ? ", cached" : "", *live,
               *pull_ns / 1e6, *compute_ns / 1e6, *push_ns / 1e6);
        fflush(stdout);
    }
    return VPU_OK;
}

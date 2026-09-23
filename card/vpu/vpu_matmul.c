/* vpu_matmul.c: whole matrix multiplies for the ggml backend, across the
 * pool. The host uploads a program's weight tensors once (UPLOAD, kept by
 * id in the card's memory), then each MUL_MAT of the model is one MATMUL
 * request: d[n][m] = a[m][k] . b[n][k]. The rows of a are split across
 * the threads; each dot product is a kernel of vpu_matmul_kernel.S (the
 * card's 16-lane fused multiply-adds), whose 16 partial sums are added
 * here.
 *
 * Float weights: float16 up-converted by the load itself, or float32,
 * one row against one or four activation rows (phi_dot*_f16/f32).
 * Quantized weights (llama.cpp's Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS): one
 * 256-weight superblock against 1, 4 or 8 activation rows per call
 * (phi_<fmt>_<rows>), the format's scalar part (scales, minimums, the
 * float16 block scales) decoded here into a 256-byte table per
 * superblock before the vector kernel runs; the reference each format
 * reproduces is ggml-quants.c's dequantize_row_<fmt>.
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
void *vpu_window(uint64_t off, size_t len);
int vpu_pool_map(void (*fn)(void *arg, int slice, int nslices), void *arg, int nslices);
int vpu_pool_threads(void);
void phi_dot_f16(const void *a, const float *b, long k16, float *out);
void phi_dot_f32(const void *a, const float *b, long k16, float *out);
void phi_dot4_f16(const void *a, const float *b, long k16, float *out, uint64_t nb);
void phi_dot4_f32(const void *a, const float *b, long k16, float *out, uint64_t nb);

/* The quantized kernels: one superblock (256 weights) of a row against
 * 1, 4 or 8 activation rows (kernelgen/quant.md has the convention). */
typedef void (*qkern)(const uint8_t *blk, const float *x, uint64_t xstride, float *scratch, float *acc, const float *consts, uint64_t next);
#define QK(fmt) \
    void phi_##fmt##_1(const uint8_t *, const float *, uint64_t, float *, float *, const float *, uint64_t); \
    void phi_##fmt##_4(const uint8_t *, const float *, uint64_t, float *, float *, const float *, uint64_t); \
    void phi_##fmt##_8(const uint8_t *, const float *, uint64_t, float *, float *, const float *, uint64_t);
QK(q4k) QK(q5k) QK(q6k) QK(q8_0) QK(iq4xs)
void phi_probe(const uint8_t *blk, const float *x, const float *consts, float *out);
void phi_bench(long kind, const void *buf, long count);

/* A mapping: huge pages when the card had some left, else 4 KiB pages;
 * `len` is what was mapped, so the unmap matches (a free of the rounded
 * huge length over a small-page mapping took the pool threads' stacks
 * with it once the huge pages ran out, 2026-09-23). */
struct mapping { void *p; size_t len; int huge; };

#define CACHE_MAX 4096
static struct { uint64_t id; struct mapping m; size_t bytes; } g_cache[CACHE_MAX];
static int g_ncache;

/* Streaming buffers, grown as needed. */
static struct { struct mapping m; size_t cap; } g_a, g_b, g_d;

/* Every buffer carries this much past its data: the unaligned load pairs
 * of the quantized kernels read up to 63 bytes beyond a block. */
#define SLACK 64

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Memory for a tensor or a buffer: huge pages when the card has them
 * (the DMA is faster with them, vpu_worker.md), else 4 KiB pages. */
static struct mapping big_alloc(size_t bytes)
{
    struct mapping m = { NULL, 0, 0 };
    bytes += SLACK;
    size_t huge = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void *p = mmap(NULL, huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) { m.p = p; m.len = huge; m.huge = 1; return m; }
    size_t small = (bytes + 4095) & ~(size_t)4095;
    p = mmap(NULL, small, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) { m.p = p; m.len = small; }
    return m;
}

static void big_free(struct mapping *m)
{
    if (m->p) munmap(m->p, m->len);
    m->p = NULL;
    m->len = 0;
}

static int grow(void *slot, size_t bytes)
{
    struct { struct mapping m; size_t cap; } *b = slot;
    if (b->cap >= bytes) return 0;
    big_free(&b->m);
    b->cap = 0;
    b->m = big_alloc(bytes);
    if (!b->m.p) return -1;
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
    big_free(&g_cache[i].m);
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

static float half_at(const uint8_t *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
    return half_to_float(h);
}

/* ------------------------------------------------------------------ */
/* The quantized formats' scalar side: the table per superblock         */

/* The kernels' scratch (kernelgen/quant.rs, TAB_*): the superblock's 16
 * scales and 16 minuends, which the kernel itself decodes from the block
 * and writes here; and the constants every call shares (C_*, bytes):
 * 16, 4, 2, 2^-1..2^-8, the IQ4_XS values, the index and shift vectors
 * of the scale decoding, a few integers. */
#define TAB_FLOATS 32
#define C_BYTES 832

static unsigned char g_consts[1024] __attribute__((aligned(64)));

static void put_f32(int off, float f) { memcpy(g_consts + off, &f, 4); }
static void put_u32(int off, uint32_t u) { memcpy(g_consts + off, &u, 4); }
static void put_vec(int off, const uint32_t v[16]) { memcpy(g_consts + off, v, 64); }

static void consts_init(void)
{
    static const float lut[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
    static const uint32_t expand[16] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 };
    static const uint32_t expand_hi[16] = { 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15 };
    static const uint32_t k4_p1[16] = { 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11 };
    static const uint32_t k4_p2[16] = { 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 7 };
    static const uint32_t d_dmin[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 };
    static const uint32_t zero[16] = { 0 };
    static const uint32_t one[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    static const uint32_t sl[16] = { 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0 };
    static const uint32_t shift_l[16] = { 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0 };
    static const uint32_t shift_h[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0 };
    put_f32(0, 16.0f);
    put_f32(4, 4.0f);
    put_f32(8, 2.0f);
    for (int j = 1; j <= 8; j++) put_f32(12 + 4 * (j - 1), 1.0f / (float)(1 << j));
    memcpy(g_consts + 64, lut, sizeof lut);
    put_vec(128, expand);
    put_vec(192, expand_hi);
    put_vec(256, k4_p1);
    put_vec(320, k4_p2);
    put_vec(384, d_dmin);
    put_vec(448, zero);
    put_vec(512, one);
    put_vec(576, sl);
    put_vec(640, shift_l);
    put_vec(704, shift_h);
    put_f32(768, 32.0f);
    put_u32(772, 63);
    put_u32(776, 15);
    put_u32(780, 0xc0);
    put_u32(784, 3);
    put_u32(788, 32);
}

static const struct qfmt {
    unsigned block_bytes;       /* per 256 weights */
    qkern k1, k4, k8;
} g_fmt[VPU_MM_TYPES] = {
    [VPU_MM_Q4_K] = { 144, phi_q4k_1, phi_q4k_4, phi_q4k_8 },
    [VPU_MM_Q5_K] = { 176, phi_q5k_1, phi_q5k_4, phi_q5k_8 },
    [VPU_MM_Q6_K] = { 210, phi_q6k_1, phi_q6k_4, phi_q6k_8 },
    [VPU_MM_Q8_0] = { 272, phi_q8_0_1, phi_q8_0_4, phi_q8_0_8 },
    [VPU_MM_IQ4_XS] = { 136, phi_iq4xs_1, phi_iq4xs_4, phi_iq4xs_8 },
};

static int quantized(uint32_t t) { return t < VPU_MM_TYPES && g_fmt[t].k1 != NULL; }

/* Q8_0 is the one format whose row may end with fewer than eight blocks
 * (k a multiple of 32, not 256): those blocks are done here. */
static float q8_0_tail(const uint8_t *blk, const float *x, int nblocks)
{
    float sum = 0.0f;
    for (int i = 0; i < nblocks; i++) {
        float d = half_at(blk + 34 * i);
        const int8_t *q = (const int8_t *)(blk + 34 * i + 2);
        float s = 0.0f;
        for (int j = 0; j < 32; j++) s += (float)q[j] * x[32 * i + j];
        sum += d * s;
    }
    return sum;
}

/* Rows go in chunks so that a chunk's weights and one group of activation
 * rows stay in the core's 512 KiB L2 together. The chunk is what the
 * host's `reserved[0]` overrides (phi-vpu matmul-check --chunk), which is
 * how the shape was measured rather than argued. */
#define ROW_CHUNK 32
#define ROW_CHUNK_MAX 64
#define POOL_SLOTS 58

struct job {
    const unsigned char *a;
    const unsigned char *b;
    float *d;
    uint64_t m, n, k, nb_a, nb_b;
    uint32_t type;
    uint32_t chunk;
    /* MUL_MAT_ID: the n columns each pick an expert. Column p (which is
     * j + t * n_used) multiplies expert ids[p], whose m rows start at
     * a + ids[p] * a_stride, by b's row (j % b_rows) + t * b_rows. */
    const int32_t *ids;
    uint64_t n_used, b_rows, a_stride;
};

static float sum16(const float *v)
{
    float s = 0.0f;
    for (int l = 0; l < 16; l++) s += v[l];
    return s;
}

static int group_of(uint64_t left) { return left >= 8 ? 8 : left >= 4 ? 4 : 1; }

/* Rows i0..i1 of a quantized tensor against all n activation rows: chunks
 * of ROW_CHUNK rows; within a chunk the activation rows go in groups of
 * 8, 4 or 1 (the kernels' variants), the group outermost so its rows are
 * read from L2 for every weight row of the chunk. The first group is
 * computed superblock by superblock right after the superblock's table
 * is prepared, so the kernels' prefetch of the superblocks ahead covers
 * the scalar preparation too; later groups reuse the tables. */
/* Which rows a slice takes. The pool pins thread t to core t % 57, slot
 * t / 57 (vpu_worker.c), and the dispatcher, slot 3 of core 56, takes the
 * last slice: so with 57 slices every core has one, and with 114 or 228
 * the slices s and s + 57 (and s + 114, s + 171) share a core. Threads
 * sharing a core work on the same chunk of rows, at the same superblock,
 * each on every nmates-th row: the activation block they read is one and
 * the same in their shared L1, which is what lets the second hardware
 * thread fill the vector unit's issue slots instead of thrashing. */
static void slice_rows(int slice, int nslices, uint64_t m, uint64_t *i0, uint64_t *i1, int *mate, int *nmates)
{
    int ncores = nslices, core = slice;
    *mate = 0;
    *nmates = 1;
    if (nslices > 57 && nslices % 57 == 0) {
        ncores = 57;
        core = slice % 57;
        *mate = slice / 57;
        *nmates = nslices / 57;
    }
    *i0 = m * (uint64_t)core / (uint64_t)ncores;
    *i1 = m * (uint64_t)(core + 1) / (uint64_t)ncores;
}

static void rows_slice_q(void *arg, int slice, int nslices)
{
    struct job *j = arg;
    const struct qfmt *f = &g_fmt[j->type];
    uint64_t i0, i1;
    int mate, nmates;
    slice_rows(slice, nslices, j->m, &i0, &i1, &mate, &nmates);
    uint64_t ns = j->k / 256, tail = j->k % 256;   /* tail: Q8_0 only */
    unsigned bb = f->block_bytes;
    const float *cs = (const float *)g_consts;
    float scratch[TAB_FLOATS] __attribute__((aligned(64)));
    /* the accumulators of this thread's rows of the chunk, T vectors each, packed */
    float acc[ROW_CHUNK_MAX * 8][16] __attribute__((aligned(64)));
    uint64_t chunk = j->chunk;
    for (uint64_t i = i0; i < i1; i += chunk) {
        uint64_t rows = i1 - i < chunk ? i1 - i : chunk;
        uint64_t c = 0;
        while (c < j->n) {
            int T = group_of(j->n - c);
            qkern kern = T == 8 ? f->k8 : T == 4 ? f->k4 : f->k1;
            const float *x = (const float *)(j->b + c * j->nb_b);
            const uint8_t *base = (const uint8_t *)j->a + i * j->nb_a;
            uint64_t mine = 0;
            for (uint64_t r = mate; r < rows; r += nmates) mine++;
            memset(acc, 0, mine * (size_t)T * 64);
            /* superblock outermost: its activation block and the accumulators
             * stay in L1 across the rows; the weights stream */
            for (uint64_t s = 0; s < ns; s++) {
                float *a = acc[0];
                for (uint64_t r = mate; r < rows; r += nmates, a += T * 16)
                    kern(base + r * j->nb_a + s * bb, x + s * 256, j->nb_b, scratch, a, cs, (uint64_t)nmates * j->nb_a);
            }
            float *a = acc[0];
            for (uint64_t r = mate; r < rows; r += nmates, a += T * 16) {
                for (int q = 0; q < T; q++) {
                    float sum = sum16(a + q * 16);
                    if (tail)
                        sum += q8_0_tail(base + r * j->nb_a + ns * bb, x + (size_t)q * (j->nb_b / 4) + ns * 256, (int)(tail / 32));
                    j->d[(c + q) * j->m + i + r] = sum;
                }
            }
            c += T;
        }
    }
}

static void rows_slice(void *arg, int slice, int nslices)
{
    struct job *j = arg;
    if (quantized(j->type)) { rows_slice_q(arg, slice, nslices); return; }
    int f16 = j->type == VPU_MM_F16;
    uint64_t i0 = j->m * (uint64_t)slice / (uint64_t)nslices, i1 = j->m * (uint64_t)(slice + 1) / (uint64_t)nslices;
    long k16 = (long)(j->k / 16);
    uint64_t tail = j->k % 16;
    float out[4][16] __attribute__((aligned(64)));
    for (uint64_t i = i0; i < i1; i++) {
        const unsigned char *arow = j->a + i * j->nb_a;
        uint64_t c = 0;
        /* Four columns at a time: the weight row is read once for the four. */
        for (; c + 4 <= j->n && k16 > 0; c += 4) {
            const float *brow = (const float *)(j->b + c * j->nb_b);
            if (f16) phi_dot4_f16(arow, brow, k16, out[0], j->nb_b);
            else phi_dot4_f32(arow, brow, k16, out[0], j->nb_b);
            for (int q = 0; q < 4; q++) {
                float sum = sum16(out[q]);
                const float *b1 = (const float *)(j->b + (c + q) * j->nb_b);
                for (uint64_t t = 0; t < tail; t++) {
                    uint64_t x = (uint64_t)k16 * 16 + t;
                    float av = f16 ? half_to_float(((const uint16_t *)arow)[x]) : ((const float *)arow)[x];
                    sum += av * b1[x];
                }
                j->d[(c + q) * j->m + i] = sum;
            }
        }
        for (; c < j->n; c++) {
            const float *brow = (const float *)(j->b + c * j->nb_b);
            float sum = 0.0f;
            if (k16 > 0) {
                if (f16) phi_dot_f16(arow, brow, k16, out[0]);
                else phi_dot_f32(arow, brow, k16, out[0]);
                sum = sum16(out[0]);
            }
            for (uint64_t t = 0; t < tail; t++) {
                uint64_t x = (uint64_t)k16 * 16 + t;
                float av = f16 ? half_to_float(((const uint16_t *)arow)[x]) : ((const float *)arow)[x];
                sum += av * brow[x];
            }
            j->d[c * j->m + i] = sum;
        }
    }
}

/* A slice that does nothing: the probe times the dispatch itself. */
static void noop_slice(void *arg, int slice, int nslices)
{
    (void)arg; (void)slice; (void)nslices;
}

/* The whole pool running phi_bench at once, each thread on its own
 * region of the buffer: the card's aggregate rates, which are what a
 * matrix multiply is measured against (one thread's rate times 57 is
 * not: the memory system is shared, the vector units are not). */
struct bench_job { long kind; unsigned char *buf; size_t per; long count; };

static void bench_slice(void *arg, int slice, int nslices)
{
    struct bench_job *b = arg;
    (void)nslices;
    phi_bench(b->kind, b->buf + (size_t)slice * b->per, b->count);
}

static uint64_t bench_pool(long kind, unsigned char *buf, size_t per, long count, int threads)
{
    struct bench_job b = { kind, buf, per, count };
    uint64_t t0 = now_ns();
    vpu_pool_map(bench_slice, &b, threads);
    return now_ns() - t0;
}

/* One expert per column: the thread's rows of expert ids[p], for every
 * column p, against that column's activation row. A column at a time
 * (the one-row kernels), which is what generation asks for anyway: there
 * every column is a different expert of the same token. */
static void rows_slice_id(void *arg, int slice, int nslices)
{
    struct job *j = arg;
    struct job one = *j;
    one.n = 1;
    for (uint64_t p = 0; p < j->n; p++) {
        int32_t e = j->ids[p];
        if (e < 0) continue;
        uint64_t t = p / j->n_used, col = p % j->n_used;
        one.a = j->a + (uint64_t)e * j->a_stride;
        one.b = j->b + ((col % j->b_rows) + t * j->b_rows) * j->nb_b;
        one.d = j->d + p * j->m;
        rows_slice(&one, slice, nslices);
    }
}

static size_t blocks(uint64_t bytes)
{
    return (size_t)((bytes + VPU_BLOCK - 1) & ~(uint64_t)(VPU_BLOCK - 1));
}

/* What a MATMUL must satisfy for its type, beyond the window rules. */
static int shape_ok(const struct vpu_matmul *mm)
{
    if (mm->a_type == VPU_MM_F32 || mm->a_type == VPU_MM_F16) return 1;
    if (!quantized(mm->a_type)) return 0;
    if (mm->k % 256 != 0 && !(mm->a_type == VPU_MM_Q8_0 && mm->k % 32 == 0)) return 0;
    if (mm->nb_b % 64 != 0) return 0;   /* the activation rows are memory operands of the kernels */
    if (mm->nb_a < mm->k / 256 * g_fmt[mm->a_type].block_bytes + (mm->k % 256) / 32 * 34) return 0;
    if ((mm->a_type == VPU_MM_Q4_K || mm->a_type == VPU_MM_Q5_K) && mm->nb_a % 16 != 0) return 0;
    return 1;
}

int vpu_matmul_run(volatile unsigned char *ctrl, uint32_t kernel, int threads, int verbose,
                   uint64_t *compute_ns, uint64_t *pull_ns, uint64_t *push_ns, int *live)
{
    struct vpu_matmul mm;
    memcpy(&mm, (const void *)(ctrl + VPU_OFF_MATMUL), sizeof mm);
    *compute_ns = *pull_ns = *push_ns = 0;
    *live = 0;
    if (g_consts[0] == 0) consts_init();

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
        struct mapping m = big_alloc(len);
        if (!m.p) return VPU_E_ALLOC;
        uint64_t t0 = now_ns();
        if (vpu_pull(m.p, len, mm.a_off) != 0) { big_free(&m); return VPU_E_PULL; }
        *pull_ns = now_ns() - t0;
        g_cache[g_ncache].id = mm.a_id;
        g_cache[g_ncache].m = m;
        g_cache[g_ncache].bytes = len;
        g_ncache++;
        if (verbose) { printf("matmul: upload id %llu, %llu bytes\n", (unsigned long long)mm.a_id, (unsigned long long)mm.bytes); fflush(stdout); }
        return VPU_OK;
    }
    /* MATMUL */
    if (verbose && (mm.m == 0 || mm.n == 0 || mm.k == 0 || mm.b_off % VPU_BLOCK != 0 || mm.d_off % VPU_BLOCK != 0 || !shape_ok(&mm))) {
        printf("matmul: rejected: type %u m %llu n %llu k %llu nb_a %llu nb_b %llu a_id %llu a_off %llu b_off %llu d_off %llu\n", mm.a_type,
               (unsigned long long)mm.m, (unsigned long long)mm.n, (unsigned long long)mm.k, (unsigned long long)mm.nb_a, (unsigned long long)mm.nb_b,
               (unsigned long long)mm.a_id, (unsigned long long)mm.a_off, (unsigned long long)mm.b_off, (unsigned long long)mm.d_off);
        fflush(stdout);
    }
    if (mm.m == 0 || mm.n == 0 || mm.k == 0 || mm.b_off % VPU_BLOCK != 0 || mm.d_off % VPU_BLOCK != 0) return VPU_E_REQUEST;
    if (kernel == VPU_K_MATMUL_ID && (mm.n_used == 0 || mm.b_rows == 0 || mm.n_tokens == 0 || mm.n != mm.n_used * mm.n_tokens || mm.ids_bytes % 64 != 0 || mm.a_id == 0)) return VPU_E_REQUEST;
    if (mm.a_type == VPU_MM_PROBE) {
        if (mm.a_off % VPU_BLOCK != 0) return VPU_E_REQUEST;
        if (grow(&g_a, VPU_BLOCK) != 0 || grow(&g_b, VPU_BLOCK) != 0 || grow(&g_d, VPU_BLOCK) != 0) return VPU_E_ALLOC;
        if (vpu_pull(g_a.m.p, VPU_BLOCK, mm.a_off) != 0 || vpu_pull(g_b.m.p, VPU_BLOCK, mm.b_off) != 0) return VPU_E_PULL;
        phi_probe(g_a.m.p, g_b.m.p, (const float *)g_consts, g_d.m.p);
        if (vpu_push(g_d.m.p, VPU_BLOCK, mm.d_off) != 0) return VPU_E_PUSH;
        /* the raw rates, on this thread: issue (compute_ns), then the streaming
         * variants of phi_bench, their times in ns as 8 u64 at d + 512 */
        uint64_t b0 = now_ns();
        phi_bench(0, NULL, 0);
        *compute_ns = now_ns() - b0;
        /* one region per thread for the pool-wide walks, and 64 MiB for
         * the single-thread ones */
        size_t per = 4u << 20, want = (size_t)threads * per;
        if (want < (64u << 20)) want = 64u << 20;
        if (grow(&g_a, want) != 0) return VPU_E_ALLOC;
        memset(g_a.m.p, 0, want);
        uint64_t *times = (uint64_t *)((unsigned char *)g_d.m.p + 512);
        for (long kind = 1; kind <= 5; kind++) {
            b0 = now_ns();
            phi_bench(kind, g_a.m.p, 0);
            times[kind] = now_ns() - b0;
        }
        /* kind 1 again on a 256 KiB walk (L2 resident): 1 M loads over the same 4096 lines */
        b0 = now_ns();
        for (int rep = 0; rep < 256; rep++) phi_bench(6, g_a.m.p, 0);
        times[6] = now_ns() - b0;
        /* the Q4_K kernels on one superblock held in L1 (compute only), 100 k calls,
         * one activation row and eight; then walking 64 MiB of weights, one row */
        {
            float tab[TAB_FLOATS] __attribute__((aligned(64))), acc[8][16] __attribute__((aligned(64)));
            const float *x = g_b.m.p, *cs = (const float *)g_consts;
            memset(acc, 0, sizeof acc);
            b0 = now_ns();
            for (int rep = 0; rep < 100000; rep++) phi_q4k_1(g_a.m.p, x, 1024, tab, acc[0], cs, 144);
            times[7] = now_ns() - b0;
            b0 = now_ns();
            for (int rep = 0; rep < 100000; rep++) phi_q4k_8(g_a.m.p, x, 1024, tab, acc[0], cs, 144);
            times[8] = now_ns() - b0;
            b0 = now_ns();
            for (int rep = 0; rep < 100000; rep++) phi_q4k_1(g_a.m.p + (size_t)rep * 144, x, 1024, tab, acc[0], cs, 144);
            times[9] = now_ns() - b0;
            b0 = now_ns();
            for (int rep = 0; rep < 100000; rep++) phi_q5k_1(g_a.m.p + (size_t)rep * 176, x, 1024, tab, acc[0], cs, 176);
            times[10] = now_ns() - b0;
        }
        /* The two ceilings, with every thread of the request working: the
         * card's aggregate read bandwidth (4 MiB per thread, prefetched)
         * and its aggregate vector issue rate (register fused
         * multiply-adds, nothing from memory). A multiply cannot beat
         * either; which one it is under tells what to work on. */
        times[12] = bench_pool(3, g_a.m.p, per, (long)(per / 64), threads);
        times[13] = bench_pool(0, NULL, 0, 1000000, threads);
        /* The two ways bytes cross the link, at the size a token's
         * activations and results are: the block device (a request each
         * way, whose cost is nearly all fixed) and the window mapped
         * straight into the card's address space (no request, but every
         * access is a link round trip). 100 rounds of 16 KiB each. */
        {
            void *win = vpu_window(mm.b_off, 65536);
            uint64_t n = 100, bytes = 16384;
            b0 = now_ns();
            for (uint64_t r = 0; r < n; r++) vpu_pull(g_b.m.p, bytes, mm.b_off);
            times[14] = now_ns() - b0;
            b0 = now_ns();
            for (uint64_t r = 0; r < n; r++) vpu_push(g_d.m.p, bytes, mm.d_off);
            times[15] = now_ns() - b0;
            if (win) {
                b0 = now_ns();
                for (uint64_t r = 0; r < n; r++) memcpy(g_b.m.p, win, bytes);
                times[16] = now_ns() - b0;
                b0 = now_ns();
                for (uint64_t r = 0; r < n; r++) memcpy(win, g_d.m.p, bytes);
                times[17] = now_ns() - b0;
            }
        }
        /* What one dispatch across `threads` threads costs with nothing to
         * do: the pool's fixed cost per request, which a small multiply
         * pays in full (1000 rounds). */
        b0 = now_ns();
        for (int rep = 0; rep < 1000; rep++) vpu_pool_map(noop_slice, NULL, threads);
        times[11] = now_ns() - b0;
        if (vpu_push(g_d.m.p, VPU_BLOCK, mm.d_off) != 0) return VPU_E_PUSH;
        return VPU_OK;
    }
    if (!shape_ok(&mm)) return VPU_E_REQUEST;
    const unsigned char *a;
    uint64_t t0 = now_ns();
    if (mm.a_id != 0) {
        int i = cache_find(mm.a_id);
        if (i < 0) return VPU_E_REQUEST;
        /* an ordinary multiply reads m rows; a mixture reads m rows of each expert, which the id check below bounds */
        if (kernel != VPU_K_MATMUL_ID && mm.m * mm.nb_a > g_cache[i].bytes) return VPU_E_REQUEST;
        a = g_cache[i].m.p;
    } else {
        if (mm.a_off % VPU_BLOCK != 0) return VPU_E_REQUEST;
        size_t len = blocks(mm.m * mm.nb_a);
        if (grow(&g_a, len) != 0) return VPU_E_ALLOC;
        if (vpu_pull(g_a.m.p, len, mm.a_off) != 0) return VPU_E_PULL;
        a = g_a.m.p;
    }
    /* The ids come first in b's area for a mixture, then the rows; one
     * pull takes both. */
    int mixture = (kernel == VPU_K_MATMUL_ID);
    size_t brows = mixture ? mm.b_rows * mm.n_tokens : mm.n;
    size_t blen = blocks(mm.ids_bytes + brows * mm.nb_b), dlen = blocks(mm.n * mm.m * 4);
    if (grow(&g_b, blen) != 0 || grow(&g_d, dlen) != 0) return VPU_E_ALLOC;
    if (vpu_pull(g_b.m.p, blen, mm.b_off) != 0) return VPU_E_PULL;
    *pull_ns = now_ns() - t0;

    uint32_t chunk = (uint32_t)mm.chunk;
    if (chunk < 1 || chunk > ROW_CHUNK_MAX) chunk = ROW_CHUNK;
    struct job j = { a, (const unsigned char *)g_b.m.p + mm.ids_bytes, g_d.m.p, mm.m, mm.n, mm.k,
                     mm.nb_a, mm.nb_b, mm.a_type, chunk, (const int32_t *)g_b.m.p,
                     mixture ? mm.n_used : 0, mixture ? mm.b_rows : 0, mixture ? mm.m * mm.nb_a : 0 };
    int max = vpu_pool_threads() + 1;
    if (max > POOL_SLOTS) max = POOL_SLOTS;
    if (threads < 1) threads = 1;
    if (threads > max) threads = max;
    if ((uint64_t)threads > mm.m) threads = (int)mm.m;
    /* An id that would read past what this tensor's slice holds is a
     * broken request, not a segfault: the ids come from the host and are
     * checked here, once, before any thread runs. */
    if (mixture) {
        uint64_t have = (mm.a_id != 0) ? g_cache[cache_find(mm.a_id)].bytes : g_a.cap;
        uint64_t experts = j.a_stride ? have / j.a_stride : 0;
        for (uint64_t p = 0; p < mm.n; p++)
            if (j.ids[p] >= (int32_t)experts) return VPU_E_REQUEST;
    }
    uint64_t c0 = now_ns();
    *live = vpu_pool_map(mixture ? rows_slice_id : rows_slice, &j, threads);
    *compute_ns = now_ns() - c0;

    uint64_t p0 = now_ns();
    if (vpu_push(g_d.m.p, dlen, mm.d_off) != 0) return VPU_E_PUSH;
    *push_ns = now_ns() - p0;
    if (verbose) {
        static const char *names[VPU_MM_TYPES] = { "f32", "f16", "q4_K", "q5_K", "q6_K", "q8_0", "iq4_xs" };
        printf("matmul: %llux%llu . %llux%llu (%s%s) on %d threads: pull %.3f compute %.3f push %.3f ms\n",
               (unsigned long long)mm.m, (unsigned long long)mm.k, (unsigned long long)mm.n, (unsigned long long)mm.k,
               mm.a_type < VPU_MM_TYPES ? names[mm.a_type] : "?", mm.a_id ? ", cached" : "", *live,
               *pull_ns / 1e6, *compute_ns / 1e6, *push_ns / 1e6);
        fflush(stdout);
    }
    return VPU_OK;
}

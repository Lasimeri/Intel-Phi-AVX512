/* vpu_matmul.h: the matrix-multiply service of the card worker, for the
 * ggml backend (host/crates/phi-ggml). The host keeps a program's weight
 * tensors resident on the card (UPLOAD once, by id), then asks for whole
 * matrix multiplies (MATMUL): d[n][m] = a[m][k] . b[n][k], the rows of a
 * split across the pool, the dot products the AVX-512-derived kernels of
 * vpu_matmul_kernel.S. The descriptor lives in the control area at
 * VPU_OFF_MATMUL; the request's kernel field says which of the three.
 * Mirrored in host/crates/phi-vpu/src/proto.rs; tools/vpu-layout-check.c
 * checks. See vpu_matmul.md. */
#ifndef VPU_MATMUL_H
#define VPU_MATMUL_H
#include <stdint.h>

#define VPU_K_UPLOAD 3   /* keep `bytes` from the window at a_off under a_id (replacing an earlier a_id) */
#define VPU_K_MATMUL 4   /* d = a . b^T with a cached (a_id) or in the window (a_off) */
#define VPU_K_FREE   5   /* drop a_id (0: everything) */
#define VPU_K_MATMUL_ID 6 /* the same, but a is one expert of a mixture chosen per column (ggml's MUL_MAT_ID) */

#define VPU_OFF_MATMUL 13312   /* struct vpu_matmul, in the control area */

#define VPU_MM_F32 0
#define VPU_MM_F16 1
/* llama.cpp quantized formats (ggml-common.h), 256-weight superblocks; the kernels are phi_<fmt>_<rows> of vpu_matmul_kernel.S */
#define VPU_MM_Q4_K 2
#define VPU_MM_Q5_K 3
#define VPU_MM_Q6_K 4
#define VPU_MM_Q8_0 5
#define VPU_MM_IQ4_XS 6
#define VPU_MM_TYPES 7
#define VPU_MM_PROBE 99   /* diagnostic: phi_probe writes 8 vectors to d (kernelgen/quant.md) */

struct vpu_matmul {
    uint64_t a_id;      /* UPLOAD: the id (nonzero); MATMUL: the cached tensor, or 0 for a in the window at a_off */
    uint64_t a_off;     /* window offset: UPLOAD the bytes; MATMUL (a_id 0) m rows of k elements, stride nb_a */
    uint64_t bytes;     /* UPLOAD: how many (the card reads whole 4 KiB blocks) */
    uint32_t a_type;    /* VPU_MM_*: the element type of a */
    uint32_t reserved0;
    uint64_t m, n, k;   /* a: m rows of k; b: n rows of k float32; d: n rows of m float32 */
    uint64_t nb_a;      /* row stride of a in bytes (32-byte aligned rows for float16) */
    uint64_t nb_b;      /* row stride of b in bytes */
    uint64_t b_off;     /* window offset of b: n rows, stride nb_b, whole blocks (MATMUL_ID: the ids first, then the rows) */
    uint64_t d_off;     /* window offset for d: n rows of m float32, contiguous, whole blocks */
    uint64_t chunk;     /* rows per chunk the card works in (0: its own default) */
    /* MUL_MAT_ID only (VPU_K_MATMUL_ID), zero otherwise. a holds `experts`
     * matrices of m rows, one after another; n is n_used * n_tokens, and
     * column p = j + t * n_used multiplies expert ids[p] by b's row
     * (j % b_rows) + t * b_rows. The ids are n int32 at b_off, the rows
     * follow at b_off + ids_bytes. */
    uint64_t n_used;
    uint64_t n_tokens;
    uint64_t b_rows;
    uint64_t ids_bytes;
};

_Static_assert(sizeof(struct vpu_matmul) == 128, "matmul descriptor layout is shared with Rust");

/* Run one request of the three kinds; fills the reply's timings and the
 * number of slices run. Returns a VPU_OK / VPU_E_* status. */
int vpu_matmul_run(volatile unsigned char *ctrl, uint32_t kernel, int threads, int verbose,
                   uint64_t *compute_ns, uint64_t *pull_ns, uint64_t *push_ns, int *live);

#endif

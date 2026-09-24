/* ggml-phi.c: the ggml side of the card backend, the glue ggml's C
 * interface requires and nothing more. It registers one device of type
 * ACCEL named "Phi" (like ggml's BLAS backend: it shares the CPU's host
 * buffers, so weights need no copy on the host), accepts MUL_MAT nodes
 * whose weight type and shape the cards take, and runs each one in three
 * steps: phi_ggml_begin (the Rust library starts the cards on their
 * rows), the host's own rows on a private ggml CPU backend here (ggml's
 * kernels, so any type ggml has works and the results match the CPU's
 * bit for bit), phi_ggml_end (the cards' rows gathered). The ggml
 * functions it needs are resolved at run time from the program that
 * loaded the backend, so this object links against nothing of ggml. See
 * ggml-phi.md. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

int phi_ggml_open(void);
int phi_ggml_supports(uint32_t a_type, uint64_t m, uint64_t k, uint64_t nb_a, uint64_t nb_b, uint64_t n);
void phi_ggml_note_weight(const uint8_t *data, uint64_t bytes);
int64_t phi_ggml_begin(const uint8_t *a, uint32_t a_type, uint64_t m, uint64_t k, uint64_t nb_a, int keep,
                       const uint8_t *b, uint64_t n, uint64_t nb_b);
int64_t phi_ggml_begin_id(const uint8_t *a, uint32_t a_type, uint64_t m, uint64_t k, uint64_t nb_a, int keep,
                          const uint8_t *b, uint64_t n, uint64_t nb_b, uint64_t experts, uint64_t nb_a2,
                          const int32_t *ids, uint64_t n_used, uint64_t n_tokens, uint64_t ids_nb1,
                          uint64_t b_rows, uint64_t nb_b2);
int phi_ggml_host_range(uint64_t i, uint64_t *from, uint64_t *to);
int phi_ggml_end(uint8_t *d, uint64_t nb_d);
int phi_ggml_end_id(uint8_t *d, uint64_t nb_d, uint64_t nb_d2);
void phi_ggml_free_all(void);

/* A feed-forward block for the fused path (Rust: ffn.rs, FfnArgs). */
struct phi_ffn_args {
    const uint8_t *gate, *up, *down, *x;
    uint32_t gate_type, up_type, down_type, pad;
    uint64_t k, inter, m_out, n, nb_gate, nb_up, nb_down, nb_x;
};
int64_t phi_ggml_ffn_begin(const struct phi_ffn_args *a);
int phi_ggml_ffn_end(uint8_t *y, uint64_t nb_y);

/* ---- ggml, resolved at run time ---- */

#define GGML_FNS(X) \
    X(ggml_backend_buffer_type_t, cpu_buft, ggml_backend_cpu_buffer_type, (void)) \
    X(ggml_backend_buffer_t, cpu_from_ptr, ggml_backend_cpu_buffer_from_ptr, (void *, size_t)) \
    X(bool, buft_is_host, ggml_backend_buft_is_host, (ggml_backend_buffer_type_t)) \
    X(struct ggml_context *, init, ggml_init, (struct ggml_init_params)) \
    X(void, free, ggml_free, (struct ggml_context *)) \
    X(size_t, tensor_overhead, ggml_tensor_overhead, (void)) \
    X(size_t, graph_overhead_custom, ggml_graph_overhead_custom, (size_t, bool)) \
    X(struct ggml_tensor *, new_tensor_2d, ggml_new_tensor_2d, (struct ggml_context *, enum ggml_type, int64_t, int64_t)) \
    X(struct ggml_tensor *, new_tensor_3d, ggml_new_tensor_3d, (struct ggml_context *, enum ggml_type, int64_t, int64_t, int64_t)) \
    X(struct ggml_tensor *, mul_mat, ggml_mul_mat, (struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *)) \
    X(struct ggml_tensor *, mul_mat_id, ggml_mul_mat_id, (struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *)) \
    X(struct ggml_cgraph *, new_graph_custom, ggml_new_graph_custom, (struct ggml_context *, size_t, bool)) \
    X(void, build_forward_expand, ggml_build_forward_expand, (struct ggml_cgraph *, struct ggml_tensor *)) \
    X(enum ggml_status, backend_graph_compute, ggml_backend_graph_compute, (ggml_backend_t, struct ggml_cgraph *)) \
    X(ggml_backend_dev_t, dev_by_type, ggml_backend_dev_by_type, (enum ggml_backend_dev_type)) \
    X(ggml_backend_t, dev_init, ggml_backend_dev_init, (ggml_backend_dev_t, const char *)) \
    X(ggml_backend_reg_t, dev_backend_reg, ggml_backend_dev_backend_reg, (ggml_backend_dev_t)) \
    X(void *, reg_get_proc_address, ggml_backend_reg_get_proc_address, (ggml_backend_reg_t, const char *)) \
    X(struct ggml_tensor *, swiglu_split, ggml_swiglu_split, (struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *)) \
    X(struct ggml_tensor *, add, ggml_add, (struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *)) \
    X(size_t, row_size, ggml_row_size, (enum ggml_type, int64_t)) \
    X(size_t, nbytes, ggml_nbytes, (const struct ggml_tensor *)) \
    X(bool, is_contiguous_1, ggml_is_contiguous_1, (const struct ggml_tensor *)) \
    X(void, threadpool_params_init, ggml_threadpool_params_init, (struct ggml_threadpool_params *, int))

#define DECL(ret, name, sym, args) static ret (*p_##name) args;
GGML_FNS(DECL)

static int resolve(void)
{
    if (p_cpu_buft) return 0;
#define LOAD(ret, name, sym, args) \
    p_##name = (ret (*) args)dlsym(RTLD_DEFAULT, #sym); \
    if (!p_##name) { fprintf(stderr, "ggml-phi: the program has no %s\n", #sym); return -1; }
    GGML_FNS(LOAD)
    return 0;
}

/* ---- the host's rows: a private CPU backend ---- */

static ggml_backend_t g_cpu;

static int host_backend(void)
{
    if (g_cpu) return 0;
    ggml_backend_dev_t dev = p_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) { fprintf(stderr, "ggml-phi: no CPU device registered\n"); return -1; }
    g_cpu = p_dev_init(dev, NULL);
    if (!g_cpu) { fprintf(stderr, "ggml-phi: the CPU backend would not initialise\n"); return -1; }
    /* 12, not one per CPU: a host thread that shares a CPU with a card
     * daemon stalls ggml.s barrier for a timeslice (7 ms per multiply at
     * 15; docs/results/2026-09-23-quantized-kernels.md). */
    int threads = 12;
    const char *e = getenv("PHI_GGML_HOST_THREADS");
    if (e && atoi(e) > 0) threads = atoi(e);
    ggml_backend_set_n_threads_t set = (ggml_backend_set_n_threads_t)p_reg_get_proc_address(p_dev_backend_reg(dev), "ggml_backend_set_n_threads");
    if (set) set(g_cpu, threads);
    /* A threadpool of its own, kept. Without one, ggml's CPU backend
     * builds a disposable pool inside every graph_compute and joins it at
     * the end (ggml/src/ggml-cpu/ggml-cpu.c, ggml_graph_compute), and this
     * backend computes about 419 small graphs a token. Its workers do not
     * spin between graphs (poll 0, PHI_GGML_HOST_POLL): between them run
     * the program's own threads, which a spinning pool would compete with,
     * the failure mode of giving the program 16 threads. PHI_GGML_HOST_POOL=0
     * leaves the backend without one, as it was. */
    const char *pe = getenv("PHI_GGML_HOST_POOL");
    if (!(pe && atoi(pe) == 0)) {
        ggml_backend_reg_t reg = p_dev_backend_reg(dev);
        struct ggml_threadpool *(*tp_new)(struct ggml_threadpool_params *) =
            (struct ggml_threadpool *(*)(struct ggml_threadpool_params *))p_reg_get_proc_address(reg, "ggml_threadpool_new");
        void (*set_tp)(ggml_backend_t, struct ggml_threadpool *) =
            (void (*)(ggml_backend_t, struct ggml_threadpool *))p_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (tp_new && set_tp) {
            struct ggml_threadpool_params tpp;
            p_threadpool_params_init(&tpp, threads);
            const char *poll = getenv("PHI_GGML_HOST_POLL");
            tpp.poll = poll ? (uint32_t)atoi(poll) : 0;
            struct ggml_threadpool *tp = tp_new(&tpp);
            if (tp) {
                set_tp(g_cpu, tp);
                fprintf(stderr, "ggml-phi: the host's rows have a threadpool of their own (poll %u)\n", tpp.poll);
            }
        }
    }
    fprintf(stderr, "ggml-phi: the host's rows run on ggml's CPU backend with %d threads\n", threads);
    fprintf(stderr, "ggml-phi: give the calling program %d threads as well (its own -t): more of them"
                    " contend with the card daemons and cost 5x at one token\n", threads);
    return 0;
}

/* A leaf that aliases rows `from..from+rows` of `t`: not a view, which
 * would bring the tensor's whole ancestry (the model graph) into the
 * graph built here. */
static struct ggml_tensor *alias(struct ggml_context *ctx, const struct ggml_tensor *t, int64_t from, int64_t rows)
{
    struct ggml_tensor *l = p_new_tensor_2d(ctx, t->type, t->ne[0], rows);
    l->data = (char *)t->data + from * t->nb[1];
    l->nb[0] = t->nb[0];
    l->nb[1] = t->nb[1];
    l->nb[2] = t->nb[1] * rows;
    l->nb[3] = l->nb[2];
    return l;
}

/* The same for a mixture's weights, which are one matrix per expert:
 * rows `from..from+rows` of every one of them. */
static struct ggml_tensor *alias3(struct ggml_context *ctx, const struct ggml_tensor *t, int64_t from, int64_t rows)
{
    struct ggml_tensor *l = p_new_tensor_3d(ctx, t->type, t->ne[0], rows, t->ne[2]);
    l->data = (char *)t->data + from * t->nb[1];
    l->nb[0] = t->nb[0];
    l->nb[1] = t->nb[1];
    l->nb[2] = t->nb[2];
    l->nb[3] = t->nb[3];
    return l;
}

/* The host's rows of one MUL_MAT_ID: the same three leaves, with the
 * expert list passed through untouched, and ggml's own kernel choosing
 * the expert per column exactly as it would have. */
static int host_rows_id(const struct ggml_tensor *node, int64_t from, int64_t to)
{
    const struct ggml_tensor *src0 = node->src[0], *src1 = node->src[1], *ids = node->src[2];
    int64_t r0 = to - from;
    struct ggml_init_params params = { p_tensor_overhead() * 16 + p_graph_overhead_custom(64, false) + 4096, NULL, true };
    struct ggml_context *ctx = p_init(params);
    if (!ctx) return -1;
    struct ggml_tensor *a = alias3(ctx, src0, from, r0);
    struct ggml_tensor *b = alias3(ctx, src1, 0, src1->ne[1]);
    struct ggml_tensor *i = alias(ctx, ids, 0, ids->ne[1]);
    struct ggml_tensor *c = p_mul_mat_id(ctx, a, b, i);
    c->data = (char *)node->data + from * 4;
    c->nb[1] = node->nb[1];
    c->nb[2] = node->nb[2];
    c->nb[3] = node->nb[3];
    struct ggml_cgraph *g = p_new_graph_custom(ctx, 64, false);
    p_build_forward_expand(g, c);
    enum ggml_status st = p_backend_graph_compute(g_cpu, g);
    p_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : -1;
}

/* d[0..n][0..r0] = a[0..r0] . b, with ggml's own MUL_MAT on the rows the
 * host keeps: two leaves, one multiply, the result pointed straight at the
 * node's data with the node's row stride. */
static int host_rows(const struct ggml_tensor *node, int64_t from, int64_t to)
{
    struct ggml_tensor *src0 = node->src[0], *src1 = node->src[1];
    int64_t r0 = to - from;
    struct ggml_init_params params = { p_tensor_overhead() * 16 + p_graph_overhead_custom(64, false) + 4096, NULL, true };
    struct ggml_context *ctx = p_init(params);
    if (!ctx) return -1;
    struct ggml_tensor *a = alias(ctx, src0, from, r0);
    struct ggml_tensor *b = alias(ctx, src1, 0, src1->ne[1]);
    struct ggml_tensor *c = p_mul_mat(ctx, a, b);
    c->data = (char *)node->data + from * 4;
    c->nb[1] = node->nb[1];
    c->nb[2] = node->nb[2];
    c->nb[3] = node->nb[3];
    struct ggml_cgraph *g = p_new_graph_custom(ctx, 64, false);
    p_build_forward_expand(g, c);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    enum ggml_status st = p_backend_graph_compute(g_cpu, g);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (getenv("PHI_GGML_VERBOSE"))
        fprintf(stderr, "ggml-phi: host rows %lld..%lld of %s %lldx%lld, n %lld: compute %.3f ms\n", (long long)from, (long long)to, ggml_type_name(src0->type),
                (long long)src0->ne[1], (long long)src0->ne[0], (long long)src1->ne[1], (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6);
    p_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : -1;
}

/* A float32 tensor whose rows are evenly spaced (ggml_is_contiguous_1),
 * as one leaf of all its rows: a mixture's SwiGLU is three-dimensional,
 * and a SwiGLU works row by row either way. */
static struct ggml_tensor *alias_rows(struct ggml_context *ctx, const struct ggml_tensor *t)
{
    int64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
    return alias(ctx, t, 0, rows);
}

/* A SwiGLU node on the host, whole: ggml's own kernel, the result pointed
 * at the node's data. This backend takes SwiGLU so that a feed-forward
 * block arrives in one sub-graph; the ones it does not fuse run here. */
static int host_glu(const struct ggml_tensor *node)
{
    struct ggml_init_params params = { p_tensor_overhead() * 8 + p_graph_overhead_custom(16, false) + 4096, NULL, true };
    struct ggml_context *ctx = p_init(params);
    if (!ctx) return -1;
    struct ggml_tensor *h = p_swiglu_split(ctx, alias_rows(ctx, node->src[0]), alias_rows(ctx, node->src[1]));
    h->data = node->data;
    h->nb[1] = node->nb[1];
    h->nb[2] = h->nb[3] = node->nb[1] * h->ne[1];
    struct ggml_cgraph *g = p_new_graph_custom(ctx, 16, false);
    p_build_forward_expand(g, h);
    enum ggml_status st = p_backend_graph_compute(g_cpu, g);
    p_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : -1;
}

/* ---- a feed-forward block, fused ---- */

/* One block as it sits in a sub-graph: the gate and up multiplies of the
 * same activations, their SwiGLU, and the down multiply of that (node
 * indices). */
struct ffn_quad { int gate, up, glu, down; };

/* The host's intermediates of a block, kept between calls: ggml's own
 * allocation would fault its pages in again every time, and at a prompt
 * this is tens of megabytes. */
static unsigned char *g_scratch;
static size_t g_scratch_cap;

static void *bump(unsigned char **at, size_t bytes)
{
    void *p = *at;
    *at += (bytes + 63) & ~(size_t)63;
    return p;
}

/* The host's runs of a block's intermediate, `ranges[r]` = from..to: for
 * each, its gate and up rows, their SwiGLU, and down over the same columns
 * (a leaf of down's rows cut at the run's superblocks, which ggml's
 * multiply takes as it takes any row stride; checked 2026-09-23 against
 * the whole multiply, docs/results/2026-09-23-ffn-per-request.md), summed
 * into the block's result. One graph, so ggml's pool runs it without
 * coming back here between the steps. */
static int host_ffn(struct ggml_tensor *const *nodes, const struct ffn_quad *q, const uint64_t (*ranges)[2], int nr)
{
    const struct ggml_tensor *gate = nodes[q->gate], *up = nodes[q->up];
    struct ggml_tensor *down = nodes[q->down];
    const struct ggml_tensor *wg = gate->src[0], *wu = up->src[0], *wd = down->src[0], *x = gate->src[1];
    int64_t n = x->ne[1], m_out = wd->ne[1];
    size_t need = 0;
    for (int r = 0; r < nr; r++)
        need += 3 * (((size_t)(ranges[r][1] - ranges[r][0]) * n * 4 + 63) & ~(size_t)63) + (((size_t)m_out * n * 4 + 63) & ~(size_t)63) * 2;
    if (need > g_scratch_cap) {
        free(g_scratch);
        g_scratch = aligned_alloc(64, need);
        g_scratch_cap = g_scratch ? need : 0;
        if (!g_scratch) return -1;
    }
    struct ggml_init_params params = { p_tensor_overhead() * (16 + 12 * (size_t)nr) + p_graph_overhead_custom(64, false) + 4096, NULL, true };
    struct ggml_context *ctx = p_init(params);
    if (!ctx) return -1;
    unsigned char *at = g_scratch;
    struct ggml_tensor *xx = alias(ctx, x, 0, n), *acc = NULL;
    for (int r = 0; r < nr; r++) {
        int64_t from = (int64_t)ranges[r][0], rows = (int64_t)(ranges[r][1] - ranges[r][0]);
        struct ggml_tensor *gg = p_mul_mat(ctx, alias(ctx, wg, from, rows), xx);
        gg->data = bump(&at, p_nbytes(gg));
        struct ggml_tensor *uu = p_mul_mat(ctx, alias(ctx, wu, from, rows), xx);
        uu->data = bump(&at, p_nbytes(uu));
        struct ggml_tensor *hh = p_swiglu_split(ctx, gg, uu);
        hh->data = bump(&at, p_nbytes(hh));
        struct ggml_tensor *wdc = p_new_tensor_2d(ctx, wd->type, rows, m_out);
        wdc->data = (char *)wd->data + p_row_size(wd->type, from);
        wdc->nb[1] = wd->nb[1];
        wdc->nb[2] = wdc->nb[3] = wd->nb[1] * m_out;
        struct ggml_tensor *yy = p_mul_mat(ctx, wdc, hh);
        yy->data = bump(&at, p_nbytes(yy));
        if (acc) {
            acc = p_add(ctx, acc, yy);
            acc->data = bump(&at, p_nbytes(acc));
        } else {
            acc = yy;
        }
    }
    acc->data = down->data;
    acc->nb[1] = down->nb[1];
    acc->nb[2] = acc->nb[3] = down->nb[1] * n;
    struct ggml_cgraph *g = p_new_graph_custom(ctx, 64, false);
    p_build_forward_expand(g, acc);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    enum ggml_status st = p_backend_graph_compute(g_cpu, g);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (getenv("PHI_GGML_VERBOSE"))
        fprintf(stderr, "ggml-phi: host part of a feed-forward block, %d run(s) of %lld, n %lld: compute %.3f ms\n", nr,
                (long long)wg->ne[1], (long long)n, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6);
    p_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : -1;
}

/* ---- what the cards take ---- */

/* ggml's type to the card service's code (proto.rs MM_*), or -1. */
static int card_type(enum ggml_type t)
{
    switch (t) {
    case GGML_TYPE_F32: return 0;
    case GGML_TYPE_F16: return 1;
    case GGML_TYPE_Q4_K: return 2;
    case GGML_TYPE_Q5_K: return 3;
    case GGML_TYPE_Q6_K: return 4;
    case GGML_TYPE_Q8_0: return 5;
    case GGML_TYPE_IQ4_XS: return 6;
    default: return -1;
    }
}

/* Is the weight in a buffer this backend reads as ggml lays it out? A
 * buffer that is not a host buffer is not: llama.cpp's CPU backend repacks
 * some quantized types into its own interleaved layout for its own kernels
 * (on this AVX2 host every Q4_K matrix whose rows are a multiple of 8,
 * ggml-cpu/repack.cpp), in a buffer type of its own that says it is not a
 * host buffer. The scheduler also checks buffers before placing a node,
 * but a multiply of such a weight must not be counted as the cards'
 * (phi_ggml_note_weight), and it is not the cards' to take in any case. A
 * weight with no buffer yet is being asked about while the model loads. */
static int weight_readable(const struct ggml_tensor *w)
{
    if (!w->buffer) return 1;
    return resolve() == 0 && p_buft_is_host(w->buffer->buft);
}

/* A mixture of experts: ggml's MUL_MAT_ID, which is what an MoE model's
 * feed-forward weights go through. src0 is one matrix per expert, src2
 * names the expert each column wants, and the cards keep the same rows
 * of every expert, so the split is by rows exactly as for a plain
 * multiply. */
static bool phi_supports_mul_mat_id(const struct ggml_tensor *op)
{
    const struct ggml_tensor *src0 = op->src[0], *src1 = op->src[1], *ids = op->src[2];
    if (!src0 || !src1 || !ids) return false;
    if (!weight_readable(src0)) return false;
    int t = card_type(src0->type);
    if (t < 0) return false;
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (src1->nb[0] != 4 || op->nb[0] != 4) return false;
    if (src0->ne[3] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
    if (!strstr(src0->name, "weight")) return false;
    /* the card holds every expert's rows: the whole tensor is the budget */
    if (phi_ggml_supports((uint32_t)t, (uint64_t)src0->ne[1], (uint64_t)src0->ne[0], src0->nb[1], src1->nb[1],
                             (uint64_t)(src1->ne[1] * src1->ne[2])) == 0) return false;
    phi_ggml_note_weight(src0->data, (uint64_t)src0->nb[2] * (uint64_t)src0->ne[2]);
    return true;
}

static bool phi_supports_mul_mat(const struct ggml_tensor *op)
{
    const struct ggml_tensor *src0 = op->src[0], *src1 = op->src[1];
    if (!src0 || !src1) return false;
    if (!weight_readable(src0)) return false;
    int t = card_type(src0->type);
    if (t < 0) return false;
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    /* rows contiguous, no batch or broadcast dimensions (attention stays on the CPU) */
    if (src1->nb[0] != 4 || op->nb[0] != 4) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
    if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) return false;
    /* only a weight is worth splitting: anything else the host does whole */
    if (!strstr(src0->name, "weight")) return false;
    if (phi_ggml_supports((uint32_t)t, (uint64_t)src0->ne[1], (uint64_t)src0->ne[0], src0->nb[1], src1->nb[1], (uint64_t)src1->ne[1]) == 0) return false;
    /* what the cards could hold, for sizing their share (Rust, settle_fraction) */
    phi_ggml_note_weight(src0->data, (uint64_t)src0->nb[1] * (uint64_t)src0->ne[1]);
    return true;
}

/* PHI_GGML_FFN=1: take SwiGLU, so that a feed-forward block's four nodes
 * arrive in one sub-graph and run as one request per card. Off by
 * default, for two reasons. It measured neutral on both models here
 * (docs/results/2026-09-23-ffn-per-request.md). And a fused block never
 * writes its gate, up and SwiGLU tensors, which nothing in ordinary
 * inference reads but a program's eval callback can: llama-imatrix asks
 * for every multiply and reads its activations, which for ffn_down is the
 * SwiGLU, and the scheduler hands this backend a sub-graph ending there,
 * so no check here can see the reader. Unset, the graph splits exactly as
 * it did before the fused path existed. */
static int ffn_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("PHI_GGML_FFN");
        const char *o = getenv("PHI_GGML_OFFLOAD");
        on = e && atoi(e) != 0;
        /* The fused path keeps its gate, up and down rows on the host and
         * hands a declined block back to it whole, which is the opposite of
         * the offload's "the cards' rows are theirs alone": with both set,
         * the offload wins and the blocks go through the plain path. */
        if (on && o && atoi(o) != 0) {
            fprintf(stderr, "ggml-phi: PHI_GGML_FFN is off under PHI_GGML_OFFLOAD (the fused path keeps its rows on the host)\n");
            on = 0;
        }
    }
    return on;
}

/* SwiGLU in the split form llama.cpp builds (ggml_swiglu_split(gate, up):
 * silu(src0) * src1), float32, rows evenly spaced. */
static bool phi_supports_glu(const struct ggml_tensor *op)
{
    const struct ggml_tensor *a = op->src[0], *b = op->src[1];
    if (!ffn_enabled() || resolve() != 0 || !a || !b) return false;
    if (ggml_get_op_params_i32(op, 0) != GGML_GLU_OP_SWIGLU) return false;
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    return p_is_contiguous_1(a) && p_is_contiguous_1(b) && p_is_contiguous_1(op);
}

/* A multiply of a quantized weight the cards take, two-dimensional, to be
 * computed in this sub-graph. */
static int weight_mm(const struct ggml_tensor *t)
{
    const struct ggml_tensor *w = t->src[0], *x = t->src[1];
    return t->op == GGML_OP_MUL_MAT && w && x && (t->flags & GGML_TENSOR_FLAG_COMPUTE) && strstr(w->name, "weight") &&
           card_type(w->type) >= 2 && w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1 &&
           x->type == GGML_TYPE_F32 && x->nb[0] == 4 && t->type == GGML_TYPE_F32 && t->nb[0] == 4;
}

static int node_index(const struct ggml_cgraph *cg, const struct ggml_tensor *t, int before)
{
    for (int i = 0; i < before; i++)
        if (cg->nodes[i] == t) return i;
    return -1;
}

/* Does any node of the sub-graph but `except` read `t`, directly or
 * through a view? */
static int read_elsewhere(const struct ggml_cgraph *cg, const struct ggml_tensor *t, const struct ggml_tensor *except)
{
    for (int i = 0; i < cg->n_nodes; i++) {
        const struct ggml_tensor *nd = cg->nodes[i];
        if (nd == except) continue;
        if (nd->view_src == t) return 1;
        for (int s = 0; s < GGML_MAX_SRC; s++)
            if (nd->src[s] == t) return 1;
    }
    return 0;
}

/* Is node i the SwiGLU of a feed-forward block the fused path can take?
 * By structure, not by name or order: its two inputs are multiplies of
 * the same activations in this sub-graph, a later multiply reads it, the
 * shapes chain, and none of the three intermediates is an output of the
 * graph or read by anything else here, since the fused path never writes
 * them. */
static int find_quad(const struct ggml_cgraph *cg, int i, struct ffn_quad *q)
{
    const struct ggml_tensor *glu = cg->nodes[i];
    if (glu->op != GGML_OP_GLU || !(glu->flags & GGML_TENSOR_FLAG_COMPUTE) || !phi_supports_glu(glu)) return 0;
    const struct ggml_tensor *g = glu->src[0], *u = glu->src[1];
    int ig = node_index(cg, g, i), iu = node_index(cg, u, i);
    if (ig < 0 || iu < 0 || !weight_mm(g) || !weight_mm(u) || g->src[1] != u->src[1]) return 0;
    int id = -1;
    for (int j = i + 1; j < cg->n_nodes && id < 0; j++)
        if (cg->nodes[j]->op == GGML_OP_MUL_MAT && cg->nodes[j]->src[1] == glu) id = j;
    if (id < 0 || !weight_mm(cg->nodes[id])) return 0;
    const struct ggml_tensor *wg = g->src[0], *wu = u->src[0], *wd = cg->nodes[id]->src[0];
    if (wg->ne[0] != wu->ne[0] || wg->ne[1] != wu->ne[1] || wd->ne[0] != wg->ne[1]) return 0;
    if ((g->flags | u->flags | glu->flags) & GGML_TENSOR_FLAG_OUTPUT) return 0;
    if (read_elsewhere(cg, g, glu) || read_elsewhere(cg, u, glu) || read_elsewhere(cg, glu, cg->nodes[id])) return 0;
    q->gate = ig;
    q->up = iu;
    q->glu = i;
    q->down = id;
    return 1;
}

/* One multiply, the plain way: the cards start on their rows (Rust,
 * phi_ggml_begin), the host computes its ranges here, the cards' rows are
 * gathered. */
static int run_mul_mat(struct ggml_tensor *node)
{
    const struct ggml_tensor *src0 = node->src[0], *src1 = node->src[1];
    int keep = strstr(src0->name, "weight") != NULL;
    int64_t nr = phi_ggml_begin(src0->data, (uint32_t)card_type(src0->type), (uint64_t)src0->ne[1], (uint64_t)src0->ne[0],
                                src0->nb[1], keep, src1->data, (uint64_t)src1->ne[1], src1->nb[1]);
    if (nr < 0) return -1;
    for (int64_t r = 0; r < nr; r++) {
        uint64_t from, to;
        if (!phi_ggml_host_range((uint64_t)r, &from, &to)) break;
        if (to > from && host_rows(node, (int64_t)from, (int64_t)to) != 0) return -1;
    }
    return phi_ggml_end(node->data, node->nb[1]) == 0 ? 0 : -1;
}

/* A block: the cards' runs through Rust (ffn.rs), the host's here, the
 * partials added. If the fused path declines, the four nodes run exactly
 * as they would have without it: the multiplies the plain way (whose
 * Rust side keeps a planned block's tensors on the host, and gives the
 * cards those of a block it never planned) and the SwiGLU on the host. */
static int run_ffn(struct ggml_tensor *const *nodes, const struct ffn_quad *q)
{
    struct ggml_tensor *gate = nodes[q->gate], *up = nodes[q->up], *glu = nodes[q->glu], *down = nodes[q->down];
    const struct ggml_tensor *wg = gate->src[0], *wu = up->src[0], *wd = down->src[0], *x = gate->src[1];
    struct phi_ffn_args a = {
        wg->data, wu->data, wd->data, x->data,
        (uint32_t)card_type(wg->type), (uint32_t)card_type(wu->type), (uint32_t)card_type(wd->type), 0,
        (uint64_t)wg->ne[0], (uint64_t)wg->ne[1], (uint64_t)wd->ne[1], (uint64_t)x->ne[1],
        wg->nb[1], wu->nb[1], wd->nb[1], x->nb[1],
    };
    int64_t nr = phi_ggml_ffn_begin(&a);
    if (nr == -2) return run_mul_mat(gate) || run_mul_mat(up) || host_glu(glu) || run_mul_mat(down) ? -1 : 0;
    if (nr < 0) return -1;
    uint64_t ranges[17][2];
    int got = 0;
    for (int64_t r = 0; r < nr && got < 17; r++)
        if (phi_ggml_host_range((uint64_t)r, &ranges[got][0], &ranges[got][1]) && ranges[got][1] > ranges[got][0]) got++;
    if (got == 0) {
        for (int64_t c = 0; c < x->ne[1]; c++) memset((char *)down->data + c * down->nb[1], 0, (size_t)wd->ne[1] * 4);
    } else if (host_ffn(nodes, q, (const uint64_t(*)[2])ranges, got) != 0) {
        return -1;
    }
    return phi_ggml_ffn_end(down->data, down->nb[1]) == 0 ? 0 : -1;
}

/* ---- backend ---- */

static const char *phi_backend_get_name(ggml_backend_t backend) { (void)backend; return "Phi"; }
static void phi_backend_free(ggml_backend_t backend) { free(backend); }

/* PHI_GGML_GRAPH=N: print the first N sub-graphs the scheduler hands
 * this backend, one line per node, with the activation tensor's address,
 * so it can be seen which multiplies arrive together and which of them
 * share an input. See ggml-phi.md. */
static void dump_graph(const struct ggml_cgraph *cgraph)
{
    static int left = -1;
    if (left < 0) {
        const char *e = getenv("PHI_GGML_GRAPH");
        left = e ? atoi(e) : 0;
    }
    if (left == 0) return;
    left--;
    fprintf(stderr, "ggml-phi: sub-graph of %d nodes\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor *n = cgraph->nodes[i];
        fprintf(stderr, "ggml-phi:   %-14s %-34s src0 %-30s src1 %p %s\n", ggml_op_desc(n), n->name,
                n->src[0] ? n->src[0]->name : "-", n->src[1] ? (void *)n->src[1] : NULL,
                n->src[1] ? n->src[1]->name : "-");
    }
}

static enum ggml_status phi_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph)
{
    (void)backend;
    dump_graph(cgraph);
    /* The feed-forward blocks of this sub-graph first: a block's gate, up
     * and SwiGLU are skipped where they stand (role 1) and the whole block
     * runs at its down multiply (role 2), which comes after all three. */
    static unsigned char *role;
    static struct ffn_quad *quads;
    static int cap;
    if (cgraph->n_nodes > cap) {
        free(role);
        free(quads);
        cap = cgraph->n_nodes;
        role = malloc((size_t)cap);
        quads = malloc((size_t)cap * sizeof *quads);
        if (!role || !quads) { cap = 0; return GGML_STATUS_ALLOC_FAILED; }
    }
    memset(role, 0, (size_t)cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ffn_quad q;
        if (!find_quad(cgraph, i, &q)) continue;
        role[q.gate] = role[q.up] = role[q.glu] = 1;
        role[q.down] = 2;
        quads[q.down] = q;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor *node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || role[i] == 1) continue;
        if (role[i] == 2) {
            if (run_ffn(cgraph->nodes, &quads[i]) != 0) return GGML_STATUS_FAILED;
            continue;
        }
        switch (node->op) {
        case GGML_OP_MUL_MAT:
            if (run_mul_mat(node) != 0) return GGML_STATUS_FAILED;
            break;
        case GGML_OP_MUL_MAT_ID: {
            const struct ggml_tensor *src0 = node->src[0], *src1 = node->src[1], *ids = node->src[2];
            int keep = strstr(src0->name, "weight") != NULL;
            int64_t n = ids->ne[0] * ids->ne[1];
            int64_t nr = phi_ggml_begin_id(src0->data, (uint32_t)card_type(src0->type), (uint64_t)src0->ne[1],
                                           (uint64_t)src0->ne[0], src0->nb[1], keep, src1->data, (uint64_t)n, src1->nb[1],
                                           (uint64_t)src0->ne[2], src0->nb[2], (const int32_t *)ids->data,
                                           (uint64_t)ids->ne[0], (uint64_t)ids->ne[1], ids->nb[1],
                                           (uint64_t)src1->ne[1], src1->nb[2]);
            if (nr < 0) return GGML_STATUS_FAILED;
            for (int64_t r = 0; r < nr; r++) {
                uint64_t from, to;
                if (!phi_ggml_host_range((uint64_t)r, &from, &to)) break;
                if (to > from && host_rows_id(node, (int64_t)from, (int64_t)to) != 0) return GGML_STATUS_FAILED;
            }
            if (phi_ggml_end_id(node->data, node->nb[1], node->nb[2]) != 0) return GGML_STATUS_FAILED;
            break;
        }
        case GGML_OP_GLU:
            if (host_glu(node) != 0) return GGML_STATUS_FAILED;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        default:
            fprintf(stderr, "ggml-phi: unsupported op %s\n", ggml_op_desc(node));
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i phi_backend_i = {
    /* .get_name           = */ phi_backend_get_name,
    /* .free               = */ phi_backend_free,
    /* .set_tensor_async   = */ NULL,
    /* .get_tensor_async   = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async   = */ NULL,
    /* .synchronize        = */ NULL,
    /* .graph_plan_create  = */ NULL,
    /* .graph_plan_free    = */ NULL,
    /* .graph_plan_update  = */ NULL,
    /* .graph_plan_compute = */ NULL,
    /* .graph_compute      = */ phi_graph_compute,
    /* .event_record       = */ NULL,
    /* .event_wait         = */ NULL,
    /* .graph_optimize     = */ NULL,
};

static ggml_guid_t phi_guid(void)
{
    static ggml_guid guid = { 0x70, 0x68, 0x69, 0x33, 0x31, 0x32, 0x30, 0x61, 0x76, 0x78, 0x35, 0x31, 0x32, 0x6b, 0x6e, 0x63 };
    return &guid;
}

/* ---- device ---- */

static const char *phi_dev_get_name(ggml_backend_dev_t dev) { (void)dev; return "Phi"; }
static const char *phi_dev_get_description(ggml_backend_dev_t dev) { (void)dev; return "Intel Xeon Phi 3120 cards beside the host (AVX-512 executed on the cards)"; }
static void phi_dev_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) { (void)dev; *free = 0; *total = 0; }
static enum ggml_backend_dev_type phi_dev_get_type(ggml_backend_dev_t dev) { (void)dev; return GGML_BACKEND_DEVICE_TYPE_ACCEL; }

static void phi_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props)
{
    props->name = phi_dev_get_name(dev);
    props->description = phi_dev_get_description(dev);
    props->type = phi_dev_get_type(dev);
    phi_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = true;
    props->caps.events = false;
}

static ggml_backend_t phi_dev_init_backend(ggml_backend_dev_t dev, const char *params)
{
    (void)params;
    if (resolve() != 0 || phi_ggml_open() < 0 || host_backend() != 0) return NULL;
    ggml_backend_t backend = calloc(1, sizeof *backend);
    if (!backend) return NULL;
    backend->guid = phi_guid();
    backend->iface = phi_backend_i;
    backend->device = dev;
    backend->context = NULL;
    return backend;
}

static ggml_backend_buffer_type_t phi_dev_get_buffer_type(ggml_backend_dev_t dev)
{
    (void)dev;
    return resolve() == 0 ? p_cpu_buft() : NULL;
}

static ggml_backend_buffer_t phi_dev_buffer_from_host_ptr(ggml_backend_dev_t dev, void *ptr, size_t size, size_t max_tensor_size)
{
    (void)dev; (void)max_tensor_size;
    return resolve() == 0 ? p_cpu_from_ptr(ptr, size) : NULL;
}

static bool phi_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor *op)
{
    (void)dev;
    switch (op->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;
    case GGML_OP_MUL_MAT:
        return phi_supports_mul_mat(op);
    case GGML_OP_MUL_MAT_ID:
        return phi_supports_mul_mat_id(op);
    case GGML_OP_GLU:
        return phi_supports_glu(op);
    default:
        return false;
    }
}

static bool phi_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft)
{
    (void)dev;
    return resolve() == 0 && p_buft_is_host(buft);
}

static const struct ggml_backend_device_i phi_device_i = {
    /* .get_name             = */ phi_dev_get_name,
    /* .get_description      = */ phi_dev_get_description,
    /* .get_memory           = */ phi_dev_get_memory,
    /* .get_type             = */ phi_dev_get_type,
    /* .get_props            = */ phi_dev_get_props,
    /* .init_backend         = */ phi_dev_init_backend,
    /* .get_buffer_type      = */ phi_dev_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ phi_dev_buffer_from_host_ptr,
    /* .supports_op          = */ phi_dev_supports_op,
    /* .supports_buft        = */ phi_dev_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

/* ---- registration ---- */

static const char *phi_reg_get_name(ggml_backend_reg_t reg) { (void)reg; return "Phi"; }
/* A device only when the cards open. llama.cpp aborts ("failed to
 * initialize") on an accelerator whose backend comes back NULL, but runs on
 * the CPU when a backend has no device: so a card that is down, or no
 * worker polling, means no device here, decided once. */
static size_t phi_reg_get_device_count(ggml_backend_reg_t reg)
{
    static int n = -1;
    (void)reg;
    if (n < 0) n = (resolve() == 0 && phi_ggml_open() >= 0 && host_backend() == 0) ? 1 : 0;
    return (size_t)n;
}

static ggml_backend_dev_t phi_reg_get_device(ggml_backend_reg_t reg, size_t index)
{
    static struct ggml_backend_device dev;
    (void)index;
    dev.iface = phi_device_i;
    dev.reg = reg;
    dev.context = NULL;
    return &dev;
}

static void *phi_reg_get_proc_address(ggml_backend_reg_t reg, const char *name)
{
    (void)reg; (void)name;
    return NULL;
}

static const struct ggml_backend_reg_i phi_reg_i = {
    /* .get_name         = */ phi_reg_get_name,
    /* .get_device_count = */ phi_reg_get_device_count,
    /* .get_device       = */ phi_reg_get_device,
    /* .get_proc_address = */ phi_reg_get_proc_address,
};

/* Called by ggml_backend_init in lib.rs, the symbol ggml_backend_load looks up. */
ggml_backend_reg_t ggml_backend_phi_reg(void)
{
    static struct ggml_backend_reg reg;
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface = phi_reg_i;
    reg.context = NULL;
    return &reg;
}

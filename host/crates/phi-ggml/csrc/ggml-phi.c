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
int64_t phi_ggml_begin(const uint8_t *a, uint32_t a_type, uint64_t m, uint64_t k, uint64_t nb_a, int keep,
                       const uint8_t *b, uint64_t n, uint64_t nb_b);
int phi_ggml_host_range(uint64_t i, uint64_t *from, uint64_t *to);
int phi_ggml_end(uint8_t *d, uint64_t nb_d);
void phi_ggml_free_all(void);

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
    X(struct ggml_tensor *, mul_mat, ggml_mul_mat, (struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *)) \
    X(struct ggml_cgraph *, new_graph_custom, ggml_new_graph_custom, (struct ggml_context *, size_t, bool)) \
    X(void, build_forward_expand, ggml_build_forward_expand, (struct ggml_cgraph *, struct ggml_tensor *)) \
    X(enum ggml_status, backend_graph_compute, ggml_backend_graph_compute, (ggml_backend_t, struct ggml_cgraph *)) \
    X(ggml_backend_dev_t, dev_by_type, ggml_backend_dev_by_type, (enum ggml_backend_dev_type)) \
    X(ggml_backend_t, dev_init, ggml_backend_dev_init, (ggml_backend_dev_t, const char *)) \
    X(ggml_backend_reg_t, dev_backend_reg, ggml_backend_dev_backend_reg, (ggml_backend_dev_t)) \
    X(void *, reg_get_proc_address, ggml_backend_reg_get_proc_address, (ggml_backend_reg_t, const char *))

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
    fprintf(stderr, "ggml-phi: the host's rows run on ggml's CPU backend with %d threads\n", threads);
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

static bool phi_supports_mul_mat(const struct ggml_tensor *op)
{
    const struct ggml_tensor *src0 = op->src[0], *src1 = op->src[1];
    if (!src0 || !src1) return false;
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
    return phi_ggml_supports((uint32_t)t, (uint64_t)src0->ne[1], (uint64_t)src0->ne[0], src0->nb[1], src1->nb[1], (uint64_t)src1->ne[1]) != 0;
}

/* ---- backend ---- */

static const char *phi_backend_get_name(ggml_backend_t backend) { (void)backend; return "Phi"; }
static void phi_backend_free(ggml_backend_t backend) { free(backend); }

static enum ggml_status phi_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph)
{
    (void)backend;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor *node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        switch (node->op) {
        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor *src0 = node->src[0], *src1 = node->src[1];
            int keep = strstr(src0->name, "weight") != NULL;
            int64_t nr = phi_ggml_begin(src0->data, (uint32_t)card_type(src0->type), (uint64_t)src0->ne[1], (uint64_t)src0->ne[0],
                                        src0->nb[1], keep, src1->data, (uint64_t)src1->ne[1], src1->nb[1]);
            if (nr < 0) return GGML_STATUS_FAILED;
            for (int64_t r = 0; r < nr; r++) {
                uint64_t from, to;
                if (!phi_ggml_host_range((uint64_t)r, &from, &to)) break;
                if (to > from && host_rows(node, (int64_t)from, (int64_t)to) != 0) return GGML_STATUS_FAILED;
            }
            if (phi_ggml_end(node->data, node->nb[1]) != 0) return GGML_STATUS_FAILED;
            break;
        }
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
static size_t phi_reg_get_device_count(ggml_backend_reg_t reg) { (void)reg; return 1; }

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

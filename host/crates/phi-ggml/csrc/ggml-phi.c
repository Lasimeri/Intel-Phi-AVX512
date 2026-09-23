/* ggml-phi.c: the ggml side of the card backend, the glue ggml's C
 * interface requires and nothing more. It registers one device of type
 * ACCEL named "Phi" (like ggml's BLAS backend: it shares the CPU's host
 * buffers, so weights need no copy on the host), accepts MUL_MAT nodes
 * whose shapes the card service takes, and hands each one to
 * phi_ggml_mul_mat in the Rust library this file is compiled into. The
 * few ggml functions it needs (the CPU buffer type) are resolved at run
 * time from the program that loaded the backend, so this object links
 * against nothing of ggml. See ggml-phi.md. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

int phi_ggml_open(void);
uint64_t phi_ggml_max_tensor(void);
uint64_t phi_ggml_max_batch(void);
int phi_ggml_mul_mat(const uint8_t *a, uint64_t a_bytes, uint32_t a_type, uint64_t m, uint64_t k, uint64_t nb_a, int keep,
                     const uint8_t *b, uint64_t n, uint64_t nb_b, uint8_t *d, uint64_t nb_d);
void phi_ggml_free_all(void);

typedef ggml_backend_buffer_type_t (*cpu_buft_fn)(void);
typedef ggml_backend_buffer_t (*cpu_from_ptr_fn)(void *, size_t);
typedef bool (*buft_is_host_fn)(ggml_backend_buffer_type_t);
static cpu_buft_fn p_cpu_buft;
static cpu_from_ptr_fn p_cpu_from_ptr;
static buft_is_host_fn p_buft_is_host;

static int resolve(void)
{
    if (p_cpu_buft) return 0;
    p_cpu_buft = (cpu_buft_fn)dlsym(RTLD_DEFAULT, "ggml_backend_cpu_buffer_type");
    p_cpu_from_ptr = (cpu_from_ptr_fn)dlsym(RTLD_DEFAULT, "ggml_backend_cpu_buffer_from_ptr");
    p_buft_is_host = (buft_is_host_fn)dlsym(RTLD_DEFAULT, "ggml_backend_buft_is_host");
    if (!p_cpu_buft || !p_cpu_from_ptr || !p_buft_is_host) {
        fprintf(stderr, "ggml-phi: the program has no ggml CPU buffer functions to share\n");
        return -1;
    }
    return 0;
}

/* ---- what the card service takes ---- */

static bool phi_supports_mul_mat(const struct ggml_tensor *op)
{
    const struct ggml_tensor *src0 = op->src[0], *src1 = op->src[1];
    if (!src0 || !src1) return false;
    if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) return false;
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    size_t ts = src0->type == GGML_TYPE_F16 ? 2 : 4;
    /* rows contiguous, no batch or broadcast dimensions (attention stays on the CPU for now) */
    if (src0->nb[0] != ts || src1->nb[0] != 4 || op->nb[0] != 4) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (src0->ne[0] != src1->ne[0]) return false;
    if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) return false;
    uint64_t m = (uint64_t)src0->ne[1], k = (uint64_t)src0->ne[0], n = (uint64_t)src1->ne[1];
    if ((uint64_t)src0->nb[1] < k * ts || (uint64_t)src1->nb[1] < k * 4) return false;
    /* float16 rows are up-converted by an aligned load: 32-byte rows and base */
    if (src0->type == GGML_TYPE_F16 && (src0->nb[1] % 32 != 0)) return false;
    if (m * src0->nb[1] > phi_ggml_max_tensor()) return false;
    if (n * src1->nb[1] > phi_ggml_max_batch() || n * m * 4 > phi_ggml_max_batch()) return false;
    return true;
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
            if (src0->type == GGML_TYPE_F16 && ((uintptr_t)src0->data % 32) != 0) {
                fprintf(stderr, "ggml-phi: %s: float16 weights at an unaligned address\n", node->name);
                return GGML_STATUS_FAILED;
            }
            int keep = strstr(src0->name, "weight") != NULL;
            int r = phi_ggml_mul_mat(src0->data, (uint64_t)src0->ne[1] * src0->nb[1],
                                     src0->type == GGML_TYPE_F16 ? 1 : 0,
                                     (uint64_t)src0->ne[1], (uint64_t)src0->ne[0], src0->nb[1], keep,
                                     src1->data, (uint64_t)src1->ne[1], src1->nb[1],
                                     node->data, node->nb[1]);
            if (r != 0) return GGML_STATUS_FAILED;
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
static const char *phi_dev_get_description(ggml_backend_dev_t dev) { (void)dev; return "Intel Xeon Phi 3120 (AVX-512 executed on the card)"; }
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
    if (resolve() != 0 || phi_ggml_open() != 0) return NULL;
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

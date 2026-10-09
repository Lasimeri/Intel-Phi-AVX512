/* ggml-layout-check.c: the ggml layout the assembly backend is built on,
 * taken from ggml's own headers as tcc reads them, and compared with the
 * numbers assembled into host/asm/ggml-phi/ggml_layout.inc.
 *
 *   tcc -I$LLAMA_CPP_DIR/ggml/include -I$LLAMA_CPP_DIR/ggml/src \
 *       -run tools/ggml-layout-check.c host/asm/ggml-phi/ggml_layout.inc          (make layout-check)
 *   tcc ... -run tools/ggml-layout-check.c --gen > host/asm/ggml-phi/ggml_layout.inc
 *
 * Every offset, size and enum value the assembly reads or writes of a
 * ggml structure (`struct ggml_tensor`, the graph, the buffer, the
 * registration, device and backend structures and their tables of
 * function pointers, the device properties, the init parameters, the
 * threadpool parameters) and every enum value it compares against is
 * here by name, so that a ggml that moves a field fails this check
 * rather than the backend reading the wrong word. Exit 1 on the first
 * difference. See ggml-layout-check.md. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

/* ggml-impl.h's inline helpers call this; nothing here does. */
void ggml_abort(const char *file, int line, const char *fmt, ...)
{
    (void)file; (void)line; (void)fmt;
    abort();
}

struct entry {
    const char *name;
    long value;
};

#define OFF(s, f) (long)offsetof(struct s, f)
#define SZ(s) (long)sizeof(struct s)

static const struct entry table[] = {
    /* struct ggml_tensor */
    { "T_BYTES", SZ(ggml_tensor) },
    { "T_TYPE", OFF(ggml_tensor, type) },
    { "T_BUFFER", OFF(ggml_tensor, buffer) },
    { "T_NE", OFF(ggml_tensor, ne) },
    { "T_NB", OFF(ggml_tensor, nb) },
    { "T_OP", OFF(ggml_tensor, op) },
    { "T_OP_PARAMS", OFF(ggml_tensor, op_params) },
    { "T_FLAGS", OFF(ggml_tensor, flags) },
    { "T_SRC", OFF(ggml_tensor, src) },
    { "T_VIEW_SRC", OFF(ggml_tensor, view_src) },
    { "T_DATA", OFF(ggml_tensor, data) },
    { "T_NAME", OFF(ggml_tensor, name) },
    { "GGML_MAX_DIMS", GGML_MAX_DIMS },
    { "GGML_MAX_SRC", GGML_MAX_SRC },
    { "GGML_MAX_NAME", GGML_MAX_NAME },
    /* struct ggml_cgraph */
    { "CG_N_NODES", OFF(ggml_cgraph, n_nodes) },
    { "CG_NODES", OFF(ggml_cgraph, nodes) },
    /* struct ggml_backend_buffer (the buffer type of a weight's buffer) */
    { "BUF_BUFT", OFF(ggml_backend_buffer, buft) },
    /* struct ggml_backend_reg and its interface */
    { "REG_BYTES", SZ(ggml_backend_reg) },
    { "REG_API_VERSION", OFF(ggml_backend_reg, api_version) },
    { "REG_IFACE", OFF(ggml_backend_reg, iface) },
    { "REG_CONTEXT", OFF(ggml_backend_reg, context) },
    { "REG_I_BYTES", SZ(ggml_backend_reg_i) },
    { "REG_I_GET_NAME", OFF(ggml_backend_reg_i, get_name) },
    { "REG_I_GET_DEVICE_COUNT", OFF(ggml_backend_reg_i, get_device_count) },
    { "REG_I_GET_DEVICE", OFF(ggml_backend_reg_i, get_device) },
    { "REG_I_GET_PROC_ADDRESS", OFF(ggml_backend_reg_i, get_proc_address) },
    /* struct ggml_backend_device and its interface */
    { "DEV_BYTES", SZ(ggml_backend_device) },
    { "DEV_IFACE", OFF(ggml_backend_device, iface) },
    { "DEV_REG", OFF(ggml_backend_device, reg) },
    { "DEV_CONTEXT", OFF(ggml_backend_device, context) },
    { "DEV_I_BYTES", SZ(ggml_backend_device_i) },
    { "DEV_I_GET_NAME", OFF(ggml_backend_device_i, get_name) },
    { "DEV_I_GET_DESCRIPTION", OFF(ggml_backend_device_i, get_description) },
    { "DEV_I_GET_MEMORY", OFF(ggml_backend_device_i, get_memory) },
    { "DEV_I_GET_TYPE", OFF(ggml_backend_device_i, get_type) },
    { "DEV_I_GET_PROPS", OFF(ggml_backend_device_i, get_props) },
    { "DEV_I_INIT_BACKEND", OFF(ggml_backend_device_i, init_backend) },
    { "DEV_I_GET_BUFFER_TYPE", OFF(ggml_backend_device_i, get_buffer_type) },
    { "DEV_I_GET_HOST_BUFFER_TYPE", OFF(ggml_backend_device_i, get_host_buffer_type) },
    { "DEV_I_BUFFER_FROM_HOST_PTR", OFF(ggml_backend_device_i, buffer_from_host_ptr) },
    { "DEV_I_SUPPORTS_OP", OFF(ggml_backend_device_i, supports_op) },
    { "DEV_I_SUPPORTS_BUFT", OFF(ggml_backend_device_i, supports_buft) },
    { "DEV_I_OFFLOAD_OP", OFF(ggml_backend_device_i, offload_op) },
    { "DEV_I_EVENT_NEW", OFF(ggml_backend_device_i, event_new) },
    { "DEV_I_EVENT_FREE", OFF(ggml_backend_device_i, event_free) },
    { "DEV_I_EVENT_SYNCHRONIZE", OFF(ggml_backend_device_i, event_synchronize) },
    /* struct ggml_backend and its interface */
    { "BE_BYTES", SZ(ggml_backend) },
    { "BE_GUID", OFF(ggml_backend, guid) },
    { "BE_IFACE", OFF(ggml_backend, iface) },
    { "BE_DEVICE", OFF(ggml_backend, device) },
    { "BE_CONTEXT", OFF(ggml_backend, context) },
    { "BE_I_BYTES", SZ(ggml_backend_i) },
    { "BE_I_GET_NAME", OFF(ggml_backend_i, get_name) },
    { "BE_I_FREE", OFF(ggml_backend_i, free) },
    { "BE_I_GRAPH_COMPUTE", OFF(ggml_backend_i, graph_compute) },
    /* struct ggml_backend_dev_props */
    { "PROPS_NAME", OFF(ggml_backend_dev_props, name) },
    { "PROPS_DESCRIPTION", OFF(ggml_backend_dev_props, description) },
    { "PROPS_TYPE", OFF(ggml_backend_dev_props, type) },
    { "PROPS_MEMORY_FREE", OFF(ggml_backend_dev_props, memory_free) },
    { "PROPS_MEMORY_TOTAL", OFF(ggml_backend_dev_props, memory_total) },
    { "PROPS_CAPS", OFF(ggml_backend_dev_props, caps) },
    { "PROPS_CAPS_ASYNC", OFF(ggml_backend_dev_props, caps.async) },
    { "PROPS_CAPS_HOST_BUFFER", OFF(ggml_backend_dev_props, caps.host_buffer) },
    { "PROPS_CAPS_BUFFER_FROM_HOST_PTR", OFF(ggml_backend_dev_props, caps.buffer_from_host_ptr) },
    { "PROPS_CAPS_EVENTS", OFF(ggml_backend_dev_props, caps.events) },
    /* struct ggml_init_params (passed by value: three words, in memory) */
    { "INIT_BYTES", SZ(ggml_init_params) },
    { "INIT_MEM_SIZE", OFF(ggml_init_params, mem_size) },
    { "INIT_MEM_BUFFER", OFF(ggml_init_params, mem_buffer) },
    { "INIT_NO_ALLOC", OFF(ggml_init_params, no_alloc) },
    /* struct ggml_threadpool_params */
    { "TPP_BYTES", SZ(ggml_threadpool_params) },
    { "TPP_N_THREADS", OFF(ggml_threadpool_params, n_threads) },
    { "TPP_POLL", OFF(ggml_threadpool_params, poll) },
    /* the enums compared against */
    { "GGML_BACKEND_API_VERSION", GGML_BACKEND_API_VERSION },
    { "GGML_TYPE_F32", GGML_TYPE_F32 },
    { "GGML_TYPE_F16", GGML_TYPE_F16 },
    { "GGML_TYPE_Q8_0", GGML_TYPE_Q8_0 },
    { "GGML_TYPE_Q4_K", GGML_TYPE_Q4_K },
    { "GGML_TYPE_Q5_K", GGML_TYPE_Q5_K },
    { "GGML_TYPE_Q6_K", GGML_TYPE_Q6_K },
    { "GGML_TYPE_IQ4_XS", GGML_TYPE_IQ4_XS },
    { "GGML_TYPE_BF16", GGML_TYPE_BF16 },
    { "GGML_TYPE_I32", GGML_TYPE_I32 },
    { "GGML_OP_NONE", GGML_OP_NONE },
    { "GGML_OP_RESHAPE", GGML_OP_RESHAPE },
    { "GGML_OP_VIEW", GGML_OP_VIEW },
    { "GGML_OP_PERMUTE", GGML_OP_PERMUTE },
    { "GGML_OP_TRANSPOSE", GGML_OP_TRANSPOSE },
    { "GGML_OP_MUL_MAT", GGML_OP_MUL_MAT },
    { "GGML_OP_MUL_MAT_ID", GGML_OP_MUL_MAT_ID },
    { "GGML_OP_GLU", GGML_OP_GLU },
    { "GGML_GLU_OP_SWIGLU", GGML_GLU_OP_SWIGLU },
    { "GGML_TENSOR_FLAG_OUTPUT", GGML_TENSOR_FLAG_OUTPUT },
    { "GGML_TENSOR_FLAG_COMPUTE", GGML_TENSOR_FLAG_COMPUTE },
    { "GGML_STATUS_SUCCESS", GGML_STATUS_SUCCESS },
    { "GGML_STATUS_FAILED", GGML_STATUS_FAILED },
    { "GGML_STATUS_ALLOC_FAILED", GGML_STATUS_ALLOC_FAILED },
    { "GGML_BACKEND_DEVICE_TYPE_CPU", GGML_BACKEND_DEVICE_TYPE_CPU },
    { "GGML_BACKEND_DEVICE_TYPE_ACCEL", GGML_BACKEND_DEVICE_TYPE_ACCEL },
    { "GGML_HINT_SRC0_IS_HADAMARD", GGML_HINT_SRC0_IS_HADAMARD },
};

#define N (sizeof table / sizeof table[0])

static void gen(void)
{
    printf("# ggml_layout.inc: the ggml layout the backend is assembled against,\n"
           "# written by tools/ggml-layout-check.c --gen from ggml's own headers\n"
           "# and checked against them by `make layout-check`. Do not edit; see\n"
           "# ggml_layout.md.\n");
    for (size_t i = 0; i < N; i++)
        printf("\t.set %s, %ld\n", table[i].name, table[i].value);
}

/* The `.set NAME, VALUE` lines of the include, compared by name. */
static int check(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ggml-layout-check: cannot open %s\n", path);
        return 1;
    }
    char line[512];
    int seen[N];
    memset(seen, 0, sizeof seen);
    int fails = 0;
    while (fgets(line, sizeof line, f)) {
        char name[128];
        long value;
        if (sscanf(line, " .set %127[A-Za-z0-9_], %ld", name, &value) != 2) continue;
        size_t i;
        for (i = 0; i < N; i++)
            if (strcmp(name, table[i].name) == 0) break;
        if (i == N) {
            printf("UNKNOWN  %s (in the include, not in the headers' table)\n", name);
            fails = 1;
            continue;
        }
        seen[i] = 1;
        if (value != table[i].value) {
            printf("MISMATCH %-32s include=%ld headers=%ld\n", name, value, table[i].value);
            fails = 1;
        }
    }
    fclose(f);
    for (size_t i = 0; i < N; i++)
        if (!seen[i]) {
            printf("MISSING  %s (in the headers' table, not in the include)\n", table[i].name);
            fails = 1;
        }
    if (!fails) printf("ggml-layout-check: ok (%zu values)\n", N);
    return fails;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--gen") == 0) {
        gen();
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: ggml-layout-check --gen | ggml-layout-check INCLUDE\n");
        return 2;
    }
    return check(argv[1]);
}

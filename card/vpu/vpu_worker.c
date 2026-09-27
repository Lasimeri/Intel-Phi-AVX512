/* phi-vpu-worker: the card side of the AVX-512 co-processor.
 *
 * Stays resident, polls a doorbell in host memory, and when work arrives
 * runs it across the card's vector units. The kernels it runs are AVX-512
 * machine code translated to this card's own instruction set ahead of
 * time by host/crates/avx512-xlate, so what executes here is the
 * program's own arithmetic, not a reimplementation of it.
 *
 * Data moves through the block device rather than the mapping, because
 * the mapping is uncached and streams at 50 MB/s while the DMA engine
 * behind the block device does 1.2 GB/s. See vpu_proto.md. The exception
 * is the matrix multiplies' small transfers, which go through the mapping
 * in whole 64-byte vectors, one transaction each (vpu_matmul.c, -m).
 *
 *   phi-vpu-worker [-v] [-s MS] [-i US] [-e N] [-m 0|1] [threads]
 *     threads 1 to 228 (57); spin MS after a job before parking (200);
 *     -e N huge pages the seamless path pools (256; 0 leaves them all to
 *     the matrix multiplies, which is what a ggml backend run wants);
 *     once parked, poll the doorbell every US microseconds (500);
 *     -m 0 sends the matrix multiplies' small transfers through the
 *     block device too (1, the default: through the mapping)
 *
 * The threads are created once. Creating one costs about 0.58 ms on
 * this card, and the first version of this worker created and joined
 * them per request, so that 57 threads took 34.9 ms to do 1.85 ms of
 * work. Every thread in the pool is pinned to its own hardware thread
 * at start-up and stays there.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include "vpu_proto.h"
#include "vpu_exec.h"
#include "vpu_matmul.h"

/* Translated from AVX-512 by avx512-xlate; see card/examples/avx512_poly.S */
void poly_kernel_x8(float *d, const float *x, const float *coef, long n);
/* 64-byte vector copies (vpu_matmul_kernel.S, kernelgen/copy.rs): whole-line
 * loads from the uncached control area. */
void phi_copy64(void *dst, const void *src, long count);

#define CTRL_BYTES 16384   /* control words, the mailbox and the exec descriptor (vpu_exec.h) */
#define WIN_BYTES (768UL << 20)   /* the window the matmul service uses: OFF_D + D_MAX (matmul.rs) */

unsigned char *g_window;   /* the whole window, mapped; NULL when the mapping failed */
void *vpu_window(uint64_t off, size_t len)
{
    if (!g_window || off + len > WIN_BYTES) return NULL;
    return g_window + off;
}
#define MAX_POOL 227          /* 228 hardware threads less the dispatcher */
#define SPIN_ROUNDS 2000      /* polls of the generation word between clock reads */
static uint64_t spin_ns = 200000000ULL;  /* spin this long after the last job, then park; -s MS */
static long idle_us = 500;               /* doorbell poll interval once idle; -i US */

/* musl on the card ships no linux/futex.h; these are the x86-64 numbers. */
#ifndef SYS_futex
#define SYS_futex 202
#endif
#define FUTEX_WAIT 0
#define FUTEX_WAKE 1

/* Knights Corner has no MFENCE (ISA reference 327364-001, appendix B).
 * A locked read-modify-write is the full fence on this core, the same
 * idiom the vector store kernels in libknc use. It is needed exactly
 * where a store must be visible before a following load: x86 orders
 * everything else. */
#define FULL_FENCE() __asm__ __volatile__("lock addq $0, (%%rsp)" ::: "memory", "cc")
#define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")

static int verbose;

/* -t: the pool's own timings, per dispatch, from the time stamp counter:
 * how long the slowest thread took to see the generation, how long the
 * longest slice ran, and how long the dispatcher took to see the last
 * one finish. Summed per slice function and printed every N
 * dispatches (-t N), so the tracing itself costs one line per N. */
static int trace, trace_every = 1000;
static inline uint64_t tsc(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
struct stamp { volatile uint64_t start, end; } __attribute__((aligned(64)));
static double tsc_per_us = 1100.0;

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

/* Knights Corner numbering: CPU 0 is core 56 thread 3, and CPUs 1+4k
 * through 4+4k are core k. Filling core-major puts one thread on every
 * core before any core gets a second, which is what the two-cycle
 * decoder wants (SSDG 2.1.2). */
/* Every pool thread handles the exec engine's signals on its own stack
 * (vpu_exec.c): the program's stack is the last place a handler frame may
 * go while the card writes the program's memory back. */
static char pool_altstacks[MAX_POOL][64 << 10] __attribute__((aligned(16)));
static void pool_altstack(int t)
{
    stack_t ss = { .ss_sp = pool_altstacks[t], .ss_size = sizeof pool_altstacks[t], .ss_flags = 0 };
    sigaltstack(&ss, NULL);
}

static int knc_cpu(int core, int slot)
{
    int cpu = 1 + core * 4 + slot;
    return (cpu >= 228) ? 0 : cpu;
}

static void pin(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof set, &set);
}

static long futex(volatile uint32_t *addr, int op, uint32_t val)
{
    return syscall(SYS_futex, addr, op, val, NULL, NULL, 0);
}

/* ------------------------------------------------------------------ */
/* The pool                                                            */

struct job {
    float *in, *out, *coef;
    long begin, count;
    void (*fn)(void *arg, int slice, int nslices);   /* a generic job (vpu_pool_map), else the polynomial */
    void *arg;
    int slice, nslices;
};

static struct {
    int nthreads;               /* pool threads; the dispatcher is one more */
    pthread_t th[MAX_POOL];
    struct job jobs[MAX_POOL + 1];   /* jobs[nthreads] is the dispatcher's own */
    volatile uint32_t done __attribute__((aligned(64)));   /* cores finished with this generation */
    volatile uint32_t parked __attribute__((aligned(64)));   /* pool threads asleep in the kernel */
} pool;

/* The generation and, for vpu_pool_map, the whole of its work, on one
 * line: a thread that sees the bump has the job in the same fetch, and
 * the threads of a core share that core's L1, so one fetch per core wakes
 * all of them. `nslices` 0 means the per-thread jobs of `pool.jobs` (the
 * polynomial's dispatch). */
static struct {
    volatile uint32_t gen;      /* bumped once per request */
    int nslices;
    void (*fn)(void *arg, int slice, int nslices);
    void *arg;
} bc __attribute__((aligned(64)));

/* Pool threads per core, and each core's own count of threads finished,
 * on a line of that core's: the threads of a core add to it (a locked add
 * on a line in their own L1, no ring traffic) and the one that completes
 * the core's count adds once to `pool.done`, so the dispatcher waits for
 * 57 cores rather than 113 threads. Never reset: every thread adds once
 * per generation, so a core's count is a multiple of its threads exactly
 * when all of them are done. */
struct corecount { volatile uint64_t n; } __attribute__((aligned(64)));
static struct corecount cores[57];
static int core_threads[57];
static uint32_t busy_cores;

/* delay r32 (ISA reference 327364-001, appendix A: VEX.128.F3.0F.W0 AE
 * /6): the thread neither fetches nor issues for `cycles`. A core's four
 * hardware threads share its issue slots, so a thread spinning while
 * another thread of its core computes takes cycles from that one; a wait
 * that delays between looks takes almost none. Measured on the card: 1000
 * cycles in 925 ns. */
static inline void knc_delay(uint32_t cycles)
{
    __asm__ __volatile__(".byte 0xc5, 0xfa, 0xae, 0xf0" : : "a"(cycles) : "memory");
}
#define WAIT_DELAY 64   /* cycles between looks at a word another core will write */

/* The hardware thread a pool thread's slot on its core becomes: a core's
 * first two threads go to its hardware threads 0 and 3, keeping 1 and 2,
 * where the card's own kernel threads were seen running (the block
 * devices' pollers on CPUs 222 and 223, core 55's threads 1 and 2), free.
 * Core 56's thread 3 is CPU 0, the dispatcher's, so there the second pool
 * thread takes 1. */
static int pool_slot(int core, int mate)
{
    static const int order[4] = { 0, 3, 1, 2 }, order56[4] = { 0, 1, 2, 3 };
    return core == 56 ? order56[mate & 3] : order[mate & 3];
}

/* -t: each thread's start and end of the current generation, on lines of
 * their own; the sums per slice function (a handful: the kernels, the
 * copies). */
static struct stamp stamps[MAX_POOL + 1];
#define TRACE_FNS 8
static struct {
    void (*fn)(void *, int, int);
    uint64_t n, see_last, run_max, run_disp, tail, total;
    uint32_t late[MAX_POOL + 1];   /* how often each thread was the last to see the generation */
} tr[TRACE_FNS];

static void trace_note(void (*fn)(void *, int, int), int nslices, uint64_t t0, uint64_t t1)
{
    int f = 0;
    while (f < TRACE_FNS && tr[f].fn && tr[f].fn != fn) f++;
    if (f == TRACE_FNS) return;
    tr[f].fn = fn;
    uint64_t see = 0, run = 0, last_end = 0;
    int late = 0;
    for (int t = 0; t < pool.nthreads; t++) {
        uint64_t s = stamps[t].start - t0, r = stamps[t].end - stamps[t].start;
        if (s > see) { see = s; late = t; }
        if (t < nslices - 1 && r > run) run = r;
        if (stamps[t].end > last_end) last_end = stamps[t].end;
    }
    uint64_t disp = stamps[pool.nthreads].end - stamps[pool.nthreads].start;
    if (stamps[pool.nthreads].end > last_end) last_end = stamps[pool.nthreads].end;
    tr[f].n++;
    tr[f].see_last += see;
    tr[f].run_max += run;
    tr[f].run_disp += disp;
    tr[f].tail += t1 - last_end;
    tr[f].total += t1 - t0;
    tr[f].late[late]++;
    if (tr[f].n % (uint64_t)trace_every) return;
    double k = 1.0 / (tsc_per_us * (double)trace_every);
    int worst = 0;
    for (int t = 1; t < pool.nthreads; t++) if (tr[f].late[t] > tr[f].late[worst]) worst = t;
    printf("trace fn %p x%d: per dispatch %.1f us: last thread sees it %.1f, longest pool slice %.1f, dispatcher's slice %.1f, "
           "last end to seen %.1f; most often last to see it: thread %d (CPU %d), %u of %d\n",
           (void *)fn, trace_every, tr[f].total * k, tr[f].see_last * k, tr[f].run_max * k, tr[f].run_disp * k, tr[f].tail * k,
           worst, knc_cpu(worst % 57, worst / 57), tr[f].late[worst], trace_every);
    fflush(stdout);
    tr[f].see_last = tr[f].run_max = tr[f].run_disp = tr[f].tail = tr[f].total = 0;
    memset(tr[f].late, 0, sizeof tr[f].late);
}

/* -t: a multiply request's stages, from the stamps vpu_matmul.c and the
 * poll loop leave in vpu_marks (0 the doorbell seen, 6 the request read,
 * 1 the descriptors read and checked, 2 pulled, 3 the groups built, 4
 * computed, 5 pushed, 7 the reply written), summed per kind of request
 * and printed every -t N of that kind. */
extern uint64_t vpu_marks[8];
static void stage_note(uint32_t kernel)
{
    static const char *names[3] = { "matmul", "matmul_id", "matmul_more" };
    static const int order[8] = { 0, 6, 1, 2, 3, 4, 5, 7 };
    static uint64_t sum[3][7], count[3];
    int k = kernel == VPU_K_MATMUL ? 0 : kernel == VPU_K_MATMUL_ID ? 1 : kernel == VPU_K_MATMUL_MORE ? 2 : -1;
    if (k < 0) return;
    for (int s = 0; s < 7; s++) sum[k][s] += vpu_marks[order[s + 1]] - vpu_marks[order[s]];
    if (++count[k] % (uint64_t)trace_every) return;
    double f = 1.0 / (tsc_per_us * (double)trace_every);
    printf("stages %s x%d, us: read request %.1f, descriptors and checks %.1f, pull %.1f, groups %.1f, compute %.1f, push %.1f, reply %.1f\n",
           names[k], trace_every, sum[k][0] * f, sum[k][1] * f, sum[k][2] * f, sum[k][3] * f, sum[k][4] * f, sum[k][5] * f, sum[k][6] * f);
    fflush(stdout);
    memset(sum[k], 0, sizeof sum[k]);
}

static void run_job(const struct job *j)
{
    if (j->fn) {
        j->fn(j->arg, j->slice, j->nslices);
        return;
    }
    if (j->count > 0) {
        poly_kernel_x8(j->out + j->begin, j->in + j->begin, j->coef, j->count);
    }
}

/* Wait for the generation to move past `seen`.
 *
 * Spin first: a request that follows another closely is picked up in
 * well under a microsecond, with no kernel involved. After SPIN_NS with
 * nothing to do, park in a futex so idle cores are actually idle and the
 * card is not drawing 57 cores' worth of power to wait. The dispatcher
 * wakes parked threads only when there are some, so the common case
 * costs it no system call at all. */
static void wait_for_work(uint32_t seen)
{
    uint64_t t0 = now_ns();
    for (;;) {
        for (int i = 0; i < SPIN_ROUNDS; i++) {
            if (bc.gen != seen) return;
            knc_delay(WAIT_DELAY);
        }
        if (now_ns() - t0 < spin_ns) continue;

        __sync_fetch_and_add(&pool.parked, 1);   /* locked: a full fence */
        /* The kernel compares gen with `seen` atomically against any
         * wake, so a bump that lands between the check above and this
         * call returns EAGAIN rather than sleeping through it. */
        futex(&bc.gen, FUTEX_WAIT, seen);
        __sync_fetch_and_sub(&pool.parked, 1);
        if (bc.gen != seen) return;
        t0 = now_ns();
    }
}

static void *pool_thread(void *arg)
{
    int t = (int)(intptr_t)arg;
    int core = t % 57;
    pin(knc_cpu(core, pool_slot(core, t / 57)));
    pool_altstack(t);
    uint32_t seen = 0;
    for (;;) {
        wait_for_work(seen);
        seen = bc.gen;
        COMPILER_BARRIER();          /* loads of the job follow the load of gen */
        if (trace) stamps[t].start = tsc();
        int nslices = bc.nslices;
        if (nslices == 0) run_job(&pool.jobs[t]);
        else if (t < nslices - 1) bc.fn(bc.arg, t, nslices);
        if (trace) stamps[t].end = tsc();
        /* locked: results are visible first; the thread completing its
         * core's count reports the core */
        if (core_threads[core] == 1 || (__sync_add_and_fetch(&cores[core].n, 1) % (uint64_t)core_threads[core]) == 0)
            __sync_fetch_and_add(&pool.done, 1);
    }
    return NULL;
}

/* Wait for every core's threads to be done with this generation. */
static void wait_done(void)
{
    while (pool.done != busy_cores) knc_delay(WAIT_DELAY);
    COMPILER_BARRIER();
}

static int pool_start(int nthreads)
{
    pool.nthreads = nthreads;
    for (int t = 0; t < nthreads; t++) core_threads[t % 57]++;
    busy_cores = 0;
    for (int c = 0; c < 57; c++) busy_cores += core_threads[c] > 0;
    for (int t = 0; t < nthreads; t++) {
        if (pthread_create(&pool.th[t], NULL, pool_thread, (void *)(intptr_t)t) != 0) {
            perror("pthread_create");
            return -1;
        }
    }
    return 0;
}

/* Cut n elements into `threads` slices of whole chunks and run them: one
 * on each pool thread, the last one on the calling thread, which would
 * otherwise sit spinning on a core that has work to do. Returns how many
 * slices had any elements. */
static int dispatch(int threads, float *in, float *out, float *coef, long n)
{
    if (threads > pool.nthreads + 1) threads = pool.nthreads + 1;
    if (threads < 1) threads = 1;

    long chunks = (n + VPU_CHUNK - 1) / VPU_CHUNK;
    long per = (chunks + threads - 1) / threads, at = 0;
    int live = 0;
    /* Slice s of `threads` goes to pool thread s, except the last slice,
     * which the dispatcher runs itself (slot nthreads). Pool threads at
     * or past the last slice get nothing this round. The loop visits the
     * dispatcher's slot last, so the slices are handed out in order. */
    for (int t = 0; t <= pool.nthreads; t++) {
        struct job *j = &pool.jobs[t];
        int slice = (t == pool.nthreads) ? threads - 1 : (t < threads - 1 ? t : -1);
        long take = 0;
        if (slice >= 0) {
            take = per;
            if (at + take > chunks) take = chunks - at;
            if (take < 0) take = 0;
        }
        j->in = in; j->out = out; j->coef = coef; j->fn = NULL;
        j->begin = at * VPU_CHUNK;
        j->count = take * VPU_CHUNK;
        if (take > 0 && j->begin + j->count > n) j->count = n - j->begin;
        if (take > 0) { at += take; live++; }
    }

    pool.done = 0;
    bc.nslices = 0;              /* the per-thread jobs */
    COMPILER_BARRIER();          /* the jobs are stored before the generation */
    bc.gen++;
    FULL_FENCE();                /* ...and the generation before parked is read */
    if (pool.parked) futex(&bc.gen, FUTEX_WAKE, INT_MAX);

    run_job(&pool.jobs[pool.nthreads]);

    wait_done();
    return live;
}

/* ------------------------------------------------------------------ */
/* Moving bulk data between the host window and card memory             */

static int blk = -1;

/* O_DIRECT is used for correctness before speed: the host changes this
 * memory behind the card's back, so anything the card's page cache
 * remembers about it is stale, and the worker would compute on whatever
 * the last reader left behind. It also wants block-aligned offsets and
 * lengths, which is why both are rounded up here and why every buffer is
 * page aligned and oversized to match.
 *
 * Large requests matter: the DMA engine reaches 1.2 GB/s with 16 MiB
 * reads and only 185 MB/s with 1 MiB ones. */
static size_t round_up(size_t n) { return (n + VPU_BLOCK - 1) & ~(size_t)(VPU_BLOCK - 1); }

static int pull(void *dst, size_t len, uint64_t off)
{
    size_t want = round_up(len), done = 0;
    while (done < want) {
        ssize_t n = pread(blk, (char *)dst + done, want - done, (off_t)(off + done));
        if (n <= 0) return -1;
        done += (size_t)n;
    }
    return 0;
}

static int push(const void *src, size_t len, uint64_t off)
{
    size_t want = round_up(len), done = 0;
    while (done < want) {
        ssize_t n = pwrite(blk, (const char *)src + done, want - done, (off_t)(off + done));
        if (n <= 0) return -1;
        done += (size_t)n;
    }
    return 0;
}

/* The same two, for the matrix-multiply service (vpu_matmul.c). */
int vpu_pull(void *dst, size_t len, uint64_t off) { return pull(dst, len, off); }
int vpu_push(const void *src, size_t len, uint64_t off) { return push(src, len, off); }

/* Buffers persist across requests and only grow. Allocating per request
 * meant every request paid a page fault per 4 KiB of data on first
 * touch; the memset here takes those faults once, outside any timing.
 *
 * They come from 2 MiB huge pages when the card has some reserved
 * (/proc/sys/vm/nr_hugepages; scripts/phi-vpu.sh start sets it), else
 * from 4 KiB pages. The difference is the whole transport: /dev/phiblk1
 * posts one record to the host per physically contiguous run of the
 * buffer, and a 4 KiB-paged buffer fresh from malloc is scattered, 15
 * records per 64 KiB and 88 per 512 KiB request, at about 20 us of host
 * work each; a huge-paged buffer is one record per 512 KiB request.
 * Measured 2026-09-22 with the stack's card/examples/blkbench.c on card
 * 0 (Intel-Phi-3120A, docs/results/2026-09-22-block-pipeline.md): a
 * 512 KiB pread went from 1.8 ms to 0.24 ms, 16 MiB from 7.4 ms to 5.2 ms
 * (the link). */
#define HUGE_BYTES (2UL << 20)
struct buf { float *p; size_t cap; int huge; };

static int reserve(struct buf *b, size_t len)
{
    size_t need = round_up(len) + VPU_BLOCK;
    if (b->cap >= need) return 0;
    if (b->huge) munmap(b->p, b->cap); else free(b->p);
    b->p = NULL;
    b->cap = 0;
    size_t hneed = (need + HUGE_BYTES - 1) & ~(HUGE_BYTES - 1);
    void *p = mmap(NULL, hneed, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) {
        b->huge = 1;
        need = hneed;
    } else {
        b->huge = 0;
        p = NULL;
        if (posix_memalign(&p, VPU_BLOCK, need) != 0) return -1;
        if (verbose) fprintf(stderr, "no huge pages for a %zu MiB buffer; 4 KiB pages (slower transport)\n", need >> 20);
    }
    memset(p, 0, need);
    b->p = p;
    b->cap = need;
    return 0;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    int max_threads = 57;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) verbose++;
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) { trace = 1; trace_every = atoi(argv[++i]); if (trace_every < 1) trace_every = 1; }
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) spin_ns = (uint64_t)atol(argv[++i]) * 1000000ULL;
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) idle_us = atol(argv[++i]);
        else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) vpu_exec_pool(atoi(argv[++i]));
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) vpu_matmul_map_small(atoi(argv[++i]));
        else max_threads = atoi(argv[i]);
    }
    if (max_threads < 1) max_threads = 1;
    if (max_threads > MAX_POOL + 1) max_threads = MAX_POOL + 1;

    int fd = open("/dev/phihost", O_RDWR);
    if (fd < 0) { perror("/dev/phihost"); return 1; }
    volatile unsigned char *ctrl = mmap(NULL, CTRL_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ctrl == MAP_FAILED) { perror("mmap"); return 1; }
    /* The whole window as well, for transfers too small to be worth a
     * block request (vpu_matmul.c, small_pull/small_push): the mapping
     * reaches host memory directly over PCIe, so it has no per-request
     * cost, but each access is a link round trip, so it is only better
     * below a few tens of kilobytes. A failure here is not fatal: the
     * block device serves everything then. */
    g_window = mmap(NULL, WIN_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (g_window == MAP_FAILED) { g_window = NULL; if (verbose) printf("window: no direct mapping, the block device serves every transfer\n"); }

    blk = open("/dev/phiblk1", O_RDWR | O_DIRECT);
    if (blk < 0) { perror("/dev/phiblk1 (O_DIRECT)"); return 1; }

    volatile struct vpu_request *req = (volatile struct vpu_request *)(ctrl + VPU_OFF_REQ);
    volatile struct vpu_reply *rep = (volatile struct vpu_reply *)(ctrl + VPU_OFF_REPLY);
    volatile uint64_t *ready = (volatile uint64_t *)(ctrl + VPU_OFF_READY);

    /* The dispatcher lives on CPU 0, which is core 56's last hardware
     * thread, so the pool's core-major fill (core 0 slot 0, core 1 slot
     * 0, ...) never lands a worker on the same core until all 227 other
     * hardware threads are taken. */
    pin(0);
    if (trace) {
        /* the counter's rate, against the kernel's clock over 20 ms */
        uint64_t n0 = now_ns(), c0 = tsc();
        struct timespec rest = { 0, 20000000L };
        nanosleep(&rest, NULL);
        tsc_per_us = (double)(tsc() - c0) * 1000.0 / (double)(now_ns() - n0);
        printf("trace: the time stamp counter runs at %.1f per us\n", tsc_per_us);
    }
    if (pool_start(max_threads - 1) != 0) return 1;
    if (vpu_exec_init() != 0) return 1;

    printf("phi-vpu-worker: %d threads pinned (%d in the pool plus this one), polling\n",
           max_threads, max_threads - 1);
    fflush(stdout);

    /* The worker owns the reset, not the host.
     *
     * Taking `last` from whatever the window already held meant a request
     * left there by a previous run was indistinguishable from no request
     * at all: the worker would sit polling a sequence number that already
     * matched, and every subsequent request that happened to reuse that
     * number was ignored for ever. Clearing both counters here makes the
     * sequence start from a known point every time the worker starts. */
    req->seq = 0;
    rep->seq = 0;
    rep->status = 0;
    uint64_t last = 0;
    *(volatile int64_t *)(ctrl + VPU_OFF_SCRATCH) = vpu_exec_scratch_tpoff();
    *ready = VPU_MAGIC;

    struct buf in = {0}, out = {0}, coef = {0};

    /* The doorbell is polled flat out for the spin window after the last
     * request, one PCIe read per iteration, which is where the 2.36 us
     * doorbell comes from. After that the poll sleeps between reads so a
     * quiet card is quiet: without this the dispatcher sat at 100 percent
     * of its hardware thread for ever, reading host memory over PCIe a
     * million times a second to learn nothing. nanosleep on this kernel
     * costs about 60 us on top of what is asked (measured 2026-09-22:
     * 10 us asks for 72, 100 us for 162), so the first doorbell after a
     * quiet spell is seen within about idle_us + 60 us. */
    uint64_t idle_since = now_ns();
    unsigned polls = 0;
    for (;;) {
        uint64_t seq = req->seq;
        if (seq == last) {
            /* Keep asserting readiness while idle: the host clears the
             * window before its first request and would otherwise wipe
             * the flag it is about to wait for. */
            *ready = VPU_MAGIC;
            if ((++polls & 63) == 0 && now_ns() - idle_since > spin_ns) {
                struct timespec rest = { 0, (long)idle_us * 1000L };
                nanosleep(&rest, NULL);
            }
            continue;
        }
        last = seq;

        uint64_t t0 = now_ns();
        vpu_marks[0] = tsc();
        /* The request's line in one 64-byte load, where reading its
         * fields one by one was a round trip over the link each (the
         * mapping is uncached): the host wrote them before the sequence
         * number just seen. */
        struct vpu_request rq[64 / sizeof(struct vpu_request) + 1] __attribute__((aligned(64)));
        phi_copy64(rq, (const void *)req, 1);
        vpu_marks[6] = tsc();
        long n = (long)rq[0].n;
        int threads = (int)rq[0].threads;
        uint32_t kernel = rq[0].kernel;
        uint64_t in_off = rq[0].in_off, out_off = rq[0].out_off;
        uint64_t aux_off = rq[0].aux_off, aux_len = rq[0].aux_len;

        int status = VPU_OK, live = 0;
        uint64_t pull_ns = 0, push_ns = 0, compute_ns = 0;
        size_t bytes = (size_t)n * 4;

        if (kernel == VPU_K_EXEC) status = vpu_exec_run(ctrl, blk, verbose);
        else if (kernel == VPU_K_UPLOAD || kernel == VPU_K_MATMUL || kernel == VPU_K_FREE || kernel == VPU_K_MATMUL_ID || kernel == VPU_K_FFN || kernel == VPU_K_MATMUL_MORE)
            status = vpu_matmul_run(ctrl, kernel, threads, verbose, &compute_ns, &pull_ns, &push_ns, &live);
        else if (kernel != VPU_K_POLY30) status = VPU_E_KERNEL;
        else if (n <= 0 || (in_off | out_off | aux_off) % VPU_BLOCK != 0) status = VPU_E_REQUEST;
        else if (reserve(&in, bytes) || reserve(&out, bytes) || reserve(&coef, aux_len)) status = VPU_E_ALLOC;

        if (status == VPU_OK && kernel == VPU_K_POLY30) {
            uint64_t p0 = now_ns();
            if (pull(in.p, bytes, in_off) || pull(coef.p, aux_len, aux_off)) status = VPU_E_PULL;
            pull_ns = now_ns() - p0;
        }
        if (status == VPU_OK && kernel == VPU_K_POLY30) {
            uint64_t c0 = now_ns();
            live = dispatch(threads, in.p, out.p, coef.p, n);
            compute_ns = now_ns() - c0;
            uint64_t p0 = now_ns();
            if (push(out.p, bytes, out_off)) status = VPU_E_PUSH;
            push_ns = now_ns() - p0;
        }

        rep->compute_ns = compute_ns;
        rep->pull_ns = pull_ns;
        rep->push_ns = push_ns;
        rep->total_ns = now_ns() - t0;
        rep->status = status;
        rep->threads = live;
        COMPILER_BARRIER();
        rep->seq = seq;   /* written last: it is what the host polls */
        if (trace && kernel != VPU_K_EXEC) { vpu_marks[7] = tsc(); stage_note(kernel); }
        idle_since = now_ns();

        /* A line per request costs the reply 35 to 60 us on this card (a
         * write to a file on the host-backed disk), which a model's
         * hundreds of small multiplies per token pay each: the matrix
         * service's requests are logged at -v -v only. */
        int service = kernel == VPU_K_UPLOAD || kernel == VPU_K_MATMUL || kernel == VPU_K_FREE || kernel == VPU_K_MATMUL_ID || kernel == VPU_K_FFN ||
                      kernel == VPU_K_MATMUL_MORE;
        if (verbose > (service ? 1 : 0)) {
            printf("seq=%llu kernel=%u n=%ld threads=%d status=%d pull=%.3fms compute=%.3fms push=%.3fms\n",
                   (unsigned long long)seq, kernel, n, live, status,
                   pull_ns / 1e6, compute_ns / 1e6, push_ns / 1e6);
            fflush(stdout);
        }
    }
}

/* Run fn(arg, slice, nslices) on nslices threads: pool threads take
 * slices 0 to nslices - 2, the caller (the dispatcher) runs the last
 * one, and everyone is waited for. The pool must be idle: the caller is
 * the dispatcher between requests, or its exec engine inside one. */
int vpu_pool_map(void (*fn)(void *arg, int slice, int nslices), void *arg, int nslices)
{
    if (nslices > pool.nthreads + 1) nslices = pool.nthreads + 1;
    if (nslices < 1) nslices = 1;
    /* One job for everyone, on the generation's line: slice s goes to pool
     * thread s and the last to this thread (the dispatcher), which would
     * otherwise wait on a core that has work to do. */
    pool.done = 0;
    bc.fn = fn;
    bc.arg = arg;
    bc.nslices = nslices;
    COMPILER_BARRIER();
    uint64_t t0 = trace ? tsc() : 0;
    bc.gen++;
    FULL_FENCE();
    if (pool.parked) futex(&bc.gen, FUTEX_WAKE, INT_MAX);
    if (trace) stamps[pool.nthreads].start = tsc();
    fn(arg, nslices - 1, nslices);
    if (trace) stamps[pool.nthreads].end = tsc();
    wait_done();
    if (trace) trace_note(fn, nslices, t0, tsc());
    return nslices;
}

int vpu_pool_threads(void)
{
    return pool.nthreads;
}

/* One slice per core, the most a copy through the uncached window can
 * use: a core's loads and stores over the link go one at a time whatever
 * number of its threads issue them, so a second thread per core only
 * adds a slice (the seamless path's copies measured 20 percent slower cut
 * for 114 than for 57, 2026-09-27). */
int vpu_pool_cores(void)
{
    int n = pool.nthreads + 1;
    return n < 57 ? n : 57;
}

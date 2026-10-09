/* avx512-f-forms-test: the AVX-512F forms llama.cpp's AVX-512F build met
 * (2026-10-08: vcomiss, vpcmpq and its kin, vinsertps, vshuff32x4,
 * vpmaxsq, vptestnmq, vpmovdb, vpscatterdd, vfmaddsub132ps), one at a
 * time, each checked against plain C. Every instruction is inline
 * assembly with the %{evex%} prefix and the high vector registers, so the
 * encoding is exactly the EVEX form a compiler emits, and the check is
 * bit-exact. The program runs the same on the card and under the
 * emulator, and both must print PASS.
 *
 *   gcc -O1 -mavx512f -mno-avx512vl -mno-avx512bw -mno-avx512dq -mno-avx512cd \
 *       -o avx512-f-forms-test tools/avx512-f-forms-test.c -lm
 *   scripts/phi512.sh --card N ./avx512-f-forms-test
 *   scripts/phi512.sh --emulate ./avx512-f-forms-test
 *
 * See avx512-f-forms-test.md. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef union {
    float f[16];
    double d[8];
    uint32_t u[16];
    int32_t i[16];
    uint64_t q[8];
    int64_t s[8];
    uint8_t b[64];
} V;

static int fails;

static void check(const char *name, const V *got, const V *want, int lanes)
{
    for (int i = 0; i < lanes; i++)
        if (got->u[i] != want->u[i]) {
            printf("FAIL %-48s lane %2d: got %08x want %08x\n", name, i, got->u[i], want->u[i]);
            fails++;
            return;
        }
    printf("ok   %s\n", name);
}

static void check_u64(const char *name, uint64_t got, uint64_t want)
{
    if (got != want) {
        printf("FAIL %-48s got %016llx want %016llx\n", name, (unsigned long long)got, (unsigned long long)want);
        fails++;
        return;
    }
    printf("ok   %s\n", name);
}

#define ASM(body, out, ...) asm volatile(body : "=m"(out) : __VA_ARGS__ : "xmm16", "xmm17", "xmm18", "xmm19", "k1", "k2", "rax", "rcx", "memory")

/* The flags vcomiss leaves, read with lahf (SF ZF - AF - PF 1 CF into AH; bit 1 is
 * the architecture's constant, not the instruction's, and is masked out)
 * and seto (OF into AL): bit 8 + n is flag bit n of the low byte, bit 0 is OF. */
#define COMI_FLAGS(insn, out, ...) \
    asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t" insn "\n\tlahf\n\tseto %%al\n\tmovzwl %%ax, %%eax\n\tmov %%eax, %0" \
                 : "=m"(out) : __VA_ARGS__ : "xmm16", "xmm17", "rax", "cc", "memory")

/* What the instruction defines: ZF, PF and CF for unordered; ZF for equal;
 * CF for less; nothing for greater. SF, AF and OF are cleared. */
static uint32_t comi_want(double x, double y)
{
    uint32_t ah = 0;
    if (isnan(x) || isnan(y)) ah |= 0x45;
    else if (x == y) ah |= 0x40;
    else if (x < y) ah |= 0x01;
    return ah << 8;
}

/* vpcmpq's eight predicates on signed and unsigned 64-bit lanes. */
static int pred_s(int p, int64_t a, int64_t b)
{
    switch (p & 7) {
    case 0: return a == b;
    case 1: return a < b;
    case 2: return a <= b;
    case 3: return 0;
    case 4: return a != b;
    case 5: return a >= b;
    case 6: return a > b;
    default: return 1;
    }
}
static int pred_u(int p, uint64_t a, uint64_t b)
{
    switch (p & 7) {
    case 0: return a == b;
    case 1: return a < b;
    case 2: return a <= b;
    case 3: return 0;
    case 4: return a != b;
    case 5: return a >= b;
    case 6: return a > b;
    default: return 1;
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0); /* every line survives a stop at a later check */
    V a, b, r, w;
    uint32_t got32;
    uint64_t got;

    /* 1. vcomiss / vucomiss / vcomisd: lane 0 compared, the flags written */
    {
        static const float xs[4] = {1.5f, 2.0f, 3.0f, NAN};
        static const float ys[4] = {2.0f, 2.0f, 1.0f, 1.0f};
        static const char *names[4] = {"vcomiss less (CF)", "vcomiss equal (ZF)", "vcomiss greater (nothing)", "vcomiss unordered (ZF PF CF)"};
        for (int t = 0; t < 4; t++) {
            memset(&a, 0, sizeof a);
            memset(&b, 0, sizeof b);
            a.f[0] = xs[t];
            b.f[0] = ys[t];
            a.f[1] = 99.0f; /* the other lanes must not matter */
            b.f[1] = -99.0f;
            COMI_FLAGS("%{evex%} vcomiss %%xmm17, %%xmm16", got32, "m"(a), "m"(b));
            check_u64(names[t], got32 & 0xd501, comi_want(xs[t], ys[t]));
        }
        memset(&a, 0, sizeof a);
        a.f[0] = 0.5f;
        b.f[3] = 0.75f;
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vcomiss %2, %%xmm16\n\tlahf\n\tseto %%al\n\tmovzwl %%ax, %%eax\n\tmov %%eax, %0"
                     : "=m"(got32) : "m"(a), "m"(b.f[3]) : "xmm16", "rax", "cc", "memory");
        check_u64("vcomiss m32 (less)", got32 & 0xd501, comi_want(0.5, 0.75));
        COMI_FLAGS("%{evex%} vucomiss %%xmm17, %%xmm16", got32, "m"(a), "m"(b));
        check_u64("vucomiss (less than lane 0 of b)", got32 & 0xd501, comi_want(0.5, b.f[0]));
        memset(&a, 0, sizeof a);
        memset(&b, 0, sizeof b);
        a.d[0] = -7.25;
        b.d[0] = -7.25;
        COMI_FLAGS("%{evex%} vcomisd %%xmm17, %%xmm16", got32, "m"(a), "m"(b));
        check_u64("vcomisd equal", got32 & 0xd501, comi_want(-7.25, -7.25));
        b.d[0] = -7.0;
        COMI_FLAGS("%{evex%} vcomisd %%xmm17, %%xmm16", got32, "m"(a), "m"(b));
        check_u64("vcomisd less", got32 & 0xd501, comi_want(-7.25, -7.0));
    }

    /* The qword vectors: pairs that differ in the high half, in the low
     * half only, across the sign, and equal. */
    static const int64_t qa[8] = {0, 1, -1, (int64_t)0x100000000LL, INT64_MAX, INT64_MIN, 5, -5};
    static const int64_t qb[8] = {0, 2, 1, (int64_t)0x0ffffffffLL, INT64_MIN, INT64_MAX, 5, 5};
    for (int i = 0; i < 8; i++) {
        a.s[i] = qa[i];
        b.s[i] = qb[i];
    }

    /* 2. vpcmpq, vpcmpuq, vpcmpeqq, vpcmpgtq, with and without a mask */
    {
        static const int preds[6] = {0, 1, 2, 4, 5, 6};
        static const char *pn[6] = {"vpcmpq eq", "vpcmpq lt", "vpcmpq le", "vpcmpq neq", "vpcmpq nlt", "vpcmpq nle"};
        uint64_t want;
#define CMPQ(name, insn, pfn, p) \
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t" insn "\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0" \
                     : "=m"(got) : "m"(a), "m"(b) : "rax", "xmm16", "xmm17", "k1", "memory"); \
        want = 0; \
        for (int i = 0; i < 8; i++) if (pfn(p, a.s[i], b.s[i])) want |= 1u << i; \
        check_u64(name, got, want)
        CMPQ(pn[0], "vpcmpq $0, %%zmm17, %%zmm16, %%k1", pred_s, preds[0]);
        CMPQ(pn[1], "vpcmpq $1, %%zmm17, %%zmm16, %%k1", pred_s, preds[1]);
        CMPQ(pn[2], "vpcmpq $2, %%zmm17, %%zmm16, %%k1", pred_s, preds[2]);
        CMPQ(pn[3], "vpcmpq $4, %%zmm17, %%zmm16, %%k1", pred_s, preds[3]);
        CMPQ(pn[4], "vpcmpq $5, %%zmm17, %%zmm16, %%k1", pred_s, preds[4]);
        CMPQ(pn[5], "vpcmpq $6, %%zmm17, %%zmm16, %%k1", pred_s, preds[5]);
#define CMPUQ(name, insn, p) \
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t" insn "\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0" \
                     : "=m"(got) : "m"(a), "m"(b) : "rax", "xmm16", "xmm17", "k1", "memory"); \
        want = 0; \
        for (int i = 0; i < 8; i++) if (pred_u(p, a.q[i], b.q[i])) want |= 1u << i; \
        check_u64(name, got, want)
        CMPUQ("vpcmpuq lt", "vpcmpuq $1, %%zmm17, %%zmm16, %%k1", 1);
        CMPUQ("vpcmpuq nle", "vpcmpuq $6, %%zmm17, %%zmm16, %%k1", 6);
        CMPQ("vpcmpeqq", "vpcmpeqq %%zmm17, %%zmm16, %%k1", pred_s, 0);
        CMPQ("vpcmpgtq", "vpcmpgtq %%zmm17, %%zmm16, %%k1", pred_s, 6);
        /* under a write mask: the masked-off bits are zero */
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tmov $0xa7, %%eax\n\tkmovw %%eax, %%k2\n\tvpcmpq $4, %%zmm17, %%zmm16, %%k1%{%%k2%}\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0"
                     : "=m"(got) : "m"(a), "m"(b) : "rax", "xmm16", "xmm17", "k1", "k2", "memory");
        want = 0;
        for (int i = 0; i < 8; i++) if ((0xa7 >> i) & 1 && a.s[i] != b.s[i]) want |= 1u << i;
        check_u64("vpcmpq neq {k2}", got, want);
        /* the memory form, an embedded broadcast */
        asm volatile("vmovups %1, %%zmm16\n\tvpcmpq $1, %2%{1to8%}, %%zmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0"
                     : "=m"(got) : "m"(a), "m"(b.s[1]) : "rax", "xmm16", "k1", "memory");
        want = 0;
        for (int i = 0; i < 8; i++) if (a.s[i] < b.s[1]) want |= 1u << i;
        check_u64("vpcmpq lt m64{1to8}", got, want);
    }

    /* 3. vpmaxsq, vpminsq, vpmaxuq, vpminuq */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvpmaxsq %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int i = 0; i < 8; i++) w.s[i] = a.s[i] > b.s[i] ? a.s[i] : b.s[i];
    check("vpmaxsq", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvpminsq %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int i = 0; i < 8; i++) w.s[i] = a.s[i] < b.s[i] ? a.s[i] : b.s[i];
    check("vpminsq", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvpmaxuq %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int i = 0; i < 8; i++) w.q[i] = a.q[i] > b.q[i] ? a.q[i] : b.q[i];
    check("vpmaxuq", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvpminuq %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int i = 0; i < 8; i++) w.q[i] = a.q[i] < b.q[i] ? a.q[i] : b.q[i];
    check("vpminuq", &r, &w, 16);
    /* under a merging mask: the masked-off lanes keep the destination */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %2, %%zmm18\n\tmov $0x5a, %%eax\n\tkmovw %%eax, %%k1\n\tvpmaxsq %%zmm17, %%zmm16, %%zmm18%{%%k1%}\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int i = 0; i < 8; i++) w.s[i] = (0x5a >> i) & 1 ? (a.s[i] > b.s[i] ? a.s[i] : b.s[i]) : b.s[i];
    check("vpmaxsq {k1} merge", &r, &w, 16);

    /* 4. vptestnmq, vptestmq, vptestnmd */
    {
        uint64_t want;
        V m1, m2;
        for (int i = 0; i < 8; i++) {
            m1.q[i] = (uint64_t)i << 32 | (i & 1);
            m2.q[i] = (i & 2) ? 0xffffffff00000000ULL : 0x1ULL;
        }
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvptestnmq %%zmm17, %%zmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0"
                     : "=m"(got) : "m"(m1), "m"(m2) : "rax", "xmm16", "xmm17", "k1", "memory");
        want = 0;
        for (int i = 0; i < 8; i++) if ((m1.q[i] & m2.q[i]) == 0) want |= 1u << i;
        check_u64("vptestnmq", got, want);
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvptestmq %%zmm17, %%zmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0"
                     : "=m"(got) : "m"(m1), "m"(m2) : "rax", "xmm16", "xmm17", "k1", "memory");
        want = 0;
        for (int i = 0; i < 8; i++) if ((m1.q[i] & m2.q[i]) != 0) want |= 1u << i;
        check_u64("vptestmq", got, want);
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvptestnmd %%zmm17, %%zmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0"
                     : "=m"(got) : "m"(m1), "m"(m2) : "rax", "xmm16", "xmm17", "k1", "memory");
        want = 0;
        for (int i = 0; i < 16; i++) if ((m1.u[i] & m2.u[i]) == 0) want |= 1u << i;
        check_u64("vptestnmd", got, want);
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvptestmd %%zmm17, %%zmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0"
                     : "=m"(got) : "m"(m1), "m"(m2) : "rax", "xmm16", "xmm17", "k1", "memory");
        want = 0;
        for (int i = 0; i < 16; i++) if ((m1.u[i] & m2.u[i]) != 0) want |= 1u << i;
        check_u64("vptestmd", got, want);
    }

    /* 5. vinsertps: a register lane, a memory dword, the zero mask */
    for (int i = 0; i < 16; i++) {
        a.f[i] = 1.0f + i;
        b.f[i] = 100.0f + i;
    }
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vinsertps $0x9c, %%xmm17, %%xmm16, %%xmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 4; i++) w.f[i] = a.f[i];
    w.f[1] = b.f[2]; /* lane 2 of the second source into lane 1 */
    w.f[2] = 0;      /* zero mask 0xc: lanes 2 and 3 */
    w.f[3] = 0;
    check("vinsertps $0x9c xmm (lane 2 to 1, zero 2 and 3)", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vinsertps $0x21, %2, %%xmm16, %%xmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b.f[5]));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 4; i++) w.f[i] = a.f[i];
    w.f[2] = b.f[5]; /* the memory dword into lane 2 */
    w.f[0] = 0;      /* zero mask 1 */
    check("vinsertps $0x21 m32 (into lane 2, zero 0)", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vinsertps $0xf0, %%xmm17, %%xmm16, %%xmm16\n\tvmovups %%zmm16, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 4; i++) w.f[i] = a.f[i];
    w.f[3] = b.f[3];
    check("vinsertps $0xf0 in place (lane 3 to 3)", &r, &w, 16);

    /* 6. vshuff32x4, vshufi32x4, vshuff64x2 */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvshuff32x4 $0x4e, %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int blk = 0; blk < 4; blk++) {
        int sel = (0x4e >> (2 * blk)) & 3;
        const V *src = blk < 2 ? &a : &b;
        for (int j = 0; j < 4; j++) w.f[blk * 4 + j] = src->f[sel * 4 + j];
    }
    check("vshuff32x4 $0x4e", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvshufi32x4 $0x1b, %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int blk = 0; blk < 4; blk++) {
        int sel = (0x1b >> (2 * blk)) & 3;
        const V *src = blk < 2 ? &a : &b;
        for (int j = 0; j < 4; j++) w.f[blk * 4 + j] = src->f[sel * 4 + j];
    }
    check("vshufi32x4 $0x1b", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %2, %%zmm18\n\tmov $0x3c, %%eax\n\tkmovw %%eax, %%k1\n\tvshuff64x2 $0xe4, %%zmm17, %%zmm16, %%zmm18%{%%k1%}\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int blk = 0; blk < 4; blk++) {
        int sel = (0xe4 >> (2 * blk)) & 3;
        const V *src = blk < 2 ? &a : &b;
        for (int j = 0; j < 2; j++) w.q[blk * 2 + j] = (0x3c >> (blk * 2 + j)) & 1 ? src->q[sel * 2 + j] : b.q[blk * 2 + j];
    }
    check("vshuff64x2 $0xe4 {k1} merge", &r, &w, 16);

    /* 7. vpmovdb: to an xmm and to memory at an odd address */
    for (int i = 0; i < 16; i++) a.u[i] = 0x11223300u + (i * 0x11u) + (i & 1 ? 0x80u : 0);
    ASM("vmovups %1, %%zmm16\n\tvpmovdb %%zmm16, %%xmm17\n\tvmovups %%zmm17, %0", r, "m"(a));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 16; i++) w.b[i] = (uint8_t)a.u[i];
    check("vpmovdb zmm to xmm (upper zero)", &r, &w, 16);
    {
        static uint8_t buf[64];
        memset(buf, 0xee, sizeof buf);
        asm volatile("vmovups %1, %%zmm16\n\tvpmovdb %%zmm16, %0" : "=m"(*(uint8_t (*)[16])(buf + 3)) : "m"(a) : "xmm16", "memory");
        V got_v, want_v;
        memcpy(&got_v, buf, 64);
        memset(&want_v, 0xee, sizeof want_v);
        for (int i = 0; i < 16; i++) want_v.b[3 + i] = (uint8_t)a.u[i];
        check("vpmovdb zmm to m128 at +3", &got_v, &want_v, 16);
    }

    /* 8. vpscatterdd and vpgatherdd: a permutation of indices under a mask */
    {
        static int32_t buf[32];
        V idx;
        for (int i = 0; i < 16; i++) {
            idx.u[i] = (i * 5 + 3) & 15; /* a permutation of 0..15 */
            a.i[i] = 1000 + i;
        }
        for (int i = 0; i < 32; i++) buf[i] = -1;
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tmov $0xbeef, %%eax\n\tkmovw %%eax, %%k1\n\tvpscatterdd %%zmm16, (%3,%%zmm17,4)%{%%k1%}\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0"
                     : "=m"(got) : "m"(a), "m"(idx), "r"(buf) : "rax", "xmm16", "xmm17", "k1", "memory");
        check_u64("vpscatterdd clears its mask", got, 0);
        int32_t want_buf[32];
        for (int i = 0; i < 32; i++) want_buf[i] = -1;
        for (int i = 0; i < 16; i++) if ((0xbeef >> i) & 1) want_buf[idx.u[i]] = a.i[i];
        int bad = 0;
        for (int i = 0; i < 32; i++) if (buf[i] != want_buf[i]) bad++;
        if (bad) { printf("FAIL %-48s %d words wrong\n", "vpscatterdd {k1}", bad); fails++; }
        else printf("ok   vpscatterdd {k1}\n");
        /* gather them back, the disabled lanes keeping the destination */
        asm volatile("vmovups %1, %%zmm17\n\tvmovups %2, %%zmm18\n\tmov $0x7ff3, %%eax\n\tkmovw %%eax, %%k1\n\tvpgatherdd (%3,%%zmm17,4), %%zmm18%{%%k1%}\n\tvmovups %%zmm18, %0"
                     : "=m"(r) : "m"(idx), "m"(b), "r"(buf) : "rax", "xmm17", "xmm18", "k1", "memory");
        for (int i = 0; i < 16; i++) w.i[i] = (0x7ff3 >> i) & 1 ? buf[idx.u[i]] : b.i[i];
        check("vpgatherdd {k1} merge", &r, &w, 16);
    }

    /* 9. vfmaddsub132ps and vfmsubadd231pd: one fused rounding per lane */
    {
        V c;
        for (int i = 0; i < 16; i++) {
            a.f[i] = 1.25f + i * 0.5f;
            b.f[i] = -3.5f + i * 0.75f;
            c.f[i] = 0.3f * i + 0.1f;
        }
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %3, %%zmm18\n\tvfmaddsub132ps %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b), "m"(c));
        for (int i = 0; i < 16; i++) w.f[i] = fmaf(c.f[i], b.f[i], i & 1 ? a.f[i] : -a.f[i]);
        check("vfmaddsub132ps", &r, &w, 16);
        for (int i = 0; i < 8; i++) {
            a.d[i] = 1.25 + i * 0.5;
            b.d[i] = -3.5 + i * 0.75;
            c.d[i] = 0.3 * i + 0.1;
        }
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %3, %%zmm18\n\tvfmsubadd231pd %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b), "m"(c));
        for (int i = 0; i < 8; i++) w.d[i] = fma(a.d[i], b.d[i], i & 1 ? -c.d[i] : c.d[i]);
        check("vfmsubadd231pd", &r, &w, 16);
    }

    if (fails) {
        printf("FAIL: %d check(s) wrong\n", fails);
        return 1;
    }
    printf("PASS: every form matched\n");
    return 0;
}

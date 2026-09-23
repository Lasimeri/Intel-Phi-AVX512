/* avx512-narrow-test: the AVX-512 forms the card has no instruction for,
 * one at a time, each checked against plain C. Every instruction is
 * written as inline assembly with the {evex} prefix and the high vector
 * registers, so the encoding is exactly the EVEX form a compiler emits
 * for AVX512VL/DQ/BW code (what llama.cpp is built with), and the check
 * is bit-exact.
 *
 * Covered: 128-bit and 256-bit forms (lane masks, upper zeroing),
 * merging and zeroing masks, inserts and extracts, vpternlogd, scalar
 * ss/sd, vmovd/vmovq, float16 conversions, float to int with NaN and
 * overflow, int to float, byte and word widening loads, 64-bit shifts,
 * valignq, interleaves and shuffles, min/max with NaN, broadcasts from
 * registers, unaligned narrow loads and stores, a narrow load at the end
 * of a mapping, mask instructions, compare predicates above 7, abs,
 * unsigned 32x32 multiply, shifts by a register count, byte alignment,
 * division and square root (float and double), scale, round, exponent,
 * scalar conversions with general registers, the permutes composed from
 * vpermd.
 *
 *   gcc -O1 -mavx512f -mavx512vl -mavx512dq -mavx512bw -mf16c \
 *       -o avx512-narrow-test tools/avx512-narrow-test.c -lm
 *   scripts/phi512.sh ./avx512-narrow-test
 *
 * See avx512-narrow-test.md. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

typedef union {
    float f[16];
    uint32_t u[16];
    uint64_t q[8];
    uint16_t h[32];
    uint8_t b[64];
} V;

static int fails;

static void check(const char *name, const V *got, const V *want, int lanes)
{
    for (int i = 0; i < lanes; i++)
        if (got->u[i] != want->u[i]) {
            printf("FAIL %-44s lane %2d: got %08x want %08x\n", name, i, got->u[i], want->u[i]);
            fails++;
            return;
        }
    printf("ok   %s\n", name);
}

static void check_u64(const char *name, uint64_t got, uint64_t want)
{
    if (got != want) {
        printf("FAIL %-44s got %016llx want %016llx\n", name, (unsigned long long)got, (unsigned long long)want);
        fails++;
        return;
    }
    printf("ok   %s\n", name);
}

static void fill(V *v, float base)
{
    for (int i = 0; i < 16; i++) v->f[i] = base + i * 1.5f;
}

static void filli(V *v, uint32_t base)
{
    for (int i = 0; i < 16; i++) v->u[i] = base * (i + 1) ^ (i << 28);
}

#define ASM(body, out, ...) asm volatile(body : "=m"(out) : __VA_ARGS__ : "xmm16", "xmm17", "xmm18", "xmm19", "k1", "k2", "k3", "rax", "rcx", "memory")

int main(void)
{
    V a, b, c, r, w;
    fill(&a, 1.0f);
    fill(&b, 100.0f);
    filli(&c, 0x01234567u);

    /* 1. a 256-bit add: 8 lanes, the top 8 zero */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vaddps %%ymm17, %%ymm16, %%ymm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 8; i++) w.f[i] = a.f[i] + b.f[i];
    check("vaddps ymm (lanes 0..7, upper zero)", &r, &w, 16);

    /* 2. a 128-bit add under a merging mask */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %2, %%zmm18\n\tmov $0x5, %%eax\n\tkmovw %%eax, %%k1\n\t%{evex%} vaddps %%xmm17, %%xmm16, %%xmm18%{%%k1%}\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    w.f[0] = a.f[0] + b.f[0];
    w.f[1] = b.f[1];
    w.f[2] = a.f[2] + b.f[2];
    w.f[3] = b.f[3];
    check("vaddps xmm{k1} merge", &r, &w, 16);

    /* 3. a 512-bit add with zeroing */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tmov $0xa5a5, %%eax\n\tkmovw %%eax, %%k1\n\t%{evex%} vaddps %%zmm17, %%zmm16, %%zmm18%{%%k1%}%{z%}\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int i = 0; i < 16; i++) w.f[i] = (0xa5a5 >> i) & 1 ? a.f[i] + b.f[i] : 0;
    check("vaddps zmm{k1}%{z%}", &r, &w, 16);

    /* 4. vinserti64x2 $1 (the instruction llama.cpp died on) */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vinserti64x2 $1, %%xmm17, %%ymm16, %%ymm16\n\tvmovups %%zmm16, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 4; i++) w.f[i] = a.f[i];
    for (int i = 0; i < 4; i++) w.f[4 + i] = b.f[i];
    check("vinserti64x2 $1, xmm, ymm, ymm", &r, &w, 16);

    /* 5. vextracti32x4 to a register and to memory */
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vextracti32x4 $2, %%zmm16, %%xmm17\n\tvmovups %%zmm17, %0", r, "m"(a));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 4; i++) w.f[i] = a.f[8 + i];
    check("vextracti32x4 $2 to xmm", &r, &w, 16);
    memset(&r, 0x77, sizeof r);
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vextracti32x4 $3, %%zmm16, %0", r, "m"(a));
    memset(&w, 0x77, sizeof w);
    for (int i = 0; i < 4; i++) w.f[i] = a.f[12 + i];
    check("vextracti32x4 $3 to memory", &r, &w, 16);

    /* 6. vpternlogd: xor3, a generic table, all ones */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %3, %%zmm18\n\t%{evex%} vpternlogd $0x96, %%zmm18, %%zmm17, %%zmm16\n\tvmovups %%zmm16, %0", r, "m"(a), "m"(b), "m"(c));
    for (int i = 0; i < 16; i++) w.u[i] = a.u[i] ^ b.u[i] ^ c.u[i];
    check("vpternlogd $0x96", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %3, %%zmm18\n\t%{evex%} vpternlogd $0xe2, %%zmm18, %%zmm17, %%zmm16\n\tvmovups %%zmm16, %0", r, "m"(a), "m"(b), "m"(c));
    for (int i = 0; i < 16; i++) {
        uint32_t x = 0;
        for (int bit = 0; bit < 32; bit++) {
            int idx = ((a.u[i] >> bit) & 1) << 2 | ((b.u[i] >> bit) & 1) << 1 | ((c.u[i] >> bit) & 1);
            x |= ((0xe2 >> idx) & 1u) << bit;
        }
        w.u[i] = x;
    }
    check("vpternlogd $0xe2 (generic)", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vpternlogd $0xff, %%zmm16, %%zmm16, %%zmm16\n\tvmovups %%zmm16, %0", r, "m"(a));
    memset(&w, 0xff, sizeof w);
    check("vpternlogd $0xff", &r, &w, 16);

    /* 7. scalar: vaddss register, vmulss memory */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vaddss %%xmm17, %%xmm16, %%xmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    w.f[0] = a.f[0] + b.f[0];
    for (int i = 1; i < 4; i++) w.f[i] = a.f[i];
    check("vaddss xmm", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vmulss %2, %%xmm16, %%xmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b.f[3]));
    memset(&w, 0, sizeof w);
    w.f[0] = a.f[0] * b.f[3];
    for (int i = 1; i < 4; i++) w.f[i] = a.f[i];
    check("vmulss m32", &r, &w, 16);

    /* 8. vmovd / vmovq with general registers and memory */
    ASM("mov $0xdeadbeef, %%eax\n\t%{evex%} vmovd %%eax, %%xmm16\n\tvmovups %%zmm16, %0", r, "m"(a));
    memset(&w, 0, sizeof w);
    w.u[0] = 0xdeadbeef;
    check("vmovd eax, xmm", &r, &w, 16);
    {
        uint64_t got = 0;
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vmovq %%xmm16, %%rax\n\tmov %%rax, %0" : "=m"(got) : "m"(c) : "xmm16", "rax", "memory");
        check_u64("vmovq xmm, rax", got, c.q[0]);
    }
    ASM("%{evex%} vmovq %1, %%xmm16\n\tvmovups %%zmm16, %0", r, "m"(c.q[2]));
    memset(&w, 0, sizeof w);
    w.q[0] = c.q[2];
    check("vmovq m64, xmm", &r, &w, 16);

    /* 9. float16: vcvtph2ps from unaligned memory and from a register; vcvtps2ph */
    {
        static uint8_t halfs[64 + 2];
        V hv;
        for (int i = 0; i < 16; i++) hv.h[i] = 0x3c00 + i * 0x0111;
        memcpy(halfs + 2, hv.h, 32);
        asm volatile("%{evex%} vcvtph2ps %1, %%zmm16\n\tvmovups %%zmm16, %0" : "=m"(r) : "m"(*(uint8_t (*)[32])(halfs + 2)) : "xmm16", "memory");
        for (int i = 0; i < 16; i++) {
            uint16_t h = hv.h[i];
            /* all test halves are normal numbers */
            uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
            w.u[i] = (s << 31) | ((e + 112) << 23) | (m << 13);
        }
        check("vcvtph2ps m256 (2-byte aligned)", &r, &w, 16);
        ASM("vmovups %1, %%zmm17\n\t%{evex%} vcvtph2ps %%ymm17, %%zmm16\n\tvmovups %%zmm16, %0", r, "m"(hv));
        check("vcvtph2ps ymm", &r, &w, 16);
        memset(&r, 0, sizeof r);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vcvtps2ph $0, %%zmm16, %%ymm17\n\tvmovups %%zmm17, %0", r, "m"(w));
        memset(&w, 0, sizeof w);
        memcpy(w.h, hv.h, 32);
        check("vcvtps2ph zmm to ymm", &r, &w, 16);
    }

    /* 10. vcvttps2dq with NaN, overflow and negatives; vcvtdq2ps */
    {
        V s;
        for (int i = 0; i < 16; i++) s.f[i] = -7.9f + i * 3.7f;
        s.f[3] = NAN;
        s.f[5] = 3e9f;
        s.f[7] = -3e9f;
        s.f[9] = 2147483648.0f;
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vcvttps2dq %%zmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(s));
        for (int i = 0; i < 16; i++) {
            float f = s.f[i];
            w.u[i] = (isnan(f) || f >= 2147483648.0f || f < -2147483648.0f) ? 0x80000000u : (uint32_t)(int32_t)f;
        }
        check("vcvttps2dq (NaN, overflow)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vcvtdq2ps %%zmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(c));
        for (int i = 0; i < 16; i++) w.f[i] = (float)(int32_t)c.u[i];
        check("vcvtdq2ps", &r, &w, 16);
    }

    /* 11. widening loads */
    ASM("%{evex%} vpmovzxbd %1, %%zmm16\n\tvmovups %%zmm16, %0", r, "m"(c.b[0]));
    for (int i = 0; i < 16; i++) w.u[i] = c.b[i];
    check("vpmovzxbd m128", &r, &w, 16);
    ASM("vmovups %1, %%zmm17\n\t%{evex%} vpmovsxwd %%ymm17, %%zmm16\n\tvmovups %%zmm16, %0", r, "m"(c));
    for (int i = 0; i < 16; i++) w.u[i] = (uint32_t)(int32_t)(int16_t)c.h[i];
    check("vpmovsxwd ymm", &r, &w, 16);

    /* 12. 64-bit shifts */
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vpsrlq $3, %%zmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(c));
    for (int i = 0; i < 8; i++) w.q[i] = c.q[i] >> 3;
    check("vpsrlq $3", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vpsllq $35, %%zmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(c));
    for (int i = 0; i < 8; i++) w.q[i] = c.q[i] << 35;
    check("vpsllq $35", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vpsraq $7, %%zmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(c));
    for (int i = 0; i < 8; i++) w.q[i] = (uint64_t)((int64_t)c.q[i] >> 7);
    check("vpsraq $7", &r, &w, 16);

    /* 13. valignq narrow, valignd full */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} valignq $1, %%ymm17, %%ymm16, %%ymm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    {
        uint64_t cat[8];
        for (int i = 0; i < 4; i++) cat[i] = b.q[i];
        for (int i = 0; i < 4; i++) cat[4 + i] = a.q[i];
        for (int i = 0; i < 4; i++) w.q[i] = cat[i + 1];
    }
    check("valignq $1 ymm", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} valignd $3, %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int i = 0; i < 16; i++) w.u[i] = i + 3 < 16 ? b.u[i + 3] : a.u[i + 3 - 16];
    check("valignd $3 zmm", &r, &w, 16);

    /* 14. interleaves and shuffles */
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vunpcklps %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int blk = 0; blk < 4; blk++) {
        w.f[blk * 4 + 0] = a.f[blk * 4 + 0];
        w.f[blk * 4 + 1] = b.f[blk * 4 + 0];
        w.f[blk * 4 + 2] = a.f[blk * 4 + 1];
        w.f[blk * 4 + 3] = b.f[blk * 4 + 1];
    }
    check("vunpcklps zmm", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vshufps $0x1b, %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int blk = 0; blk < 4; blk++) {
        w.f[blk * 4 + 0] = a.f[blk * 4 + 3];
        w.f[blk * 4 + 1] = a.f[blk * 4 + 2];
        w.f[blk * 4 + 2] = b.f[blk * 4 + 1];
        w.f[blk * 4 + 3] = b.f[blk * 4 + 0];
    }
    check("vshufps $0x1b zmm", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vmovlhps %%xmm17, %%xmm16, %%xmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    memset(&w, 0, sizeof w);
    w.f[0] = a.f[0];
    w.f[1] = a.f[1];
    w.f[2] = b.f[0];
    w.f[3] = b.f[1];
    check("vmovlhps xmm", &r, &w, 16);

    /* 15. vmaxps with NaN and signed zeros: the second source wins */
    {
        V x, y;
        fill(&x, -3.0f);
        fill(&y, -2.0f);
        x.f[1] = NAN;
        y.f[2] = NAN;
        x.f[4] = -0.0f;
        y.f[4] = 0.0f;
        x.f[5] = 0.0f;
        y.f[5] = -0.0f;
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vmaxps %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(x), "m"(y));
        for (int i = 0; i < 16; i++) w.f[i] = (x.f[i] > y.f[i]) ? x.f[i] : y.f[i];
        check("vmaxps (NaN, signed zero)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vminps %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(x), "m"(y));
        for (int i = 0; i < 16; i++) w.f[i] = (x.f[i] < y.f[i]) ? x.f[i] : y.f[i];
        check("vminps (NaN, signed zero)", &r, &w, 16);
    }

    /* 16. broadcasts from a register and from eax */
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vpbroadcastd %%xmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(c));
    for (int i = 0; i < 16; i++) w.u[i] = c.u[0];
    check("vpbroadcastd xmm, zmm", &r, &w, 16);
    ASM("mov $0x5a5aa5a5, %%eax\n\t%{evex%} vpbroadcastd %%eax, %%ymm17\n\tvmovups %%zmm17, %0", r, "m"(c));
    memset(&w, 0, sizeof w);
    for (int i = 0; i < 8; i++) w.u[i] = 0x5a5aa5a5;
    check("vpbroadcastd eax, ymm", &r, &w, 16);

    /* 17. unaligned narrow store, and an arithmetic memory operand off alignment */
    {
        static uint8_t buf[128];
        memset(buf, 0x11, sizeof buf);
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vmovdqu64 %%ymm16, %0" : "=m"(*(uint8_t (*)[32])(buf + 4)) : "m"(c) : "xmm16", "memory");
        V got, want;
        memcpy(&got, buf, 64);
        memset(&want, 0x11, sizeof want);
        memcpy(want.b + 4, c.b, 32);
        check("vmovdqu64 ymm to [+4]", &got, &want, 16);
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vaddps %2, %%ymm16, %%ymm17\n\tvmovups %%zmm17, %0" : "=m"(r) : "m"(a), "m"(*(float (*)[8])(buf + 4)) : "xmm16", "xmm17", "memory");
        memset(&w, 0, sizeof w);
        for (int i = 0; i < 8; i++) w.f[i] = a.f[i] + c.f[i];
        check("vaddps [+4], ymm", &r, &w, 16);
    }

    /* 18. a 128-bit load at the very end of a mapping, the next page unmapped */
    {
        long pg = 4096;
        uint8_t *p = mmap(NULL, 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { perror("mmap"); return 1; }
        mprotect(p + pg, pg, PROT_NONE);
        memcpy(p + pg - 16, b.b, 16);
        asm volatile("%{evex%} vmovups %1, %%xmm16\n\tvmovups %%zmm16, %0" : "=m"(r) : "m"(*(float (*)[4])(p + pg - 16)) : "xmm16", "memory");
        memset(&w, 0, sizeof w);
        for (int i = 0; i < 4; i++) w.f[i] = b.f[i];
        check("vmovups xmm at the end of a mapping", &r, &w, 16);
        munmap(p, 2 * pg);
    }

    /* 19. mask instructions */
    {
        uint64_t got;
        asm volatile("mov $0xf0f0, %%eax\n\tkmovw %%eax, %%k1\n\tmov $0x3c3c, %%eax\n\tkmovw %%eax, %%k2\n\tkandw %%k2, %%k1, %%k3\n\tkmovw %%k3, %%eax\n\tmov %%rax, %0" : "=m"(got) : : "rax", "k1", "k2", "k3", "memory");
        check_u64("kandw k3 = k1 & k2", got, 0xf0f0 & 0x3c3c);
        asm volatile("mov $0x8421, %%eax\n\tkmovw %%eax, %%k1\n\tkshiftlw $3, %%k1, %%k2\n\tkmovw %%k2, %%eax\n\tmov %%rax, %0" : "=m"(got) : : "rax", "k1", "k2", "memory");
        check_u64("kshiftlw $3", got, (0x8421u << 3) & 0xffff);
        asm volatile("mov $0x00ff, %%eax\n\tkmovw %%eax, %%k1\n\tmov $0xff00, %%eax\n\tkmovw %%eax, %%k2\n\tkortestw %%k2, %%k1\n\tsetc %%al\n\tmovzbl %%al, %%eax\n\tmov %%rax, %0" : "=m"(got) : : "rax", "k1", "k2", "memory");
        check_u64("kortestw sets CF for all ones", got, 1);
    }

    /* 20. compare predicates above 7, and a narrow integer compare */
    {
        uint64_t got;
        V x, y;
        fill(&x, 0.0f);
        fill(&y, 0.0f);
        y.f[2] = 5.0f;
        x.f[5] = 99.0f;
        x.f[9] = NAN;
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vcmpps $0x0d, %%zmm17, %%zmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0" : "=m"(got) : "m"(x), "m"(y) : "rax", "xmm16", "xmm17", "k1", "memory");
        uint64_t want = 0;
        for (int i = 0; i < 16; i++) if (x.f[i] >= y.f[i]) want |= 1u << i;
        check_u64("vcmpps GE_OS (0x0d)", got, want);
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vcmpps $0x09, %%zmm17, %%zmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0" : "=m"(got) : "m"(x), "m"(y) : "rax", "xmm16", "xmm17", "k1", "memory");
        want = 0;
        for (int i = 0; i < 16; i++) if (!(x.f[i] >= y.f[i])) want |= 1u << i;
        check_u64("vcmpps NGE_US (0x09)", got, want);
        asm volatile("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vpcmpud $2, %%xmm17, %%xmm16, %%k1\n\tkmovw %%k1, %%eax\n\tmov %%rax, %0" : "=m"(got) : "m"(c), "m"(a) : "rax", "xmm16", "xmm17", "k1", "memory");
        want = 0;
        for (int i = 0; i < 4; i++) if (c.u[i] <= a.u[i]) want |= 1u << i;
        check_u64("vpcmpud LE xmm (upper bits zero)", got, want);
    }

    /* 21. abs, unsigned multiply, shift by a register count, byte alignment */
    ASM("vmovups %1, %%zmm16\n\t%{evex%} vpabsd %%zmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(c));
    for (int i = 0; i < 16; i++) w.u[i] = (int32_t)c.u[i] < 0 ? -c.u[i] : c.u[i];
    check("vpabsd", &r, &w, 16);
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vpmuludq %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(c), "m"(a));
    for (int i = 0; i < 8; i++) w.q[i] = (uint64_t)c.u[2 * i] * (uint64_t)a.u[2 * i];
    check("vpmuludq", &r, &w, 16);
    {
        static const uint64_t cnt[2] = {5, 0};
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vmovq %2, %%xmm17\n\t%{evex%} vpslld %%xmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0" : "=m"(r) : "m"(c), "m"(cnt[0]) : "xmm16", "xmm17", "xmm18", "memory");
        for (int i = 0; i < 16; i++) w.u[i] = c.u[i] << 5;
        check("vpslld xmm count", &r, &w, 16);
    }
    ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vpalignr $8, %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(b));
    for (int blk = 0; blk < 4; blk++) {
        w.u[blk * 4 + 0] = b.u[blk * 4 + 2];
        w.u[blk * 4 + 1] = b.u[blk * 4 + 3];
        w.u[blk * 4 + 2] = a.u[blk * 4 + 0];
        w.u[blk * 4 + 3] = a.u[blk * 4 + 1];
    }
    check("vpalignr $8 zmm", &r, &w, 16);

    /* 22. division and square root, which the card synthesises */
    {
        V x, y;
        for (int i = 0; i < 16; i++) { x.f[i] = 3.0f + i * 7.25f; y.f[i] = 0.7f + i * 1.3f; }
        x.f[5] = 0.0f; y.f[6] = 0.0f; x.f[7] = INFINITY; y.f[9] = -2.5f; x.f[11] = 1e-20f;
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vdivps %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(x), "m"(y));
        for (int i = 0; i < 16; i++) w.f[i] = x.f[i] / y.f[i];
        check("vdivps (Newton from vrcp23ps)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vsqrtps %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(x));
        for (int i = 0; i < 16; i++) w.f[i] = sqrtf(x.f[i]);
        check("vsqrtps (Newton from vrsqrt23ps)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vdivss %%xmm17, %%xmm16, %%xmm18\n\tvmovups %%zmm18, %0", r, "m"(x), "m"(y));
        memset(&w, 0, sizeof w);
        w.f[0] = x.f[0] / y.f[0];
        for (int i = 1; i < 4; i++) w.f[i] = x.f[i];
        check("vdivss", &r, &w, 16);
        V xd, yd;
        double *xq = (double *)xd.q, *yq = (double *)yd.q;
        for (int i = 0; i < 8; i++) { xq[i] = 3.0 + i * 7.25; yq[i] = 0.7 + i * 1.3; }
        xq[3] = 0.0; yq[4] = -0.001; xq[6] = 1e300;
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vdivpd %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(xd), "m"(yd));
        for (int i = 0; i < 8; i++) ((double *)w.q)[i] = xq[i] / yq[i];
        check("vdivpd (seeded through float)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vsqrtpd %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(xd));
        for (int i = 0; i < 8; i++) ((double *)w.q)[i] = sqrt(xq[i]);
        check("vsqrtpd", &r, &w, 16);
    }

    /* 23. scale, round, exponent */
    {
        V x, y;
        for (int i = 0; i < 16; i++) { x.f[i] = 1.5f + i * 0.37f; y.f[i] = -3.7f + i * 0.9f; }
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vscalefps %%zmm17, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(x), "m"(y));
        for (int i = 0; i < 16; i++) w.f[i] = x.f[i] * powf(2.0f, floorf(y.f[i]));
        check("vscalefps", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vrndscaleps $9, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(y));
        for (int i = 0; i < 16; i++) w.f[i] = floorf(y.f[i]);
        check("vrndscaleps $9 (floor)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vrndscalesd $11, %%xmm16, %%xmm16, %%xmm18\n\tvmovups %%zmm18, %0", r, "m"(y));
        memset(&w, 0, sizeof w);
        ((double *)w.q)[0] = trunc(((double *)y.q)[0]);
        w.q[1] = y.q[1];
        check("vrndscalesd $11 (truncate)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vgetexpps %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(x));
        for (int i = 0; i < 16; i++) w.f[i] = floorf(log2f(x.f[i]));
        check("vgetexpps", &r, &w, 16);
    }

    /* 24. scalar conversions with general registers */
    {
        uint64_t got;
        /* lanes above the converted one come from the first source (xmm16 = a) */
        ASM("vmovups %1, %%zmm16\n\tmov $-123456789, %%eax\n\t%{evex%} vcvtsi2sd %%eax, %%xmm16, %%xmm17\n\tvmovups %%zmm17, %0", r, "m"(a));
        memset(&w, 0, sizeof w);
        ((double *)w.q)[0] = (double)-123456789;
        w.q[1] = a.q[1];
        check("vcvtsi2sd eax", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\tmov $0xf0000001, %%eax\n\t%{evex%} vcvtusi2sd %%eax, %%xmm16, %%xmm17\n\tvmovups %%zmm17, %0", r, "m"(a));
        memset(&w, 0, sizeof w);
        ((double *)w.q)[0] = (double)0xf0000001u;
        w.q[1] = a.q[1];
        check("vcvtusi2sd eax (unsigned)", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\tmov $16777217, %%eax\n\t%{evex%} vcvtsi2ss %%eax, %%xmm16, %%xmm17\n\tvmovups %%zmm17, %0", r, "m"(a));
        memset(&w, 0, sizeof w);
        w.f[0] = (float)16777217;
        for (int i = 1; i < 4; i++) w.f[i] = a.f[i];
        check("vcvtsi2ss eax (rounds)", &r, &w, 16);
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vcvttss2si %%xmm16, %%eax\n\tmov %%rax, %0" : "=m"(got) : "m"(b) : "xmm16", "rax", "memory");
        check_u64("vcvttss2si eax", got, (uint64_t)(uint32_t)(int32_t)b.f[0]);
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vcvttss2si %%xmm16, %%rax\n\tmov %%rax, %0" : "=m"(got) : "m"(b) : "xmm16", "rax", "memory");
        check_u64("vcvttss2si rax", got, (uint64_t)(int64_t)b.f[0]);
        V xd;
        ((double *)xd.q)[0] = -2.75e9;
        asm volatile("vmovups %1, %%zmm16\n\t%{evex%} vcvtsd2si %%xmm16, %%rax\n\tmov %%rax, %0" : "=m"(got) : "m"(xd) : "xmm16", "rax", "memory");
        check_u64("vcvtsd2si rax (rounds to nearest)", got, (uint64_t)(int64_t)llrint(-2.75e9));
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vcvtss2sd %%xmm16, %%xmm16, %%xmm17\n\tvmovups %%zmm17, %0", r, "m"(b));
        memset(&w, 0, sizeof w);
        ((double *)w.q)[0] = (double)b.f[0];
        w.q[1] = b.q[1];
        check("vcvtss2sd", &r, &w, 16);
    }

    /* 25. permutes composed from vpermd */
    {
        V idx;
        for (int i = 0; i < 16; i++) idx.u[i] = (i * 7 + 3) & 31;
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %3, %%zmm18\n\t%{evex%} vpermi2ps %%zmm18, %%zmm17, %%zmm16\n\tvmovups %%zmm16, %0", r, "m"(idx), "m"(a), "m"(b));
        for (int i = 0; i < 16; i++) w.u[i] = idx.u[i] & 16 ? b.u[idx.u[i] & 15] : a.u[idx.u[i] & 15];
        check("vpermi2ps", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\tvmovups %3, %%zmm18\n\t%{evex%} vpermt2d %%zmm18, %%zmm16, %%zmm17\n\tvmovups %%zmm17, %0", r, "m"(idx), "m"(a), "m"(b));
        for (int i = 0; i < 16; i++) w.u[i] = idx.u[i] & 16 ? b.u[idx.u[i] & 15] : a.u[idx.u[i] & 15];
        check("vpermt2d", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\t%{evex%} vpermq $0x93, %%zmm16, %%zmm18\n\tvmovups %%zmm18, %0", r, "m"(c));
        for (int h = 0; h < 2; h++)
            for (int j = 0; j < 4; j++) w.q[h * 4 + j] = c.q[h * 4 + ((0x93 >> (2 * j)) & 3)];
        check("vpermq $0x93", &r, &w, 16);
        ASM("vmovups %1, %%zmm16\n\tvmovups %2, %%zmm17\n\t%{evex%} vpermd %%ymm16, %%ymm17, %%ymm18\n\tvmovups %%zmm18, %0", r, "m"(a), "m"(idx));
        memset(&w, 0, sizeof w);
        for (int i = 0; i < 8; i++) w.u[i] = a.u[idx.u[i] & 7];
        check("vpermd ymm (3-bit indices)", &r, &w, 16);
    }

    if (fails) {
        printf("FAIL: %d check(s) wrong\n", fails);
        return 1;
    }
    printf("PASS: every form matched\n");
    return 0;
}

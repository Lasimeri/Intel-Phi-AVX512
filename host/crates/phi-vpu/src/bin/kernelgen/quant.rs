//! The quantized-weight kernels: one 256-weight superblock of a row in
//! llama.cpp's Q4_K, Q5_K, Q6_K, Q8_0 or IQ4_XS format against 1, 4 or 8
//! rows of float32 activations, accumulated into sixteen partial sums per
//! activation row. The card has no byte arithmetic, so the bytes load as
//! floats (the `{uint8}` and `{sint8}` up-conversions), nibbles and
//! two-bit fields are split with a multiply by a power of two, a floor
//! and a fused subtract, and each sixteen-weight vector is finished as
//! `w = q * scale - minuend` with the two values broadcast from a table
//! the caller fills per superblock (the scalar work of the format, in C).
//! See quant.md.
//!
//! Calling convention (System V), the same for every kernel:
//!
//! ```text
//!   rdi  the superblock (the block's first byte; Q8_0: eight consecutive blocks)
//!   rsi  the first activation row, 256 floats, 64-byte aligned
//!   rdx  an array of T pointers, one per activation row (entry 0 is rsi
//!        again); each row is 256 floats, 64-byte aligned
//!   rcx  the superblock's table: 16 scales, 16 minuends (`TAB_SC`, `TAB_MN`)
//!   r8   the accumulators, T x 16 floats, 64-byte aligned, read and written
//!   r9   the constants, 64-byte aligned, the same for every call (`C_*`)
//!   [rsp+8]  the byte distance to the superblock this thread processes next
//!            (the prefetch stride: the row stride when rows are swept
//!            superblock by superblock, the block size along one row)
//! ```

use knc_mvex::*;

/// The superblock's table: scales and minuends per sixteen-weight vector
/// (the C side fills 32 floats per superblock).
pub const TAB_SC: i32 = 0;
pub const TAB_MN: i32 = 64;
pub const TAB_BYTES: i32 = 128;
/// The constants: 16, 4, 2, then `2^-j` for j in 1..=8 at `C_POW2NEG + 4 * (j - 1)`,
/// and the sixteen-entry value table of IQ4_XS at 64.
pub const C_SIXTEEN: i32 = 0;
pub const C_FOUR: i32 = 4;
pub const C_TWO: i32 = 8;
pub const C_POW2NEG: i32 = 12;
pub const C_LUT: i32 = 64;
/// Index vectors for `vpermd` and shift vectors, 64-byte aligned (the
/// preparation of the scales, `prep_*` below), then scalar constants.
pub const C_IDX_EXPAND: i32 = 128; // [0,0,1,1,..,7,7]
pub const C_IDX_EXPAND_HI: i32 = 192; // [8,8,9,9,..,15,15]
pub const C_IDX_K4_P1: i32 = 256; // [0,1,2,3, 8,9,10,11, 4,5,6,7, 8,9,10,11]
pub const C_IDX_K4_P2: i32 = 320; // [0,0,0,0, 0,1,2,3, 0,0,0,0, 4,5,6,7]
pub const C_IDX_D_DMIN: i32 = 384; // [0 x8, 1 x8]
pub const C_IDX_ZERO: i32 = 448;
pub const C_IDX_ONE: i32 = 512;
pub const C_IDX_SL: i32 = 576; // [2,2,3,3,4,4,5,5, 0..]
pub const C_SHIFT_L: i32 = 640; // [0,4,0,4,0,4,0,4, 0..]
pub const C_SHIFT_H: i32 = 704; // [0,2,4,..,14, 0..]
pub const C_F32_32: i32 = 768;
pub const C_U_63: i32 = 772;
pub const C_U_15: i32 = 776;
pub const C_U_C0: i32 = 780;
pub const C_U_3: i32 = 784;
pub const C_U_32: i32 = 788;
pub const C_BYTES: i32 = 832;
/// Prefetch distances, in superblocks: the L1 line two ahead, the L2 six
/// ahead (about 500 ns of work at the kernels' rate, past GDDR latency).
const PF_L1: i32 = 2;
const PF_L2: i32 = 4;

const BLK: Gpr = Gpr::Rdi;
const STRIDE: Gpr = Gpr::Rdx;
const TAB: Gpr = Gpr::Rcx;
const ACC: Gpr = Gpr::R8;
const CONSTS: Gpr = Gpr::R9;
/// The activation rows' base registers; the ones past the fourth are
/// callee-saved and pushed by the eight-row kernels.
const ROWS: [Gpr; 8] = [Gpr::Rsi, Gpr::R10, Gpr::R11, Gpr::Rax, Gpr::Rbx, Gpr::R13, Gpr::R14, Gpr::R15];

pub struct Asm(pub String);

impl Asm {
    fn i(&mut self, x: &Insn) {
        self.0.push_str(&x.gas());
        self.0.push('\n');
    }
    fn t(&mut self, x: &str) {
        self.0.push_str("    ");
        self.0.push_str(x);
        self.0.push('\n');
    }
}

fn pow2neg(j: i32) -> Src {
    assert!((1..=8).contains(&j));
    Src::MemConv(Mem::new(CONSTS, C_POW2NEG + 4 * (j - 1)), Conv::Bcast1)
}

fn constant(off: i32) -> Src {
    Src::MemConv(Mem::new(CONSTS, off), Conv::Bcast1)
}

/// Prefetch the superblocks `PF_L1` and `PF_L2` calls ahead of this one,
/// every line they can touch (`bytes` per superblock), with the stride
/// the caller passes on the stack. Runs first, so rax and r11 are free.
fn prefetch_ahead(a: &mut Asm, bytes: i32) {
    let lines = (bytes + 63) / 64 + 1;
    a.t("mov 8(%rsp), %r11");
    a.t(&format!("lea (%rdi,%r11,{PF_L1}), %rax"));
    for l in 0..lines {
        a.i(&vprefetch(Mem::new(Gpr::Rax, 64 * l), Cache::L1));
    }
    a.t(&format!("lea (%rax,%r11,{}), %rax", PF_L2 - PF_L1));
    for l in 0..lines {
        a.i(&vprefetch(Mem::new(Gpr::Rax, 64 * l), Cache::L2));
    }
}

/// The accumulator a vector `v` of activation row `i` adds into: one row
/// rotates over four registers so that consecutive vectors' fused adds
/// are independent (the vector unit's result latency is four cycles and
/// the core is in order; ISA reference, section 1.2 on the pipeline), the
/// four- and eight-row kernels have a register per row anyway.
fn acc(t: usize, i: usize, v: i32) -> Zmm {
    if t == 1 {
        Zmm((v % 4) as u8)
    } else {
        Zmm(i as u8)
    }
}

fn prologue(a: &mut Asm, name: &str, t: usize, what: &str, block_bytes: i32, prep: fn(&mut Asm)) {
    a.0.push_str(&format!("# {name}: {what}, against {t} activation row(s)\n"));
    a.0.push_str(&format!("    .globl {name}\n    .type {name}, @function\n{name}:\n"));
    prefetch_ahead(a, block_bytes);
    if t > 4 {
        a.t("push %rbx");
        a.t("push %r13");
        a.t("push %r14");
        a.t("push %r15");
    }
    prep(a);
    // The rows after the first come from the caller's array, not from a
    // stride: a mixture of experts groups the columns that chose the
    // same expert, and those columns' activation rows are wherever the
    // tokens that chose it happen to be. It is the same instruction
    // count as the chain of `lea`s this replaces, and rdx is read by
    // nothing else in these kernels.
    for (i, &row) in ROWS.iter().enumerate().take(t).skip(1) {
        a.t(&format!("mov {}(%{}), %{}", 8 * i, STRIDE, row));
    }
    for i in 0..t {
        a.i(&vmovaps_load(Zmm(i as u8), Mem::new(ACC, 64 * i as i32)));
    }
    if t == 1 {
        for r in 1..4u8 {
            a.i(&vpxord(Zmm(r), Zmm(r), Src::Reg(Zmm(r)), K(0)));
        }
    }
}

fn epilogue(a: &mut Asm, name: &str, t: usize) {
    if t == 1 {
        a.i(&vaddps(Zmm(0), Zmm(0), Src::Reg(Zmm(1)), K(0)));
        a.i(&vaddps(Zmm(2), Zmm(2), Src::Reg(Zmm(3)), K(0)));
        a.i(&vaddps(Zmm(0), Zmm(0), Src::Reg(Zmm(2)), K(0)));
    }
    for i in 0..t {
        a.i(&vmovaps_store(Mem::new(ACC, 64 * i as i32), Zmm(i as u8)));
    }
    if t > 4 {
        a.t("pop %r15");
        a.t("pop %r14");
        a.t("pop %r13");
        a.t("pop %rbx");
    }
    a.t("ret");
    a.0.push_str(&format!("    .size {name}, .-{name}\n\n"));
}

/// Sixteen bytes at `mem`, as floats, from any alignment (the unpack pair).
fn bytes_unaligned(a: &mut Asm, dst: Zmm, mem: Mem, conv: Conv) {
    a.i(&vloadunpacklps_conv(dst, mem, conv, K(0)));
    a.i(&vloadunpackhps_conv(dst, mem.offset(64).expect("displacement"), conv, K(0)));
}

/// Sixteen bytes at a 16-byte aligned `mem`, as floats.
fn bytes_aligned(a: &mut Asm, dst: Zmm, mem: Mem, conv: Conv) {
    a.i(&vmovaps_load_conv(dst, mem, conv, K(0)));
}

// Every stage below is emitted for a whole batch of registers before the
// next stage, so dependent instructions sit a batch apart.

/// Each `b` holds sixteen bytes as floats (0 to 255): afterwards the
/// matching `hi` is the high nibbles and `b` the low ones
/// (`floor(b / 16)`, then `b - 16 hi`).
fn split_nibbles(a: &mut Asm, pairs: &[(Zmm, Zmm)]) {
    for &(b, hi) in pairs {
        a.i(&vmulps(hi, b, pow2neg(4), K(0)));
    }
    for &(_, hi) in pairs {
        a.i(&vrndfxpntps(hi, Src::Reg(hi), Round::Down, K(0)));
    }
    for &(b, hi) in pairs {
        a.i(&vfnmadd231ps(b, hi, constant(C_SIXTEEN), K(0)));
    }
}

/// `w = w * sc[v] - mn[v]` for each (register, vector index) of the batch.
fn finish(a: &mut Asm, batch: &[(Zmm, i32)], minuend: bool) {
    for &(w, v) in batch {
        a.i(&vmulps(w, w, Src::MemConv(Mem::new(TAB, TAB_SC + 4 * v), Conv::Bcast1), K(0)));
    }
    if minuend {
        for &(w, v) in batch {
            a.i(&vsubps(w, w, Src::MemConv(Mem::new(TAB, TAB_MN + 4 * v), Conv::Bcast1), K(0)));
        }
    }
}

/// `acc += w * x[i][16 v .. 16 v + 16]` for each activation row and each
/// vector of the batch, ordered so the same accumulator is not written
/// twice in a row.
/// The activations a kernel multiplies by: float32, or float16 that the
/// memory operand itself up-converts. The second halves what crosses the
/// link and halves the activation bytes the core reads per call, for the
/// same instruction: the conversion is a field of the operand, not work
/// (`knc-mvex/src/conv.md`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Act {
    F32,
    F16,
}

impl Act {
    /// The suffix its kernels carry, and the bytes one vector of
    /// activations occupies.
    fn suffix(self) -> &'static str {
        match self {
            Act::F32 => "",
            Act::F16 => "h",
        }
    }
    fn stride(self) -> i32 {
        match self {
            Act::F32 => 64,
            Act::F16 => 32,
        }
    }
    fn src(self, row: Gpr, v: i32) -> Src {
        let m = Mem::new(row, self.stride() * v);
        match self {
            Act::F32 => Src::Mem(m),
            Act::F16 => Src::MemConv(m, Conv::F16),
        }
    }
}

fn fmas(a: &mut Asm, t: usize, batch: &[(Zmm, i32)], x: Act) {
    for &(w, v) in batch {
        for (i, &row) in ROWS.iter().enumerate().take(t) {
            a.i(&vfmadd231ps(acc(t, i, v), w, x.src(row, v), K(0)));
        }
    }
}

/// `w += 16 * bits` for each pair: the fifth or sixth bit joins the nibble.
fn add_high_bits(a: &mut Asm, pairs: &[(Zmm, Zmm)]) {
    for &(w, bits) in pairs {
        a.i(&vfmadd231ps(w, bits, constant(C_SIXTEEN), K(0)));
    }
}

// ---- the scales: the scalar part of each format, done on the vector unit ----
//
// Each `prep_*` reads the format's scale bytes from the block and writes
// the sixteen per-vector scales to [rcx + TAB_SC] and the sixteen
// minuends to [rcx + TAB_MN], so the kernel's `finish` can broadcast them.
// It runs before the row registers are set, so it may use rax; it uses
// zmm16 upward. The card's scalar float unit is x87 and the packed
// 6-bit fields are branchy in C (measured 478 ns per superblock against
// 192 ns for the whole vector kernel), which is why this is vector code.

fn idx(a: &mut Asm, reg: Zmm, off: i32) {
    a.i(&vmovaps_load(reg, Mem::new(CONSTS, off)));
}

fn u32c(off: i32) -> Src {
    Src::MemConv(Mem::new(CONSTS, off), Conv::Bcast1)
}

/// The int32 unpack pair: sixteen bytes at `mem` as integers.
fn ints_unaligned(a: &mut Asm, dst: Zmm, mem: Mem, conv: Conv) {
    a.i(&vloadunpackld_conv(dst, mem, conv, K(0)));
    a.i(&vloadunpackhd_conv(dst, mem.offset(64).expect("displacement"), conv, K(0)));
}

/// Sixteen halfs at `mem` as floats, any alignment.
fn halfs_unaligned(a: &mut Asm, dst: Zmm, mem: Mem) {
    a.i(&vloadunpacklps_conv(dst, mem, Conv::F16, K(0)));
    a.i(&vloadunpackhps_conv(dst, mem.offset(64).expect("displacement"), Conv::F16, K(0)));
}

fn kmask(a: &mut Asm, bits: u32) {
    a.t(&format!("mov ${bits}, %eax"));
    a.i(&kmov_k_r32(K(1), Gpr::Rax));
}

/// `v` (lanes 0..7 the scales, 8..15 the minimums as floats) times the
/// block's `d` and `dmin` (halfs at `blk + 0`, `+ 2`), expanded to the
/// sixteen vectors and stored.
fn store_expanded(a: &mut Asm, v: Zmm, minuends: bool) {
    let (i, w) = (Zmm(30), Zmm(31));
    idx(a, i, C_IDX_EXPAND);
    a.i(&vpermd(w, i, Src::Reg(v), K(0)));
    a.i(&vmovaps_store(Mem::new(TAB, TAB_SC), w));
    if minuends {
        idx(a, i, C_IDX_EXPAND_HI);
        a.i(&vpermd(w, i, Src::Reg(v), K(0)));
        a.i(&vmovaps_store(Mem::new(TAB, TAB_MN), w));
    }
}

/// Q4_K and Q5_K: twelve bytes at `blk + 4` hold eight 6-bit scales and
/// eight 6-bit minimums (ggml-quants.c, get_scale_min_k4): for j < 4
/// `sc = q[j] & 63`, `m = q[j+4] & 63`; for j >= 4
/// `sc = (q[j+4] & 15) | (q[j-4] >> 6) << 4`, `m = (q[j+4] >> 4) | (q[j] >> 6) << 4`.
/// Lanes 0..7 become the scales, 8..15 the minimums: two permutes pick
/// the main byte and the high-bits byte of every lane, the two forms are
/// computed for all lanes and the right one kept by mask.
fn prep_k4(a: &mut Asm) {
    let (qi, p1, p2, v, tlo, thi, hb, zero, i, hv, mul) = (
        Zmm(16),
        Zmm(17),
        Zmm(18),
        Zmm(19),
        Zmm(20),
        Zmm(21),
        Zmm(22),
        Zmm(23),
        Zmm(24),
        Zmm(25),
        Zmm(26),
    );
    ints_unaligned(a, qi, Mem::new(BLK, 4), Conv::U8);
    idx(a, i, C_IDX_K4_P1);
    a.i(&vpermd(p1, i, Src::Reg(qi), K(0)));
    idx(a, i, C_IDX_K4_P2);
    a.i(&vpermd(p2, i, Src::Reg(qi), K(0)));
    a.i(&vpandd(v, p1, u32c(C_U_63), K(0)));
    a.i(&vpandd(tlo, p1, u32c(C_U_15), K(0)));
    a.i(&vpsrld(thi, Src::Reg(p1), 4, K(0)));
    a.i(&vpandd(hb, p2, u32c(C_U_C0), K(0)));
    a.i(&vpsrld(hb, Src::Reg(hb), 2, K(0)));
    a.i(&vpord(tlo, tlo, Src::Reg(hb), K(0)));
    a.i(&vpord(thi, thi, Src::Reg(hb), K(0)));
    a.i(&vpxord(zero, zero, Src::Reg(zero), K(0)));
    kmask(a, 0x00f0);
    a.i(&vpord(v, tlo, Src::Reg(zero), K(1)));
    kmask(a, 0xf000);
    a.i(&vpord(v, thi, Src::Reg(zero), K(1)));
    a.i(&vcvtfxpntdq2ps(v, Src::Reg(v), K(0)));
    halfs_unaligned(a, hv, Mem::new(BLK, 0));
    idx(a, i, C_IDX_D_DMIN);
    a.i(&vpermd(mul, i, Src::Reg(hv), K(0)));
    a.i(&vmulps(v, v, Src::Reg(mul), K(0)));
    store_expanded(a, v, true);
}

/// Q6_K: sixteen signed scales at `blk + 192`, one per vector, times `d`
/// (a half at `blk + 208`); the minuend is 32 times the scale.
fn prep_q6k(a: &mut Asm) {
    let (v, hv, i, d, w) = (Zmm(16), Zmm(17), Zmm(18), Zmm(19), Zmm(20));
    ints_unaligned(a, v, Mem::new(BLK, 192), Conv::S8);
    a.i(&vcvtfxpntdq2ps(v, Src::Reg(v), K(0)));
    halfs_unaligned(a, hv, Mem::new(BLK, 208));
    idx(a, i, C_IDX_ZERO);
    a.i(&vpermd(d, i, Src::Reg(hv), K(0)));
    a.i(&vmulps(v, v, Src::Reg(d), K(0)));
    a.i(&vmovaps_store(Mem::new(TAB, TAB_SC), v));
    a.i(&vmulps(w, v, constant(C_F32_32), K(0)));
    a.i(&vmovaps_store(Mem::new(TAB, TAB_MN), w));
}

/// Q8_0: block i's `d` is the half at `34 i`. The unpack loads are expand
/// loads (consecutive elements from the address into the unmasked lanes),
/// so the pair from `34 i` with lane i alone unmasked puts that half in
/// lane i.
fn prep_q8_0(a: &mut Asm) {
    let dv = Zmm(16);
    a.i(&vpxord(dv, dv, Src::Reg(dv), K(0)));
    for i in 0..8 {
        kmask(a, 1 << i);
        a.i(&vloadunpacklps_conv(dv, Mem::new(BLK, 34 * i), Conv::F16, K(1)));
        a.i(&vloadunpackhps_conv(dv, Mem::new(BLK, 34 * i + 64), Conv::F16, K(1)));
    }
    store_expanded(a, dv, false);
}

/// IQ4_XS: `scales_h` (u16 at 2) and `scales_l` (4 bytes at 4) hold eight
/// 6-bit scales: `ls = ((sl[ib/2] >> 4 (ib % 2)) & 15) | ((sh >> 2 ib) & 3) << 4`,
/// the scale `d * (ls - 32)` (dequantize_row_iq4_xs).
fn prep_iq4xs(a: &mut Asm) {
    let (bi, i, sl8, lo4, s0, s1, hi2, hv, d) = (Zmm(16), Zmm(17), Zmm(18), Zmm(19), Zmm(20), Zmm(21), Zmm(22), Zmm(23), Zmm(24));
    ints_unaligned(a, bi, Mem::new(BLK, 2), Conv::U8);
    idx(a, i, C_IDX_SL);
    a.i(&vpermd(sl8, i, Src::Reg(bi), K(0)));
    a.i(&vpsrlvd(lo4, sl8, Src::Mem(Mem::new(CONSTS, C_SHIFT_L)), K(0)));
    a.i(&vpandd(lo4, lo4, u32c(C_U_15), K(0)));
    idx(a, i, C_IDX_ZERO);
    a.i(&vpermd(s0, i, Src::Reg(bi), K(0)));
    idx(a, i, C_IDX_ONE);
    a.i(&vpermd(s1, i, Src::Reg(bi), K(0)));
    a.i(&vpslld(s1, Src::Reg(s1), 8, K(0)));
    a.i(&vpord(s0, s0, Src::Reg(s1), K(0)));
    a.i(&vpsrlvd(hi2, s0, Src::Mem(Mem::new(CONSTS, C_SHIFT_H)), K(0)));
    a.i(&vpandd(hi2, hi2, u32c(C_U_3), K(0)));
    a.i(&vpslld(hi2, Src::Reg(hi2), 4, K(0)));
    a.i(&vpord(lo4, lo4, Src::Reg(hi2), K(0)));
    a.i(&vpsubd(lo4, lo4, u32c(C_U_32), K(0)));
    a.i(&vcvtfxpntdq2ps(lo4, Src::Reg(lo4), K(0)));
    halfs_unaligned(a, hv, Mem::new(BLK, 0));
    idx(a, i, C_IDX_ZERO);
    a.i(&vpermd(d, i, Src::Reg(hv), K(0)));
    a.i(&vmulps(lo4, lo4, Src::Reg(d), K(0)));
    store_expanded(a, lo4, false);
}

/// Q4_K: `qs` at 16 (128 bytes), eight sub-blocks of 32; the low nibbles
/// of bytes 32g..32g+32 are sub-block 2g, the high ones sub-block 2g+1
/// (ggml-quants.c, dequantize_row_q4_K). The block is 144 bytes, so `qs`
/// is 16-byte aligned when the row is. Two groups (eight vectors) per batch.
fn q4k(a: &mut Asm, t: usize, x: Act) {
    let name = format!("phi_q4k_{t}{}", x.suffix());
    prologue(a, &name, t, "one Q4_K superblock", 144, prep_k4);
    for pair in 0..2 {
        let b = [Zmm(8), Zmm(9), Zmm(10), Zmm(11)];
        let h = [Zmm(12), Zmm(13), Zmm(14), Zmm(15)];
        for (i, &r) in b.iter().enumerate() {
            bytes_aligned(a, r, Mem::new(BLK, 16 + 64 * pair + 16 * i as i32), Conv::U8);
        }
        split_nibbles(a, &[(b[0], h[0]), (b[1], h[1]), (b[2], h[2]), (b[3], h[3])]);
        let g0 = 2 * pair;
        let batch = [
            (b[0], 4 * g0),
            (b[1], 4 * g0 + 1),
            (h[0], 4 * g0 + 2),
            (h[1], 4 * g0 + 3),
            (b[2], 4 * g0 + 4),
            (b[3], 4 * g0 + 5),
            (h[2], 4 * g0 + 6),
            (h[3], 4 * g0 + 7),
        ];
        finish(a, &batch, true);
        fmas(a, t, &batch, x);
    }
    epilogue(a, &name, t);
}

/// The eight bits of the byte vectors `hs` (as floats), each into its own
/// register as 0.0 or 1.0: `h_j = floor(h * 2^-j)` for j = 1..7, then
/// `bit_j = h_j - 2 h_{j+1}` in increasing j (bit 7 is `h_7` itself, as
/// `h < 256`); bit 0 lands in `h`. Stage by stage across the vectors.
#[allow(clippy::needless_range_loop)] // j is the bit position, not only an index
fn bits_of(a: &mut Asm, hs: &[(Zmm, u8)]) -> Vec<[Zmm; 8]> {
    let regs: Vec<[Zmm; 8]> = hs
        .iter()
        .map(|&(h, scratch)| {
            let mut r = [h; 8];
            for j in 1..8u8 {
                r[j as usize] = Zmm(scratch + j - 1);
            }
            r
        })
        .collect();
    for j in 1..8 {
        for (n, &(h, _)) in hs.iter().enumerate() {
            a.i(&vmulps(regs[n][j], h, pow2neg(j as i32), K(0)));
        }
    }
    for j in 1..8 {
        for r in &regs {
            a.i(&vrndfxpntps(r[j], Src::Reg(r[j]), Round::Down, K(0)));
        }
    }
    for j in 0..7 {
        for r in &regs {
            a.i(&vfnmadd231ps(r[j], r[j + 1], constant(C_TWO), K(0)));
        }
    }
    regs
}

/// Q5_K: `qh` at 16 (32 bytes), `qs` at 48 (128 bytes); group g's low
/// nibbles take bit 2g of `qh`, its high nibbles bit 2g+1
/// (dequantize_row_q5_K: `u1 = 1, u2 = 2`, both shifted by 2 per group).
/// The block is 176 bytes: 16-byte aligned rows keep both arrays aligned.
fn q5k(a: &mut Asm, t: usize, x: Act) {
    let name = format!("phi_q5k_{t}{}", x.suffix());
    prologue(a, &name, t, "one Q5_K superblock", 176, prep_k4);
    let (qh0, qh1) = (Zmm(12), Zmm(13));
    bytes_aligned(a, qh0, Mem::new(BLK, 16), Conv::U8);
    bytes_aligned(a, qh1, Mem::new(BLK, 32), Conv::U8);
    let bits = bits_of(a, &[(qh0, 14), (qh1, 21)]);
    for pair in 0..2usize {
        let b = [Zmm(8), Zmm(9), Zmm(28), Zmm(29)];
        let h = [Zmm(10), Zmm(11), Zmm(30), Zmm(31)];
        for (i, &r) in b.iter().enumerate() {
            bytes_aligned(a, r, Mem::new(BLK, 48 + 64 * pair as i32 + 16 * i as i32), Conv::U8);
        }
        split_nibbles(a, &[(b[0], h[0]), (b[1], h[1]), (b[2], h[2]), (b[3], h[3])]);
        let (g0, g1) = (2 * pair, 2 * pair + 1);
        add_high_bits(
            a,
            &[
                (b[0], bits[0][2 * g0]),
                (b[1], bits[1][2 * g0]),
                (h[0], bits[0][2 * g0 + 1]),
                (h[1], bits[1][2 * g0 + 1]),
                (b[2], bits[0][2 * g1]),
                (b[3], bits[1][2 * g1]),
                (h[2], bits[0][2 * g1 + 1]),
                (h[3], bits[1][2 * g1 + 1]),
            ],
        );
        let batch = [
            (b[0], 4 * g0 as i32),
            (b[1], 4 * g0 as i32 + 1),
            (h[0], 4 * g0 as i32 + 2),
            (h[1], 4 * g0 as i32 + 3),
            (b[2], 4 * g1 as i32),
            (b[3], 4 * g1 as i32 + 1),
            (h[2], 4 * g1 as i32 + 2),
            (h[3], 4 * g1 as i32 + 3),
        ];
        finish(a, &batch, true);
        fmas(a, t, &batch, x);
    }
    epilogue(a, &name, t);
}

/// The four two-bit fields of the byte vectors `hs`: `h1 = floor(h/4)`,
/// `h2 = floor(h/16)`, `h3 = floor(h/64)`, then `f0 = h - 4 h1`,
/// `f1 = h1 - 4 h2`, `f2 = h2 - 4 h3`, `f3 = h3`. Field 0 lands in `h`.
#[allow(clippy::needless_range_loop)] // j is the field position, not only an index
fn fields_of(a: &mut Asm, hs: &[(Zmm, u8)]) -> Vec<[Zmm; 4]> {
    let regs: Vec<[Zmm; 4]> = hs.iter().map(|&(h, s)| [h, Zmm(s), Zmm(s + 1), Zmm(s + 2)]).collect();
    for j in 1..4 {
        for (n, &(h, _)) in hs.iter().enumerate() {
            a.i(&vmulps(regs[n][j], h, pow2neg(2 * j as i32), K(0)));
        }
    }
    for j in 1..4 {
        for r in &regs {
            a.i(&vrndfxpntps(r[j], Src::Reg(r[j]), Round::Down, K(0)));
        }
    }
    for j in 0..3 {
        for r in &regs {
            a.i(&vfnmadd231ps(r[j], r[j + 1], constant(C_FOUR), K(0)));
        }
    }
    regs
}

/// Q6_K: `ql` at 0 (128 bytes), `qh` at 128 (64), scales at 192 (16 bytes,
/// one per sixteen weights), `d` at 208; 210 bytes, so nothing here is
/// aligned and every load is the unpack pair. Per half of 128 weights
/// (dequantize_row_q6_K): weights 0..32 are the low nibbles of ql[0..32)
/// with field 0 of qh[0..32), 32..64 the low nibbles of ql[32..64) with
/// field 1, 64..96 the high nibbles of ql[0..32) with field 2, 96..128 the
/// high nibbles of ql[32..64) with field 3; the value is `q - 32`, which
/// the caller folds into the minuend (`32 * d * scale`).
fn q6k(a: &mut Asm, t: usize, x: Act) {
    let name = format!("phi_q6k_{t}{}", x.suffix());
    prologue(a, &name, t, "one Q6_K superblock", 210, prep_q6k);
    for h in 0..2i32 {
        let lq = [Zmm(8), Zmm(9), Zmm(10), Zmm(11)];
        let hi = [Zmm(20), Zmm(21), Zmm(22), Zmm(23)];
        for (i, r) in lq.iter().enumerate() {
            bytes_unaligned(a, *r, Mem::new(BLK, 64 * h + 16 * i as i32), Conv::U8);
        }
        let (qh0, qh1) = (Zmm(12), Zmm(13));
        bytes_unaligned(a, qh0, Mem::new(BLK, 128 + 32 * h), Conv::U8);
        bytes_unaligned(a, qh1, Mem::new(BLK, 144 + 32 * h), Conv::U8);
        let f = fields_of(a, &[(qh0, 14), (qh1, 17)]);
        split_nibbles(a, &[(lq[0], hi[0]), (lq[1], hi[1]), (lq[2], hi[2]), (lq[3], hi[3])]);
        // (register, field of qh0 or qh1, vector index within the half)
        let order = [
            (lq[0], f[0][0], 0),
            (lq[1], f[1][0], 1),
            (lq[2], f[0][1], 2),
            (lq[3], f[1][1], 3),
            (hi[0], f[0][2], 4),
            (hi[1], f[1][2], 5),
            (hi[2], f[0][3], 6),
            (hi[3], f[1][3], 7),
        ];
        let pairs: Vec<(Zmm, Zmm)> = order.iter().map(|&(w, f, _)| (w, f)).collect();
        add_high_bits(a, &pairs);
        let batch: Vec<(Zmm, i32)> = order.iter().map(|&(w, _, i)| (w, 8 * h + i)).collect();
        finish(a, &batch, true);
        fmas(a, t, &batch, x);
    }
    epilogue(a, &name, t);
}

/// Q8_0: eight blocks of 34 bytes (`d` then 32 signed bytes) make the 256
/// weights; block i's bytes are at 34 i + 2, any alignment. No minuend.
/// Four blocks (eight vectors) per batch.
fn q8_0(a: &mut Asm, t: usize, x: Act) {
    let name = format!("phi_q8_0_{t}{}", x.suffix());
    prologue(a, &name, t, "eight Q8_0 blocks", 272, prep_q8_0);
    for half in 0..2i32 {
        let mut batch = Vec::new();
        for i in 0..4i32 {
            let blk = 4 * half + i;
            let (w0, w1) = (Zmm(8 + 2 * i as u8), Zmm(9 + 2 * i as u8));
            bytes_unaligned(a, w0, Mem::new(BLK, 34 * blk + 2), Conv::S8);
            bytes_unaligned(a, w1, Mem::new(BLK, 34 * blk + 18), Conv::S8);
            batch.push((w0, 2 * blk));
            batch.push((w1, 2 * blk + 1));
        }
        finish(a, &batch, false);
        fmas(a, t, &batch, x);
    }
    epilogue(a, &name, t);
}

/// IQ4_XS: `d` at 0, `scales_h` at 2, `scales_l` at 4 (4 bytes), `qs` at 8
/// (128 bytes); 136 bytes, so `qs` is only 8-byte aligned. Each nibble
/// indexes the sixteen-entry value table (`kvalues_iq4nl`, in the
/// constants at `C_LUT` as floats); the low nibbles of bytes 16 ib..16 ib+16
/// are weights 32 ib..32 ib+16, the high ones the next sixteen
/// (dequantize_row_iq4_xs). No minuend. Four sub-blocks (eight vectors)
/// per batch.
fn iq4xs(a: &mut Asm, t: usize, x: Act) {
    let name = format!("phi_iq4xs_{t}{}", x.suffix());
    prologue(a, &name, t, "one IQ4_XS superblock", 136, prep_iq4xs);
    for half in 0..2i32 {
        let b = [Zmm(8), Zmm(9), Zmm(10), Zmm(11)];
        let h = [Zmm(12), Zmm(13), Zmm(14), Zmm(15)];
        let w = [Zmm(16), Zmm(17), Zmm(18), Zmm(19), Zmm(20), Zmm(21), Zmm(22), Zmm(23)];
        for (i, &r) in b.iter().enumerate() {
            bytes_unaligned(a, r, Mem::new(BLK, 8 + 64 * half + 16 * i as i32), Conv::U8);
        }
        split_nibbles(a, &[(b[0], h[0]), (b[1], h[1]), (b[2], h[2]), (b[3], h[3])]);
        for i in 0..4 {
            a.i(&vcvtfxpntps2dq(b[i], Src::Reg(b[i]), Round::Zero, K(0)));
            a.i(&vcvtfxpntps2dq(h[i], Src::Reg(h[i]), Round::Zero, K(0)));
        }
        for i in 0..4 {
            a.i(&vpermd(w[2 * i], b[i], Src::Mem(Mem::new(CONSTS, C_LUT)), K(0)));
            a.i(&vpermd(w[2 * i + 1], h[i], Src::Mem(Mem::new(CONSTS, C_LUT)), K(0)));
        }
        let mut batch = Vec::new();
        for i in 0..4i32 {
            let ib = 4 * half + i;
            batch.push((w[2 * i as usize], 2 * ib));
            batch.push((w[2 * i as usize + 1], 2 * ib + 1));
        }
        finish(a, &batch, false);
        fmas(a, t, &batch, x);
    }
    epilogue(a, &name, t);
}

pub fn kernels() -> String {
    let mut a = Asm(String::new());
    a.0.push_str(&format!("# quantized kernels: rcx = the superblock's table ({TAB_BYTES} bytes: scales {TAB_SC}, minuends {TAB_MN}), r9 = the constants ({C_BYTES} bytes: 16, 4, 2, 2^-1..2^-8, the IQ4_XS values at {C_LUT})\n"));
    for x in [Act::F32, Act::F16] {
        for t in [1, 4, 8] {
            q4k(&mut a, t, x);
            q5k(&mut a, t, x);
            q6k(&mut a, t, x);
            q8_0(&mut a, t, x);
            iq4xs(&mut a, t, x);
        }
    }
    a.0
}

/// `phi_probe(blk, x, tab, out)`: what the instructions the quantized
/// kernels rely on produce, one 64-byte vector each into `out`, for
/// `phi-vpu matmul-check --probe` to print (the card's own answer is the
/// only source of truth for an encoding). rdi = 256 bytes, rsi = 16
/// floats, rdx = the constants, rcx = out (8 vectors).
pub fn probe() -> String {
    let mut a = Asm(String::new());
    let name = "phi_probe";
    a.0.push_str(&format!(
        "# {name}: instruction probe (quant.md)\n    .globl {name}\n    .type {name}, @function\n{name}:\n"
    ));
    let out = |i: i32| Mem::new(Gpr::Rcx, 64 * i);
    let tab = Gpr::Rdx;
    // the prefetches first: a wrong encoding shows as a dead worker
    a.i(&vprefetch(Mem::new(Gpr::Rdi, 128), Cache::L1));
    a.i(&vprefetch(Mem::new(Gpr::Rdi, 192), Cache::L2));
    // 0: float unpack pair {uint8} at +3; 1: the int32 pair at +3 (integers); 2: aligned {uint8}; 3: float pair {sint8} at +3
    a.i(&vloadunpacklps_conv(Zmm(8), Mem::new(Gpr::Rdi, 3), Conv::U8, K(0)));
    a.i(&vloadunpackhps_conv(Zmm(8), Mem::new(Gpr::Rdi, 67), Conv::U8, K(0)));
    a.i(&vmovaps_store(out(0), Zmm(8)));
    a.i(&vloadunpackld_conv(Zmm(9), Mem::new(Gpr::Rdi, 3), Conv::U8, K(0)));
    a.i(&vloadunpackhd_conv(Zmm(9), Mem::new(Gpr::Rdi, 67), Conv::U8, K(0)));
    a.i(&vmovaps_store(out(1), Zmm(9)));
    bytes_aligned(&mut a, Zmm(10), Mem::new(Gpr::Rdi, 0), Conv::U8);
    a.i(&vmovaps_store(out(2), Zmm(10)));
    a.i(&vloadunpacklps_conv(Zmm(11), Mem::new(Gpr::Rdi, 3), Conv::S8, K(0)));
    a.i(&vloadunpackhps_conv(Zmm(11), Mem::new(Gpr::Rdi, 67), Conv::S8, K(0)));
    a.i(&vmovaps_store(out(3), Zmm(11)));
    // 4: x to int32 (stored as raw bits); 5: vpermd of the table by those
    a.i(&vmovaps_load(Zmm(12), Mem::new(Gpr::Rsi, 0)));
    a.i(&vcvtfxpntps2dq(Zmm(13), Src::Reg(Zmm(12)), Round::Zero, K(0)));
    a.i(&vmovaps_store(out(4), Zmm(13)));
    a.i(&vpermd(Zmm(14), Zmm(13), Src::Mem(Mem::new(tab, C_LUT)), K(0)));
    a.i(&vmovaps_store(out(5), Zmm(14)));
    // 6: floor(x / 4); 7: x - 4 floor(x / 4)
    a.i(&vmulps(
        Zmm(15),
        Zmm(12),
        Src::MemConv(Mem::new(tab, C_POW2NEG + 4), Conv::Bcast1),
        K(0),
    ));
    a.i(&vrndfxpntps(Zmm(15), Src::Reg(Zmm(15)), Round::Down, K(0)));
    a.i(&vmovaps_store(out(6), Zmm(15)));
    a.i(&vfnmadd231ps(
        Zmm(12),
        Zmm(15),
        Src::MemConv(Mem::new(tab, C_FOUR), Conv::Bcast1),
        K(0),
    ));
    a.i(&vmovaps_store(out(7), Zmm(12)));
    a.t("ret");
    a.0.push_str(&format!("    .size {name}, .-{name}\n\n"));
    a.0
}

/// `phi_bench(kind, buf, count)`: the core's raw rates, timed by the
/// caller. kind 0: `count` iterations of eight independent register
/// fused multiply-adds (8 vector instructions each); kind 1: `count`
/// aligned 64-byte loads walking `buf`. `count` 0 means 1 M, which walks
/// 64 MiB. Both leave the vector registers as they find them, which
/// nothing depends on. The whole pool runs it at once for the card's
/// aggregate rates (`vpu_matmul.c`, the probe).
pub fn bench() -> String {
    let mut a = Asm(String::new());
    let name = "phi_bench";
    a.0.push_str(&format!(
        "# {name}: raw issue and streaming rates (quant.md)\n    .globl {name}\n    .type {name}, @function\n{name}:\n"
    ));
    // A count in rdx replaces the default. Not a cmov: the card's scalar
    // core is a P54C and has none (an invalid opcode trap, 2026-09-23).
    a.t("mov %rdx, %rax");
    a.t("test %rax, %rax");
    a.t("jnz 8f");
    a.t("mov $1000000, %rax");
    a.t("8:");
    a.t("test %rdi, %rdi");
    a.t("jnz 2f");
    a.t("1:");
    for i in 0..8u8 {
        a.i(&vfmadd231ps(Zmm(i), Zmm(8 + i), Src::Reg(Zmm(16 + i)), K(0)));
    }
    a.t("dec %rax");
    a.t("jnz 1b");
    a.t("ret");
    // kinds 1..5: 1 M aligned loads walking buf; 2 with vprefetch0 4 lines ahead, 3 with
    // vprefetch1 16 ahead and vprefetch0 4 ahead, 4 with vprefetch1 32 ahead only,
    // 5 the {uint8} converting load (16 bytes a step). The caller sizes the walk.
    a.t("2:");
    a.t("mov %rsi, %rcx");
    a.t("cmp $2, %rdi");
    a.t("je 4f");
    a.t("cmp $3, %rdi");
    a.t("je 5f");
    a.t("cmp $4, %rdi");
    a.t("je 6f");
    a.t("cmp $5, %rdi");
    a.t("je 7f");
    a.t("cmp $6, %rdi");
    a.t("jne 3f");
    a.t("mov $4096, %rax");
    a.t("3:");
    a.i(&vmovaps_load(Zmm(0), Mem::new(Gpr::Rcx, 0)));
    a.t("add $64, %rcx");
    a.t("dec %rax");
    a.t("jnz 3b");
    a.t("ret");
    a.t("4:");
    a.i(&vprefetch(Mem::new(Gpr::Rcx, 256), Cache::L1));
    a.i(&vmovaps_load(Zmm(0), Mem::new(Gpr::Rcx, 0)));
    a.t("add $64, %rcx");
    a.t("dec %rax");
    a.t("jnz 4b");
    a.t("ret");
    a.t("5:");
    a.i(&vprefetch(Mem::new(Gpr::Rcx, 1024), Cache::L2));
    a.i(&vprefetch(Mem::new(Gpr::Rcx, 256), Cache::L1));
    a.i(&vmovaps_load(Zmm(0), Mem::new(Gpr::Rcx, 0)));
    a.t("add $64, %rcx");
    a.t("dec %rax");
    a.t("jnz 5b");
    a.t("ret");
    a.t("6:");
    a.i(&vprefetch(Mem::new(Gpr::Rcx, 2048), Cache::L2));
    a.i(&vmovaps_load(Zmm(0), Mem::new(Gpr::Rcx, 0)));
    a.t("add $64, %rcx");
    a.t("dec %rax");
    a.t("jnz 6b");
    a.t("ret");
    a.t("7:");
    a.i(&vmovaps_load_conv(Zmm(0), Mem::new(Gpr::Rcx, 0), Conv::U8, K(0)));
    a.t("add $16, %rcx");
    a.t("dec %rax");
    a.t("jnz 7b");
    a.t("ret");
    a.0.push_str(&format!("    .size {name}, .-{name}\n\n"));
    a.0
}

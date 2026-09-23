//! EVEX to MVEX rewriting for the seamless path.
//!
//! The card's MVEX prefix and AVX-512's EVEX prefix share the four-byte
//! shape, the opcode maps, and the ModRM, SIB and displacement bytes
//! that follow, so for a 512-bit register-to-register instruction the
//! card has one to one, the rewrite is the prefix payload alone and the
//! instruction keeps its length (`Rewrite::InPlace`): P1 bit 2 clears,
//! and P2's `z L'L b V' aaa` becomes `E SSS V' aaa`, with SSS 001 for an
//! embedded broadcast.
//!
//! Everything else becomes a sequence placed out of line: the site
//! becomes `jmp rel32` into a thunk area that runs the sequence and
//! jumps back (`Rewrite::Thunk`). The sequences are built by `Em`, an
//! emitter that encodes MVEX, VEX mask and a few integer instructions
//! directly, and that owns the per-thread scratch area the card provides
//! through `fs` (`Target::scratch`, `vpu_exec.h`): what a sequence
//! clobbers (rax, rcx, one mask register, temporary vector registers)
//! is saved there first and restored last, never on the program's stack,
//! which the threads of a split loop share.
//!
//! What the card cannot do directly and how each is expressed:
//!
//! - a 128-bit or 256-bit form (AVX512VL): the same 512-bit operation
//!   under a lane mask, then the lanes above the vector length zeroed,
//!   as the EVEX semantics require;
//! - zeroing masking `{z}`: the operation under the mask, then the
//!   unselected lanes zeroed;
//! - a memory operand that is not an aligned move: the card needs every
//!   memory operand aligned to its size (ISA reference 327364-001, 2.1.1)
//!   and EVEX does not, so the operand is staged through the card's
//!   alignment-free unpack pair into a temporary register (a 16-byte or
//!   32-byte aligned move reads its block with a block broadcast instead);
//! - scalar `ss`/`sd` operations: lane 0 under a mask, the other lanes of
//!   the low 128 bits from the first source;
//! - moves between vector and general registers (`vmovd`, `vmovq`):
//!   through the scratch area, since the card has no such instruction;
//! - inserts, extracts, shuffles within 128-bit blocks: `vpermf32x4` and
//!   `vpshufd` under masks;
//! - `vpternlogd`: the truth table split on the first operand into two
//!   two-input functions;
//! - conversions: the card's fixed-point conversions with the rounding in
//!   the immediate, and a fix-up for NaN and overflow, which AVX-512
//!   makes the integer indefinite and the card does not;
//! - `float16`: the card's memory up-conversion and store down-conversion;
//! - mask instructions the program encodes with VEX (`kmovw`, `kandw`, ...):
//!   the card's two-operand forms.
//!
//! Encodings are from the ISA reference 327364-001 (opcode lines quoted
//! where they are used) and every sequence is checked on the card by
//! `tools/avx512-narrow-test.c` (results in `docs/results/`).

use iced_x86::{EncodingKind, Instruction, Mnemonic, OpKind, Register};

use crate::Unsupported;

/// What the emitter needs to know about the card it targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// `fs`-relative displacement of the per-thread scratch area
    /// (`SCRATCH_BYTES` bytes, 64-byte aligned) the card worker provides;
    /// its layout is the `S_*` constants here.
    pub scratch: i32,
}

/// Bytes of the per-thread scratch area (`vpu_exec.h` reserves the same).
pub const SCRATCH_BYTES: u32 = 512;
const S_ZMM: [i32; 6] = [0, 64, 128, 192, 256, 320];
const S_RAX: i32 = 384;
const S_RCX: i32 = 392;
const S_K: i32 = 400;
const S_GPR: i32 = 408;
const S_FLAGS: i32 = 416;
const S_CW: i32 = 424; // the x87 control word, and a copy with truncation set
const S_CW2: i32 = 428;
const S_XFER: i32 = 448; // a 64-byte slot for register <-> memory conversions

/// What a site becomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rewrite {
    /// The same length, in place.
    InPlace(Vec<u8>),
    /// Out of line: the site becomes `jmp` to the sequence, which jumps back.
    Thunk(Thunk),
}

/// An out-of-line sequence with what the layout must patch.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Thunk {
    pub seq: Vec<u8>,
    pub fixups: Vec<Fixup>,
    /// 64-byte constants placed after the sequence, on 64-byte boundaries.
    pub consts: Vec<[u8; 64]>,
}

/// A 32-bit displacement in `seq` that depends on where the thunk lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fixup {
    /// Offset in `seq` of the displacement.
    pub disp_at: usize,
    /// Offset in `seq` of the end of the instruction (RIP-relative base).
    pub insn_end: usize,
    pub what: FixupKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixupKind {
    /// The constant of that index.
    Const(usize),
    /// An absolute address of the program.
    Rip(u64),
}

fn refuse(insn: &Instruction, reason: &str) -> Unsupported {
    Unsupported {
        text: format!("{insn}"),
        reason: reason.to_string(),
    }
}

/// zmm number of a zmm, ymm or xmm register operand.
fn vreg(r: Register) -> Option<u8> {
    if r.is_zmm() {
        Some((r as u32 - Register::ZMM0 as u32) as u8)
    } else if r.is_ymm() {
        Some((r as u32 - Register::YMM0 as u32) as u8)
    } else if r.is_xmm() {
        Some((r as u32 - Register::XMM0 as u32) as u8)
    } else {
        None
    }
}

fn kreg(r: Register) -> Option<u8> {
    if r.is_k() {
        Some((r as u32 - Register::K0 as u32) as u8)
    } else {
        None
    }
}

/// Number of a 64-bit or 32-bit general register, in encoding order.
fn gpr(r: Register) -> Option<(u8, bool)> {
    if r.is_gpr64() {
        Some(((r as u32 - Register::RAX as u32) as u8, true))
    } else if r.is_gpr32() {
        Some(((r as u32 - Register::EAX as u32) as u8, false))
    } else {
        None
    }
}

/// The EVEX prefix fields of an instruction, as encoded.
#[derive(Clone, Copy, Debug)]
struct Ev {
    map: u8,
    pp: u8,
    w: u8,
    ll: u8,
    z: bool,
    b: bool,
    aaa: u8,
    op: u8,
    modrm: u8,
    /// ModRM.reg with R and R' (5 bits).
    reg: u8,
    /// vvvv with V' (5 bits).
    vvvv: u8,
    /// ModRM.rm with B and X, when mod is 11 (5 bits).
    rm: u8,
    is_reg_rm: bool,
}

fn parse(bytes: &[u8]) -> Ev {
    let (p0, p1, p2) = (bytes[1], bytes[2], bytes[3]);
    let modrm = bytes[5];
    Ev {
        map: p0 & 3,
        pp: p1 & 3,
        w: p1 >> 7,
        ll: (p2 >> 5) & 3,
        z: p2 >> 7 == 1,
        b: (p2 >> 4) & 1 == 1,
        aaa: p2 & 7,
        op: bytes[4],
        modrm,
        reg: ((modrm >> 3) & 7) | ((!p0 >> 7) & 1) << 3 | ((!p0 >> 4) & 1) << 4,
        vvvv: ((!p1 >> 3) & 0xf) | ((!p2 >> 3) & 1) << 4,
        rm: (modrm & 7) | ((!p0 >> 5) & 1) << 3 | ((!p0 >> 6) & 1) << 4,
        is_reg_rm: modrm >> 6 == 3,
    }
}

/// A memory operand, decoded, to be re-encoded with a 32-bit displacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MemOp {
    base: Option<u8>,
    index: Option<u8>,
    scale: u8,
    disp: i64,
    /// Absolute target of a RIP-relative operand.
    rip: Option<u64>,
}

impl MemOp {
    fn of(insn: &Instruction) -> Option<MemOp> {
        if !(0..insn.op_count()).any(|i| insn.op_kind(i) == OpKind::Memory) {
            return None;
        }
        if insn.is_ip_rel_memory_operand() {
            return Some(MemOp {
                base: None,
                index: None,
                scale: 0,
                disp: 0,
                rip: Some(insn.ip_rel_memory_address()),
            });
        }
        let base = gpr(insn.memory_base()).map(|(n, _)| n);
        let index = gpr(insn.memory_index()).map(|(n, _)| n);
        let scale = match insn.memory_index_scale() {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => 3,
        };
        Some(MemOp {
            base,
            index,
            scale,
            disp: insn.memory_displacement64() as i64,
            rip: None,
        })
    }

    fn plus(self, delta: i64) -> MemOp {
        MemOp {
            disp: self.disp + delta,
            rip: self.rip.map(|r| r.wrapping_add(delta as u64)),
            ..self
        }
    }
}

/// The r/m operand of an emitted instruction.
#[derive(Clone, Copy, Debug)]
enum Rm {
    Reg(u8),
    Mem(MemOp),
    /// `[fs: scratch + offset]`.
    Fs(i32),
    /// A 64-byte constant of the thunk.
    Const(usize),
}

/// Emits one thunk: MVEX, VEX mask and integer instructions, keeping
/// what it clobbers in the scratch area.
struct Em<'a> {
    tg: &'a Target,
    t: Thunk,
    /// zmm registers in use: the instruction's operands and the temporaries.
    reserved: u32,
    temps: Vec<(u8, usize)>,
    kscr: u8,
    k_used: bool,
    rax_used: bool,
    rcx_used: bool,
    flags_used: bool,
    /// General registers loaded from the scratch area after the restores.
    tail: Vec<(u8, bool, i32)>,
}

impl<'a> Em<'a> {
    fn new(tg: &'a Target, insn: &Instruction) -> Em<'a> {
        let mut reserved = 0u32;
        let mut kmask = 0u8;
        for i in 0..insn.op_count() {
            if insn.op_kind(i) == OpKind::Register {
                if let Some(z) = vreg(insn.op_register(i)) {
                    reserved |= 1 << z;
                }
                if let Some(k) = kreg(insn.op_register(i)) {
                    kmask |= 1 << k;
                }
            }
        }
        if let Some(k) = kreg(insn.op_mask()) {
            kmask |= 1 << k;
        }
        let kscr = (1..8).rev().find(|k| kmask & (1 << k) == 0).unwrap_or(7);
        Em {
            tg,
            t: Thunk::default(),
            reserved,
            temps: Vec::new(),
            kscr,
            k_used: false,
            rax_used: false,
            rcx_used: false,
            flags_used: false,
            tail: Vec::new(),
        }
    }

    fn fs(&self, off: i32) -> i32 {
        self.tg.scratch.wrapping_add(off)
    }

    /// One MVEX.512 instruction. `reg` is the ModRM.reg register (zmm or
    /// k, 5 bits), `vvvv` the first source (0 when unused), `k` the write
    /// mask (0 = none), `sss` the swizzle/conversion or broadcast field.
    #[allow(clippy::too_many_arguments)]
    fn mvex(&mut self, map: u8, pp: u8, w: u8, op: u8, reg: u8, vvvv: u8, rm: Rm, k: u8, sss: u8, imm: Option<u8>) {
        let (xbar, bbar, seg) = match rm {
            Rm::Reg(r) => (!(r >> 4) & 1, !(r >> 3) & 1, false),
            Rm::Mem(m) => (m.index.map_or(1, |i| !(i >> 3) & 1), m.base.map_or(1, |b| !(b >> 3) & 1), false),
            Rm::Fs(_) => (1, 1, true),
            Rm::Const(_) => (1, 1, false),
        };
        let p0 = ((!(reg >> 3) & 1) << 7) | (xbar << 6) | (bbar << 5) | ((!(reg >> 4) & 1) << 4) | map;
        let p1 = (w << 7) | ((!vvvv & 0xf) << 3) | pp;
        let p2 = (sss << 4) | ((!(vvvv >> 4) & 1) << 3) | (k & 7);
        if seg {
            self.t.seq.push(0x64);
        }
        self.t.seq.extend_from_slice(&[0x62, p0, p1, p2, op]);
        let r3 = (reg & 7) << 3;
        let mut fix: Option<(usize, FixupKind)> = None;
        match rm {
            Rm::Reg(r) => self.t.seq.push(0xc0 | r3 | (r & 7)),
            Rm::Fs(off) => {
                self.t.seq.extend_from_slice(&[0x04 | r3, 0x25]);
                self.t.seq.extend_from_slice(&self.fs(off).to_le_bytes());
            }
            Rm::Const(i) => {
                self.t.seq.push(0x05 | r3);
                fix = Some((self.t.seq.len(), FixupKind::Const(i)));
                self.t.seq.extend_from_slice(&[0; 4]);
            }
            Rm::Mem(m) => {
                if let Some(target) = m.rip {
                    self.t.seq.push(0x05 | r3);
                    fix = Some((self.t.seq.len(), FixupKind::Rip(target)));
                    self.t.seq.extend_from_slice(&[0; 4]);
                } else {
                    let disp = m.disp as i32;
                    match (m.base, m.index) {
                        (Some(b), None) if b & 7 != 4 => {
                            self.t.seq.push(0x80 | r3 | (b & 7));
                        }
                        (Some(_), None) => {
                            // rsp or r12 as the base: a SIB byte with no index.
                            self.t.seq.extend_from_slice(&[0x84 | r3, 0x24]);
                        }
                        (Some(b), Some(i)) => {
                            self.t
                                .seq
                                .extend_from_slice(&[0x84 | r3, (m.scale << 6) | ((i & 7) << 3) | (b & 7)]);
                        }
                        (None, Some(i)) => {
                            self.t.seq.extend_from_slice(&[0x04 | r3, (m.scale << 6) | ((i & 7) << 3) | 5]);
                        }
                        (None, None) => {
                            self.t.seq.extend_from_slice(&[0x04 | r3, 0x25]);
                        }
                    }
                    self.t.seq.extend_from_slice(&disp.to_le_bytes());
                }
            }
        }
        if let Some(i) = imm {
            self.t.seq.push(i);
        }
        if let Some((disp_at, what)) = fix {
            self.t.fixups.push(Fixup {
                disp_at,
                insn_end: self.t.seq.len(),
                what,
            });
        }
    }

    /// A VEX-encoded mask instruction with two mask (or one general)
    /// register operands: `op` from the KAND/KMOV/KNOT/KOR/KXOR lines
    /// (VEX.128.0F.W0 41 44 45 47 90 92 93 98 /r).
    fn vexk(&mut self, op: u8, reg: u8, rm: u8) {
        if reg < 8 && rm < 8 {
            self.t.seq.extend_from_slice(&[0xc5, 0xf8, op, 0xc0 | (reg << 3) | rm]);
        } else {
            let b1 = ((!(reg >> 3) & 1) << 7) | (1 << 6) | ((!(rm >> 3) & 1) << 5) | 1;
            self.t
                .seq
                .extend_from_slice(&[0xc4, b1, 0x78, op, 0xc0 | ((reg & 7) << 3) | (rm & 7)]);
        }
    }
    fn kand(&mut self, k1: u8, k2: u8) {
        self.vexk(0x41, k1, k2);
    }
    fn knot(&mut self, k1: u8, k2: u8) {
        self.vexk(0x44, k1, k2);
    }
    fn kor(&mut self, k1: u8, k2: u8) {
        self.vexk(0x45, k1, k2);
    }
    fn kmov_kk(&mut self, k1: u8, k2: u8) {
        self.vexk(0x90, k1, k2);
    }
    fn kmov_k_r(&mut self, k: u8, r: u8) {
        self.vexk(0x92, k, r);
    }
    fn kmov_r_k(&mut self, r: u8, k: u8) {
        self.vexk(0x93, r, k);
    }

    /// `mov [fs:scratch+off], r` (89 /r) or `mov r, [fs:scratch+off]` (8B /r).
    fn gpr_scratch(&mut self, r: u8, wide: bool, off: i32, store: bool) {
        self.t.seq.push(0x64);
        let rex = 0x40 | ((wide as u8) << 3) | ((r >> 3) & 1);
        if rex != 0x40 {
            self.t.seq.push(rex);
        }
        self.t
            .seq
            .extend_from_slice(&[if store { 0x89 } else { 0x8b }, 0x04 | ((r & 7) << 3), 0x25]);
        self.t.seq.extend_from_slice(&self.fs(off).to_le_bytes());
    }

    fn rax(&mut self) {
        if !self.rax_used {
            self.rax_used = true;
            self.gpr_scratch(0, true, S_RAX, true);
        }
    }
    fn rcx(&mut self) {
        if !self.rcx_used {
            self.rcx_used = true;
            self.gpr_scratch(1, true, S_RCX, true);
        }
    }
    /// The scratch mask register, saved on first use.
    fn k(&mut self) -> u8 {
        if !self.k_used {
            self.k_used = true;
            self.rax();
            self.kmov_r_k(0, self.kscr);
            self.gpr_scratch(0, true, S_K, true);
        }
        self.kscr
    }
    /// Save the flags (rax is clobbered, so it is saved first).
    fn flags(&mut self) {
        if !self.flags_used {
            self.flags_used = true;
            self.rax();
            self.t.seq.extend_from_slice(&[0x9f, 0x0f, 0x90, 0xc0]); // lahf; seto al
            self.gpr_scratch(0, true, S_FLAGS, true);
        }
    }
    fn mov_eax(&mut self, imm: u32) {
        self.rax();
        self.t.seq.push(0xb8);
        self.t.seq.extend_from_slice(&imm.to_le_bytes());
    }
    /// The scratch mask register set to `mask`.
    fn mask_imm(&mut self, mask: u16) -> u8 {
        let k = self.k();
        self.mov_eax(u32::from(mask));
        self.kmov_k_r(k, 0);
        k
    }

    /// A temporary zmm register, its value kept in the scratch area.
    fn temp(&mut self) -> Result<u8, &'static str> {
        let z = (0..32u8).find(|z| self.reserved & (1 << z) == 0).ok_or("no free vector register")?;
        let slot = self.temps.len();
        if slot >= S_ZMM.len() {
            return Err("more temporaries than scratch slots");
        }
        self.reserved |= 1 << z;
        self.temps.push((z, slot));
        self.mvex(1, 0, 0, 0x29, z, 0, Rm::Fs(S_ZMM[slot]), 0, 0, None);
        Ok(z)
    }

    fn constant(&mut self, c: [u8; 64]) -> usize {
        if let Some(i) = self.t.consts.iter().position(|x| *x == c) {
            return i;
        }
        self.t.consts.push(c);
        self.t.consts.len() - 1
    }
    fn const_dws(&mut self, v: [u32; 16]) -> usize {
        let mut c = [0u8; 64];
        for (l, x) in v.iter().enumerate() {
            c[l * 4..l * 4 + 4].copy_from_slice(&x.to_le_bytes());
        }
        self.constant(c)
    }
    fn const_dw(&mut self, v: u32) -> usize {
        let mut c = [0u8; 64];
        for l in 0..16 {
            c[l * 4..l * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        self.constant(c)
    }

    // ---- the vector instructions the sequences use ----

    /// vmovaps d{k}, s (MVEX.512.0F.W0 28 /r); vmovapd for W1 (66, 28).
    fn vmov(&mut self, d: u8, s: Rm, k: u8, w: u8) {
        self.mvex(1, w, w, 0x28, d, 0, s, k, 0, None);
    }
    /// vpxord d{k}, d, d (MVEX.NDS.512.66.0F.W0 EF /r); vpxorq for W1.
    fn zero(&mut self, d: u8, k: u8, w: u8) {
        self.mvex(1, 1, w, 0xEF, d, d, Rm::Reg(d), k, 0, None);
    }
    /// Zero the dword lanes selected by `mask`.
    fn zero_lanes(&mut self, d: u8, mask: u16) {
        if mask != 0 {
            let k = self.mask_imm(mask);
            self.zero(d, k, 0);
        }
    }
    /// vpshufd d{k}, s, imm (MVEX.512.66.0F.W0 70 /r ib).
    fn vpshufd(&mut self, d: u8, s: Rm, imm: u8, k: u8) {
        self.mvex(1, 1, 0, 0x70, d, 0, s, k, 0, Some(imm));
    }
    /// vpermf32x4 d{k}, s, imm (MVEX.512.66.0F3A.W0 07 /r ib).
    fn vpermf32x4(&mut self, d: u8, s: Rm, imm: u8, k: u8) {
        self.mvex(3, 1, 0, 0x07, d, 0, s, k, 0, Some(imm));
    }
    /// vpermd d{k}, idx, s (MVEX.NDS.512.66.0F38.W0 36 /r).
    fn vpermd(&mut self, d: u8, idx: u8, s: Rm, k: u8) {
        self.mvex(2, 1, 0, 0x36, d, idx, s, k, 0, None);
    }
    /// valignd d{k}, s1, s2, imm (MVEX.NDS.512.66.0F3A.W0 03 /r ib).
    fn valignd(&mut self, d: u8, s1: u8, s2: Rm, imm: u8, k: u8) {
        self.mvex(3, 1, 0, 0x03, d, s1, s2, k, 0, Some(imm));
    }
    /// A three-operand 0F38 integer op (vpmulld 40, vpsrlvd 45, vpsravd 46,
    /// vpsllvd 47, vpminsd 39, vpminud 3B, vpmaxsd 3D, vpmaxud 3F, vpmulhud 86, vpmulhd 87).
    fn i38(&mut self, op: u8, d: u8, s1: u8, s2: Rm, k: u8) {
        self.mvex(2, 1, 0, op, d, s1, s2, k, 0, None);
    }
    /// A three-operand 0F integer op (vpaddd FE, vpsubd FA, vpandd DB,
    /// vpandnd DF, vpord EB, vpxord EF); W1 for the q forms of the bitwise ones.
    fn i0f(&mut self, op: u8, d: u8, s1: u8, s2: Rm, k: u8, w: u8) {
        self.mvex(1, 1, w, op, d, s1, s2, k, 0, None);
    }
    /// vpslld /6, vpsrld /2, vpsrad /4 by immediate (MVEX.NDD.512.66.0F.W0 72 /n ib).
    fn shift_imm(&mut self, which: u8, d: u8, s: Rm, n: u8, k: u8) {
        self.mvex(1, 1, 0, 0x72, which, d, s, k, 0, Some(n));
    }
    /// vcmpps k2{k1}, s1, s2, pred (MVEX.NDS.512.0F.W0 C2 /r ib); vcmppd (66, W1).
    #[allow(clippy::too_many_arguments)]
    fn vcmp(&mut self, kd: u8, s1: u8, s2: Rm, pred: u8, k: u8, w: u8, sss: u8) {
        self.mvex(1, w, w, 0xC2, kd, s1, s2, k, sss, Some(pred));
    }
    /// The unaligned load pair: vloadunpackld/hd (MVEX.512.0F38.W0 D0/D4,
    /// W1 D0/D4 for qwords, D1/D5 for the float forms that take the
    /// float16 up-conversion), each masked, the high half at +64.
    fn load_unaligned(&mut self, d: u8, m: MemOp, k: u8, w: u8, float: bool, sss: u8) {
        let lo = if float { 0xD1 } else { 0xD0 };
        self.mvex(2, 0, w, lo, d, 0, Rm::Mem(m), k, sss, None);
        self.mvex(2, 0, w, lo + 4, d, 0, Rm::Mem(m.plus(64)), k, sss, None);
    }
    /// The unaligned store pair: vpackstoreld/hd (MVEX.512.66.0F38.W0
    /// D0/D4; W1 for qwords; D1/D5 float forms with the float16
    /// down-conversion), masked.
    fn store_unaligned(&mut self, m: MemOp, s: u8, k: u8, w: u8, float: bool, sss: u8) {
        let lo = if float { 0xD1 } else { 0xD0 };
        self.mvex(2, 1, w, lo, s, 0, Rm::Mem(m), k, sss, None);
        self.mvex(2, 1, w, lo + 4, s, 0, Rm::Mem(m.plus(64)), k, sss, None);
    }
    /// A 16-byte block into every block (vbroadcasti32x4, MVEX.512.66.0F38.W0
    /// 5A /r) or a 32-byte half into both halves (vbroadcasti64x4, W1 5B).
    fn load_block(&mut self, d: u8, m: Rm, k: u8, half: bool) {
        if half {
            self.mvex(2, 1, 1, 0x5B, d, 0, m, k, 0, None);
        } else {
            self.mvex(2, 1, 0, 0x5A, d, 0, m, k, 0, None);
        }
    }
    /// vbroadcastss d{k}, m32 (MVEX.512.66.0F38.W0 18 /r); vbroadcastsd W1 19.
    fn broadcast(&mut self, d: u8, m: Rm, k: u8, w: u8) {
        self.mvex(2, 1, w, 0x18 | w, d, 0, m, k, 0, None);
    }

    /// The scratch mask set to the dword-lane form of the qword-lane mask
    /// in `k` (bit i to bits 2i and 2i+1): the unaligned pairs move
    /// dwords, since qword elements would need 8-byte alignment that
    /// AVX-512 does not require.
    fn qmask_to_dmask(&mut self, k: u8, limit: u16) -> u8 {
        let ks = self.k();
        self.flags();
        self.rcx();
        self.kmov_r_k(0, k);
        // movzx eax, al; mov ecx, eax; shl ecx, 4; or eax, ecx; and eax, 0x0f0f
        self.t.seq.extend_from_slice(&[
            0x0f, 0xb6, 0xc0, 0x89, 0xc1, 0xc1, 0xe1, 0x04, 0x09, 0xc8, 0x25, 0x0f, 0x0f, 0x00, 0x00,
        ]);
        // mov ecx, eax; shl ecx, 2; or eax, ecx; and eax, 0x3333
        self.t
            .seq
            .extend_from_slice(&[0x89, 0xc1, 0xc1, 0xe1, 0x02, 0x09, 0xc8, 0x25, 0x33, 0x33, 0x00, 0x00]);
        // mov ecx, eax; shl ecx, 1; or eax, ecx; and eax, 0x5555
        self.t
            .seq
            .extend_from_slice(&[0x89, 0xc1, 0xd1, 0xe1, 0x09, 0xc8, 0x25, 0x55, 0x55, 0x00, 0x00]);
        // mov ecx, eax; shl ecx, 1; or eax, ecx; and eax, limit
        self.t.seq.extend_from_slice(&[0x89, 0xc1, 0xd1, 0xe1, 0x09, 0xc8, 0x25]);
        self.t.seq.extend_from_slice(&u32::from(limit).to_le_bytes());
        self.kmov_k_r(ks, 0);
        ks
    }

    /// `mov r, [fs:scratch+off]` after everything is restored: for a
    /// general register the sequence produces (rax included).
    fn gpr_result(&mut self, r: u8, wide: bool, off: i32) {
        self.tail.push((r, wide, off));
    }

    /// Restore everything and hand the sequence over.
    fn finish(mut self) -> Thunk {
        let temps = std::mem::take(&mut self.temps);
        for &(z, slot) in temps.iter().rev() {
            self.mvex(1, 0, 0, 0x28, z, 0, Rm::Fs(S_ZMM[slot]), 0, 0, None);
        }
        if self.flags_used {
            self.gpr_scratch(0, true, S_FLAGS, false);
            self.t.seq.extend_from_slice(&[0x04, 0x7f, 0x9e]); // add al, 127; sahf
        }
        if self.k_used {
            self.gpr_scratch(0, true, S_K, false);
            self.kmov_k_r(self.kscr, 0);
        }
        if self.rcx_used {
            self.gpr_scratch(1, true, S_RCX, false);
        }
        if self.rax_used {
            self.gpr_scratch(0, true, S_RAX, false);
        }
        let tail = std::mem::take(&mut self.tail);
        for (r, wide, off) in tail {
            self.gpr_scratch(r, wide, off, false);
        }
        self.t
    }
}

/// Write the result held in `r` into `d` under the instruction's mask and
/// vector length: merge under `kn`, zero under `{z}`, zero above the length.
fn commit(em: &mut Em, d: u8, r: u8, ev: &Ev) {
    let vl = vl_lanes(ev.ll, ev.w);
    let full = ev.ll == 2;
    if r != d || !full || ev.aaa != 0 {
        let k = if full && ev.aaa == 0 {
            0
        } else {
            let k = em.mask_imm(vl);
            if ev.aaa != 0 {
                em.kand(k, ev.aaa);
            }
            k
        };
        if r != d {
            em.vmov(d, Rm::Reg(r), k, ev.w);
        }
        if ev.z && ev.aaa != 0 {
            let kz = em.k();
            em.knot(kz, k);
            em.zero(d, kz, ev.w);
        }
    }
    if !full {
        em.zero_lanes(d, upper(ev.ll));
    }
}

/// The second source of a three-operand instruction as a register: the
/// register itself, or a memory operand staged into a temporary. `elems`
/// is the lane mask the load needs (dword or qword lanes per W).
fn src2(em: &mut Em, insn: &Instruction, ev: &Ev, _elems: u16) -> Result<u8, Unsupported> {
    match MemOp::of(insn) {
        None => Ok(ev.rm),
        Some(m) => {
            let t = em.temp().map_err(|e| refuse(insn, e))?;
            if ev.b {
                em.mvex(1, ev.w, ev.w, 0x28, t, 0, Rm::Mem(m), 0, 1, None);
            } else {
                let dm = vl_lanes(ev.ll, 0);
                let k = if dm == 0xffff { 0 } else { em.mask_imm(dm) };
                em.load_unaligned(t, m, k, 0, false, 0);
            }
            Ok(t)
        }
    }
}

/// The instructions with their own operand shapes. `Ok(None)` means the
/// instruction is not one of them.
fn special(insn: &Instruction, bytes: &[u8], ev: &Ev, tg: &Target) -> Result<Option<Rewrite>, Unsupported> {
    use Mnemonic as M;
    let m = insn.mnemonic();
    let r = match m {
        M::Vextractf64x4
        | M::Vextractf32x4
        | M::Vextracti64x4
        | M::Vextracti32x4
        | M::Vextractf64x2
        | M::Vextracti64x2
        | M::Vextractf32x8
        | M::Vextracti32x8 => extract(insn, ev, tg)?,
        M::Vinsertf64x4
        | M::Vinsertf32x4
        | M::Vinserti64x4
        | M::Vinserti32x4
        | M::Vinsertf64x2
        | M::Vinserti64x2
        | M::Vinsertf32x8
        | M::Vinserti32x8 => insert(insn, ev, tg)?,
        M::Vpternlogd | M::Vpternlogq => ternlog(insn, ev, tg)?,
        M::Vmovups | M::Vmovupd | M::Vmovdqu32 | M::Vmovdqu64 | M::Vmovdqu8 | M::Vmovdqu16 => unaligned_move(insn, bytes, ev, tg)?,
        M::Vmovss | M::Vmovsd => scalar_move(insn, ev, tg)?,
        M::Vaddss
        | M::Vsubss
        | M::Vmulss
        | M::Vaddsd
        | M::Vsubsd
        | M::Vmulsd
        | M::Vfmadd132ss
        | M::Vfmadd213ss
        | M::Vfmadd231ss
        | M::Vfmsub132ss
        | M::Vfmsub213ss
        | M::Vfmsub231ss
        | M::Vfnmadd132ss
        | M::Vfnmadd213ss
        | M::Vfnmadd231ss
        | M::Vfnmsub132ss
        | M::Vfnmsub213ss
        | M::Vfnmsub231ss
        | M::Vfmadd132sd
        | M::Vfmadd213sd
        | M::Vfmadd231sd
        | M::Vfmsub132sd
        | M::Vfmsub213sd
        | M::Vfmsub231sd
        | M::Vfnmadd132sd
        | M::Vfnmadd213sd
        | M::Vfnmadd231sd
        | M::Vfnmsub132sd
        | M::Vfnmsub213sd
        | M::Vfnmsub231sd => scalar_op(insn, ev, tg)?,
        M::Vmaxss | M::Vminss | M::Vmaxsd | M::Vminsd => scalar_minmax(insn, ev, tg)?,
        M::Vmaxps | M::Vminps | M::Vmaxpd | M::Vminpd => minmax(insn, ev, tg)?,
        M::Vmovd | M::Vmovq => movdq(insn, ev, tg)?,
        M::Vpbroadcastd | M::Vpbroadcastq | M::Vbroadcastss | M::Vbroadcastsd if MemOp::of(insn).is_none() => broadcast_reg(insn, ev, tg)?,
        M::Vmovlhps
        | M::Vmovhlps
        | M::Vunpcklps
        | M::Vunpckhps
        | M::Vunpcklpd
        | M::Vunpckhpd
        | M::Vpunpckldq
        | M::Vpunpckhdq
        | M::Vpunpcklqdq
        | M::Vpunpckhqdq
        | M::Vshufps
        | M::Vshufpd => interleave(insn, ev, tg)?,
        M::Valignq => valignq(insn, ev, tg)?,
        M::Vpalignr | M::Vpsrldq | M::Vpslldq => byte_shift(insn, ev, tg)?,
        M::Vcvttps2dq | M::Vcvtps2dq => cvt_to_int(insn, ev, tg)?,
        M::Vcvtph2ps => cvt_ph2ps(insn, ev, tg)?,
        M::Vcvtpd2ps => cvt_pd2ps(insn, ev, tg)?,
        M::Vscalefps => scalef(insn, ev, tg)?,
        M::Vdivps | M::Vdivpd | M::Vdivss | M::Vdivsd | M::Vsqrtps | M::Vsqrtpd | M::Vsqrtss | M::Vsqrtsd => divsqrt(insn, ev, tg)?,
        M::Vrndscaleps | M::Vrndscalepd | M::Vrndscaless | M::Vrndscalesd => rndscale(insn, ev, tg)?,
        M::Vcvtsi2ss
        | M::Vcvtsi2sd
        | M::Vcvtusi2ss
        | M::Vcvtusi2sd
        | M::Vcvttss2si
        | M::Vcvtss2si
        | M::Vcvttsd2si
        | M::Vcvtsd2si
        | M::Vcvttss2usi
        | M::Vcvtss2usi
        | M::Vcvttsd2usi
        | M::Vcvtsd2usi
        | M::Vcvtss2sd
        | M::Vcvtsd2ss => scalar_cvt(insn, ev, tg)?,
        M::Vpermi2ps
        | M::Vpermi2d
        | M::Vpermt2ps
        | M::Vpermt2d
        | M::Vpermi2pd
        | M::Vpermi2q
        | M::Vpermt2pd
        | M::Vpermt2q
        | M::Vpermq
        | M::Vpermpd
        | M::Vpermilpd => permute(insn, ev, tg)?,
        M::Vpermilps | M::Vpermd | M::Vpermps if ev.is_reg_rm || !ev.b => match insn.op_kind(insn.op_count() - 1) {
            OpKind::Immediate8 => return Ok(None),
            _ if insn.mnemonic() == M::Vpermilps || ev.ll != 2 => permute(insn, ev, tg)?,
            _ => return Ok(None),
        },
        M::Vcvtps2ph => cvt_ps2ph(insn, ev, tg)?,
        M::Vpmovzxbd | M::Vpmovsxbd | M::Vpmovzxwd | M::Vpmovsxwd => widen(insn, ev, tg)?,
        M::Vpsrlq | M::Vpsllq | M::Vpsraq if insn.op_kind(insn.op_count() - 1) == OpKind::Immediate8 => shift_q(insn, ev, tg)?,
        M::Vpabsd => abs_d(insn, ev, tg)?,
        M::Vpmuludq => muludq(insn, ev, tg)?,
        M::Vpslld | M::Vpsrld | M::Vpsrad if insn.op_kind(insn.op_count() - 1) != OpKind::Immediate8 => shift_by_xmm(insn, ev, tg)?,
        _ => return Ok(None),
    };
    Ok(Some(r))
}

fn extract(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let s = ev.reg;
    let imm = insn.immediate8();
    // 128-bit block (x4 of dwords, x2 of qwords) or 256-bit half (x8, x4)
    let half = matches!(
        insn.mnemonic(),
        M::Vextractf64x4 | M::Vextracti64x4 | M::Vextractf32x8 | M::Vextracti32x8
    );
    let perm = if half {
        if imm & 1 == 1 {
            Some(0x4e)
        } else {
            None
        }
    } else if imm & 3 != 0 {
        Some((imm & 3) * 0x55)
    } else {
        None
    };
    let lanes: u16 = if half { 0x00ff } else { 0x000f };
    match MemOp::of(insn) {
        Some(m) => {
            // To memory: the block brought to the bottom of a temporary, then packed out.
            let t = em.temp().map_err(|e| refuse(insn, e))?;
            match perm {
                Some(p) => em.vpermf32x4(t, Rm::Reg(s), p, 0),
                None => em.vmov(t, Rm::Reg(s), 0, 0),
            }
            let k = if ev.aaa != 0 && ev.w == 1 {
                em.qmask_to_dmask(ev.aaa, lanes)
            } else {
                let k = em.mask_imm(lanes);
                if ev.aaa != 0 {
                    em.kand(k, ev.aaa);
                }
                k
            };
            em.store_unaligned(m, t, k, 0, false, 0);
        }
        None => {
            let d = ev.rm;
            let src = match perm {
                Some(p) => {
                    let t = if ev.aaa != 0 { em.temp().map_err(|e| refuse(insn, e))? } else { d };
                    em.vpermf32x4(t, Rm::Reg(s), p, 0);
                    t
                }
                None => s,
            };
            let ev2 = Ev {
                ll: if half { 1 } else { 0 },
                ..*ev
            };
            commit(&mut em, d, src, &ev2);
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

fn insert(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let d = ev.reg;
    let s1 = ev.vvvv;
    let imm = insn.immediate8();
    let half = matches!(
        insn.mnemonic(),
        M::Vinsertf64x4 | M::Vinserti64x4 | M::Vinsertf32x8 | M::Vinserti32x8
    );
    let lanes: u16 = if half { 0x00ff } else { 0x000f };
    let s2 = src2(&mut em, insn, ev, lanes)?;
    let (perm, block_mask) = if half {
        (0x44u8, if imm & 1 == 1 { 0xff00u16 } else { 0x00ffu16 })
    } else {
        (0x00u8, 0x000fu16 << (4 * (imm & 3)))
    };
    // The result in a temporary when the destination is also the second
    // source, or is masked: s1 with one block replaced.
    let r = if d == s2 || ev.aaa != 0 {
        em.temp().map_err(|e| refuse(insn, e))?
    } else {
        d
    };
    if r != s1 {
        em.vmov(r, Rm::Reg(s1), 0, 0);
    }
    let k = em.mask_imm(block_mask);
    em.vpermf32x4(r, Rm::Reg(s2), perm, k);
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// A two-input boolean function of (b, c) from its 4-bit truth table
/// (bit index = b<<1 | c), into `t`.
fn bool2(em: &mut Em, t: u8, b: u8, c: u8, table: u8, w: u8) -> Result<(), &'static str> {
    match table & 0xf {
        0x0 => em.zero(t, 0, w),
        0xf => {
            let i = em.const_dw(0xffff_ffff);
            em.vmov(t, Rm::Const(i), 0, 0);
        }
        0xc => em.vmov(t, Rm::Reg(b), 0, w),
        0xa => em.vmov(t, Rm::Reg(c), 0, w),
        0x8 => em.i0f(0xDB, t, b, Rm::Reg(c), 0, w),
        0xe => em.i0f(0xEB, t, b, Rm::Reg(c), 0, w),
        0x6 => em.i0f(0xEF, t, b, Rm::Reg(c), 0, w),
        0x2 => em.i0f(0xDF, t, b, Rm::Reg(c), 0, w), // ~b & c
        0x4 => em.i0f(0xDF, t, c, Rm::Reg(b), 0, w), // ~c & b
        0x9 => {
            // ~(b ^ c)
            em.i0f(0xEF, t, b, Rm::Reg(c), 0, w);
            let i = em.const_dw(0xffff_ffff);
            em.i0f(0xEF, t, t, Rm::Const(i), 0, 0);
        }
        0x1 => {
            // ~(b | c)
            em.i0f(0xEB, t, b, Rm::Reg(c), 0, w);
            let i = em.const_dw(0xffff_ffff);
            em.i0f(0xEF, t, t, Rm::Const(i), 0, 0);
        }
        0x7 => {
            // ~(b & c)
            em.i0f(0xDB, t, b, Rm::Reg(c), 0, w);
            let i = em.const_dw(0xffff_ffff);
            em.i0f(0xEF, t, t, Rm::Const(i), 0, 0);
        }
        0x3 => {
            // ~b
            let i = em.const_dw(0xffff_ffff);
            em.i0f(0xEF, t, b, Rm::Const(i), 0, 0);
        }
        0x5 => {
            let i = em.const_dw(0xffff_ffff);
            em.i0f(0xEF, t, c, Rm::Const(i), 0, 0);
        }
        0xb => {
            // b | ~c
            let i = em.const_dw(0xffff_ffff);
            em.i0f(0xEF, t, c, Rm::Const(i), 0, 0);
            em.i0f(0xEB, t, t, Rm::Reg(b), 0, w);
        }
        0xd => {
            // ~b | c
            let i = em.const_dw(0xffff_ffff);
            em.i0f(0xEF, t, b, Rm::Const(i), 0, 0);
            em.i0f(0xEB, t, t, Rm::Reg(c), 0, w);
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn ternlog(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let a = ev.reg; // also the destination
    let b = ev.vvvv;
    let imm = insn.immediate8();
    let w = ev.w;
    match imm {
        0xff => {
            let i = em.const_dw(0xffff_ffff);
            let r = if ev.aaa != 0 || ev.ll != 2 {
                em.temp().map_err(|e| refuse(insn, e))?
            } else {
                a
            };
            em.vmov(r, Rm::Const(i), 0, 0);
            commit(&mut em, a, r, ev);
        }
        0x00 => {
            let r = if ev.aaa != 0 || ev.ll != 2 {
                em.temp().map_err(|e| refuse(insn, e))?
            } else {
                a
            };
            em.zero(r, 0, w);
            commit(&mut em, a, r, ev);
        }
        _ => {
            let c = src2(&mut em, insn, ev, vl_lanes(ev.ll, ev.w))?;
            let t1 = em.temp().map_err(|e| refuse(insn, e))?;
            let t2 = em.temp().map_err(|e| refuse(insn, e))?;
            if imm == 0x96 {
                em.i0f(0xEF, t1, a, Rm::Reg(b), 0, w);
                em.i0f(0xEF, t1, t1, Rm::Reg(c), 0, w);
            } else {
                // f(a,b,c) = (a & f(1,b,c)) | (~a & f(0,b,c))
                bool2(&mut em, t1, b, c, imm >> 4, w).map_err(|e| refuse(insn, e))?;
                bool2(&mut em, t2, b, c, imm & 0xf, w).map_err(|e| refuse(insn, e))?;
                em.i0f(0xDB, t1, a, Rm::Reg(t1), 0, w);
                em.i0f(0xDF, t2, a, Rm::Reg(t2), 0, w);
                em.i0f(0xEB, t1, t1, Rm::Reg(t2), 0, w);
            }
            commit(&mut em, a, t1, ev);
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

fn unaligned_move(insn: &Instruction, bytes: &[u8], ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let byte_wise = matches!(insn.mnemonic(), M::Vmovdqu8 | M::Vmovdqu16);
    if byte_wise && ev.aaa != 0 {
        return Err(refuse(insn, "a byte or word mask: the card masks 32-bit lanes only"));
    }
    let w = if byte_wise { 0 } else { ev.w };
    let full = ev.ll == 2;
    let load = insn.op0_kind() == OpKind::Register;
    match MemOp::of(insn) {
        None => {
            // Register to register: the aligned move does the same thing.
            if full && !ev.z {
                let mut v = bytes[..insn.len()].to_vec();
                v[1] = (v[1] & 0xfc) | 1;
                v[2] = (v[2] & !0x04 & !0x03) | w;
                v[3] = bytes[3] & 0x0f;
                v[4] = if load { 0x28 } else { 0x29 };
                return Ok(Rewrite::InPlace(v));
            }
            let (d, s) = if load { (ev.reg, ev.rm) } else { (ev.rm, ev.reg) };
            commit(&mut em, d, s, &Ev { w, ..*ev });
        }
        Some(m) => {
            // The pairs move dwords whatever the element size (qword elements
            // would need 8-byte alignment); a qword mask is expanded.
            let dvl = vl_lanes(ev.ll, 0);
            let k = if ev.aaa != 0 && w == 1 {
                em.qmask_to_dmask(ev.aaa, dvl)
            } else if full && ev.aaa == 0 {
                0
            } else if full {
                ev.aaa
            } else {
                let k = em.mask_imm(dvl);
                if ev.aaa != 0 {
                    em.kand(k, ev.aaa);
                }
                k
            };
            if load {
                let d = ev.reg;
                em.load_unaligned(d, m, k, 0, false, 0);
                if ev.z && ev.aaa != 0 {
                    let kz = em.k();
                    em.knot(kz, k);
                    em.zero(d, kz, 0);
                }
                if !full {
                    em.zero_lanes(d, upper(ev.ll));
                }
            } else {
                em.store_unaligned(m, ev.reg, k, 0, false, 0);
            }
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

fn scalar_move(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let w = ev.w;
    let lane0: u16 = 1;
    let rest: u16 = if w == 0 { 0xe } else { 0x2 }; // the other lanes of the low 128 bits
    let above: u16 = 0xfff0; // dword lanes above 128 bits
    if ev.z && ev.aaa != 0 {
        return Err(refuse(insn, "masked scalar move with zeroing: not expressed yet"));
    }
    match (MemOp::of(insn), insn.op0_kind() == OpKind::Register) {
        (Some(m), true) => {
            // Load: lane 0 from memory, everything else zero (or, under a
            // mask that is off, unchanged in lane 0 and zero above).
            let d = ev.reg;
            let k = if ev.aaa != 0 {
                let k = em.mask_imm(lane0);
                em.kand(k, ev.aaa);
                k
            } else {
                0
            };
            em.broadcast(d, Rm::Mem(m), k, w);
            em.zero_lanes(d, 0xfffe << (w as u16));
        }
        (Some(m), false) => {
            let dl: u16 = if w == 1 { 3 } else { 1 };
            let k = if ev.aaa != 0 && w == 1 {
                em.qmask_to_dmask(ev.aaa, dl)
            } else {
                let k = em.mask_imm(dl);
                if ev.aaa != 0 {
                    em.kand(k, ev.aaa);
                }
                k
            };
            em.store_unaligned(m, ev.reg, k, 0, false, 0);
        }
        (None, _) => {
            // d = s1 in the low 128 bits except lane 0 from s2; zero above.
            let (d, s1, s2) = (ev.reg, ev.vvvv, ev.rm);
            let k0 = if ev.aaa != 0 {
                let k = em.mask_imm(lane0);
                em.kand(k, ev.aaa);
                k
            } else {
                em.mask_imm(lane0)
            };
            if d != s2 {
                em.vmov(d, Rm::Reg(s2), k0, w);
            }
            if d != s1 {
                let k = em.mask_imm(rest);
                em.vmov(d, Rm::Reg(s1), k, w);
            }
            em.zero_lanes(d, above);
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// The scalar arithmetic: the 512-bit operation in lane 0, the rest of
/// the low 128 bits from the first source, zero above.
fn scalar_op(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let w = ev.w;
    if ev.z && ev.aaa != 0 {
        return Err(refuse(insn, "masked scalar operation with zeroing: not expressed yet"));
    }
    if ev.b {
        return Err(refuse(insn, "scalar operation with embedded rounding: not expressed yet"));
    }
    let (d, s1) = (ev.reg, ev.vvvv);
    // The packed opcode is the scalar one minus the F3/F2 prefix: same map, same byte.
    let (map, pp, op) = (ev.map, if ev.map == 1 { w } else { 1 }, ev.op);
    let (rm, sss) = match MemOp::of(insn) {
        Some(m) => (Rm::Mem(m), 1u8), // {1to16}: reads the scalar's bytes exactly
        None => (Rm::Reg(ev.rm), 0),
    };
    let k = em.mask_imm(1);
    if ev.aaa != 0 {
        em.kand(k, ev.aaa);
    }
    em.mvex(map, pp, w, op, d, s1, rm, k, sss, None);
    if d != s1 {
        let k = em.mask_imm(if w == 0 { 0xe } else { 0x2 });
        em.vmov(d, Rm::Reg(s1), k, w);
    }
    em.zero_lanes(d, 0xfff0);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vmaxps / vminps with AVX-512 semantics (the second source when either
/// is NaN or both are zero): the second source everywhere, the first where
/// an ordered compare holds.
fn minmax(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let w = ev.w;
    let is_max = matches!(insn.mnemonic(), M::Vmaxps | M::Vmaxpd);
    let (d, s1) = (ev.reg, ev.vvvv);
    let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, w))?;
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let kc = em.k();
    // r = s2; r{s1 <ordered> s2} = s1 (max: s1 > s2, which is s2 < s1)
    em.vmov(r, Rm::Reg(s2), 0, w);
    if is_max {
        em.vcmp(kc, s2, Rm::Reg(s1), 1, 0, w, 0);
    } else {
        em.vcmp(kc, s1, Rm::Reg(s2), 1, 0, w, 0);
    }
    em.vmov(r, Rm::Reg(s1), kc, w);
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

fn scalar_minmax(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let w = ev.w;
    if ev.aaa != 0 {
        return Err(refuse(insn, "masked scalar min/max: not expressed yet"));
    }
    let is_max = matches!(insn.mnemonic(), M::Vmaxss | M::Vmaxsd);
    let (d, s1) = (ev.reg, ev.vvvv);
    let s2 = match MemOp::of(insn) {
        Some(m) => {
            let t = em.temp().map_err(|e| refuse(insn, e))?;
            em.broadcast(t, Rm::Mem(m), 0, w);
            t
        }
        None => ev.rm,
    };
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let kc = em.k();
    em.vmov(r, Rm::Reg(s2), 0, w);
    if is_max {
        em.vcmp(kc, s2, Rm::Reg(s1), 1, 0, w, 0);
    } else {
        em.vcmp(kc, s1, Rm::Reg(s2), 1, 0, w, 0);
    }
    em.vmov(r, Rm::Reg(s1), kc, w);
    // lane 0 from r, the rest of the low 128 bits from s1, zero above
    let k = em.mask_imm(1);
    em.vmov(d, Rm::Reg(r), k, w);
    if d != s1 {
        let k = em.mask_imm(if w == 0 { 0xe } else { 0x2 });
        em.vmov(d, Rm::Reg(s1), k, w);
    }
    em.zero_lanes(d, 0xfff0);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vmovd / vmovq between a vector register and a general register or memory.
fn movdq(insn: &Instruction, _ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let wide = insn.mnemonic() == Mnemonic::Vmovq;
    let w = wide as u8;
    let above: u16 = if wide { 0xfffc } else { 0xfffe };
    let to_vec = insn.op0_kind() == OpKind::Register && vreg(insn.op0_register()).is_some();
    if to_vec {
        let d = vreg(insn.op0_register()).unwrap();
        let src = match insn.op1_kind() {
            OpKind::Memory => Rm::Mem(MemOp::of(insn).unwrap()),
            OpKind::Register => {
                if let Some(s) = vreg(insn.op1_register()) {
                    // vmovq xmm, xmm: the low 64 bits, the rest zero
                    let k = em.mask_imm(1);
                    em.vmov(d, Rm::Reg(s), k, 1);
                    em.zero_lanes(d, 0xfffc);
                    return Ok(Rewrite::Thunk(em.finish()));
                }
                let (r, _) = gpr(insn.op1_register()).ok_or_else(|| refuse(insn, "vmovd source is not a general register"))?;
                em.gpr_scratch(r, wide, S_GPR, true);
                Rm::Fs(S_GPR)
            }
            _ => return Err(refuse(insn, "vmovd operand shape")),
        };
        em.broadcast(d, src, 0, w);
        em.zero_lanes(d, above);
    } else {
        let s = vreg(insn.op1_register()).ok_or_else(|| refuse(insn, "vmovd source is not a vector register"))?;
        let k = em.mask_imm(if wide { 3 } else { 1 });
        match insn.op0_kind() {
            OpKind::Memory => em.store_unaligned(MemOp::of(insn).unwrap(), s, k, 0, false, 0),
            OpKind::Register => {
                let (r, _) = gpr(insn.op0_register()).ok_or_else(|| refuse(insn, "vmovd destination is not a general register"))?;
                // Lane 0 packed into the (line-aligned) scratch slot, then into the register.
                em.mvex(2, 1, 0, 0xD0, s, 0, Rm::Fs(S_GPR), k, 0, None);
                em.gpr_result(r, wide, S_GPR);
            }
            _ => return Err(refuse(insn, "vmovd operand shape")),
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// vpbroadcastd/q, vbroadcastss/sd from a vector or general register.
fn broadcast_reg(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let d = ev.reg;
    let wide = ev.w == 1;
    let r = if ev.aaa != 0 { em.temp().map_err(|e| refuse(insn, e))? } else { d };
    if let Some((g, _)) = gpr(insn.op1_register()) {
        em.gpr_scratch(g, wide, S_GPR, true);
        em.broadcast(r, Rm::Fs(S_GPR), 0, ev.w);
    } else {
        // Block 0 to every block, then lane 0 (or lanes 0,1) to every lane.
        em.vpermf32x4(r, Rm::Reg(ev.rm), 0x00, 0);
        em.vpshufd(r, Rm::Reg(r), if wide { 0x44 } else { 0x00 }, 0);
    }
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// Interleaves and shuffles within 128-bit blocks: two per-block dword
/// shuffles merged under a lane mask.
fn interleave(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let (d, s1) = (ev.reg, ev.vvvv);
    let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
    let imm = insn.immediate8();
    // (shuffle of s1, shuffle of s2, lanes taken from the second) per block
    let (sh1, sh2, from2): (u8, u8, u16) = match insn.mnemonic() {
        M::Vmovlhps => (0x44, 0x44, 0xcccc), // d = s1[0,1], s2[0,1]
        M::Vmovhlps => (0xee, 0xee, 0x3333), // d = s2[2,3], s1[2,3]
        M::Vunpcklps | M::Vpunpckldq => (0x50, 0x50, 0xaaaa),
        M::Vunpckhps | M::Vpunpckhdq => (0xfa, 0xfa, 0xaaaa),
        M::Vunpcklpd | M::Vpunpcklqdq => (0x44, 0x44, 0xcccc),
        M::Vunpckhpd | M::Vpunpckhqdq => (0xee, 0xee, 0xcccc),
        M::Vshufps => (imm, imm, 0xcccc),
        M::Vshufpd => {
            // Per block, bit 2i picks the qword of s1 for lane 0 and bit 2i+1
            // the qword of s2 for lane 1; vpshufd is the same for every block,
            // so only immediates with the same two bits per block are expressed.
            let n = if ev.ll == 2 {
                4
            } else if ev.ll == 1 {
                2
            } else {
                1
            };
            let p = imm & 3;
            if (0..n).any(|i| (imm >> (2 * i)) & 3 != p) {
                return Err(refuse(insn, "vshufpd with a different selection per block: not expressed yet"));
            }
            let a = if p & 1 == 1 { 0xee } else { 0x44 }; // lane 0 <- qword (p&1) of s1
            let b = if p & 2 == 2 { 0xee } else { 0x44 };
            (a, b, 0xcccc)
        }
        _ => unreachable!(),
    };
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let t = em.temp().map_err(|e| refuse(insn, e))?;
    em.vpshufd(r, Rm::Reg(s1), sh1, 0);
    em.vpshufd(t, Rm::Reg(s2), sh2, 0);
    let k = em.mask_imm(from2);
    em.vmov(r, Rm::Reg(t), k, 0);
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// valignq: valignd by twice the count at full width; a narrow form
/// concatenates the two sources' low lanes first.
fn valignq(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let (d, s1) = (ev.reg, ev.vvvv);
    let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 1))?;
    let n = (insn.immediate8() & 7) * 2;
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    if ev.ll == 2 {
        em.valignd(r, s1, Rm::Reg(s2), n, 0);
    } else {
        // t = [s2 low half | s1 low half] at the vector length, then align within it.
        let t = em.temp().map_err(|e| refuse(insn, e))?;
        em.vmov(t, Rm::Reg(s2), 0, 0);
        let (perm, hi): (u8, u16) = if ev.ll == 1 { (0x4e, 0xff00) } else { (0x00, 0x00f0) };
        let k = em.mask_imm(hi);
        em.vpermf32x4(t, Rm::Reg(s1), perm, k);
        em.valignd(r, t, Rm::Reg(t), n, 0);
    }
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vpalignr, vpsrldq, vpslldq with a shift that is a whole number of
/// dwords: per-block rotations merged (or zero-filled) under a mask.
fn byte_shift(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let imm = insn.immediate8();
    if imm % 4 != 0 {
        return Err(refuse(
            insn,
            "a byte shift within 128-bit blocks that is not a whole number of dwords: not expressed yet",
        ));
    }
    let n = imm / 4;
    let rot = |by: u8| -> u8 {
        // lane i takes lane (i + by) mod 4
        let mut v = 0u8;
        for i in 0..4u8 {
            v |= ((i + by) & 3) << (2 * i);
        }
        v
    };
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    match insn.mnemonic() {
        M::Vpsrldq => {
            let (d, s) = (if dest_is_vvvv(ev) { ev.vvvv } else { ev.reg }, ev.rm);
            if n >= 4 {
                em.zero(r, 0, 0);
            } else {
                em.vpshufd(r, Rm::Reg(s), rot(n), 0);
                let vacated = (0xf << (4 - n) & 0xf) as u16;
                em.zero_lanes(r, vacated * 0x1111);
            }
            commit(&mut em, d, r, ev);
        }
        M::Vpslldq => {
            let (d, s) = (if dest_is_vvvv(ev) { ev.vvvv } else { ev.reg }, ev.rm);
            if n >= 4 {
                em.zero(r, 0, 0);
            } else {
                em.vpshufd(r, Rm::Reg(s), rot(4 - n), 0);
                let vacated = ((1u16 << n) - 1) * 0x1111;
                em.zero_lanes(r, vacated);
            }
            commit(&mut em, d, r, ev);
        }
        _ => {
            // vpalignr d, a, b, n*4: per block, concat(a, b) >> n dwords
            let (d, a) = (ev.reg, ev.vvvv);
            let b = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
            if n >= 8 {
                em.zero(r, 0, 0);
            } else if n >= 4 {
                em.vpshufd(r, Rm::Reg(a), rot(n - 4), 0);
                let vacated = (0xf << (8 - n) & 0xf) as u16;
                em.zero_lanes(r, vacated * 0x1111);
            } else {
                let t = em.temp().map_err(|e| refuse(insn, e))?;
                em.vpshufd(r, Rm::Reg(b), rot(n), 0);
                em.vpshufd(t, Rm::Reg(a), rot(n), 0);
                let from_a = ((0xf << (4 - n)) & 0xf) as u16 * 0x1111;
                let k = em.mask_imm(from_a);
                em.vmov(r, Rm::Reg(t), k, 0);
            }
            commit(&mut em, d, r, ev);
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// vcvttps2dq / vcvtps2dq: the card's fixed-point conversion (truncate or
/// MXCSR rounding), then NaN and overflow made the integer indefinite
/// 0x80000000, which the card does not produce for NaN or +overflow.
fn cvt_to_int(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let d = ev.reg;
    let truncate = insn.mnemonic() == Mnemonic::Vcvttps2dq;
    let s = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    em.mvex(3, 1, 0, 0xCB, r, 0, Rm::Reg(s), 0, 0, Some(if truncate { 3 } else { 0 }));
    // Lanes not less than 2^31 (unordered included) become 0x80000000.
    let big = em.const_dw(0x4f00_0000); // 2147483648.0f
    let kc = em.k();
    em.vcmp(kc, s, Rm::Const(big), 5, 0, 0, 0);
    let ind = em.const_dw(0x8000_0000);
    em.vmov(r, Rm::Const(ind), kc, 0);
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vcvtph2ps: the card's float16 up-conversion from memory (the unpack
/// pair, so any alignment) or from a register through the scratch area.
fn cvt_ph2ps(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let d = ev.reg;
    let r = if ev.aaa != 0 || ev.ll != 2 {
        em.temp().map_err(|e| refuse(insn, e))?
    } else {
        d
    };
    let lanes = vl_lanes(ev.ll, 0);
    match MemOp::of(insn) {
        Some(m) => {
            let k = if ev.ll == 2 { 0 } else { em.mask_imm(lanes) };
            em.load_unaligned(r, m, k, 0, true, 3);
        }
        None => {
            // The source's halves (16 per 256 bits) to the scratch slot, then converted in.
            let s = ev.rm;
            let n_dwords: u16 = if ev.ll == 2 {
                0xff
            } else if ev.ll == 1 {
                0xf
            } else {
                0x3
            };
            let k = em.mask_imm(n_dwords);
            em.mvex(2, 1, 0, 0xD0, s, 0, Rm::Fs(S_XFER), k, 0, None);
            let k = if ev.ll == 2 { 0 } else { em.mask_imm(lanes) };
            em.mvex(1, 0, 0, 0x28, r, 0, Rm::Fs(S_XFER), k, 3, None);
        }
    }
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vcvtps2ph: the card's float16 store down-conversion (MXCSR rounding,
/// which the immediate's 0 and 4 both mean here).
fn cvt_ps2ph(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let imm = insn.immediate8();
    if imm & 3 != 0 && imm & 4 == 0 {
        return Err(refuse(insn, "vcvtps2ph with a rounding other than nearest: not expressed yet"));
    }
    let s = ev.reg;
    let src_ll = ev.ll;
    let lanes = vl_lanes(src_ll, 0);
    let k = if src_ll == 2 { 0 } else { em.mask_imm(lanes) };
    match MemOp::of(insn) {
        Some(m) => {
            let k = if ev.aaa != 0 {
                let k = em.mask_imm(lanes);
                em.kand(k, ev.aaa);
                k
            } else {
                k
            };
            em.store_unaligned(m, s, k, 0, true, 3);
        }
        None => {
            let d = ev.rm;
            em.mvex(1, 0, 0, 0x29, s, 0, Rm::Fs(S_XFER), k, 3, None);
            let r = if ev.aaa != 0 { em.temp().map_err(|e| refuse(insn, e))? } else { d };
            // The halves back as dwords: 8 for a 512-bit source, 4, 2.
            let n_dwords: u16 = if src_ll == 2 {
                0xff
            } else if src_ll == 1 {
                0xf
            } else {
                0x3
            };
            let k = em.mask_imm(n_dwords);
            em.vmov(r, Rm::Fs(S_XFER), k, 0);
            let out_ll = if src_ll == 2 { 1 } else { 0 };
            commit(&mut em, d, r, &Ev { ll: out_ll, w: 0, ..*ev });
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// vpmovzxbd / sxbd / zxwd / sxwd: the card's memory up-conversions,
/// from memory through the unpack pair (element alignment only) or from
/// a register through the scratch area.
fn widen(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let d = ev.reg;
    let (conv, elem_bytes): (u8, u16) = match insn.mnemonic() {
        M::Vpmovzxbd => (4, 1),
        M::Vpmovsxbd => (5, 1),
        M::Vpmovzxwd => (6, 2),
        _ => (7, 2),
    };
    let lanes = vl_lanes(ev.ll, 0);
    let r = if ev.aaa != 0 || ev.ll != 2 {
        em.temp().map_err(|e| refuse(insn, e))?
    } else {
        d
    };
    match MemOp::of(insn) {
        Some(m) => {
            let k = if ev.ll == 2 { 0 } else { em.mask_imm(lanes) };
            em.load_unaligned(r, m, k, 0, false, conv);
        }
        None => {
            // The source's bytes (16 lanes x elem_bytes) to the scratch slot.
            let n_dwords = (lanes.count_ones() as u16 * elem_bytes).div_ceil(4);
            let k = em.mask_imm((1u16 << n_dwords) - 1);
            em.mvex(2, 1, 0, 0xD0, ev.rm, 0, Rm::Fs(S_XFER), k, 0, None);
            let k = if ev.ll == 2 { 0 } else { em.mask_imm(lanes) };
            em.mvex(1, 1, 0, 0x6F, r, 0, Rm::Fs(S_XFER), k, conv, None);
        }
    }
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vpsrlq / vpsllq / vpsraq by immediate on dword lanes: the two halves
/// of each qword shifted and combined.
fn shift_q(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let d = ev.vvvv;
    let s = src2(&mut em, insn, &Ev { vvvv: 0, ..*ev }, vl_lanes(ev.ll, 1))?;
    let n = insn.immediate8();
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let t = em.temp().map_err(|e| refuse(insn, e))?;
    let (even, odd): (u16, u16) = (0x5555, 0xaaaa);
    match insn.mnemonic() {
        M::Vpsrlq | M::Vpsraq => {
            let arith = insn.mnemonic() == M::Vpsraq;
            if n >= 64 {
                if arith {
                    em.vpshufd(r, Rm::Reg(s), 0xf5, 0);
                    em.shift_imm(4, r, Rm::Reg(r), 31, 0);
                } else {
                    em.zero(r, 0, 0);
                }
            } else if n >= 32 {
                // low <- high >> (n-32); high <- 0 or sign
                em.vpshufd(r, Rm::Reg(s), 0xf5, 0); // high dword in both lanes
                let k = em.mask_imm(even);
                em.shift_imm(if arith { 4 } else { 2 }, r, Rm::Reg(r), n - 32, k);
                let k = em.mask_imm(odd);
                if arith {
                    em.shift_imm(4, r, Rm::Reg(r), 31, k);
                } else {
                    em.zero(r, k, 0);
                }
            } else if n == 0 {
                em.vmov(r, Rm::Reg(s), 0, 0);
            } else {
                // even lanes: (lo >> n) | (hi << (32-n)); odd lanes: hi >> n (logical or arithmetic)
                em.shift_imm(2, r, Rm::Reg(s), n, 0);
                if arith {
                    let k = em.mask_imm(odd);
                    em.shift_imm(4, r, Rm::Reg(s), n, k);
                }
                em.vpshufd(t, Rm::Reg(s), 0xf5, 0);
                em.shift_imm(6, t, Rm::Reg(t), 32 - n, 0);
                let k = em.mask_imm(even);
                em.i0f(0xEB, r, r, Rm::Reg(t), k, 0);
            }
        }
        _ => {
            if n >= 64 {
                em.zero(r, 0, 0);
            } else if n >= 32 {
                em.vpshufd(r, Rm::Reg(s), 0xa0, 0); // low dword in both lanes
                let k = em.mask_imm(odd);
                em.shift_imm(6, r, Rm::Reg(r), n - 32, k);
                let k = em.mask_imm(even);
                em.zero(r, k, 0);
            } else if n == 0 {
                em.vmov(r, Rm::Reg(s), 0, 0);
            } else {
                // odd lanes: (hi << n) | (lo >> (32-n)); even lanes: lo << n
                em.shift_imm(6, r, Rm::Reg(s), n, 0);
                em.vpshufd(t, Rm::Reg(s), 0xa0, 0);
                em.shift_imm(2, t, Rm::Reg(t), 32 - n, 0);
                let k = em.mask_imm(odd);
                em.i0f(0xEB, r, r, Rm::Reg(t), k, 0);
            }
        }
    }
    commit(&mut em, d, r, &Ev { w: 1, ..*ev });
    Ok(Rewrite::Thunk(em.finish()))
}

/// vpabsd: (s ^ (s >> 31)) - (s >> 31).
fn abs_d(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let d = ev.reg;
    let s = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let t = em.temp().map_err(|e| refuse(insn, e))?;
    em.shift_imm(4, t, Rm::Reg(s), 31, 0);
    em.i0f(0xEF, r, s, Rm::Reg(t), 0, 0);
    em.i0f(0xFA, r, r, Rm::Reg(t), 0, 0);
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vpmuludq: the 64-bit product of the even dword lanes, as low and high
/// 32-bit products (vpmulld, vpmulhud) interleaved.
fn muludq(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let (d, s1) = (ev.reg, ev.vvvv);
    let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 1))?;
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let t = em.temp().map_err(|e| refuse(insn, e))?;
    em.i38(0x40, r, s1, Rm::Reg(s2), 0);
    em.i38(0x86, t, s1, Rm::Reg(s2), 0);
    em.vpshufd(t, Rm::Reg(t), 0xa0, 0);
    let k = em.mask_imm(0xaaaa);
    em.vmov(r, Rm::Reg(t), k, 0);
    commit(&mut em, d, r, &Ev { w: 1, ..*ev });
    Ok(Rewrite::Thunk(em.finish()))
}

/// vpslld / vpsrld / vpsrad by the count in an xmm register or memory:
/// the count broadcast, then the card's variable shift (counts of 32 or
/// more give zero, or the sign, as AVX-512 does).
fn shift_by_xmm(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let (d, s1) = (ev.reg, ev.vvvv);
    let t = em.temp().map_err(|e| refuse(insn, e))?;
    match MemOp::of(insn) {
        Some(m) => em.broadcast(t, Rm::Mem(m), 0, 0),
        None => {
            em.vpermf32x4(t, Rm::Reg(ev.rm), 0x00, 0);
            em.vpshufd(t, Rm::Reg(t), 0x00, 0);
        }
    }
    // The count is 64 bits; a high dword other than zero means a count >= 32.
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let op = match insn.mnemonic() {
        M::Vpslld => 0x47,
        M::Vpsrld => 0x45,
        _ => 0x46,
    };
    em.i38(op, r, s1, Rm::Reg(t), 0);
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

/// vcmpps / vcmppd with a predicate above 7, and vpcmpd / vpcmpud with
/// FALSE or TRUE: the card's eight predicates, operands swapped or two
/// compares joined, where AVX-512 has thirty-two.
fn compare(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let kd = ev.reg & 7;
    let (s1, w) = (ev.vvvv, ev.w);
    let vl = vl_lanes(ev.ll, w);
    let full = ev.ll == 2;
    let keff = if full {
        ev.aaa
    } else {
        let k = em.mask_imm(vl);
        if ev.aaa != 0 {
            em.kand(k, ev.aaa);
        }
        k
    };
    let pred = insn.immediate8();
    if ev.map == 3 {
        // vpcmpd FALSE (3) or TRUE (7)
        if pred & 7 == 3 {
            em.mov_eax(0);
            em.kmov_k_r(kd, 0);
        } else if full && ev.aaa == 0 {
            em.mov_eax(0xffff);
            em.kmov_k_r(kd, 0);
        } else {
            em.kmov_kk(kd, keff);
        }
        return Ok(Rewrite::Thunk(em.finish()));
    }
    let s2 = src2(&mut em, insn, ev, vl)?;
    // (card predicate, swap operands, then OR with a second compare)
    let plan: (u8, bool, Option<(u8, bool)>) = match pred {
        8 | 24 => (0, false, Some((3, false))), // EQ_UQ, EQ_US: eq | unord
        9 | 25 => (6, true, None),              // NGE: not(s1 >= s2) = nle(s2, s1)
        10 | 26 => (5, true, None),             // NGT: nlt(s2, s1)
        11 | 27 => {
            em.mov_eax(0);
            em.kmov_k_r(kd, 0);
            return Ok(Rewrite::Thunk(em.finish()));
        }
        12 | 28 => (1, false, Some((1, true))), // NEQ_O: lt | gt
        13 | 29 => (2, true, None),             // GE_O: le(s2, s1)
        14 | 30 => (1, true, None),             // GT_O: lt(s2, s1)
        15 | 31 => {
            if full && ev.aaa == 0 {
                em.mov_eax(0xffff);
                em.kmov_k_r(kd, 0);
            } else {
                em.kmov_kk(kd, keff);
            }
            return Ok(Rewrite::Thunk(em.finish()));
        }
        16 => (0, false, None),
        17 => (1, false, None),
        18 => (2, false, None),
        19 => (3, false, None),
        20 => (4, false, None),
        21 => (5, false, None),
        22 => (6, false, None),
        23 => (7, false, None),
        _ => unreachable!(),
    };
    let (p, swap, second) = plan;
    if swap {
        em.vcmp(kd, s2, Rm::Reg(s1), p, keff, w, 0);
    } else {
        em.vcmp(kd, s1, Rm::Reg(s2), p, keff, w, 0);
    }
    if let Some((p2, swap2)) = second {
        // The second compare into a temporary mask, joined by kor.
        // The scratch mask holds keff when narrow or masked; a second scratch is needed.
        let k2 = (1..8).rev().find(|k| *k != kd && *k != em.kscr && *k != ev.aaa).unwrap_or(1);
        em.rcx();
        em.kmov_r_k(1, k2);
        if swap2 {
            em.vcmp(k2, s2, Rm::Reg(s1), p2, keff, w, 0);
        } else {
            em.vcmp(k2, s1, Rm::Reg(s2), p2, keff, w, 0);
        }
        em.kor(kd, k2);
        em.kmov_k_r(k2, 1);
    }
    if !full || ev.aaa != 0 {
        em.kand(kd, keff);
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// The mask instructions AVX-512 encodes with VEX (kmovw, kandw, korw,
/// kxorw, kxnorw, kandnw, knotw, kortestw, kunpckbw, kshiftlw, kshiftrw
/// and the d/q widths for 16-bit values): the card's two-operand forms
/// (VEX.128.0F.W0 41 42 44 45 46 47 90 92 93 98 /r), through general
/// registers where it has none.
fn kops(insn: &Instruction, bytes: &[u8], tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let k = |i: u32| kreg(insn.op_register(i));
    let l1 = bytes.len() >= 3 && bytes[0] == 0xc5 && (bytes[1] >> 2) & 1 == 1
        || bytes.len() >= 4 && bytes[0] == 0xc4 && (bytes[2] >> 2) & 1 == 1;
    let three = |em: &mut Em, op: u8| -> Result<(), Unsupported> {
        // k1 = k2 op k3 on the card's k1 = k1 op k2
        let (k1, k2, k3) = (k(0).unwrap(), k(1).unwrap(), k(2).unwrap());
        if k1 == k2 {
            em.vexk(op, k1, k3);
        } else if k1 == k3 && matches!(op, 0x41 | 0x45 | 0x47 | 0x46) {
            em.vexk(op, k1, k2); // commutative
        } else if k1 != k3 {
            em.kmov_kk(k1, k2);
            em.vexk(op, k1, k3);
        } else {
            // k1 = k2 op k1 with a non-commutative op: through the scratch mask
            let ks = em.k();
            em.kmov_kk(ks, k2);
            em.vexk(op, ks, k3);
            em.kmov_kk(k1, ks);
        }
        Ok(())
    };
    // The forms whose bytes the card reads the same: kmovw between mask
    // and general registers (VEX.L0.0F.W0 90 92 93), knotw (44), kortestw
    // (98), all L0. They stay in place (they are 4 bytes, too short for a jump).
    if !l1
        && matches!(insn.mnemonic(), M::Kmovw | M::Knotw | M::Kortestw)
        && insn.op0_kind() == OpKind::Register
        && insn.op1_kind() == OpKind::Register
    {
        return Ok(Rewrite::InPlace(bytes[..insn.len()].to_vec()));
    }
    match insn.mnemonic() {
        M::Kmovw | M::Kmovd | M::Kmovq | M::Kmovb => {
            match (insn.op0_kind(), insn.op1_kind()) {
                (OpKind::Register, OpKind::Register) => match (k(0), k(1)) {
                    (Some(a), Some(b)) => em.kmov_kk(a, b),
                    (Some(a), None) => {
                        let (r, _) = gpr(insn.op1_register()).ok_or_else(|| refuse(insn, "kmov source"))?;
                        em.kmov_k_r(a, r);
                    }
                    (None, Some(b)) => {
                        let (r, _) = gpr(insn.op0_register()).ok_or_else(|| refuse(insn, "kmov destination"))?;
                        em.kmov_r_k(r, b);
                    }
                    _ => return Err(refuse(insn, "kmov between general registers")),
                },
                (OpKind::Register, OpKind::Memory) => {
                    // k <- m16: through eax
                    let a = k(0).unwrap();
                    let m = MemOp::of(insn).unwrap();
                    em.rax();
                    // movzx eax, word [m]: 0F B7 /r
                    em.t.seq.extend_from_slice(&[0x0f, 0xb7]);
                    em.mem_modrm(0, m);
                    em.kmov_k_r(a, 0);
                }
                (OpKind::Memory, OpKind::Register) => {
                    let b = k(1).unwrap();
                    let m = MemOp::of(insn).unwrap();
                    em.rax();
                    em.kmov_r_k(0, b);
                    // mov word [m], ax: 66 89 /r
                    em.t.seq.extend_from_slice(&[0x66, 0x89]);
                    em.mem_modrm(0, m);
                }
                _ => return Err(refuse(insn, "kmov operand shape")),
            }
        }
        M::Kandw | M::Kandd | M::Kandq | M::Kandb => three(&mut em, 0x41)?,
        M::Kandnw | M::Kandnd | M::Kandnq | M::Kandnb => three(&mut em, 0x42)?,
        M::Korw | M::Kord | M::Korq | M::Korb => three(&mut em, 0x45)?,
        M::Kxnorw | M::Kxnord | M::Kxnorq | M::Kxnorb => three(&mut em, 0x46)?,
        M::Kxorw | M::Kxord | M::Kxorq | M::Kxorb => three(&mut em, 0x47)?,
        M::Knotw | M::Knotd | M::Knotq | M::Knotb => em.knot(k(0).unwrap(), k(1).unwrap()),
        M::Kortestw | M::Kortestd | M::Kortestq | M::Kortestb => {
            if !l1 && bytes[0] == 0xc5 && bytes[1] == 0xf8 {
                return Ok(Rewrite::InPlace(bytes[..insn.len()].to_vec()));
            }
            em.vexk(0x98, k(0).unwrap(), k(1).unwrap());
        }
        M::Kunpckbw => {
            // k1 = (k2[7:0] << 8) | k3[7:0]
            let (k1, k2, k3) = (k(0).unwrap(), k(1).unwrap(), k(2).unwrap());
            em.flags();
            em.rcx();
            em.kmov_r_k(0, k3);
            em.kmov_r_k(1, k2);
            em.t.seq.extend_from_slice(&[0x0f, 0xb6, 0xc0]); // movzx eax, al
            em.t.seq.extend_from_slice(&[0xc1, 0xe1, 0x08]); // shl ecx, 8
            em.t.seq.extend_from_slice(&[0x0f, 0xb6, 0xc9]); // movzx ecx, cl  (garbage above bit 7 of k2 dropped first)
            em.t.seq.extend_from_slice(&[0xc1, 0xe1, 0x08]);
            em.t.seq.extend_from_slice(&[0x09, 0xc8]); // or eax, ecx
            em.kmov_k_r(k1, 0);
        }
        M::Kshiftlw | M::Kshiftrw | M::Kshiftlb | M::Kshiftrb | M::Kshiftld | M::Kshiftrd | M::Kshiftlq | M::Kshiftrq => {
            let (k1, k2) = (k(0).unwrap(), k(1).unwrap());
            let n = insn.immediate8();
            let left = matches!(insn.mnemonic(), M::Kshiftlw | M::Kshiftlb | M::Kshiftld | M::Kshiftlq);
            em.flags();
            em.kmov_r_k(0, k2);
            if n >= 16 {
                em.t.seq.extend_from_slice(&[0x31, 0xc0]); // xor eax, eax
            } else {
                em.t.seq.extend_from_slice(&[0xc1, if left { 0xe0 } else { 0xe8 }, n]); // shl/shr eax, n
                if left {
                    em.t.seq.extend_from_slice(&[0x0f, 0xb7, 0xc0]); // movzx eax, ax
                }
            }
            em.kmov_k_r(k1, 0);
        }
        _ => return Err(refuse(insn, "a VEX instruction the card does not have")),
    }
    Ok(Rewrite::Thunk(em.finish()))
}

impl Em<'_> {
    /// ModRM, SIB and displacement of a general-register memory operand
    /// for a legacy instruction, with `reg` in ModRM.reg (r8-r15 bases
    /// and indexes are not expressed: refused earlier as a REX prefix
    /// would be needed before the opcode).
    fn mem_modrm(&mut self, reg: u8, m: MemOp) {
        let r3 = (reg & 7) << 3;
        if let Some(target) = m.rip {
            self.t.seq.push(0x05 | r3);
            let disp_at = self.t.seq.len();
            self.t.seq.extend_from_slice(&[0; 4]);
            self.t.fixups.push(Fixup {
                disp_at,
                insn_end: self.t.seq.len(),
                what: FixupKind::Rip(target),
            });
            return;
        }
        let disp = m.disp as i32;
        match (m.base, m.index) {
            (Some(b), None) if b & 7 != 4 => self.t.seq.push(0x80 | r3 | (b & 7)),
            (Some(_), None) => self.t.seq.extend_from_slice(&[0x84 | r3, 0x24]),
            (Some(b), Some(i)) => self
                .t
                .seq
                .extend_from_slice(&[0x84 | r3, (m.scale << 6) | ((i & 7) << 3) | (b & 7)]),
            (None, Some(i)) => self.t.seq.extend_from_slice(&[0x04 | r3, (m.scale << 6) | ((i & 7) << 3) | 5]),
            (None, None) => self.t.seq.extend_from_slice(&[0x04 | r3, 0x25]),
        }
        self.t.seq.extend_from_slice(&disp.to_le_bytes());
    }
}

/// Lay a thunk out at `at` for a site at `site` of `len` bytes whose next
/// instruction is `next`: the sequence, `jmp next`, then the constants on
/// 64-byte boundaries, with every fixup patched. Returns the thunk bytes
/// and the site's replacement (`jmp at` padded with NOPs to `len`).
pub fn thunk_bytes(thunk: &Thunk, at: u64, site: u64, len: usize, next: u64) -> (Vec<u8>, Vec<u8>) {
    let mut t = thunk.seq.clone();
    let back = next as i64 - (at as i64 + t.len() as i64 + 5);
    t.push(0xe9);
    t.extend_from_slice(&(back as i32).to_le_bytes());
    let mut const_at = Vec::with_capacity(thunk.consts.len());
    for c in &thunk.consts {
        // The card faults on a misaligned vector memory operand: every
        // constant goes on a 64-byte boundary.
        while (at + t.len() as u64) % 64 != 0 {
            t.push(0xcc);
        }
        const_at.push(at + t.len() as u64);
        t.extend_from_slice(c);
    }
    for f in &thunk.fixups {
        let target = match f.what {
            FixupKind::Const(i) => const_at[i],
            FixupKind::Rip(a) => a,
        };
        let rel = (target as i64 - (at as i64 + f.insn_end as i64)) as i32;
        t[f.disp_at..f.disp_at + 4].copy_from_slice(&rel.to_le_bytes());
    }
    let mut s = vec![0xe9];
    s.extend_from_slice(&((at as i64 - (site as i64 + 5)) as i32).to_le_bytes());
    while s.len() < len {
        s.push(0x90);
    }
    (t, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions};

    const TG: Target = Target { scratch: -0x1000 };

    fn dec(bytes: &[u8]) -> Instruction {
        Decoder::with_ip(64, bytes, 0x1000, DecoderOptions::NONE).decode()
    }

    fn thunk(bytes: &[u8]) -> Thunk {
        match rewrite(&dec(bytes), bytes, &TG).unwrap() {
            Rewrite::Thunk(t) => t,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn indexed_fmadd_with_broadcast_keeps_its_sib_and_gets_the_swizzle() {
        // vfmadd213ps (%r9,%rax,4){1to16},%zmm1,%zmm0 as gcc emitted it
        let b = [0x62, 0xd2, 0x75, 0x58, 0xa8, 0x04, 0x81];
        match rewrite(&dec(&b), &b, &TG).unwrap() {
            Rewrite::InPlace(v) => {
                assert_eq!(v.len(), b.len());
                assert_eq!(&v[4..], &b[4..], "opcode, ModRM and SIB untouched");
                assert_eq!(v[2], 0x71, "P1 bit 2 cleared");
                assert_eq!(v[3], 0x18, "SSS = 001 for {{1to16}}, V' and aaa kept");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn aligned_load_with_sib_is_one_to_one() {
        // vmovaps (%r14,%rdx,4),%zmm1
        let b = [0x62, 0xc1, 0x7c, 0x48, 0x28, 0x0c, 0x96];
        match rewrite(&dec(&b), &b, &TG).unwrap() {
            Rewrite::InPlace(v) => assert_eq!(v, vec![0x62, 0xc1, 0x78, 0x08, 0x28, 0x0c, 0x96]),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn masked_mullo_keeps_its_mask() {
        // vpmulld %zmm3,%zmm1,%zmm0{%k1}
        let b = [0x62, 0xf2, 0x75, 0x49, 0x40, 0xc3];
        match rewrite(&dec(&b), &b, &TG).unwrap() {
            Rewrite::InPlace(v) => assert_eq!(v, vec![0x62, 0xf2, 0x71, 0x09, 0x40, 0xc3]),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn extract_high_half_is_a_permute_and_a_masked_zero() {
        // vextractf64x4 $0x1,%zmm0,%ymm1
        let t = thunk(&[0x62, 0xf3, 0xfd, 0x48, 0x1b, 0xc1, 0x01]);
        let pos = t.seq.windows(7).position(|w| w == [0x62, 0xf3, 0x79, 0x08, 0x07, 0xc8, 0x4e]);
        assert!(pos.is_some(), "vpermf32x4 zmm1, zmm0, 0x4e somewhere in {:02x?}", t.seq);
        assert!(t.seq.len() > 20, "then the masked zero of the upper lanes");
    }

    #[test]
    fn a_ymm_add_runs_under_a_lane_mask_and_clears_the_top() {
        // vaddps %ymm2,%ymm1,%ymm0
        let t = thunk(&[0x62, 0xf1, 0x74, 0x28, 0x58, 0xc2]);
        // mov eax, 0xff (the 8 lanes) then kmov k7, eax
        let pos = t
            .seq
            .windows(9)
            .position(|w| w == [0xb8, 0xff, 0x00, 0x00, 0x00, 0xc5, 0xf8, 0x92, 0xf8])
            .expect("lane mask into k7");
        // the add under k7: MVEX vaddps zmm0{k7}, zmm1, zmm2
        assert_eq!(&t.seq[pos + 9..pos + 15], &[0x62, 0xf1, 0x70, 0x0f, 0x58, 0xc2]);
        // then lanes 8..15 zeroed: mov eax, 0xff00; kmov; vpxord zmm0{k7}, zmm0, zmm0
        let z = t.seq.windows(6).position(|w| w == [0x62, 0xf1, 0x79, 0x0f, 0xef, 0xc0]);
        assert!(z.is_some(), "vpxord under k7 in {:02x?}", t.seq);
        // and rax and k7 restored at the end
        let n = t.seq.len();
        assert_eq!(&t.seq[n - 9..n - 4], &[0x64, 0x48, 0x8b, 0x04, 0x25], "mov rax, [fs:...] last");
    }

    #[test]
    fn a_narrow_insert_places_the_block_under_a_mask() {
        // vinserti64x2 $0x1,%xmm0,%ymm1,%ymm1 (the instruction llama.cpp died on)
        let b = [0x62, 0xf3, 0xf5, 0x28, 0x38, 0xc8, 0x01];
        let t = thunk(&b);
        // block 1 mask 0xf0 into k7, then vpermf32x4 zmm1{k7}, zmm0, 0x00
        assert!(t
            .seq
            .windows(9)
            .any(|w| w == [0xb8, 0xf0, 0x00, 0x00, 0x00, 0xc5, 0xf8, 0x92, 0xf8]));
        assert!(
            t.seq.windows(7).any(|w| w == [0x62, 0xf3, 0x79, 0x0f, 0x07, 0xc8, 0x00]),
            "{:02x?}",
            t.seq
        );
    }

    #[test]
    fn zeroing_full_width_add_zeroes_the_unselected_lanes() {
        // vaddps %zmm2,%zmm1,%zmm0{%k1}{z}
        let t = thunk(&[0x62, 0xf1, 0x74, 0xc9, 0x58, 0xc2]);
        // the add under k1, then knot k7, k1 and vpxord zmm0{k7}
        assert!(t.seq.windows(6).any(|w| w == [0x62, 0xf1, 0x70, 0x09, 0x58, 0xc2]));
        assert!(
            t.seq.windows(4).any(|w| w == [0xc5, 0xf8, 0x44, 0xf9]),
            "knot k7, k1 in {:02x?}",
            t.seq
        );
    }

    #[test]
    fn a_memory_operand_of_arithmetic_is_staged_through_the_unpack_pair() {
        // vaddps (%rax),%zmm1,%zmm0: the card needs 64-byte alignment, the program promises none
        let t = thunk(&[0x62, 0xf1, 0x74, 0x48, 0x58, 0x00]);
        // vloadunpackld zmmT, [rax+0] (MVEX.512.0F38.W0 D0) then hd at +64
        let ld = t.seq.windows(5).position(|w| w[0] == 0x62 && w[4] == 0xd0).expect("unpack low");
        let hd = t.seq.windows(5).position(|w| w[0] == 0x62 && w[4] == 0xd4).expect("unpack high");
        assert!(hd > ld);
        assert_eq!(t.seq[hd + 5] & 0xc7, 0x80, "mod=10 [rax+disp32]");
        assert_eq!(&t.seq[hd + 6..hd + 10], &64i32.to_le_bytes());
    }

    #[test]
    fn vmovd_from_a_general_register_goes_through_the_scratch_slot() {
        // vmovd %eax,%xmm16 (EVEX: the high register forces it)
        let t = thunk(&[0x62, 0xe1, 0x7d, 0x08, 0x6e, 0xc0]);
        // mov [fs:scratch+280], eax  = 64 89 04 25 disp32
        let disp = (TG.scratch + S_GPR).to_le_bytes();
        let pos = t.seq.windows(8).position(|w| w[..4] == [0x64, 0x89, 0x04, 0x25] && w[4..] == disp);
        assert!(pos.is_some(), "{:02x?}", t.seq);
    }

    #[test]
    fn kandw_three_operands_become_a_move_and_the_card_form() {
        // kandw %k3,%k2,%k1: VEX.L1.0F.W0 41 /r
        let b = [0xc5, 0xec, 0x41, 0xcb];
        let t = thunk(&b);
        assert_eq!(
            &t.seq[..8],
            &[0xc5, 0xf8, 0x90, 0xca, 0xc5, 0xf8, 0x41, 0xcb],
            "kmov k1,k2; kand k1,k3"
        );
    }

    #[test]
    fn thunk_layout_patches_constants_and_rip_relative_operands() {
        let mut th = Thunk::default();
        // vmovaps zmm0, [rip+const0]: 62 f1 7c 48 28 05 disp32
        th.seq.extend_from_slice(&[0x62, 0xf1, 0x78, 0x08, 0x28, 0x05, 0, 0, 0, 0]);
        th.fixups.push(Fixup {
            disp_at: 6,
            insn_end: 10,
            what: FixupKind::Const(0),
        });
        th.consts.push([0xaa; 64]);
        th.seq.extend_from_slice(&[0x62, 0xf1, 0x78, 0x08, 0x28, 0x0d, 0, 0, 0, 0]);
        th.fixups.push(Fixup {
            disp_at: 16,
            insn_end: 20,
            what: FixupKind::Rip(0x5000),
        });
        let (t, s) = thunk_bytes(&th, 0x2000, 0x1000, 7, 0x1007);
        assert_eq!(s, vec![0xe9, 0xfb, 0x0f, 0x00, 0x00, 0x90, 0x90]);
        assert_eq!(t[20], 0xe9, "jmp back after the sequence");
        let c0 = t.len() - 64;
        assert_eq!((0x2000 + c0) % 64, 0, "constant on a 64-byte boundary");
        let rel = i32::from_le_bytes(t[6..10].try_into().unwrap()) as i64;
        assert_eq!(0x2000 + 10 + rel, 0x2000 + c0 as i64);
        let rel = i32::from_le_bytes(t[16..20].try_into().unwrap()) as i64;
        assert_eq!(0x2000 + 20 + rel, 0x5000);
    }
}

/// Lanes of a vector length: the dword-lane (W0) or qword-lane (W1) mask
/// of the elements a 128-bit (`ll` 0), 256-bit (1) or 512-bit (2) form has.
fn vl_lanes(ll: u8, w: u8) -> u16 {
    match (ll, w) {
        (0, 0) => 0x000f,
        (1, 0) => 0x00ff,
        (2, 0) => 0xffff,
        (0, _) => 0x0003,
        (1, _) => 0x000f,
        _ => 0x00ff,
    }
}

/// The dword lanes above a vector length, which a write of that length zeroes.
fn upper(ll: u8) -> u16 {
    match ll {
        0 => 0xfff0,
        1 => 0xff00,
        _ => 0,
    }
}

/// How a whitelisted instruction is spelled on the card: the same
/// (map, pp, W, opcode), or another one, possibly with an immediate the
/// card form takes and the AVX-512 form has not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Canon {
    map: u8,
    pp: u8,
    w: u8,
    op: u8,
    imm: Option<u8>,
}

/// The whitelist: EVEX (map, pp, W, opcode[, ModRM.reg]) to the card's
/// instruction. Each row names the card instruction whose opcode line
/// in 327364-001 it was checked against.
fn canon(map: u8, pp: u8, w: u8, op: u8, reg: u8) -> Option<Canon> {
    let same = Some(Canon { map, pp, w, op, imm: None });
    let to = |map, pp, w, op| Some(Canon { map, pp, w, op, imm: None });
    let with = |map, pp, w, op, imm| {
        Some(Canon {
            map,
            pp,
            w,
            op,
            imm: Some(imm),
        })
    };
    match (map, pp, w, op) {
        // vmovaps / vmovapd load, store, register (MVEX.512.0F.W0 28, 29; .66.W1 for pd)
        (1, 0, 0, 0x28) | (1, 0, 0, 0x29) | (1, 1, 1, 0x28) | (1, 1, 1, 0x29) => same,
        // vaddps vmulps vsubps (MVEX.NDS.512.0F.W0 58, 59, 5C) and the pd forms
        (1, 0, 0, 0x58) | (1, 0, 0, 0x59) | (1, 0, 0, 0x5C) => same,
        (1, 1, 1, 0x58) | (1, 1, 1, 0x59) | (1, 1, 1, 0x5C) => same,
        // vmovdqa32 / vmovdqa64 (MVEX.512.66.0F.W0 6F, 7F; W1)
        (1, 1, 0, 0x6F) | (1, 1, 0, 0x7F) | (1, 1, 1, 0x6F) | (1, 1, 1, 0x7F) => same,
        // vpaddd vpsubd vpandd vpandnd vpord vpxord (MVEX.NDS.512.66.0F.W0 FE FA DB DF EB EF)
        (1, 1, 0, 0xFE) | (1, 1, 0, 0xFA) | (1, 1, 0, 0xDB) | (1, 1, 0, 0xDF) | (1, 1, 0, 0xEB) | (1, 1, 0, 0xEF) => same,
        // vpandq vpandnq vporq vpxorq (W1)
        (1, 1, 1, 0xDB) | (1, 1, 1, 0xDF) | (1, 1, 1, 0xEB) | (1, 1, 1, 0xEF) => same,
        // vandps vandnps vorps vxorps (EVEX.0F.W0 54 55 56 57): bitwise, the card's integer forms
        (1, 0, 0, 0x54) => to(1, 1, 0, 0xDB),
        (1, 0, 0, 0x55) => to(1, 1, 0, 0xDF),
        (1, 0, 0, 0x56) => to(1, 1, 0, 0xEB),
        (1, 0, 0, 0x57) => to(1, 1, 0, 0xEF),
        // vandpd vandnpd vorpd vxorpd (EVEX.66.0F.W1 54..57)
        (1, 1, 1, 0x54) => to(1, 1, 1, 0xDB),
        (1, 1, 1, 0x55) => to(1, 1, 1, 0xDF),
        (1, 1, 1, 0x56) => to(1, 1, 1, 0xEB),
        (1, 1, 1, 0x57) => to(1, 1, 1, 0xEF),
        // vpcmpeqd vpcmpgtd into a mask (MVEX.NDS.512.66.0F.W0 76, 66)
        (1, 1, 0, 0x76) | (1, 1, 0, 0x66) => same,
        // vpsrld /2, vpsrad /4, vpslld /6 by immediate (MVEX.NDD.512.66.0F.W0 72)
        (1, 1, 0, 0x72) if matches!(reg, 2 | 4 | 6) => same,
        // vpshufd imm (MVEX.512.66.0F.W0 70)
        (1, 1, 0, 0x70) => same,
        // vpermilps imm (EVEX.66.0F3A.W0 04): the same per-block shuffle as vpshufd
        (3, 1, 0, 0x04) => to(1, 1, 0, 0x70),
        // vmovshdup (F3.0F.W0 16), vmovsldup (F3.0F.W0 12), vmovddup (F2.0F.W1 12): fixed shuffles
        (1, 2, 0, 0x16) => with(1, 1, 0, 0x70, 0xf5),
        (1, 2, 0, 0x12) => with(1, 1, 0, 0x70, 0xa0),
        (1, 3, 1, 0x12) => with(1, 1, 0, 0x70, 0x44),
        // vpcmpd / vpcmpud with a predicate into a mask (MVEX.NDS.512.66.0F3A.W0 1F, 1E ib)
        (3, 1, 0, 0x1F) | (3, 1, 0, 0x1E) => same,
        // vcmpps / vcmppd (MVEX.NDS.512.0F.W0 C2 ib; .66.W1)
        (1, 0, 0, 0xC2) | (1, 1, 1, 0xC2) => same,
        // vbroadcastss / vbroadcastsd from memory (MVEX.512.66.0F38.W0 18; W1 19)
        (2, 1, 0, 0x18) | (2, 1, 1, 0x19) => same,
        // vpbroadcastd / vpbroadcastq from memory (MVEX.512.66.0F38.W0 58; W1 59)
        (2, 1, 0, 0x58) | (2, 1, 1, 0x59) => same,
        // vbroadcasti32x4 (5A), vbroadcasti64x4 (W1 5B), vbroadcastf32x4 (1A), vbroadcastf64x4 (W1 1B)
        (2, 1, 0, 0x5A) | (2, 1, 1, 0x5B) | (2, 1, 0, 0x1A) | (2, 1, 1, 0x1B) => same,
        // vfmadd, vfmsub, vfnmadd, vfnmsub 132/213/231 ps (W0) and pd (W1)
        (2, 1, _, 0x98)
        | (2, 1, _, 0xA8)
        | (2, 1, _, 0xB8)
        | (2, 1, _, 0x9A)
        | (2, 1, _, 0xAA)
        | (2, 1, _, 0xBA)
        | (2, 1, _, 0x9C)
        | (2, 1, _, 0xAC)
        | (2, 1, _, 0xBC)
        | (2, 1, _, 0x9E)
        | (2, 1, _, 0xAE)
        | (2, 1, _, 0xBE) => same,
        // vpmulld, vpermd, vpsrlvd, vpsravd, vpsllvd, vpminsd, vpminud, vpmaxsd, vpmaxud
        // (MVEX.NDS.512.66.0F38.W0 40 36 45 46 47 39 3B 3D 3F)
        (2, 1, 0, 0x40)
        | (2, 1, 0, 0x36)
        | (2, 1, 0, 0x45)
        | (2, 1, 0, 0x46)
        | (2, 1, 0, 0x47)
        | (2, 1, 0, 0x39)
        | (2, 1, 0, 0x3B)
        | (2, 1, 0, 0x3D)
        | (2, 1, 0, 0x3F) => same,
        // vpermps (EVEX.66.0F38.W0 16): the card's vpermd
        (2, 1, 0, 0x16) => to(2, 1, 0, 0x36),
        // vpblendmd / vpblendmq (MVEX.NDS.512.66.0F38.W0 64; W1), vblendmps / vblendmpd (65)
        (2, 1, _, 0x64) | (2, 1, _, 0x65) => same,
        // vgetexpps (MVEX.512.66.0F38.W0 42), vgetmantps (MVEX.512.66.0F3A.W0 26 ib): the same opcodes and immediate as AVX-512
        (2, 1, 0, 0x42) | (3, 1, 0, 0x26) => same,
        // valignd (MVEX.NDS.512.66.0F3A.W0 03 ib)
        (3, 1, 0, 0x03) => same,
        // vcvtdq2ps (EVEX.0F.W0 5B): vcvtfxpntdq2ps (MVEX.512.0F3A.W0 CB ib), no exponent adjust, MXCSR rounding
        (1, 0, 0, 0x5B) => with(3, 0, 0, 0xCB, 0x00),
        // vcvtudq2ps (EVEX.F2.0F.W0 7A): vcvtfxpntudq2ps (MVEX.512.0F3A.W0 CA ib)
        (1, 3, 0, 0x7A) => with(3, 0, 0, 0xCA, 0x00),
        // vcvtps2pd zmm, ymm/m256 (EVEX.0F.W0 5A): the card's vcvtps2pd (MVEX.512.0F.W0 5A) converts the low 8 floats the same way
        (1, 0, 0, 0x5A) => same,
        _ => None,
    }
}

/// A byte or word compare for equality whose only consumer is
/// `kortestq`/`kortestd` on its result (a scan for a differing byte,
/// as compilers emit for memcmp-like loops): the card's masks are 16
/// bits, one per dword, so the compare becomes the dword compare, whose
/// result is zero exactly when the byte compare's is. The region
/// builder checks the consumer and calls this instead of `rewrite`.
pub fn rewrite_bytecmp_for_kortest(insn: &Instruction, bytes: &[u8], tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    if insn.encoding() != EncodingKind::EVEX || bytes.len() < 6 {
        return Err(refuse(insn, "not an EVEX instruction"));
    }
    let mut v = bytes[..insn.len()].to_vec();
    let ev = parse(bytes);
    match insn.mnemonic() {
        // vpcmpeqb (EVEX.66.0F.W0 74), vpcmpeqw (75): vpcmpeqd (76)
        M::Vpcmpeqb | M::Vpcmpeqw => v[4] = 0x76,
        // vpcmpb / vpcmpub / vpcmpw / vpcmpuw with EQ (0) or NEQ (4)
        // (EVEX.66.0F3A 3F 3E; W1 for words): vpcmpd (1F) with the same predicate
        M::Vpcmpb | M::Vpcmpub | M::Vpcmpw | M::Vpcmpuw if ev.map == 3 && matches!(insn.immediate8() & 7, 0 | 4) => {
            v[4] = 0x1F;
            v[2] &= 0x7f; // W0
        }
        _ => return Err(refuse(insn, "a byte or word compare the card cannot express")),
    }
    let dinsn = iced_x86::Decoder::with_ip(64, &v, insn.ip(), iced_x86::DecoderOptions::NONE).decode();
    rewrite(&dinsn, &v, tg)
}

/// Whether the destination is the vvvv register (the shift-by-immediate
/// groups 71, 72, 73, where ModRM.reg is the opcode extension).
fn dest_is_vvvv(ev: &Ev) -> bool {
    ev.map == 1 && ev.pp == 1 && (ev.op == 0x72 || ev.op == 0x73 || ev.op == 0x71)
}

/// Rewrite one AVX-512 instruction (EVEX, or a VEX-encoded mask
/// instruction) for the card.
pub fn rewrite(insn: &Instruction, bytes: &[u8], tg: &Target) -> Result<Rewrite, Unsupported> {
    if insn.encoding() == EncodingKind::VEX {
        return kops(insn, bytes, tg);
    }
    if insn.encoding() != EncodingKind::EVEX || bytes.len() < 6 || bytes[0] != 0x62 {
        return Err(refuse(insn, "not an EVEX instruction"));
    }
    let ev = parse(bytes);
    if let Some(r) = special(insn, bytes, &ev, tg)? {
        return Ok(r);
    }
    let c = canon(ev.map, ev.pp, ev.w, ev.op, (ev.modrm >> 3) & 7).ok_or_else(|| refuse(insn, "not in the card's instruction table"))?;
    if ev.map == 1 && ev.op == 0xC2 && insn.immediate8() > 7 {
        return compare(insn, &ev, tg);
    }
    if ev.map == 3 && (ev.op == 0x1F || ev.op == 0x1E) && matches!(insn.immediate8() & 7, 3 | 7) {
        return compare(insn, &ev, tg);
    }
    let same_shape = c
        == Canon {
            map: ev.map,
            pp: ev.pp,
            w: ev.w,
            op: ev.op,
            imm: None,
        };
    let has_mem = !ev.is_reg_rm;
    let aligned_move = matches!((ev.map, ev.op), (1, 0x28) | (1, 0x29) | (1, 0x6F) | (1, 0x7F));
    let broadcast_load = ev.map == 2 && matches!(ev.op, 0x18 | 0x19 | 0x58 | 0x59 | 0x5A | 0x5B | 0x1A | 0x1B);
    if ev.b && !has_mem {
        return Err(refuse(insn, "embedded rounding or SAE: not expressed on the card yet"));
    }
    // In place: the 512-bit form of an instruction spelled the same, with
    // no zeroing, whose memory operand (if any) is a broadcast, an
    // aligned move or a block broadcast: what the card can address as is.
    if ev.ll == 2 && !ev.z && same_shape && (!has_mem || ev.b || aligned_move || broadcast_load) {
        let mut v = bytes[..insn.len()].to_vec();
        v[2] = bytes[2] & !0x04;
        v[3] = (if ev.b { 1 << 4 } else { 0 }) | (bytes[3] & 0x0f);
        return Ok(Rewrite::InPlace(v));
    }
    generic(insn, &ev, c, tg)
}

/// The general masked form of a whitelisted instruction: the 512-bit
/// operation under a lane mask, its memory operand staged when the card
/// could not address it, the unselected or upper lanes zeroed after.
fn generic(insn: &Instruction, ev: &Ev, c: Canon, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let narrow = ev.ll != 2;
    let kn = ev.aaa;
    let is_store = ev.map == 1 && (ev.op == 0x29 || ev.op == 0x7F);
    let is_cmp = (ev.map == 1 && (ev.op == 0xC2 || ev.op == 0x76 || ev.op == 0x66)) || (ev.map == 3 && (ev.op == 0x1F || ev.op == 0x1E));
    let mem = MemOp::of(insn);
    let vl = vl_lanes(ev.ll, ev.w);
    let full = ev.ll == 2;
    if is_store {
        // A narrow or unaligned store: the pack pair writes only the
        // selected lanes, at any alignment.
        let m = mem.ok_or_else(|| refuse(insn, "store without a memory operand"))?;
        let dvl = vl_lanes(ev.ll, 0);
        let k = if kn != 0 && ev.w == 1 {
            em.qmask_to_dmask(kn, dvl)
        } else if full {
            kn
        } else {
            let k = em.mask_imm(dvl);
            if kn != 0 {
                em.kand(k, kn);
            }
            k
        };
        em.store_unaligned(m, ev.reg, k, 0, false, 0);
        return Ok(Rewrite::Thunk(em.finish()));
    }
    let d = if dest_is_vvvv(ev) { ev.vvvv } else { ev.reg };
    let imm = c.imm.or_else(|| {
        if insn.op_count() > 0 && insn.op_kind(insn.op_count() - 1) == OpKind::Immediate8 {
            Some(insn.immediate8())
        } else {
            None
        }
    });
    // The second operand, staged into a temporary when the card cannot
    // address it as the program did.
    let (rm, sss) = match mem {
        None => (Rm::Reg(ev.rm), 0),
        Some(m) if ev.b => (Rm::Mem(m), 1),
        Some(m) => {
            let aligned_move = matches!((ev.map, ev.op), (1, 0x28) | (1, 0x6F));
            let block = ev.map == 2 && matches!(ev.op, 0x5A | 0x5B | 0x1A | 0x1B);
            let bcast = ev.map == 2 && matches!(ev.op, 0x18 | 0x19 | 0x58 | 0x59);
            if bcast || (block && full) {
                (Rm::Mem(m), 0)
            } else if aligned_move && !full {
                // An aligned 16-byte or 32-byte move: its block, broadcast, is
                // exactly the bytes it reads, at the alignment it guarantees.
                let t = em.temp().map_err(|e| refuse(insn, e))?;
                em.load_block(t, Rm::Mem(m), 0, ev.ll == 1);
                (Rm::Reg(t), 0)
            } else {
                let t = em.temp().map_err(|e| refuse(insn, e))?;
                let k = if full { 0 } else { em.mask_imm(vl_lanes(ev.ll, 0)) };
                em.load_unaligned(t, m, k, 0, false, 0);
                (Rm::Reg(t), 0)
            }
        }
    };
    // The mask the operation runs under.
    let keff = if full {
        kn
    } else {
        let k = em.mask_imm(vl);
        if kn != 0 {
            em.kand(k, kn);
        }
        k
    };
    let (reg_field, vvvv_field) = if dest_is_vvvv(ev) { (ev.reg & 7, d) } else { (ev.reg, ev.vvvv) };
    em.mvex(c.map, c.pp, c.w, c.op, reg_field, vvvv_field, rm, keff, sss, imm);
    if is_cmp {
        // A compare writes its result mask: bits of masked-off lanes must be
        // zero, as AVX-512 defines them.
        if !full || kn != 0 {
            em.kand(ev.reg & 7, keff);
        }
    } else {
        if ev.z {
            let kz = em.k();
            em.knot(kz, keff);
            em.zero(d, kz, ev.w);
        }
        if narrow {
            em.zero_lanes(d, upper(ev.ll));
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// vcvtpd2ps ymm, zmm/m512: the card's vcvtpd2ps (MVEX.512.66.0F.W1 5A)
/// gives the 8 floats in the low lanes; the lanes above are zeroed
/// here, as a ymm destination requires.
fn cvt_pd2ps(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let d = ev.reg;
    let s = match MemOp::of(insn) {
        Some(m) => {
            let t = em.temp().map_err(|e| refuse(insn, e))?;
            let k = if ev.ll == 2 { 0 } else { em.mask_imm(vl_lanes(ev.ll, 0)) };
            em.load_unaligned(t, m, k, 0, false, 0);
            t
        }
        None => ev.rm,
    };
    let r = if ev.aaa != 0 { em.temp().map_err(|e| refuse(insn, e))? } else { d };
    em.mvex(1, 1, 1, 0x5A, r, 0, Rm::Reg(s), 0, 0, None);
    // 8 doubles give 8 floats (ll 2); 4 give 4 (ll 1); 2 give 2 (ll 0)
    let out_ll = if ev.ll == 2 { 1 } else { 0 };
    let lanes: u16 = if ev.ll == 2 {
        0xff
    } else if ev.ll == 1 {
        0xf
    } else {
        0x3
    };
    if ev.ll != 2 {
        em.zero_lanes(r, !lanes);
    }
    commit(&mut em, d, r, &Ev { ll: out_ll, w: 0, ..*ev });
    Ok(Rewrite::Thunk(em.finish()))
}

/// The permutes the card composes from `vpermd` (indices, bits 3:0) and
/// `vptestmd`: two-table permutes (`vpermi2*`, `vpermt2*`), qword
/// permutes by immediate or by vector (`vpermq`, `vpermpd`), the
/// variable and qword `vpermilp*`, and the narrow `vpermd`/`vpermps`
/// (whose indices use fewer bits).
fn permute(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let m = insn.mnemonic();
    let d = ev.reg;
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let t = em.temp().map_err(|e| refuse(insn, e))?;
    let odd = em.const_dws([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    // A qword index vector (3 or 4 bits per qword) as a dword index vector:
    // lane 2i gets 2*q, lane 2i+1 gets 2*q+1 (bit 4 keeps a table selector).
    let qidx_to_didx = |em: &mut Em, out: u8, q: u8| {
        em.vpshufd(out, Rm::Reg(q), 0xa0, 0);
        em.shift_imm(6, out, Rm::Reg(out), 1, 0);
        em.i0f(0xFE, out, out, Rm::Const(odd), 0, 0);
    };
    match m {
        M::Vpermi2ps | M::Vpermi2d | M::Vpermt2ps | M::Vpermt2d | M::Vpermi2pd | M::Vpermi2q | M::Vpermt2pd | M::Vpermt2q => {
            let is_i = matches!(m, M::Vpermi2ps | M::Vpermi2d | M::Vpermi2pd | M::Vpermi2q);
            let qword = matches!(m, M::Vpermi2pd | M::Vpermi2q | M::Vpermt2pd | M::Vpermt2q);
            let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
            // vpermi2: dest = index, vvvv = table 1; vpermt2: dest = table 1, vvvv = index
            let (idx, s1) = if is_i { (ev.reg, ev.vvvv) } else { (ev.vvvv, ev.reg) };
            let ix = if qword {
                qidx_to_didx(&mut em, t, idx);
                t
            } else {
                idx
            };
            let sel = em.const_dw(0x10);
            let k = em.k();
            em.mvex(2, 1, 0, 0x27, k, ix, Rm::Const(sel), 0, 0, None); // vptestmd k, ix, [0x10]: table 2 lanes
            em.vpermd(r, ix, Rm::Reg(s1), 0);
            em.vpermd(r, ix, Rm::Reg(s2), k);
        }
        M::Vpermq | M::Vpermpd if insn.op_kind(insn.op_count() - 1) == OpKind::Immediate8 => {
            // Per 256-bit half: qword j from qword (imm >> 2j) & 3 of the same half.
            let imm = insn.immediate8();
            let mut v = [0u32; 16];
            for (i, x) in v.iter_mut().enumerate() {
                let half = (i / 8) as u32;
                let q = ((i / 2) % 4) as u32;
                let src_q = u32::from(imm >> (2 * q)) & 3;
                *x = half * 8 + 2 * src_q + (i as u32 & 1);
            }
            let c = em.const_dws(v);
            let s = src2(&mut em, insn, &Ev { vvvv: 0, ..*ev }, vl_lanes(ev.ll, 0))?;
            em.vmov(t, Rm::Const(c), 0, 0);
            em.vpermd(r, t, Rm::Reg(s), 0);
        }
        M::Vpermq | M::Vpermpd => {
            // By vector: qword indices (3 bits) in vvvv, the table in rm.
            let s = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
            qidx_to_didx(&mut em, t, ev.vvvv);
            em.vpermd(r, t, Rm::Reg(s), 0);
        }
        M::Vpermilpd if insn.op_kind(insn.op_count() - 1) == OpKind::Immediate8 => {
            // Bit (2b + j) of the immediate picks the qword for lane j of block b.
            let imm = insn.immediate8();
            let mut v = [0u32; 16];
            for (i, x) in v.iter_mut().enumerate() {
                let b = (i / 4) as u32;
                let j = ((i / 2) % 2) as u32;
                let sel = u32::from(imm >> (2 * b + j)) & 1;
                *x = b * 4 + 2 * sel + (i as u32 & 1);
            }
            let c = em.const_dws(v);
            let s = src2(&mut em, insn, &Ev { vvvv: 0, ..*ev }, vl_lanes(ev.ll, 0))?;
            em.vmov(t, Rm::Const(c), 0, 0);
            em.vpermd(r, t, Rm::Reg(s), 0);
        }
        M::Vpermilpd => {
            // Bit 1 of each qword index picks the qword within its 128-bit block.
            let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
            em.vpshufd(t, Rm::Reg(s2), 0xa0, 0);
            em.shift_imm(2, t, Rm::Reg(t), 1, 0);
            let one = em.const_dw(1);
            em.i0f(0xDB, t, t, Rm::Const(one), 0, 0);
            em.shift_imm(6, t, Rm::Reg(t), 1, 0);
            let base = em.const_dws([0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13]);
            em.i0f(0xFE, t, t, Rm::Const(base), 0, 0);
            em.vpermd(r, t, Rm::Reg(ev.vvvv), 0);
        }
        M::Vpermilps => {
            // Two bits of each dword index pick the dword within its block.
            let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
            let three = em.const_dw(3);
            em.i0f(0xDB, t, s2, Rm::Const(three), 0, 0);
            let base = em.const_dws([0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]);
            em.i0f(0xFE, t, t, Rm::Const(base), 0, 0);
            em.vpermd(r, t, Rm::Reg(ev.vvvv), 0);
        }
        _ => {
            // vpermd / vpermps, narrow: the index uses 3 bits (ymm) or 2 (xmm)
            let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
            let mask = em.const_dw(if ev.ll == 1 { 7 } else { 3 });
            em.i0f(0xDB, t, ev.vvvv, Rm::Const(mask), 0, 0);
            em.vpermd(r, t, Rm::Reg(s2), 0);
        }
    }
    commit(&mut em, d, r, &Ev { w: 0, ..*ev });
    Ok(Rewrite::Thunk(em.finish()))
}

impl Em<'_> {
    /// An x87 or integer instruction with one `[fs:scratch+off]` memory
    /// operand: `opcodes` then ModRM with `reg` as the extension.
    fn fs_op(&mut self, opcodes: &[u8], reg: u8, off: i32) {
        self.t.seq.push(0x64);
        self.t.seq.extend_from_slice(opcodes);
        self.t.seq.extend_from_slice(&[0x04 | (reg << 3), 0x25]);
        self.t.seq.extend_from_slice(&self.fs(off).to_le_bytes());
    }
}

/// The scalar conversions between general registers and floats, and
/// between float widths. Integer to float goes through x87 (`fild`, then
/// `fstp` to the width: one rounding, from an exact value), float to a
/// 32-bit integer through the card's fixed-point conversion with the
/// indefinite fix-up, float to a 64-bit integer through x87 with the
/// control word set to truncate when the instruction truncates.
fn scalar_cvt(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let m = insn.mnemonic();
    if ev.b {
        return Err(refuse(insn, "scalar conversion with embedded rounding: not expressed yet"));
    }
    match m {
        M::Vcvtsi2ss | M::Vcvtsi2sd | M::Vcvtusi2ss | M::Vcvtusi2sd => {
            let to_double = matches!(m, M::Vcvtsi2sd | M::Vcvtusi2sd);
            let unsigned = matches!(m, M::Vcvtusi2ss | M::Vcvtusi2sd);
            let (d, s1) = (ev.reg, ev.vvvv);
            // The integer to the scratch slot as a qword: sign- or zero-extended.
            match MemOp::of(insn) {
                Some(mm) => {
                    let wide = insn.memory_size() == iced_x86::MemorySize::UInt64 || insn.memory_size() == iced_x86::MemorySize::Int64;
                    em.rax();
                    if wide {
                        em.t.seq.extend_from_slice(&[0x48, 0x8b]); // mov rax, [m]
                        em.mem_modrm(0, mm);
                    } else if unsigned {
                        em.t.seq.push(0x8b); // mov eax, [m] (zero-extends)
                        em.mem_modrm(0, mm);
                    } else {
                        em.t.seq.extend_from_slice(&[0x48, 0x63]); // movsxd rax, [m]
                        em.mem_modrm(0, mm);
                    }
                    em.gpr_scratch(0, true, S_XFER, true);
                }
                None => {
                    let (r, wide) = gpr(insn.op2_register()).ok_or_else(|| refuse(insn, "conversion source is not a general register"))?;
                    if wide {
                        em.gpr_scratch(r, true, S_XFER, true);
                    } else if unsigned {
                        // mov eax, r32 zero-extends; then the qword
                        em.rax();
                        if r != 0 {
                            if r >= 8 {
                                em.t.seq.push(0x41);
                            }
                            em.t.seq.extend_from_slice(&[0x8b, 0xc0 | (r & 7)]);
                        }
                        em.gpr_scratch(0, true, S_XFER, true);
                    } else {
                        // movsxd rax, r32
                        em.rax();
                        em.t.seq.extend_from_slice(&[0x48 | ((r >> 3) & 1), 0x63, 0xc0 | (r & 7)]);
                        em.gpr_scratch(0, true, S_XFER, true);
                    }
                    if unsigned && wide {
                        return Err(refuse(insn, "unsigned 64-bit integer to float: not expressed yet"));
                    }
                }
            }
            em.fs_op(&[0xdf], 5, S_XFER); // fild qword
            if to_double {
                em.fs_op(&[0xdd], 3, S_XFER); // fstp qword
                let k = em.mask_imm(1);
                em.broadcast(d, Rm::Fs(S_XFER), k, 1);
                if d != s1 {
                    let k = em.mask_imm(2);
                    em.vmov(d, Rm::Reg(s1), k, 1);
                }
                em.zero_lanes(d, 0xfff0);
            } else {
                em.fs_op(&[0xd9], 3, S_XFER); // fstp dword
                let k = em.mask_imm(1);
                em.broadcast(d, Rm::Fs(S_XFER), k, 0);
                if d != s1 {
                    let k = em.mask_imm(0xe);
                    em.vmov(d, Rm::Reg(s1), k, 0);
                }
                em.zero_lanes(d, 0xfff0);
            }
        }
        M::Vcvttss2si | M::Vcvtss2si | M::Vcvttsd2si | M::Vcvtsd2si | M::Vcvttss2usi | M::Vcvtss2usi | M::Vcvttsd2usi | M::Vcvtsd2usi => {
            let truncate = matches!(m, M::Vcvttss2si | M::Vcvttsd2si | M::Vcvttss2usi | M::Vcvttsd2usi);
            let from_double = matches!(m, M::Vcvttsd2si | M::Vcvtsd2si | M::Vcvttsd2usi | M::Vcvtsd2usi);
            let unsigned = matches!(m, M::Vcvttss2usi | M::Vcvtss2usi | M::Vcvttsd2usi | M::Vcvtsd2usi);
            let (r, wide) = gpr(insn.op0_register()).ok_or_else(|| refuse(insn, "conversion destination is not a general register"))?;
            // The float in a register, or staged from memory.
            let s = match MemOp::of(insn) {
                Some(mm) => {
                    let t = em.temp().map_err(|e| refuse(insn, e))?;
                    em.broadcast(t, Rm::Mem(mm), 0, from_double as u8);
                    t
                }
                None => ev.rm,
            };
            if wide || unsigned {
                // x87: exact, and the integer indefinite for NaN and overflow.
                let t = em.temp().map_err(|e| refuse(insn, e))?;
                em.vmov(t, Rm::Reg(s), 0, from_double as u8);
                em.mvex(2, 1, 0, 0xD0, t, 0, Rm::Fs(S_XFER), 0, 0, None); // the lane(s) to the slot
                if from_double {
                    em.fs_op(&[0xdd], 0, S_XFER); // fld qword
                } else {
                    em.fs_op(&[0xd9], 0, S_XFER); // fld dword
                }
                if truncate {
                    em.flags();
                    em.fs_op(&[0xd9], 7, S_CW); // fnstcw
                    em.fs_op(&[0x0f, 0xb7], 0, S_CW); // movzx eax, word [cw]
                    em.t.seq.extend_from_slice(&[0x80, 0xcc, 0x0c]); // or ah, 0x0c: round toward zero
                    em.t.seq.push(0x66);
                    em.fs_op(&[0x89], 0, S_CW2); // mov [cw2], ax
                    em.fs_op(&[0xd9], 5, S_CW2); // fldcw
                }
                em.fs_op(&[0xdf], 7, S_XFER); // fistp qword
                if truncate {
                    em.fs_op(&[0xd9], 5, S_CW); // fldcw back
                }
                em.gpr_result(r, wide, S_XFER);
                if !wide {
                    // The 32-bit unsigned result: the low dword of the qword (values
                    // up to 2^32 fit the 64-bit conversion exactly).
                }
            } else {
                let t = em.temp().map_err(|e| refuse(insn, e))?;
                let imm = if truncate { 3 } else { 0 };
                if from_double {
                    em.mvex(3, 3, 1, 0xE6, t, 0, Rm::Reg(s), 0, 0, Some(imm)); // vcvtfxpntpd2dq
                    let big = em.constant({
                        let mut c = [0u8; 64];
                        for l in 0..8 {
                            c[l * 8..l * 8 + 8].copy_from_slice(&0x41E0_0000_0000_0000u64.to_le_bytes());
                        }
                        c
                    });
                    let kc = em.k();
                    em.vcmp(kc, s, Rm::Const(big), 5, 0, 1, 0); // not less than 2^31, NaN included
                    let ind = em.const_dw(0x8000_0000);
                    em.vmov(t, Rm::Const(ind), kc, 0);
                } else {
                    em.mvex(3, 1, 0, 0xCB, t, 0, Rm::Reg(s), 0, 0, Some(imm)); // vcvtfxpntps2dq
                    let big = em.const_dw(0x4f00_0000);
                    let kc = em.k();
                    em.vcmp(kc, s, Rm::Const(big), 5, 0, 0, 0);
                    let ind = em.const_dw(0x8000_0000);
                    em.vmov(t, Rm::Const(ind), kc, 0);
                }
                let k = em.mask_imm(1);
                em.mvex(2, 1, 0, 0xD0, t, 0, Rm::Fs(S_XFER), k, 0, None);
                em.gpr_result(r, false, S_XFER);
            }
        }
        M::Vcvtss2sd => {
            let (d, s1) = (ev.reg, ev.vvvv);
            let s = match MemOp::of(insn) {
                Some(mm) => {
                    let t = em.temp().map_err(|e| refuse(insn, e))?;
                    em.broadcast(t, Rm::Mem(mm), 0, 0);
                    t
                }
                None => ev.rm,
            };
            let r = em.temp().map_err(|e| refuse(insn, e))?;
            em.mvex(1, 0, 0, 0x5A, r, 0, Rm::Reg(s), 0, 0, None); // vcvtps2pd
            let k = em.mask_imm(1);
            em.vmov(d, Rm::Reg(r), k, 1);
            if d != s1 {
                let k = em.mask_imm(2);
                em.vmov(d, Rm::Reg(s1), k, 1);
            }
            em.zero_lanes(d, 0xfff0);
        }
        _ => {
            // vcvtsd2ss
            let (d, s1) = (ev.reg, ev.vvvv);
            let s = match MemOp::of(insn) {
                Some(mm) => {
                    let t = em.temp().map_err(|e| refuse(insn, e))?;
                    em.broadcast(t, Rm::Mem(mm), 0, 1);
                    t
                }
                None => ev.rm,
            };
            let r = em.temp().map_err(|e| refuse(insn, e))?;
            em.mvex(1, 1, 1, 0x5A, r, 0, Rm::Reg(s), 0, 0, None); // vcvtpd2ps
            let k = em.mask_imm(1);
            em.vmov(d, Rm::Reg(r), k, 0);
            if d != s1 {
                let k = em.mask_imm(0xe);
                em.vmov(d, Rm::Reg(s1), k, 0);
            }
            em.zero_lanes(d, 0xfff0);
        }
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// vrndscaleps/pd/ss/sd with M = 0 (to an integer) and an explicit
/// rounding mode: the card's vrndfxpntps/pd (MVEX.512.66.0F3A.W0 52 ib;
/// W1 for pd), whose immediate takes the same two rounding-mode bits.
fn rndscale(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let imm = insn.immediate8();
    if imm >> 4 != 0 {
        return Err(refuse(insn, "vrndscale to a fraction of a bit (M above 0): not expressed yet"));
    }
    if imm & 4 != 0 {
        return Err(refuse(insn, "vrndscale with the MXCSR rounding mode: not expressed yet"));
    }
    let rc = imm & 3;
    let w = ev.w;
    let scalar = matches!(insn.mnemonic(), M::Vrndscaless | M::Vrndscalesd);
    if scalar {
        if ev.aaa != 0 {
            return Err(refuse(insn, "masked scalar round: not expressed yet"));
        }
        let (d, s1) = (ev.reg, ev.vvvv);
        let (rm, sss) = match MemOp::of(insn) {
            Some(m) => (Rm::Mem(m), 1u8),
            None => (Rm::Reg(ev.rm), 0),
        };
        let k = em.mask_imm(1);
        em.mvex(3, 1, w, 0x52, d, 0, rm, k, sss, Some(rc));
        if d != s1 {
            let k = em.mask_imm(if w == 0 { 0xe } else { 0x2 });
            em.vmov(d, Rm::Reg(s1), k, w);
        }
        em.zero_lanes(d, 0xfff0);
    } else {
        let d = ev.reg;
        let s = src2(&mut em, insn, &Ev { vvvv: 0, ..*ev }, vl_lanes(ev.ll, 0))?;
        let r = if ev.aaa != 0 || ev.ll != 2 {
            em.temp().map_err(|e| refuse(insn, e))?
        } else {
            d
        };
        em.mvex(3, 1, w, 0x52, r, 0, Rm::Reg(s), 0, 0, Some(rc));
        commit(&mut em, d, r, ev);
    }
    Ok(Rewrite::Thunk(em.finish()))
}

/// vscalefps: x * 2^floor(y). The card's vscaleps (MVEX.NDS.512.66.0F38.W0
/// 84) takes the exponent as an int32 vector: floor(y) as an integer
/// first (vcvtfxpntps2dq rounding down).
fn scalef(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    let mut em = Em::new(tg, insn);
    let (d, s1) = (ev.reg, ev.vvvv);
    let s2 = src2(&mut em, insn, ev, vl_lanes(ev.ll, 0))?;
    let t = em.temp().map_err(|e| refuse(insn, e))?;
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    em.mvex(3, 1, 0, 0xCB, t, 0, Rm::Reg(s2), 0, 0, Some(1)); // floor(y) as int32
    em.mvex(2, 1, 0, 0x84, r, s1, Rm::Reg(t), 0, 0, None);
    commit(&mut em, d, r, ev);
    Ok(Rewrite::Thunk(em.finish()))
}

impl Em<'_> {
    /// A fused multiply-add: `op` from the 0F38 map (A8 213, B8 231,
    /// AC fn213, BC fn231, and the pd forms with W1): d = f(d, s1, s2).
    fn fma(&mut self, op: u8, d: u8, s1: u8, s2: Rm, w: u8) {
        self.mvex(2, 1, w, op, d, s1, s2, 0, 0, None);
    }
    fn vmulp(&mut self, d: u8, s1: u8, s2: Rm, w: u8) {
        self.mvex(1, w, w, 0x59, d, s1, s2, 0, 0, None);
    }
    fn const_f32(&mut self, f: f32) -> usize {
        self.const_dw(f.to_bits())
    }
    fn const_f64(&mut self, f: f64) -> usize {
        let mut c = [0u8; 64];
        for l in 0..8 {
            c[l * 8..l * 8 + 8].copy_from_slice(&f.to_bits().to_le_bytes());
        }
        self.constant(c)
    }
    /// 1/b to float precision: vrcp23ps (MVEX.512.66.0F38.W0 CA), one
    /// Newton step (r = r + r*(1 - b*r)).
    fn recip_ps(&mut self, r: u8, b: u8, e: u8) {
        self.mvex(2, 1, 0, 0xCA, r, 0, Rm::Reg(b), 0, 0, None);
        let one = self.const_f32(1.0);
        self.vmov(e, Rm::Const(one), 0, 0);
        self.fma(0xBC, e, b, Rm::Reg(r), 0); // e = 1 - b*r
        self.fma(0xB8, r, e, Rm::Reg(r), 0); // r = e*r + r
    }
}

impl Em<'_> {
    /// A double's biased exponent field (bits 62:52) into the odd (high)
    /// dword lanes of `e` (the even lanes are garbage), from `x`.
    fn dexp(&mut self, e: u8, x: u8) {
        self.shift_imm(2, e, Rm::Reg(x), 20, 0);
        let m = self.const_dw(0x7ff);
        self.i0f(0xDB, e, e, Rm::Const(m), 0, 0);
    }
    /// `x` with its exponent field replaced: `out` = x, high dwords
    /// (odd lanes) = (x & 0x800fffff) | (expo << 20), `expo` a dword vector.
    fn dsetexp(&mut self, out: u8, x: u8, expo: u8, t: u8) {
        let odd = self.mask_imm(0xaaaa);
        let keep = self.const_dw(0x800f_ffff);
        self.vmov(out, Rm::Reg(x), 0, 0);
        self.i0f(0xDB, out, out, Rm::Const(keep), odd, 0);
        self.shift_imm(6, t, Rm::Reg(expo), 20, 0);
        self.i0f(0xEB, out, out, Rm::Reg(t), odd, 0);
    }
    /// A power of two as a double vector: `out` = 2^(expo - 1023), `expo`
    /// the biased exponent in the odd lanes (the even lanes become zero).
    fn dpow2(&mut self, out: u8, expo: u8) {
        self.shift_imm(6, out, Rm::Reg(expo), 20, 0);
        let even = self.mask_imm(0x5555);
        self.zero(out, even, 0);
    }
}

#[allow(clippy::needless_late_init)]
/// Division and square root, which the card lacks: Newton-Raphson from
/// its 23-bit estimates (vrcp23ps, vrsqrt23ps), then one residual step
/// with a fused multiply-add, which makes the result correctly rounded
/// in all but rare cases. Doubles are first scaled by a power of two
/// into [1, 4) through their exponent field (so the float seed cannot
/// overflow or underflow), seeded through float, refined twice, and
/// scaled back exactly. Zero, infinity and NaN operands are fixed up
/// from the operands themselves. Scalar forms work lane 0.
fn divsqrt(insn: &Instruction, ev: &Ev, tg: &Target) -> Result<Rewrite, Unsupported> {
    use Mnemonic as M;
    let mut em = Em::new(tg, insn);
    let m = insn.mnemonic();
    let w = ev.w;
    let scalar = matches!(m, M::Vdivss | M::Vdivsd | M::Vsqrtss | M::Vsqrtsd);
    let is_div = matches!(m, M::Vdivps | M::Vdivpd | M::Vdivss | M::Vdivsd);
    if scalar && ev.aaa != 0 {
        return Err(refuse(insn, "masked scalar divide or square root: not expressed yet"));
    }
    if ev.b && MemOp::of(insn).is_none() {
        return Err(refuse(insn, "embedded rounding: not expressed yet"));
    }
    let d = ev.reg;
    let stage = |em: &mut Em, use_vvvv: bool| -> Result<u8, Unsupported> {
        if scalar {
            match MemOp::of(insn) {
                Some(mm) => {
                    let t = em.temp().map_err(|e| refuse(insn, e))?;
                    em.broadcast(t, Rm::Mem(mm), 0, w);
                    Ok(t)
                }
                None => Ok(ev.rm),
            }
        } else {
            let e2 = if use_vvvv { *ev } else { Ev { vvvv: 0, ..*ev } };
            src2(em, insn, &e2, vl_lanes(ev.ll, 0))
        }
    };
    // a / b, or sqrt(a)
    let (a, b) = if is_div {
        (ev.vvvv, stage(&mut em, true)?)
    } else {
        (stage(&mut em, false)?, 0)
    };
    let r = em.temp().map_err(|e| refuse(insn, e))?;
    let e = em.temp().map_err(|e| refuse(insn, e))?;
    let q = em.temp().map_err(|e| refuse(insn, e))?;
    let kx = em.k();
    // 64-byte constants for the double paths
    let dconst = |em: &mut Em, v: u64| -> usize {
        let mut c = [0u8; 64];
        for l in 0..8 {
            c[l * 8..l * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        em.constant(c)
    };
    // The register holding the result at the end.
    let result: u8;
    if is_div && w == 0 {
        em.recip_ps(r, b, e);
        em.vmulp(q, a, Rm::Reg(r), 0);
        em.vmov(e, Rm::Reg(a), 0, 0);
        em.fma(0xBC, e, b, Rm::Reg(q), 0); // e = a - b*q
        em.fma(0xB8, q, e, Rm::Reg(r), 0); // q = e*r + q
                                           // b zero or infinite (the refinement made NaN of it): a times the raw estimate
        em.mvex(2, 1, 0, 0xCA, r, 0, Rm::Reg(b), 0, 0, None);
        em.vmulp(r, a, Rm::Reg(r), 0);
        let absm = em.const_dw(0x7fff_ffff);
        em.i0f(0xDB, e, b, Rm::Const(absm), 0, 0);
        let zero = em.const_f32(0.0);
        em.vcmp(kx, e, Rm::Const(zero), 0, 0, 0, 0);
        em.vmov(q, Rm::Reg(r), kx, 0);
        let inf = em.const_f32(f32::INFINITY);
        em.vcmp(kx, e, Rm::Const(inf), 0, 0, 0, 0);
        em.vmov(q, Rm::Reg(r), kx, 0);
        // a infinite: the residual is inf - inf; the raw product is right
        em.i0f(0xDB, e, a, Rm::Const(absm), 0, 0);
        em.vcmp(kx, e, Rm::Const(inf), 0, 0, 0, 0);
        em.vmov(q, Rm::Reg(r), kx, 0);
        result = q;
    } else if is_div {
        let ex = em.temp().map_err(|e| refuse(insn, e))?;
        // r = b' (b with exponent 1023, so in [1, 2)); ex = a * 2^(1023 - E), exact
        em.dexp(ex, b);
        let c1023 = em.const_dw(1023);
        em.vmov(e, Rm::Const(c1023), 0, 0);
        em.dsetexp(r, b, e, q);
        let c2046 = em.const_dw(2046);
        em.vmov(e, Rm::Const(c2046), 0, 0);
        em.i0f(0xFA, e, e, Rm::Reg(ex), 0, 0);
        em.dpow2(ex, e);
        em.vmulp(ex, ex, Rm::Reg(a), 1);
        // q = 1/b': seeded through float, two Newton steps
        em.mvex(1, 1, 1, 0x5A, e, 0, Rm::Reg(r), 0, 0, None);
        em.mvex(2, 1, 0, 0xCA, e, 0, Rm::Reg(e), 0, 0, None);
        em.mvex(1, 0, 0, 0x5A, q, 0, Rm::Reg(e), 0, 0, None);
        let one = em.const_f64(1.0);
        for _ in 0..2 {
            em.vmov(e, Rm::Const(one), 0, 1);
            em.fma(0xBC, e, r, Rm::Reg(q), 1); // e = 1 - b'*q
            em.fma(0xB8, q, e, Rm::Reg(q), 1); // q = e*q + q
        }
        // e = quotient; ex = residual; e += residual * reciprocal
        em.vmulp(e, ex, Rm::Reg(q), 1);
        em.fma(0xBC, ex, r, Rm::Reg(e), 1); // ex = aS - b'*e
        em.fma(0xB8, e, ex, Rm::Reg(q), 1); // e = ex*q + e
                                            // fix-ups: b zero (signed infinity, NaN for 0/0), b infinite (signed zero), NaN in
        let absm = dconst(&mut em, 0x7fff_ffff_ffff_ffff);
        let signm = dconst(&mut em, 0x8000_0000_0000_0000);
        let zero = dconst(&mut em, 0);
        let inf = dconst(&mut em, 0x7ff0_0000_0000_0000);
        let nan = dconst(&mut em, 0xfff8_0000_0000_0000);
        em.i0f(0xEF, q, a, Rm::Reg(b), 0, 1);
        em.i0f(0xDB, q, q, Rm::Const(signm), 0, 1); // q = the result's sign
        em.i0f(0xDB, r, b, Rm::Const(absm), 0, 1); // r = |b|
        em.vcmp(kx, r, Rm::Const(zero), 0, 0, 1, 0);
        em.i0f(0xEB, ex, q, Rm::Const(inf), 0, 1);
        em.vmov(e, Rm::Reg(ex), kx, 1); // a/0 = signed infinity
        em.vcmp(kx, r, Rm::Const(inf), 0, 0, 1, 0);
        em.vmov(e, Rm::Reg(q), kx, 1); // a/inf = signed zero
        em.i0f(0xDB, ex, a, Rm::Const(absm), 0, 1); // ex = |a|
        em.mvex(1, 1, 1, 0x58, r, r, Rm::Reg(ex), 0, 0, None); // |a| + |b|
        em.vcmp(kx, r, Rm::Const(zero), 0, 0, 1, 0);
        em.vmov(e, Rm::Const(nan), kx, 1); // 0/0
        em.vcmp(kx, a, Rm::Reg(a), 3, 0, 1, 0);
        em.vmov(e, Rm::Reg(a), kx, 1);
        em.vcmp(kx, b, Rm::Reg(b), 3, 0, 1, 0);
        em.vmov(e, Rm::Reg(b), kx, 1);
        result = e;
    } else if w == 0 {
        // sqrt(a) = a * rsqrt(a), refined; 0 and infinity give themselves.
        em.mvex(2, 1, 0, 0xCB, r, 0, Rm::Reg(a), 0, 0, None);
        let half = em.const_f32(0.5);
        let three_halves = em.const_f32(1.5);
        em.vmulp(e, a, Rm::Reg(r), 0);
        em.vmulp(e, e, Rm::Reg(r), 0);
        em.vmov(q, Rm::Const(three_halves), 0, 0);
        em.fma(0xBC, q, e, Rm::Const(half), 0);
        em.vmulp(r, r, Rm::Reg(q), 0);
        em.vmulp(q, a, Rm::Reg(r), 0);
        em.vmov(e, Rm::Reg(a), 0, 0);
        em.fma(0xBC, e, q, Rm::Reg(q), 0);
        em.vmulp(r, r, Rm::Const(half), 0);
        em.fma(0xB8, q, e, Rm::Reg(r), 0);
        let zero = em.const_f32(0.0);
        em.vcmp(kx, a, Rm::Const(zero), 0, 0, 0, 0);
        em.vmov(q, Rm::Reg(a), kx, 0);
        let inf = em.const_f32(f32::INFINITY);
        em.vcmp(kx, a, Rm::Const(inf), 0, 0, 0, 0);
        em.vmov(q, Rm::Reg(a), kx, 0);
        result = q;
    } else {
        let ex = em.temp().map_err(|e| refuse(insn, e))?;
        // r = a' = a with exponent 1023 + p, p the parity of E - 1023, so a' is in [1, 4)
        // and the shift back is even; ex = 2^((E - 1023 - p) / 2)
        em.dexp(ex, a);
        let one_dw = em.const_dw(1);
        em.i0f(0xFE, e, ex, Rm::Const(one_dw), 0, 0);
        em.i0f(0xDB, e, e, Rm::Const(one_dw), 0, 0); // p
        let c1023 = em.const_dw(1023);
        em.i0f(0xFE, e, e, Rm::Const(c1023), 0, 0); // 1023 + p
        em.dsetexp(r, a, e, q);
        let c2046 = em.const_dw(2046);
        em.i0f(0xFA, q, ex, Rm::Reg(e), 0, 0); // E - (1023 + p)
        em.i0f(0xFE, q, q, Rm::Const(c2046), 0, 0); // + 2046
        em.shift_imm(4, q, Rm::Reg(q), 1, 0); // / 2: the biased exponent of the scale
        em.dpow2(ex, q);
        // q = rsqrt(a') seeded through float, two Newton steps: y = 1.5y - 0.5 a' y^3
        em.mvex(1, 1, 1, 0x5A, e, 0, Rm::Reg(r), 0, 0, None);
        em.mvex(2, 1, 0, 0xCB, e, 0, Rm::Reg(e), 0, 0, None);
        em.mvex(1, 0, 0, 0x5A, q, 0, Rm::Reg(e), 0, 0, None);
        let half = em.const_f64(0.5);
        let mhalf = em.const_f64(-0.5);
        for _ in 0..2 {
            em.vmulp(e, r, Rm::Reg(q), 1);
            em.vmulp(e, e, Rm::Reg(q), 1);
            em.vmulp(e, e, Rm::Reg(q), 1);
            em.vmulp(e, e, Rm::Const(mhalf), 1); // -0.5 a' y^3
            em.fma(0xB8, q, q, Rm::Const(half), 1); // q = 0.5q + q
            em.mvex(1, 1, 1, 0x58, q, q, Rm::Reg(e), 0, 0, None); // + e
        }
        // s = a' y; s += (a' - s*s) * 0.5y; then scaled
        em.vmulp(e, r, Rm::Reg(q), 1);
        em.vmulp(q, q, Rm::Const(half), 1);
        em.fma(0xBC, r, e, Rm::Reg(e), 1); // r = a' - s*s
        em.fma(0xB8, e, r, Rm::Reg(q), 1); // e = r*q + e
        em.vmulp(e, e, Rm::Reg(ex), 1);
        let zero = dconst(&mut em, 0);
        let inf = dconst(&mut em, 0x7ff0_0000_0000_0000);
        em.vcmp(kx, a, Rm::Const(zero), 0, 0, 1, 0);
        em.vmov(e, Rm::Reg(a), kx, 1);
        em.vcmp(kx, a, Rm::Const(inf), 0, 0, 1, 0);
        em.vmov(e, Rm::Reg(a), kx, 1);
        em.vcmp(kx, a, Rm::Reg(a), 3, 0, 1, 0);
        em.vmov(e, Rm::Reg(a), kx, 1);
        result = e;
    }
    if scalar {
        let s1 = ev.vvvv;
        let k = em.mask_imm(1);
        em.vmov(d, Rm::Reg(result), k, w);
        if d != s1 {
            let k = em.mask_imm(if w == 0 { 0xe } else { 0x2 });
            em.vmov(d, Rm::Reg(s1), k, w);
        }
        em.zero_lanes(d, 0xfff0);
    } else {
        commit(&mut em, d, result, ev);
    }
    Ok(Rewrite::Thunk(em.finish()))
}

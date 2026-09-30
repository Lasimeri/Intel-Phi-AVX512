/* mvex-decode.c: read assembly on stdin and rewrite every `.byte` line
 * that encodes one MVEX or mask-register instruction into the macro line
 * of card/vpu/mvex.inc that encodes the same bytes, keeping the line's
 * comment; every other line passes through. Used once to turn the
 * generated card/vpu/vpu_matmul_kernel.S into card/vpu/kernels.S, and kept
 * for reading MVEX bytes back (a rewriter thunk, a worker dump):
 *
 *   tcc -run tools/mvex-decode.c < in.S > out.S
 *   printf '\t.byte 0x62, 0xf1, 0x79, 0x08, 0xef, 0xc0\n' | tcc -run tools/mvex-decode.c
 *
 * A line whose bytes it does not know is an error (exit 1, the line on
 * stderr): nothing is silently left as bytes. See mvex-decode.md. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One instruction row: map (P0 mm), pp, W, opcode, the ModRM.reg
 * extension for the /n forms (or -1), the shape, the macro name. */
enum shape { NDS, LOAD, STORE, RMI, RM2, SHIFT, CMP, PREFETCH };
struct row { int map, pp, w, opcode, ext; enum shape shape; const char *name; int imm; };
static const struct row rows[] = {
	{ 1, 0, 0, 0x58, -1, NDS, "VADDPS", 0 },     { 1, 0, 0, 0x5c, -1, NDS, "VSUBPS", 0 },
	{ 1, 0, 0, 0x59, -1, NDS, "VMULPS", 0 },     { 1, 1, 1, 0x58, -1, NDS, "VADDPD", 0 },
	{ 1, 1, 1, 0x5c, -1, NDS, "VSUBPD", 0 },     { 1, 1, 1, 0x59, -1, NDS, "VMULPD", 0 },
	{ 2, 1, 0, 0xb8, -1, NDS, "VFMADD231PS", 0 }, { 2, 1, 0, 0xa8, -1, NDS, "VFMADD213PS", 0 },
	{ 2, 1, 0, 0xbc, -1, NDS, "VFNMADD231PS", 0 }, { 2, 1, 0, 0xaa, -1, NDS, "VFMSUB213PS", 0 },
	{ 2, 1, 0, 0xba, -1, NDS, "VFMSUB231PS", 0 }, { 2, 1, 1, 0xb8, -1, NDS, "VFMADD231PD", 0 },
	{ 2, 1, 1, 0xa8, -1, NDS, "VFMADD213PD", 0 },
	{ 1, 1, 0, 0xfe, -1, NDS, "VPADDD", 0 },     { 1, 1, 0, 0xfa, -1, NDS, "VPSUBD", 0 },
	{ 1, 1, 0, 0xdb, -1, NDS, "VPANDD", 0 },     { 1, 1, 0, 0xdf, -1, NDS, "VPANDND", 0 },
	{ 1, 1, 0, 0xeb, -1, NDS, "VPORD", 0 },      { 1, 1, 0, 0xef, -1, NDS, "VPXORD", 0 },
	{ 2, 1, 0, 0x47, -1, NDS, "VPSLLVD", 0 },    { 2, 1, 0, 0x45, -1, NDS, "VPSRLVD", 0 },
	{ 2, 1, 0, 0x36, -1, NDS, "VPERMD", 0 },
	{ 1, 1, 0, 0x72, 6, SHIFT, "VPSLLD", 1 },     { 1, 1, 0, 0x72, 2, SHIFT, "VPSRLD", 1 },
	{ 1, 1, 0, 0x72, 4, SHIFT, "VPSRAD", 1 },
	{ 1, 0, 0, 0x28, -1, LOAD, "VMOVAPS", 0 },   { 1, 1, 1, 0x28, -1, LOAD, "VMOVAPD", 0 },
	{ 1, 1, 0, 0x6f, -1, LOAD, "VMOVDQA32", 0 },
	{ 2, 0, 0, 0xd0, -1, LOAD, "VLOADUNPACKLD", 0 }, { 2, 0, 0, 0xd4, -1, LOAD, "VLOADUNPACKHD", 0 },
	{ 2, 0, 0, 0xd1, -1, LOAD, "VLOADUNPACKLPS", 0 }, { 2, 0, 0, 0xd5, -1, LOAD, "VLOADUNPACKHPS", 0 },
	{ 2, 1, 0, 0x58, -1, LOAD, "VPBROADCASTD", 0 },
	{ 1, 0, 0, 0x29, -1, STORE, "VMOVAPS_ST", 0 }, { 1, 1, 1, 0x29, -1, STORE, "VMOVAPD_ST", 0 },
	{ 1, 1, 0, 0x7f, -1, STORE, "VMOVDQA32_ST", 0 },
	{ 1, 3, 0, 0x29, -1, STORE, "VMOVNRAPS_ST", 0 },
	{ 2, 1, 0, 0xd0, -1, STORE, "VPACKSTORELD", 0 }, { 2, 1, 0, 0xd4, -1, STORE, "VPACKSTOREHD", 0 },
	{ 3, 0, 0, 0xcb, -1, RMI, "VCVTFXPNTDQ2PS", 1 }, { 3, 1, 0, 0xcb, -1, RMI, "VCVTFXPNTPS2DQ", 1 },
	{ 3, 1, 0, 0x52, -1, RMI, "VRNDFXPNTPS", 1 },
	{ 2, 1, 0, 0xc8, -1, RM2, "VEXP223PS", 0 },   { 2, 1, 0, 0xca, -1, RM2, "VRCP23PS", 0 },
	{ 1, 0, 0, 0xc2, -1, CMP, "VCMPPS", 1 },     { 1, 1, 1, 0xc2, -1, CMP, "VCMPPD", 1 },
	{ 1, 0, 0, 0x18, 1, PREFETCH, "VPREFETCH0", 0 }, { 1, 0, 0, 0x18, 2, PREFETCH, "VPREFETCH1", 0 },
};

static const char *gpr[16] = { "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
	"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15" };
static const char *gpr32[8] = { "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi" };
static const char *cv[8] = { "CV_NONE", "CV_1TO16", "CV_4TO16", "CV_F16", "CV_U8", "CV_S8", "CV_U16", "CV_S16" };
static const char *sw[8] = { "SW_NONE", "SW_CDAB", "SW_BADC", "SW_DACB", "SW_AAAA", "SW_BBBB", "SW_CCCC", "SW_DDDD" };

static int fail(const char *what, const char *line)
{
	fprintf(stderr, "mvex-decode: %s: %s", what, line);
	return 1;
}

/* the optional named arguments */
static void tail(char *out, int sss, int mem, int k, int eh)
{
	if (sss) sprintf(out + strlen(out), ", conv=%s", mem ? cv[sss] : sw[sss]);
	if (k) sprintf(out + strlen(out), ", k=k%d", k);
	if (eh) strcat(out, ", eh=1");
}

/* Decode n bytes; on success write the macro line (without the comment)
 * into out and return the bytes consumed, else 0. */
static int decode(const unsigned char *b, int n, char *out)
{
	if (n >= 4 && b[0] == 0xc5) {
		/* two-byte VEX: the mask family, or delay */
		if (b[1] == 0xfa && b[2] == 0xae && b[3] == 0xf0) { strcpy(out, "KNC_DELAY_EAX"); return 4; }
		int vvvv = (~(b[1] >> 3)) & 0xf, reg = (b[3] >> 3) & 7, rm = b[3] & 7;
		if ((b[1] & 0x87) != 0x80 || (b[3] & 0xc0) != 0xc0) return 0;
		switch (b[2]) {
		case 0x92: if (vvvv) return 0; sprintf(out, "KMOV_KR k%d, %s", reg, gpr32[rm]); return 4;
		case 0x93: if (vvvv) return 0; sprintf(out, "KMOV_RK %s, k%d", gpr32[reg], rm); return 4;
		case 0x90: if (vvvv) return 0; sprintf(out, "KMOV_KK k%d, k%d", reg, rm); return 4;
		case 0x98: if (vvvv) return 0; sprintf(out, "KORTEST k%d, k%d", reg, rm); return 4;
		case 0x44: if (vvvv) return 0; sprintf(out, "KNOT k%d, k%d", reg, rm); return 4;
		case 0x41: sprintf(out, "KAND k%d, k%d, k%d", reg, vvvv, rm); return 4;
		case 0x42: sprintf(out, "KANDN k%d, k%d, k%d", reg, vvvv, rm); return 4;
		case 0x45: sprintf(out, "KOR k%d, k%d, k%d", reg, vvvv, rm); return 4;
		case 0x47: sprintf(out, "KXOR k%d, k%d, k%d", reg, vvvv, rm); return 4;
		case 0x46: sprintf(out, "KXNOR k%d, k%d, k%d", reg, vvvv, rm); return 4;
		default: return 0;
		}
	}
	if (n < 6 || b[0] != 0x62) return 0;
	int p0 = b[1], p1 = b[2], p2 = b[3], opcode = b[4], modrm = b[5];
	int map = p0 & 3, R = !(p0 >> 7 & 1), X = !(p0 >> 6 & 1), B = !(p0 >> 5 & 1), R2 = !(p0 >> 4 & 1);
	if (p0 & 0x0c) return 0;
	int w = p1 >> 7, vvvv = (~(p1 >> 3)) & 0xf, pp = p1 & 3;
	if (p1 & 4) return 0;
	int eh = p2 >> 7, sss = (p2 >> 4) & 7, V2 = !(p2 >> 3 & 1), k = p2 & 7;
	int mod = modrm >> 6, regf = (modrm >> 3) & 7, rmf = modrm & 7;
	int reg = R2 << 4 | R << 3 | regf, vfull = V2 << 4 | vvvv;
	int mem = mod == 2, rm = 0, base = 0, disp = 0, used = 6;
	if (mod == 3) { rm = X << 4 | B << 3 | rmf; }
	else if (mod == 2) { if (n < 10) return 0; base = B << 3 | rmf; if (X) return 0;
		disp = b[6] | b[7] << 8 | b[8] << 16 | (int)((unsigned)b[9] << 24); used = 10; }
	else return 0;
	const struct row *r = NULL;
	for (unsigned i = 0; i < sizeof rows / sizeof rows[0]; i++) {
		const struct row *c = &rows[i];
		if (c->map == map && c->pp == pp && c->w == w && c->opcode == opcode && (c->ext < 0 || c->ext == regf)) { r = c; break; }
	}
	if (!r) return 0;
	int imm = 0;
	if (r->imm) { if (n < used + 1) return 0; imm = b[used++]; }
	char m[64];
	if (mem) sprintf(m, "%s, %d", gpr[base], disp); else sprintf(m, "zmm%d", rm);
	switch (r->shape) {
	case NDS:
		if (r->pp == 3 && eh) return 0;
		sprintf(out, "%s zmm%d, zmm%d, %s", r->name, reg, vfull, m);
		tail(out, sss, mem, k, eh);
		return used;
	case LOAD:
		if (vfull) return 0;
		sprintf(out, "%s zmm%d, %s", r->name, reg, m);
		tail(out, sss, mem, k, eh);
		return used;
	case STORE:
		if (vfull || !mem) return 0;
		if (r->pp == 3) { /* the no-read stores: EH distinguishes them */
			if (sss || k) return 0;
			sprintf(out, "%s %s, zmm%d", eh ? "VMOVNRNGOAPS_ST" : "VMOVNRAPS_ST", m, reg);
			return used;
		}
		sprintf(out, "%s %s, zmm%d", r->name, m, reg);
		tail(out, sss, 1, k, eh);
		return used;
	case RMI:
		if (vfull) return 0;
		sprintf(out, "%s zmm%d, %s, %s%d", r->name, reg, m, imm == 0x50 ? "EXP_Q8_24 + " : "", imm == 0x50 ? 0 : imm);
		if (imm == 0x50) sprintf(out, "%s zmm%d, %s, EXP_Q8_24", r->name, reg, m);
		tail(out, sss, mem, k, eh);
		return used;
	case RM2:
		if (vfull) return 0;
		sprintf(out, "%s zmm%d, %s", r->name, reg, m);
		tail(out, sss, mem, k, eh);
		return used;
	case SHIFT:
		/* destination in vvvv, source in r/m, the extension consumed */
		if (sss || eh) return 0;
		sprintf(out, "%s zmm%d, %s, %d", r->name, vfull, m, imm);
		if (k) sprintf(out + strlen(out), ", k=k%d", k);
		return used;
	case CMP:
		if (reg > 7 || sss || eh) return 0;
		sprintf(out, "%s k%d, zmm%d, %s, %d", r->name, reg, vfull, m, imm);
		if (k) sprintf(out + strlen(out), ", k=k%d", k);
		return used;
	case PREFETCH:
		if (!mem || vfull || sss || k || eh || R || R2) return 0;
		sprintf(out, "%s %s", r->name, m);
		return used;
	}
	return 0;
}

int main(void)
{
	char line[4096], out[256];
	unsigned char bytes[32];
	while (fgets(line, sizeof line, stdin)) {
		const char *p = line;
		while (*p == ' ' || *p == '\t') p++;
		if (strncmp(p, ".byte", 5) != 0) { fputs(line, stdout); continue; }
		p += 5;
		int n = 0;
		for (;;) {
			while (*p == ' ' || *p == '\t' || *p == ',') p++;
			if (n == 32) return fail("more than 32 bytes", line);
			char *end;
			long v = strtol(p, &end, 0);
			if (end == p || v < 0 || v > 255) break;
			bytes[n++] = (unsigned char)v;
			p = end;
		}
		const char *comment = strchr(p, '#');
		int used = decode(bytes, n, out);
		if (used == 0 || used != n) return fail("unknown instruction", line);
		if (comment) {
			const char *e = comment + strlen(comment);
			while (e > comment && (e[-1] == '\n' || e[-1] == ' ' || e[-1] == '\t')) e--;
			printf("\t%s\t%.*s\n", out, (int)(e - comment), comment);
		} else {
			printf("\t%s\n", out);
		}
	}
	return 0;
}

# cdma.S: one card-owned channel of the SBOX DMA engine, programmed from
# user space on the card: the channel's registers through /dev/mem (the
# SBOX page, uncached), its descriptor ring, status word and landing area
# in one 2 MiB huge page whose physical address comes from
# /proc/self/pagemap, the host window's card address from the ring
# region's header. Copies run from the host window into the landing area,
# one 64-byte line of descriptors each (the copy, a status descriptor that
# writes the copy's sequence number, two NOPs); completion is read from
# the status word, never from the tail pointer. Every descriptor is
# checked against the window and the landing area before it is written.
# See cdma.md. Written for the card's scalar core: no SSE, no cmov, the
# fence is a locked add.

	.include "defs.inc"

	.set O_SYNC, 0x101000
	.set MAP_LOCKED, 0x2000
	.set MAP_POPULATE, 0x8000
	.set SYS_flock, 73
	.set LOCK_EX, 2
	.set LOCK_NB, 4

# The SBOX's DMA registers (SSDG 328207-002 2.1.8.2.1; the stack's
# docs/spec/sbox-registers.md): the page at SBOX + 0xA000 holds every
# channel's registers (0x40 apart) and DCR (0xA280). Only channel
# CD_CHAN's registers and DCR are written.
	.set SBOX_PHYS, 0x08007D0000
	.set DMA_PAGE_PHYS, SBOX_PHYS + 0xA000
	.set CD_CHAN, 7
	.set CH, 0x40 * CD_CHAN
	.set R_DCAR, CH + 0x00
	.set R_DHPR, CH + 0x04
	.set R_DTPR, CH + 0x08
	.set R_DRAR_LO, CH + 0x14
	.set R_DRAR_HI, CH + 0x18
	.set R_DSTAT, CH + 0x20
	.set R_DCHERR, CH + 0x2C
	.set R_DCHERRMSK, CH + 0x30
	.set R_DCR, 0x280
	.set DCR_BITS, 3 << (2 * CD_CHAN)	# bit 2n owner (1 host), bit 2n+1 enable
	.set DCR_ENABLE, 2 << (2 * CD_CHAN)
	.set DCAR_IM0, 1 << 24
	.set DCAR_IM1, 1 << 25

# The ring region the host writes at boot (the stack's phictl defs.inc,
# RING_REGION_BASE; kernel asm/knc_ring.h): magic "PHIR" at 0, the host
# window's card address at 32 and its size at 40.
	.set REGION_PHYS, 0x10000000
	.set REGION_MAGIC_V, 0x52494850
	.set REGION_HOSTMEM_ADDR, 32
	.set REGION_HOSTMEM_SIZE, 40

# The huge page: the ring (256 descriptors, 4 KiB, at the page's base,
# which the DRAR base alignment wants), the status word on its own line,
# the landing area from 8 KiB to the page's end.
	.set RING_N, 256
	.set CD_STATUS, 4096
	.set CD_LAND, 8192
	.globl CD_LAND_MAX
	.set CD_LAND_MAX, HUGE_BYTES - CD_LAND
	.set CD_LEN_MAX, 16383 * 64		# the length field: 14 bits of lines
	.set CD_IN_FLIGHT, RING_N / 4 - 1
	.set CD_TIMEOUT_TICKS, 2200000000	# 2 s at the card's 1.1 GHz counter
	.set CD_WAIT_DELAY, 32			# cycles between looks at the status word

	.bss
	.balign 64
	.globl cdma_ready, cdma_regs, cdma_page, cdma_page_phys, cdma_land
	.globl cdma_win_phys, cdma_win_bytes, cdma_seq, cdma_done, cdma_timeout
	.globl cdma_dcr_before, cdma_dcr_after, cdma_tail0, cdma_err
cdma_ready:	.skip 8		# 1 while the channel is up
cdma_regs:	.skip 8		# the register page, mapped uncached
cdma_page:	.skip 8		# the huge page
cdma_page_phys:	.skip 8	# its card physical address
cdma_land:	.skip 8		# the landing area (cdma_page + CD_LAND)
cdma_win_phys:	.skip 8	# the host window's card address
cdma_win_bytes:	.skip 8	# its size
cdma_head:	.skip 8		# the next descriptor index
cdma_seq:	.skip 8		# copies submitted
cdma_done:	.skip 8		# the last sequence number seen in the status word
cdma_timeout:	.skip 8	# ticks a wait may take (0: CD_TIMEOUT_TICKS)
cdma_dcr_before:	.skip 8
cdma_dcr_after:	.skip 8
cdma_tail0:	.skip 8		# the tail found at open
cdma_err:	.skip 8		# why the last open or wait failed (CE_*)
cdma_claimed:	.skip 8	# 1 once this process wrote the channel's DCR bits
cdma_lock_fd:	.skip 8	# the lock on CD_LOCK, kept open
	.globl cdma_nodoorbell
cdma_nodoorbell:	.skip 8	# test hook: descriptors written, head not advanced
pagemap_buf:	.skip 4096

	.text

# Error codes in cdma_err and as cdma_open's negative return.
	.set CE_DEVMEM, 1
	.set CE_REGION, 2
	.set CE_HUGE, 3
	.set CE_PAGEMAP, 4
	.set CE_BUSY, 5
	.set CE_DCR, 6
	.set CE_DRAIN, 7
	.set CE_NONCE, 8
	.set CE_TIMEOUT, 9
	.set CE_AHEAD, 10

# TICKS: the time-stamp counter in rax (clobbers rdx).
	.macro TICKS
	rdtsc
	shlq $32, %rdx
	orq %rdx, %rax
	.endm

# rd32 off / wr32 off: a channel register through the uncached page in rbx.
	.macro RD32 off
	movl \off(%rbx), %eax
	.endm
	.macro WR32 off
	movl %eax, \off(%rbx)
	.endm

# map_phys(rdi = physical address, rsi = bytes, edx = prot, ecx = fd) ->
# rax = the mapping or -errno.
map_phys:
	movq %rdi, %r9
	movl %ecx, %r8d
	xorl %edi, %edi
	movl $MAP_SHARED, %r10d
	SYS SYS_mmap
	ret

	.globl cdma_open
# cdma_open(rdi = a 64-byte line of the host window mapped by the caller,
# rsi = that line's window offset) -> rax 0 when the channel is up, else
# -CE_* (also in cdma_err; nothing is left enabled). The line is the
# nonce's: its first 8 bytes are overwritten.
cdma_open:
	pushq %rbx
	pushq %rbp
	pushq %r12
	pushq %r13
	pushq %r14
	pushq %r15
	subq $8, %rsp
	movq %rdi, %r12			# nonce line (mapped)
	movq %rsi, %r13			# its window offset
	movq $0, cdma_ready(%rip)
	movq $0, cdma_claimed(%rip)
	# one owner of the channel on the card: an exclusive lock on
	# CD_LOCK, held until the process ends (a dead owner frees it)
	cmpl $0, cdma_lock_fd(%rip)
	jne 10f
	leaq s_lock(%rip), %rdi
	movl $O_RDWR | O_CREAT, %esi
	movl $0600, %edx
	SYS SYS_open
	testq %rax, %rax
	js 95f
	movq %rax, %r14
	movl %r14d, %edi
	movl $LOCK_EX | LOCK_NB, %esi
	SYS SYS_flock
	testq %rax, %rax
	jz 11f
	movl %r14d, %edi
	SYS SYS_close
	jmp 95f
11:	movl %r14d, cdma_lock_fd(%rip)
10:	# /dev/mem
	leaq s_devmem(%rip), %rdi
	movl $O_RDWR | O_SYNC, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js 91f
	movq %rax, %r14			# fd
	# the ring region's header: the host window's card address and size
	movq $REGION_PHYS, %rdi
	movl $4096, %esi
	movl $PROT_READ, %edx
	movl %r14d, %ecx
	call map_phys
	cmpq $MAP_FAILED_MAX, %rax
	ja 922f
	movq %rax, %rdi
	movl (%rdi), %eax
	cmpl $REGION_MAGIC_V, %eax
	jne 921f
	movq REGION_HOSTMEM_ADDR(%rdi), %rax
	movq %rax, cdma_win_phys(%rip)
	movq REGION_HOSTMEM_SIZE(%rdi), %rax
	movq %rax, cdma_win_bytes(%rip)
	movl $4096, %esi
	SYS SYS_munmap
	movq cdma_win_phys(%rip), %rax
	testq %rax, %rax
	jz 922f
	testq $4095, %rax
	jnz 922f
	cmpq $0, cdma_win_bytes(%rip)
	je 922f
	cmpq %r13, cdma_win_bytes(%rip)
	jbe 922f
	# the register page
	movq $DMA_PAGE_PHYS, %rdi
	movl $4096, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl %r14d, %ecx
	call map_phys
	cmpq $MAP_FAILED_MAX, %rax
	ja 912f
	movq %rax, cdma_regs(%rip)
	movq %rax, %rbx
	movl %r14d, %edi
	SYS SYS_close
	# the huge page, populated and locked
	xorl %edi, %edi
	movl $HUGE_BYTES, %esi
	movl $PROT_READ | PROT_WRITE, %edx
	movl $MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | MAP_LOCKED, %r10d
	movq $-1, %r8
	xorl %r9d, %r9d
	SYS SYS_mmap
	cmpq $MAP_FAILED_MAX, %rax
	ja 93f
	movq %rax, cdma_page(%rip)
	leaq CD_LAND(%rax), %rcx
	movq %rcx, cdma_land(%rip)
	# its physical address: 512 pagemap entries, present and consecutive
	leaq s_pagemap(%rip), %rdi
	movl $O_RDONLY, %esi
	xorl %edx, %edx
	SYS SYS_open
	testq %rax, %rax
	js 94f
	movq %rax, %r14
	movl %r14d, %edi
	leaq pagemap_buf(%rip), %rsi
	movl $4096, %edx
	movq cdma_page(%rip), %r10
	shrq $12, %r10
	shlq $3, %r10
	SYS SYS_pread64
	movq %rax, %r15
	movl %r14d, %edi
	SYS SYS_close
	cmpq $4096, %r15
	jne 94f
	leaq pagemap_buf(%rip), %rsi
	movq (%rsi), %rax
	btq $63, %rax
	jnc 94f
	movabsq $(1 << 55) - 1, %rdx
	andq %rdx, %rax
	jz 94f				# a zero frame: no CAP_SYS_ADMIN
	testq $511, %rax
	jnz 94f
	movq %rax, %r8			# first frame
	xorl %ecx, %ecx
1:	movq (%rsi,%rcx,8), %rax
	btq $63, %rax
	jnc 94f
	andq %rdx, %rax
	leaq (%r8,%rcx), %r9
	cmpq %r9, %rax
	jne 94f
	incl %ecx
	cmpl $512, %ecx
	jb 1b
	shlq $12, %r8
	movq %r8, cdma_page_phys(%rip)
	movabsq $1 << 36, %rax		# the ring base is a 36-bit address
	cmpq %rax, %r8
	jae 94f
	# the channel: with the lock held, one left enabled was an earlier
	# owner's that died; its last copies drain before it is taken over
	TICKS
	movq %rax, %r15
5:	RD32 R_DCR
	movl %eax, %eax
	movq %rax, cdma_dcr_before(%rip)
	testl $DCR_ENABLE, %eax
	jz 2f
	RD32 R_DHPR
	movl %eax, %ecx
	RD32 R_DTPR
	cmpl %eax, %ecx
	je 2f
	TICKS
	subq %r15, %rax
	movl $CD_TIMEOUT_TICKS, %ecx
	cmpq %rcx, %rax
	ja 95f
	jmp 5b
2:	# owner card, disabled; the other channels' bits checked unchanged
	movq $1, cdma_claimed(%rip)
	movq cdma_dcr_before(%rip), %rax
	andl $~DCR_BITS, %eax
	movl %eax, %r15d
	WR32 R_DCR
	RD32 R_DCR
	cmpl %r15d, %eax
	jne 96f
	# interrupts masked (the card kernel has no handler), errors unmasked
	RD32 R_DCAR
	orl $DCAR_IM0 | DCAR_IM1, %eax
	WR32 R_DCAR
	xorl %eax, %eax
	WR32 R_DCHERRMSK
	# the ring: base, size, local (no SYS bit, no SMPT page)
	movq cdma_page_phys(%rip), %rax
	WR32 R_DRAR_LO
	movq cdma_page_phys(%rip), %rax
	shrq $32, %rax
	andl $0xf, %eax
	orl $RING_N << 4, %eax
	WR32 R_DRAR_HI
	# the status word cleared, the head on the line at or after the tail
	# (the slots between are NOPs: the page is zero)
	movq cdma_page(%rip), %rdi
	movq $0, CD_STATUS(%rdi)
	RD32 R_DTPR
	andl $RING_N - 1, %eax
	movq %rax, cdma_tail0(%rip)
	addl $3, %eax
	andl $-4, %eax
	andl $RING_N - 1, %eax
	movq %rax, cdma_head(%rip)
	FENCE
	WR32 R_DHPR
	# enabled
	movl %r15d, %eax
	orl $DCR_ENABLE, %eax
	movl %eax, %ebp
	WR32 R_DCR
	RD32 R_DCR
	cmpl %ebp, %eax
	jne 96f
	movl %eax, %eax
	movq %rax, cdma_dcr_after(%rip)
	# the NOPs drained: the tail reaches the head
	TICKS
	movq %rax, %r15
3:	RD32 R_DTPR
	andl $RING_N - 1, %eax
	cmpq cdma_head(%rip), %rax
	je 4f
	TICKS
	subq %r15, %rax
	movl $CD_TIMEOUT_TICKS, %ecx
	cmpq %rcx, %rax
	ja 97f
	jmp 3b
4:	movq $0, cdma_seq(%rip)
	movq $0, cdma_done(%rip)
	movq $1, cdma_ready(%rip)
	# the nonce: written through the caller's mapping, read back (the
	# posted write completed), copied by the engine, compared
	TICKS
	orq $1, %rax
	movq %rax, %r15
	movq %r15, (%r12)
	movq (%r12), %rax
	cmpq %r15, %rax
	jne 98f
	movq cdma_land(%rip), %rdi
	movq $0, (%rdi)
	movq %r13, %rdi
	andq $-64, %rdi
	xorl %esi, %esi
	movl $64, %edx
	call cdma_submit
	testq %rax, %rax
	jz 98f
	movq %rax, %rdi
	call cdma_wait
	testl %eax, %eax
	jz 98f
	movq cdma_land(%rip), %rdi
	movq %r13, %rax
	andl $63, %eax
	cmpq %r15, (%rdi,%rax)
	jne 98f
	movq $0, cdma_err(%rip)
	xorl %eax, %eax
	jmp 99f
91:	movl $CE_DEVMEM, %eax
	jmp 90f
921:	movl $4096, %esi		# rdi still the header's mapping
	SYS SYS_munmap
922:	movl $CE_REGION, %eax
	jmp 89f
912:	movl $CE_DEVMEM, %eax
89:	movq %rax, %r15			# /dev/mem is open (r14): closed first
	movl %r14d, %edi
	SYS SYS_close
	movq %r15, %rax
	jmp 90f
93:	movl $CE_HUGE, %eax
	jmp 90f
94:	movl $CE_PAGEMAP, %eax
	jmp 90f
95:	movl $CE_BUSY, %eax
	jmp 90f
96:	movl $CE_DCR, %eax
	jmp 90f
97:	movl $CE_DRAIN, %eax
	jmp 90f
98:	movl $CE_NONCE, %eax
90:	movq %rax, cdma_err(%rip)
	call cdma_close
	movq cdma_err(%rip), %rax
	negq %rax
99:	addq $8, %rsp
	popq %r15
	popq %r14
	popq %r13
	popq %r12
	popq %rbp
	popq %rbx
	ret

	.globl cdma_submit
# cdma_submit(rdi = window offset, rsi = landing offset, rdx = bytes) ->
# rax = the copy's sequence number, or 0 when refused: the channel down,
# anything not a multiple of 64, the source outside the window, the
# destination outside the landing area, or more than the length field
# holds. Waits only while the ring is full. Clobbers rcx, rdx, rsi, rdi,
# r8 to r11.
cdma_submit:
	cmpq $0, cdma_ready(%rip)
	je 90f
	movq %rdi, %rax
	orq %rsi, %rax
	orq %rdx, %rax
	testq $63, %rax
	jnz 90f
	testq %rdx, %rdx
	jz 90f
	cmpq $CD_LEN_MAX, %rdx
	ja 90f
	movq cdma_win_bytes(%rip), %rax
	cmpq %rax, %rdi
	jae 90f
	subq %rdi, %rax
	cmpq %rax, %rdx
	ja 90f
	cmpq $CD_LAND_MAX, %rsi
	jae 90f
	movq $CD_LAND_MAX, %rax
	subq %rsi, %rax
	cmpq %rax, %rdx
	ja 90f
	movq %rdx, %r10			# bytes
	# room in the ring
1:	movq cdma_seq(%rip), %rax
	subq cdma_done(%rip), %rax
	cmpq $CD_IN_FLIGHT, %rax
	jb 2f
	pushq %rdi
	pushq %rsi
	pushq %r10
	call cdma_poll
	popq %r10
	popq %rsi
	popq %rdi
	testq %rax, %rax
	js 90f
	jmp 1b
2:	movq cdma_seq(%rip), %r11
	incq %r11			# seq
	movq cdma_head(%rip), %r8
	shlq $4, %r8
	addq cdma_page(%rip), %r8	# the line of descriptors
	# memcpy: src[39:0] | lines << 46; dst[39:0] | 1 << 60
	movq cdma_win_phys(%rip), %rax
	addq %rdi, %rax
	movq %r10, %rcx
	shrq $6, %rcx
	shlq $46, %rcx
	orq %rcx, %rax
	movq %rax, 0(%r8)
	movq cdma_page_phys(%rip), %rax
	addq $CD_LAND, %rax
	addq %rsi, %rax
	movabsq $1 << 60, %rcx
	orq %rcx, %rax
	movq %rax, 8(%r8)
	# status: the sequence number into the status word
	movq %r11, 16(%r8)
	movq cdma_page_phys(%rip), %rax
	addq $CD_STATUS, %rax
	movabsq $2 << 60, %rcx
	orq %rcx, %rax
	movq %rax, 24(%r8)
	movq $0, 32(%r8)
	movq $0, 40(%r8)
	movq $0, 48(%r8)
	movq $0, 56(%r8)
	FENCE
	movq %r11, cdma_seq(%rip)
	cmpq $0, cdma_nodoorbell(%rip)	# the bench's negative control
	jne 3f
	movq cdma_head(%rip), %rax
	addq $4, %rax
	andq $RING_N - 1, %rax
	movq %rax, cdma_head(%rip)
	movq cdma_regs(%rip), %rcx
	movl %eax, R_DHPR(%rcx)
3:	movq %r11, %rax
	ret
90:	xorl %eax, %eax
	ret

	.globl cdma_poll
# cdma_poll -> rax = the last completed sequence number (the status word),
# or -1 when the word is ahead of what was submitted (the channel then
# closed). Clobbers rcx, rdx.
cdma_poll:
	movq cdma_page(%rip), %rcx
	movq CD_STATUS(%rcx), %rax
	cmpq cdma_seq(%rip), %rax
	ja 8f
	cmpq cdma_done(%rip), %rax
	jbe 1f
	movq %rax, cdma_done(%rip)
1:	movq cdma_done(%rip), %rax
	ret
8:	movq $CE_AHEAD, cdma_err(%rip)
	pushq %rdi
	pushq %rsi
	pushq %r8
	pushq %r9
	pushq %r10
	pushq %r11
	call cdma_close
	popq %r11
	popq %r10
	popq %r9
	popq %r8
	popq %rsi
	popq %rdi
	movq $-1, %rax
	ret

	.globl cdma_wait
# cdma_wait(rdi = seq) -> eax 1 when that copy is in the landing area, 0
# when it did not complete within the timeout (cdma_timeout ticks, or 2 s)
# or the channel failed: the channel is then closed and cdma_err set.
# The looks at the status word are CD_WAIT_DELAY cycles apart (delay: no
# issue slots taken from the other threads of this core). Clobbers rcx,
# rdx, rsi, rdi, r8.
cdma_wait:
	movq %rdi, %rsi
	movq cdma_page(%rip), %rcx
	movq CD_STATUS(%rcx), %rax
	cmpq %rsi, %rax
	jb 1f
	cmpq cdma_seq(%rip), %rax
	ja 1f
	movq %rax, cdma_done(%rip)
	movl $1, %eax
	ret
1:	TICKS
	movq %rax, %r8
2:	cmpq $0, cdma_ready(%rip)
	je 8f
	movl $CD_WAIT_DELAY, %eax
	DELAY_EAX
	call cdma_poll
	testq %rax, %rax
	js 8f
	cmpq %rsi, %rax
	jae 7f
	TICKS
	subq %r8, %rax
	movq cdma_timeout(%rip), %rdx
	testq %rdx, %rdx
	jnz 3f
	movl $CD_TIMEOUT_TICKS, %edx
3:	cmpq %rdx, %rax
	ja 6f
	jmp 2b
6:	movq $CE_TIMEOUT, cdma_err(%rip)
	pushq %rsi
	call cdma_close
	popq %rsi
8:	xorl %eax, %eax
	ret
7:	movl $1, %eax
	ret

	.globl cdma_close
# cdma_close: the channel disabled (left card-owned) when this process
# claimed it, cdma_ready 0. The
# huge page stays mapped: a copy still in flight lands in memory this
# process owns.
cdma_close:
	pushq %rbx
	movq $0, cdma_ready(%rip)
	cmpq $0, cdma_claimed(%rip)
	je 9f
	movq cdma_regs(%rip), %rbx
	testq %rbx, %rbx
	jz 9f
	RD32 R_DCR
	testl $DCR_ENABLE, %eax
	jz 9f
	andl $~DCR_ENABLE, %eax
	WR32 R_DCR
9:	popq %rbx
	ret

	.globl cdma_report
# cdma_report(edi = descriptor): one line with the channel's registers
# and this side's counters ("cdma: channel 7 ...").
cdma_report:
	pushq %rbx
	pushq %r12
	pushq %r13
	movl %edi, %r12d
	movq cdma_regs(%rip), %rbx
	call l_begin
	leaq s_rep1(%rip), %rsi
	call w_str
	movl $CD_CHAN, %edi
	call w_dec
	testq %rbx, %rbx
	jz 1f
	leaq s_rep_dcr(%rip), %rsi
	call w_str
	RD32 R_DCR
	movl %eax, %edi
	call w_hex0x
	leaq s_rep_head(%rip), %rsi
	call w_str
	RD32 R_DHPR
	movl %eax, %edi
	call w_dec
	leaq s_rep_tail(%rip), %rsi
	call w_str
	RD32 R_DTPR
	movl %eax, %edi
	call w_dec
	leaq s_rep_dstat(%rip), %rsi
	call w_str
	RD32 R_DSTAT
	movl %eax, %edi
	call w_hex0x
	leaq s_rep_err(%rip), %rsi
	call w_str
	RD32 R_DCHERR
	movl %eax, %edi
	call w_hex0x
	leaq s_rep_drar(%rip), %rsi
	call w_str
	RD32 R_DRAR_HI
	movl %eax, %edi
	call w_hex0x
	movb $'/', %al
	call w_byte
	RD32 R_DRAR_LO
	movl %eax, %edi
	call w_hex0x
1:	leaq s_rep_seq(%rip), %rsi
	call w_str
	movq cdma_seq(%rip), %rdi
	call w_dec
	leaq s_rep_done(%rip), %rsi
	call w_str
	movq cdma_page(%rip), %rax
	testq %rax, %rax
	jz 2f
	movq CD_STATUS(%rax), %rdi
	call w_dec
2:	leaq s_rep_errc(%rip), %rsi
	call w_str
	movq cdma_err(%rip), %rdi
	call w_dec
	movl %r12d, %edi
	call l_end
	popq %r13
	popq %r12
	popq %rbx
	ret

	.section .rodata
s_devmem:	.asciz "/dev/mem"
s_pagemap:	.asciz "/proc/self/pagemap"
s_lock:	.asciz "/tmp/phi-cdma.lock"
s_rep1:	.asciz "cdma: channel "
s_rep_dcr:	.asciz ": DCR "
s_rep_head:	.asciz ", head "
s_rep_tail:	.asciz ", tail "
s_rep_dstat:	.asciz ", DSTAT "
s_rep_err:	.asciz ", DCHERR "
s_rep_drar:	.asciz ", DRAR "
s_rep_seq:	.asciz "; submitted "
s_rep_done:	.asciz ", status word "
s_rep_errc:	.asciz ", error "

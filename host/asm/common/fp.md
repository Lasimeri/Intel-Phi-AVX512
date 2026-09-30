# fp.S

Floating point on the host, where Rust had it: the text forms the
backend's lines use, the roundings and conversions in the order Rust
performed them, the float16 conversion of the activations, the CPUID
check for it, a decimal parser for the environment. SSE2 and SSE4.1
throughout (`roundsd`); AVX and F16C only in `to_f16`, behind
`have_f16c`.

## Conversions

| routine | arguments | result |
| --- | --- | --- |
| `f64_of_u64` | rdi | xmm0 the value as a double, round to nearest (Rust's `as f64`; values of 2^63 and up halved, converted and doubled) |
| `u64_of_f64` | xmm0 | rax Rust's `as u64`: truncated toward zero, 0 below zero or NaN, saturated above 2^64 |
| `f64_round` | xmm0 | rounded half away from zero (Rust's `f64::round`): `roundsd` toward zero, then one added with the value's sign when the fraction is at least a half |
| `f64_clamp` | xmm0, xmm1 lo, xmm2 hi | Rust's `clamp` |
| `f64_secs` | rdi ns | Rust's `Duration::as_secs_f64`: the whole seconds as a double plus the nanoseconds as a double over 1e9, two roundings in that order |
| `parse_f64` | rdi | xmm0 a decimal such as `0.75`, `12`, `-1.5`; eax 1 when the whole string parsed. Up to eighteen digits become an integer divided by a power of ten: one correctly rounded operation, which is what a short decimal reads as in Rust's parser too |

## Text

| routine | arguments | writes |
| --- | --- | --- |
| `w_f64_fixed` | xmm0, esi decimals | Rust's `{:.N}`: the exact decimal expansion rounded half to even. The double is taken apart into mantissa and exponent; a value with a non-negative exponent is an integer (a 128-bit product, printed by `w_u128_dec`); otherwise the mantissa is scaled by 10^N and shifted, the dropped bits deciding the rounding exactly. A negative value (a negative zero included) leads with `-`; `inf` and `NaN` as Rust prints them |
| `w_f64_signed` | xmm0, esi | Rust's `{:+.N}`: `+` before a non-negative value |
| `w_f32_exp` | xmm0 (a float) | Rust's `{:e}` of an f32: the shortest digits that read back to the same float, as `d.ddde<exp>`. Exact for finite values from 1 up to 1e17 (the activation magnitudes it is used for: a value past 65504 is what sends a multiply as float32); outside that range the fixed form without decimals is written, which is at least the right number. The shortest form is found by trying 1 to 9 digits and testing each candidate against the float's rounding interval, the bounds excluded for an odd mantissa |

Checked against a Rust program printing the same values with the same
format strings: 47 of 48 lines identical, the one difference 3.4e38
(outside the documented range).

## Float16

`have_f16c`: eax 1 when CPUID reports OSXSAVE, AVX and F16C and
`xgetbv` says the OS keeps the AVX state (Rust's
`is_x86_feature_detected!("f16c") && ("avx")`).

`to_f16(rdi src, rsi dst, rdx n)`: n floats to halves by `vcvtps2ph`
(round to nearest even), eight at a time then one by one, and the
largest magnitude seen returned in xmm0 as a float: a half stops at
65504 and anything larger becomes infinity, which the caller compares
with `F16_MAX` to send the multiply as float32 instead. The eight-lane
maximum is folded as Rust folded its lanes, so the same value results.
Ends with `vzeroupper`.

## Constants

`c_1e9`, `c_1e6`, `c_1e3`, `c_100`, `c_mib` (1048576.0), the power
of ten table for the parser; all local to the file (backend.S carries
its own copies, `k_*`, exported to ffn.S).

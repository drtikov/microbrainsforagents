//! Zero-allocation 2-bit packed ternary integer add/subtract matvec kernels.
//!
//! Encoding matches `BitNetQuantizer.pack_ternary_2bit` in `bitnet_ternary_engine.py`:
//! - 4 trits per `u8` byte in LSB-first order: bits `0..1`, `2..3`, `4..5`, `6..7`
//! - 2-bit code mapping: `0 => -1`, `1 => 0`, `2 => +1`, `3 => 0`
//! - Matrix `(out_features, in_features)` flattened row-major and padded at end to 4 trits.
//!
//! Pure `core` math: safe on native targets, embedded `no_std`, and `wasm32`.

#[inline(always)]
pub fn decode_trit_i32(code: u8) -> i32 {
    match code & 0x03 {
        0 => -1,
        2 => 1,
        _ => 0,
    }
}

#[inline(always)]
fn apply_trit_i32(acc: i32, code: u8, x: i32) -> i32 {
    match code & 0x03 {
        0 => acc - x,
        2 => acc + x,
        _ => acc,
    }
}

/// Pack ternary `{-1, 0, +1}` slice into `packed_out` (4 trits/byte, padding `1` => `0`).
/// Returns the number of bytes written, or `0` if `packed_out` is too small.
pub fn pack_ternary_2bit_slice(ternary: &[i8], packed_out: &mut [u8]) -> usize {
    let needed = (ternary.len() + 3) / 4;
    if packed_out.len() < needed {
        return 0;
    }
    let mut byte_idx = 0usize;
    let mut i = 0usize;
    while i < ternary.len() {
        let mut byte_val = 0u8;
        let mut k = 0usize;
        while k < 4 {
            let code = if i + k < ternary.len() {
                let v = ternary[i + k].clamp(-1, 1);
                (v + 1) as u8
            } else {
                1u8 // padding represents 0
            };
            byte_val |= (code & 0x03) << (2 * k);
            k += 1;
        }
        packed_out[byte_idx] = byte_val;
        byte_idx += 1;
        i += 4;
    }
    needed
}

/// Zero-allocation 2-bit packed ternary matvec:
/// `out[r] = sum_{c=0..in_features} W[r, c] * (x_q[c] as i32)`
/// using integer additions/subtractions only.
///
/// Returns `true` on success, `false` if any input/output buffer is undersized.
pub fn ternary_packed_matvec_i8(
    packed_weights: &[u8],
    x_q: &[i8],
    out_features: usize,
    in_features: usize,
    out: &mut [i32],
) -> bool {
    if out.len() < out_features || x_q.len() < in_features {
        return false;
    }
    let total_trits = match out_features.checked_mul(in_features) {
        Some(v) => v,
        None => return false,
    };
    let required_bytes = (total_trits + 3) / 4;
    if packed_weights.len() < required_bytes {
        return false;
    }
    if out_features == 0 || in_features == 0 {
        let mut r = 0usize;
        while r < out_features {
            out[r] = 0;
            r += 1;
        }
        return true;
    }

    if in_features % 4 == 0 {
        let bytes_per_row = in_features / 4;
        let mut r = 0usize;
        while r < out_features {
            let row_bytes = &packed_weights[r * bytes_per_row..(r + 1) * bytes_per_row];
            let mut acc0: i32 = 0;
            let mut acc1: i32 = 0;
            let mut acc2: i32 = 0;
            let mut acc3: i32 = 0;
            let mut g = 0usize;
            while g < bytes_per_row {
                let b = row_bytes[g];
                let base = g * 4;
                acc0 = apply_trit_i32(acc0, b, x_q[base] as i32);
                acc1 = apply_trit_i32(acc1, b >> 2, x_q[base + 1] as i32);
                acc2 = apply_trit_i32(acc2, b >> 4, x_q[base + 2] as i32);
                acc3 = apply_trit_i32(acc3, b >> 6, x_q[base + 3] as i32);
                g += 1;
            }
            out[r] = acc0 + acc1 + acc2 + acc3;
            r += 1;
        }
    } else {
        let mut r = 0usize;
        while r < out_features {
            let start_trit = r * in_features;
            let mut c = 0usize;
            let mut acc: i32 = 0;

            // Unaligned prefix until trit index is 4-aligned
            while c < in_features && ((start_trit + c) & 3) != 0 {
                let flat = start_trit + c;
                let b = packed_weights[flat >> 2];
                let shift = (flat & 3) * 2;
                acc = apply_trit_i32(acc, b >> shift, x_q[c] as i32);
                c += 1;
            }

            // Aligned 4-trit interior
            while c + 4 <= in_features {
                let flat = start_trit + c;
                let b = packed_weights[flat >> 2];
                acc = apply_trit_i32(acc, b, x_q[c] as i32);
                acc = apply_trit_i32(acc, b >> 2, x_q[c + 1] as i32);
                acc = apply_trit_i32(acc, b >> 4, x_q[c + 2] as i32);
                acc = apply_trit_i32(acc, b >> 6, x_q[c + 3] as i32);
                c += 4;
            }

            // Unaligned tail
            while c < in_features {
                let flat = start_trit + c;
                let b = packed_weights[flat >> 2];
                let shift = (flat & 3) * 2;
                acc = apply_trit_i32(acc, b >> shift, x_q[c] as i32);
                c += 1;
            }

            out[r] = acc;
            r += 1;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_matvec_i32(w: &[i8], x_q: &[i8], out_features: usize, in_features: usize) -> [i32; 32] {
        let mut res = [0i32; 32];
        for r in 0..out_features {
            let mut sum = 0i32;
            for c in 0..in_features {
                sum += (w[r * in_features + c] as i32) * (x_q[c] as i32);
            }
            res[r] = sum;
        }
        res
    }

    #[test]
    fn packed_matvec_matches_naive_int32_aligned_and_unaligned() {
        // Case 1: aligned in_features = 8, out_features = 4
        let w_aligned: [i8; 32] = [
            -1, 0, 1, -1, 1, 1, 0, -1,
             0, 1, 0,  1, -1, 0, 1, 0,
             1, 1, -1, 0, 0, -1, 1, 1,
            -1, -1, 0, 1, 1, 0, 0, -1,
        ];
        let x_aligned: [i8; 8] = [12, -5, 127, -127, 0, 44, -19, 3];
        let mut packed_a = [0u8; 8];
        assert_eq!(pack_ternary_2bit_slice(&w_aligned, &mut packed_a), 8);

        let mut out_a = [0i32; 4];
        assert!(ternary_packed_matvec_i8(&packed_a, &x_aligned, 4, 8, &mut out_a));
        let ref_a = naive_matvec_i32(&w_aligned, &x_aligned, 4, 8);
        assert_eq!(&out_a[..4], &ref_a[..4]);

        // Case 2: unaligned in_features = 5, out_features = 3 (15 trits -> 4 bytes)
        let w_unaligned: [i8; 15] = [
            -1, 1, 0, -1, 1,
             0, -1, 1, 1, 0,
             1, 0, -1, 1, -1,
        ];
        let x_unaligned: [i8; 5] = [7, -11, 25, -4, 9];
        let mut packed_u = [0u8; 4];
        assert_eq!(pack_ternary_2bit_slice(&w_unaligned, &mut packed_u), 4);

        let mut out_u = [0i32; 3];
        assert!(ternary_packed_matvec_i8(&packed_u, &x_unaligned, 3, 5, &mut out_u));
        let ref_u = naive_matvec_i32(&w_unaligned, &x_unaligned, 3, 5);
        assert_eq!(&out_u[..3], &ref_u[..3]);
    }
}

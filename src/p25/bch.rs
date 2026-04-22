//! BCH(63,16,23) decoder for the P25 NID codeword.
//!
//! Ported from OP25 `op25/gr-op25_repeater/lib/bch.cc` (Copyright 2010, KA1RBI).
//! Returns the number of errors corrected, or a negative value on failure.

const BCH_GF_EXP: [i32; 64] = [
    1, 2, 4, 8, 16, 32, 3, 6, 12, 24, 48, 35, 5, 10, 20, 40, 19, 38, 15, 30, 60, 59, 53, 41, 17,
    34, 7, 14, 28, 56, 51, 37, 9, 18, 36, 11, 22, 44, 27, 54, 47, 29, 58, 55, 45, 25, 50, 39, 13,
    26, 52, 43, 21, 42, 23, 46, 31, 62, 63, 61, 57, 49, 33, 0,
];

const BCH_GF_LOG: [i32; 64] = [
    -1, 0, 1, 6, 2, 12, 7, 26, 3, 32, 13, 35, 8, 48, 27, 18, 4, 24, 33, 16, 14, 52, 36, 54, 9, 45,
    49, 38, 28, 41, 19, 56, 5, 62, 25, 11, 34, 31, 17, 47, 15, 23, 53, 51, 37, 44, 55, 40, 10, 61,
    46, 30, 50, 22, 39, 43, 29, 60, 42, 21, 20, 59, 57, 58,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BchResult {
    NoErrors,
    Corrected(u32),
    Uncorrectable,
}

/// `codeword` is 64 bits, LSB at index 0, containing the BCH(63,16,23) codeword
/// (only indices 0..=62 are meaningful; bit 63 is not used by the code and is
/// preserved). The vector is updated in place with the corrected bits on success.
pub fn bch_decode(codeword: &mut [bool; 64]) -> BchResult {
    let mut s = [0i32; 23];
    let mut syn_error = false;

    for i in 1..=22 {
        let mut acc = 0i32;
        for j in 0..=62 {
            if codeword[j] {
                acc ^= BCH_GF_EXP[(i * j) % 63];
            }
        }
        if acc != 0 {
            syn_error = true;
        }
        s[i] = BCH_GF_LOG[acc as usize];
    }

    if !syn_error {
        return BchResult::NoErrors;
    }

    // Berlekamp–Massey (polynomial-index form, from KA1RBI)
    let mut elp = [[0i32; 22]; 24];
    let mut d = [0i32; 23];
    let mut l = [0i32; 24];
    let mut ulu = [0i32; 24];

    l[0] = 0;
    ulu[0] = -1;
    d[0] = 0;
    elp[0][0] = 0;
    l[1] = 0;
    ulu[1] = 0;
    d[1] = s[1];
    elp[1][0] = 1;
    for e in elp[0].iter_mut().skip(1).take(21) {
        *e = -1;
    }
    for e in elp[1].iter_mut().skip(1).take(21) {
        *e = 0;
    }

    let mut u: usize = 0;
    loop {
        u += 1;
        if d[u] == -1 {
            l[u + 1] = l[u];
            for i in 0..=(l[u] as usize) {
                elp[u + 1][i] = elp[u][i];
                elp[u][i] = BCH_GF_LOG[elp[u][i] as usize];
            }
        } else {
            let mut q = (u as i32) - 1;
            while q > 0 && d[q as usize] == -1 {
                q -= 1;
            }
            if q > 0 {
                let mut j = q;
                loop {
                    j -= 1;
                    if d[j as usize] != -1 && ulu[q as usize] < ulu[j as usize] {
                        q = j;
                    }
                    if j <= 0 {
                        break;
                    }
                }
            }
            let q_usize = q as usize;
            l[u + 1] = if l[u] > l[q_usize] + (u as i32) - q {
                l[u]
            } else {
                l[q_usize] + (u as i32) - q
            };
            for e in elp[u + 1].iter_mut().take(22) {
                *e = 0;
            }
            for i in 0..=(l[q_usize] as usize) {
                if elp[q_usize][i] != -1 {
                    let idx = (d[u] + 63 - d[q_usize] + elp[q_usize][i]).rem_euclid(63);
                    let shifted = i + (u - q_usize);
                    if shifted < 22 {
                        elp[u + 1][shifted] = BCH_GF_EXP[idx as usize];
                    }
                }
            }
            for i in 0..=(l[u] as usize) {
                elp[u + 1][i] ^= elp[u][i];
                elp[u][i] = BCH_GF_LOG[elp[u][i] as usize];
            }
        }
        ulu[u + 1] = (u as i32) - l[u + 1];
        if u < 22 {
            let sym = s[u + 1];
            d[u + 1] = if sym != -1 { BCH_GF_EXP[sym as usize] } else { 0 };
            for i in 1..=(l[u + 1] as usize) {
                if s[u + 1 - i] != -1 && elp[u + 1][i] != 0 {
                    let idx =
                        (s[u + 1 - i] + BCH_GF_LOG[elp[u + 1][i] as usize]).rem_euclid(63);
                    d[u + 1] ^= BCH_GF_EXP[idx as usize];
                }
            }
            d[u + 1] = BCH_GF_LOG[d[u + 1] as usize];
        }
        if u >= 22 || l[u + 1] > 11 {
            break;
        }
    }

    u += 1;
    if l[u] > 11 {
        return BchResult::Uncorrectable;
    }
    for i in 0..=(l[u] as usize) {
        elp[u][i] = BCH_GF_LOG[elp[u][i] as usize];
    }

    let mut reg = [0i32; 12];
    let ll = l[u] as usize;
    reg[1..=ll].copy_from_slice(&elp[u][1..=ll]);
    let mut locn = [0i32; 12];
    let mut count: usize = 0;
    for i in 1..=63 {
        let mut q = 1i32;
        for j in 1..=(l[u] as usize) {
            if reg[j] != -1 {
                reg[j] = (reg[j] + j as i32) % 63;
                q ^= BCH_GF_EXP[reg[j] as usize];
            }
        }
        if q == 0 {
            if count >= locn.len() {
                return BchResult::Uncorrectable;
            }
            locn[count] = 63 - i;
            count += 1;
        }
    }
    if count != l[u] as usize {
        return BchResult::Uncorrectable;
    }
    for &loc in locn.iter().take(count) {
        codeword[loc as usize] ^= true;
    }
    BchResult::Corrected(count as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_zero_codeword() {
        let mut cw = [false; 64];
        assert_eq!(bch_decode(&mut cw), BchResult::NoErrors);
    }

    #[test]
    fn recovers_single_bit_error() {
        let mut cw = [false; 64];
        cw[5] = true;
        match bch_decode(&mut cw) {
            BchResult::Corrected(n) => {
                assert_eq!(n, 1);
                assert!(!cw[5], "bit should have been corrected back to 0");
            }
            other => panic!("expected single-bit correction, got {other:?}"),
        }
    }

    #[test]
    fn fails_when_errors_too_many() {
        let mut cw = [false; 64];
        for b in cw.iter_mut().take(20) {
            *b = true;
        }
        let result = bch_decode(&mut cw);
        assert!(matches!(result, BchResult::Uncorrectable), "got {result:?}");
    }
}

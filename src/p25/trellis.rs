//! P25 1/2-rate trellis decoder for TSBK / PDU blocks.
//!
//! Ported from OP25 `block_deinterleave()` in `p25p1_fdma.cc`
//! (originally from Wireshark, Copyright 2008 Michael Ossmann).
//!
//! Input: a 196-bit block (as a slice of `bool`).
//! Output: 12 bytes (96 bits) of decoded data.

const DEINTERLEAVE_TB: [u16; 196] = [
    0, 1, 2, 3, 52, 53, 54, 55, 100, 101, 102, 103, 148, 149, 150, 151, 4, 5, 6, 7, 56, 57, 58, 59,
    104, 105, 106, 107, 152, 153, 154, 155, 8, 9, 10, 11, 60, 61, 62, 63, 108, 109, 110, 111, 156,
    157, 158, 159, 12, 13, 14, 15, 64, 65, 66, 67, 112, 113, 114, 115, 160, 161, 162, 163, 16, 17,
    18, 19, 68, 69, 70, 71, 116, 117, 118, 119, 164, 165, 166, 167, 20, 21, 22, 23, 72, 73, 74, 75,
    120, 121, 122, 123, 168, 169, 170, 171, 24, 25, 26, 27, 76, 77, 78, 79, 124, 125, 126, 127,
    172, 173, 174, 175, 28, 29, 30, 31, 80, 81, 82, 83, 128, 129, 130, 131, 176, 177, 178, 179, 32,
    33, 34, 35, 84, 85, 86, 87, 132, 133, 134, 135, 180, 181, 182, 183, 36, 37, 38, 39, 88, 89, 90,
    91, 136, 137, 138, 139, 184, 185, 186, 187, 40, 41, 42, 43, 92, 93, 94, 95, 140, 141, 142, 143,
    188, 189, 190, 191, 44, 45, 46, 47, 96, 97, 98, 99, 144, 145, 146, 147, 192, 193, 194, 195, 48,
    49, 50, 51,
];

const NEXT_WORDS: [[u8; 4]; 4] = [
    [0x2, 0xC, 0x1, 0xF],
    [0xE, 0x0, 0xD, 0x3],
    [0x9, 0x7, 0xA, 0x4],
    [0x5, 0xB, 0x6, 0x8],
];

/// Decodes a single 196-bit trellis-coded block into 12 bytes.
/// Returns `None` if a unique minimum Hamming distance cannot be chosen at some step.
pub fn trellis_decode_block(bits: &[bool; 196]) -> Option<[u8; 12]> {
    let mut buf = [0u8; 12];
    let mut state: usize = 0;

    let mut b = 0usize;
    while b < 196 {
        let codeword = ((bits[DEINTERLEAVE_TB[b] as usize] as u8) << 3)
            | ((bits[DEINTERLEAVE_TB[b + 1] as usize] as u8) << 2)
            | ((bits[DEINTERLEAVE_TB[b + 2] as usize] as u8) << 1)
            | (bits[DEINTERLEAVE_TB[b + 3] as usize] as u8);

        let mut best = 5u32;
        let mut best_j: Option<usize> = None;
        let mut unique = true;
        for (j, &nw) in NEXT_WORDS[state].iter().enumerate() {
            let hd = (codeword ^ nw).count_ones();
            if hd < best {
                best = hd;
                best_j = Some(j);
                unique = true;
            } else if hd == best {
                unique = false;
            }
        }
        if !unique {
            return None;
        }
        state = best_j?;

        let d = b >> 2;
        if d < 48 {
            buf[d >> 2] |= (state as u8) << (6 - ((d % 4) * 2));
        }
        b += 4;
    }

    Some(buf)
}

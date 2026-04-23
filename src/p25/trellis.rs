//! P25 1/2-rate trellis decoder for TSBK / PDU blocks — Viterbi variant.
//!
//! The block is 49 four-bit codewords (196 bits) produced by a rate-1/2
//! convolutional code with a 4-state shift register (state = previous input
//! dibit). Each transition emits the 4-bit word `NEXT_WORDS[prev_state][input]`
//! and advances the state to `input`. The encoder starts and ends at state 0,
//! with the 49th (tail) dibit forced to 0 to flush the register — so the
//! payload is the first 48 input dibits = 12 bytes.
//!
//! The deinterleave table is ported from OP25 `block_deinterleave()` in
//! `p25p1_fdma.cc` (originally Wireshark, Copyright 2008 Michael Ossmann).
//! OP25's decoder picks the minimum-distance transition greedily each step
//! and bails whenever two candidates tie — which trips on every single-bit
//! error because `NEXT_WORDS[state]` has pairwise distance 2. We do full
//! Viterbi instead: carry a path metric per state, pick the surviving
//! predecessor per (step, state), terminate at state 0, traceback the input
//! sequence. That turns a dropped block into at-worst a couple of corrected
//! bits.

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

const N_STEPS: usize = 49;

/// Rolling decode statistics. `hd_hist[i]` counts per-step Hamming distances
/// along the *winning* Viterbi path (i.e. the bit errors that had to be
/// corrected). `metric_sum` / `blocks_decoded` gives the average path metric
/// per successful block — i.e. the mean number of bit errors per block.
#[derive(Default, Debug, Clone, Copy)]
pub struct TrellisStats {
    pub hd_hist: [u32; 5],
    pub steps: u32,
    pub metric_sum: u64,
    pub blocks_decoded: u32,
}

/// Soft-output Viterbi decode of a single 196-bit trellis block. Returns 12
/// bytes of payload (48 data dibits) on success. The only failure mode is
/// state 0 being unreachable after the forward pass, which is essentially
/// impossible given the `NEXT_WORDS` table is a permutation of all 16 4-bit
/// codewords — so None is really just a safety valve.
pub fn trellis_decode_block(bits: &[bool; 196], stats: &mut TrellisStats) -> Option<[u8; 12]> {
    const INF: u32 = u32::MAX / 4;

    let mut pm = [INF; 4];
    pm[0] = 0;
    let mut preds = [[0u8; 4]; N_STEPS];
    let mut codewords = [0u8; N_STEPS];

    let mut b = 0usize;
    for step in 0..N_STEPS {
        let codeword = ((bits[DEINTERLEAVE_TB[b] as usize] as u8) << 3)
            | ((bits[DEINTERLEAVE_TB[b + 1] as usize] as u8) << 2)
            | ((bits[DEINTERLEAVE_TB[b + 2] as usize] as u8) << 1)
            | (bits[DEINTERLEAVE_TB[b + 3] as usize] as u8);
        codewords[step] = codeword;
        b += 4;

        let mut new_pm = [INF; 4];
        for new_state in 0..4usize {
            for prev_state in 0..4usize {
                if pm[prev_state] >= INF {
                    continue;
                }
                let nw = NEXT_WORDS[prev_state][new_state];
                let hd = (codeword ^ nw).count_ones();
                let cand = pm[prev_state] + hd;
                if cand < new_pm[new_state] {
                    new_pm[new_state] = cand;
                    preds[step][new_state] = prev_state as u8;
                }
            }
        }
        pm = new_pm;
    }

    // Terminate at state 0 — that's where the encoder's tail dibit drives us.
    if pm[0] >= INF {
        return None;
    }

    // Traceback: at each step the state *after* the transition equals the
    // input dibit at that step, so walking predecessors backwards recovers
    // the input sequence directly.
    let mut input_dibits = [0u8; N_STEPS];
    let mut state = 0usize;
    for step in (0..N_STEPS).rev() {
        input_dibits[step] = state as u8;
        state = preds[step][state] as usize;
    }

    // Per-step HD along the winning path, for diagnostics.
    let mut s = 0usize;
    for step in 0..N_STEPS {
        let input = input_dibits[step] as usize;
        let nw = NEXT_WORDS[s][input];
        let hd = (codewords[step] ^ nw).count_ones();
        let bin = (hd as usize).min(4);
        stats.hd_hist[bin] = stats.hd_hist[bin].saturating_add(1);
        stats.steps = stats.steps.saturating_add(1);
        s = input;
    }
    stats.metric_sum = stats.metric_sum.saturating_add(pm[0] as u64);
    stats.blocks_decoded = stats.blocks_decoded.saturating_add(1);

    // Pack 48 input dibits into 12 bytes, MSB-first within each byte.
    let mut buf = [0u8; 12];
    for d in 0..48 {
        buf[d >> 2] |= input_dibits[d] << (6 - ((d % 4) * 2));
    }
    Some(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: encode a known dibit sequence, flatten into a 196-bit
    /// interleaved block, and verify Viterbi recovers the input.
    fn encode_block(input_dibits: &[u8; 48]) -> [bool; 196] {
        let mut flat = [false; 196];
        let mut state = 0usize;
        let mut b = 0usize;
        for &d in input_dibits.iter() {
            let cw = NEXT_WORDS[state][d as usize];
            flat[DEINTERLEAVE_TB[b] as usize] = (cw >> 3) & 1 != 0;
            flat[DEINTERLEAVE_TB[b + 1] as usize] = (cw >> 2) & 1 != 0;
            flat[DEINTERLEAVE_TB[b + 2] as usize] = (cw >> 1) & 1 != 0;
            flat[DEINTERLEAVE_TB[b + 3] as usize] = cw & 1 != 0;
            b += 4;
            state = d as usize;
        }
        // Tail dibit = 0 to flush to state 0.
        let cw = NEXT_WORDS[state][0];
        flat[DEINTERLEAVE_TB[b] as usize] = (cw >> 3) & 1 != 0;
        flat[DEINTERLEAVE_TB[b + 1] as usize] = (cw >> 2) & 1 != 0;
        flat[DEINTERLEAVE_TB[b + 2] as usize] = (cw >> 1) & 1 != 0;
        flat[DEINTERLEAVE_TB[b + 3] as usize] = cw & 1 != 0;
        flat
    }

    #[test]
    fn decodes_clean_block() {
        let mut dibits = [0u8; 48];
        for (i, d) in dibits.iter_mut().enumerate() {
            *d = (i % 4) as u8;
        }
        let block = encode_block(&dibits);
        let mut stats = TrellisStats::default();
        let got = trellis_decode_block(&block, &mut stats).expect("decode");
        let mut expected = [0u8; 12];
        for d in 0..48 {
            expected[d >> 2] |= dibits[d] << (6 - ((d % 4) * 2));
        }
        assert_eq!(got, expected);
        assert_eq!(stats.hd_hist, [49, 0, 0, 0, 0]);
        assert_eq!(stats.metric_sum, 0);
        assert_eq!(stats.blocks_decoded, 1);
    }

    #[test]
    fn corrects_single_bit_error() {
        let dibits = [2u8; 48];
        let mut block = encode_block(&dibits);
        // Flip one bit in the middle of the block.
        block[97] = !block[97];
        let mut stats = TrellisStats::default();
        let got = trellis_decode_block(&block, &mut stats).expect("decode");
        let mut expected = [0u8; 12];
        for d in 0..48 {
            expected[d >> 2] |= dibits[d] << (6 - ((d % 4) * 2));
        }
        assert_eq!(got, expected, "Viterbi should recover from a single bit flip");
        // Exactly one step on the winning path had hd>0.
        assert_eq!(stats.metric_sum, 1);
    }
}

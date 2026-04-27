use crate::p25::nid::{Duid, decode_nid};

pub const FRAME_SYNC_MAGIC: u64 = 0x5575_F5FF_77FF;
pub const FRAME_SYNC_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

/// NID symbol index at which the first in-frame status dibit sits.
const STATUS_DIBIT_IN_NID_POS: u32 = 12;
/// Total dibits that make up NID: 32 NID dibits plus one inserted status dibit.
const NID_TOTAL_DIBITS: u32 = 33;
/// 24 frame-sync dibits plus the 33 NID+status dibits; anything beyond this is body.
const SYNC_AND_NID_DIBITS: usize = 57;

/// Frame-body max lengths in bits (from OP25 `p25_framer.cc`).
const MAX_FRAME_BITS: [usize; 16] = [
    792,  // 0 - HDU
    0, 0, // 1, 2 - undef
    144,  // 3 - TDU
    0,    // 4 - undef
    1728, // 5 - LDU1
    0,    // 6 - undef
    720,  // 7 - TSBK
    0,    // 8 - undef
    0,    // 9 - VSELP
    1728, // a - LDU2
    0,    // b - undef
    962,  // c - PDU (triple MBT)
    0, 0, // d, e - undef
    432,  // f - TDU_LC
];

#[derive(Debug, Clone)]
pub struct Frame {
    pub duid: Duid,
    pub dibits: Vec<u8>,
}

pub struct Framer {
    sync_sr: u64,
    nid_syms: u32,
    nid_accum: u64,
    state: State,
    body: Vec<u8>,
    body_target_dibits: usize,
    current_duid: Option<Duid>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Hunt,
    CollectNid,
    CollectBody,
}

impl Default for Framer {
    fn default() -> Self {
        Self::new()
    }
}

impl Framer {
    pub fn new() -> Self {
        Self {
            sync_sr: 0,
            nid_syms: 0,
            nid_accum: 0,
            state: State::Hunt,
            body: Vec::with_capacity(360),
            body_target_dibits: 0,
            current_duid: None,
        }
    }

    /// Drop any in-progress hunt/NID/body state. Call after a retune so a
    /// half-collected frame from the previous tune doesn't get stitched
    /// together with fresh dibits.
    pub fn reset(&mut self) {
        self.sync_sr = 0;
        self.nid_syms = 0;
        self.nid_accum = 0;
        self.state = State::Hunt;
        self.body.clear();
        self.body_target_dibits = 0;
        self.current_duid = None;
    }

    /// Feed one dibit. Returns `Some(Frame)` when a complete frame is ready.
    pub fn push_dibit(&mut self, dibit: u8) -> Option<Frame> {
        let dibit = dibit & 0x3;
        self.sync_sr = ((self.sync_sr << 2) | dibit as u64) & FRAME_SYNC_MASK;
        self.nid_accum = (self.nid_accum << 2) | dibit as u64;

        match self.state {
            State::Hunt => {
                if self.is_sync() {
                    self.state = State::CollectNid;
                    self.nid_syms = 0;
                }
                None
            }
            State::CollectNid => {
                self.nid_syms += 1;
                if self.nid_syms == STATUS_DIBIT_IN_NID_POS {
                    // drop the just-accumulated status dibit
                    self.nid_accum >>= 2;
                }
                if self.nid_syms < NID_TOTAL_DIBITS {
                    return None;
                }
                let nid_word = self.nid_accum;
                self.nid_accum = 0;
                let Some(duid) = decode_nid(nid_word) else {
                    self.state = State::Hunt;
                    self.current_duid = None;
                    return None;
                };
                let body_bits = MAX_FRAME_BITS[duid as usize];
                if body_bits == 0 {
                    self.state = State::Hunt;
                    self.current_duid = None;
                    return None;
                }
                self.current_duid = Some(duid);
                self.body.clear();
                self.body_target_dibits = (body_bits / 2).saturating_sub(SYNC_AND_NID_DIBITS);
                self.state = State::CollectBody;
                None
            }
            State::CollectBody => {
                self.body.push(dibit);
                if self.body.len() < self.body_target_dibits {
                    return None;
                }
                let duid = self.current_duid.take()?;
                let dibits = std::mem::take(&mut self.body);
                self.state = State::Hunt;
                Some(Frame { duid, dibits })
            }
        }
    }

    pub fn push_dibits(&mut self, dibits: &[u8], out: &mut Vec<Frame>) {
        for &d in dibits {
            if let Some(f) = self.push_dibit(d) {
                out.push(f);
            }
        }
    }

    fn is_sync(&self) -> bool {
        (self.sync_sr ^ FRAME_SYNC_MAGIC).count_ones() <= 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_matches_exact_pattern() {
        let mut f = Framer::new();
        let sync_bits = FRAME_SYNC_MAGIC;
        for i in (0..48).step_by(2).rev() {
            let d = ((sync_bits >> i) & 0x3) as u8;
            let _ = f.push_dibit(d);
        }
        assert!(f.is_sync(), "sync register should match the magic");
    }
}

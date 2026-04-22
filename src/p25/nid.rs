use crate::p25::bch::{BchResult, bch_decode};

/// Data Unit IDs (P25 TIA-102.BAAA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Duid {
    Hdu = 0x0,
    Tdu = 0x3,
    Ldu1 = 0x5,
    Tsbk = 0x7,
    Ldu2 = 0xA,
    Pdu = 0xC,
    TduLc = 0xF,
    Other = 0xFF,
}

impl Duid {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0x0 => Duid::Hdu,
            0x3 => Duid::Tdu,
            0x5 => Duid::Ldu1,
            0x7 => Duid::Tsbk,
            0xA => Duid::Ldu2,
            0xC => Duid::Pdu,
            0xF => Duid::TduLc,
            _ => Duid::Other,
        }
    }
}

/// Decodes the 64-bit NID word (bits are packed MSB-first in `acc_msb_first`).
/// The LSB of the 64-bit word is the parity bit; the upper 63 bits are the
/// BCH(63,16,23) codeword. Returns the DUID on a parity-consistent decode.
pub fn decode_nid(acc_msb_first: u64) -> Option<Duid> {
    let acc_parity = (acc_msb_first & 1) != 0;

    let mut cw = [false; 64];
    let mut tmp = acc_msb_first;
    for b in &mut cw {
        tmp >>= 1;
        *b = (tmp & 1) != 0;
    }
    match bch_decode(&mut cw) {
        BchResult::NoErrors => {}
        BchResult::Corrected(n) if n <= 4 => {}
        _ => return None,
    }

    let mut acc: u64 = 0;
    for i in (0..64).rev() {
        acc |= cw[i] as u64;
        acc <<= 1;
    }

    if (acc >> 1) == 0 {
        return None;
    }

    let duid_raw = ((acc >> 48) & 0xF) as u8;

    // TIA-102-BAAC parity rule
    let expected_parity = matches!(duid_raw, 0x5 | 0xA);
    if acc_parity != expected_parity {
        return None;
    }

    Some(Duid::from_u8(duid_raw))
}

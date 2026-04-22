//! TSBK (Trunking Signaling Block) opcode parsing.
//!
//! Bit-field offsets match OP25 `apps/trunking.py:534-811`.
//! Each TSBK is 96 bits (12 bytes): the last 16 bits are the CRC-16.
//! `tsbk` is the full 96-bit value with the MSB at bit 95.
//! `(tsbk >> 88) & 0x3F` = opcode.

use crate::p25::freq_table::{FreqTable, IdentEntry};

/// TSBK opcode values (TIA-102.AABF).
mod opcode {
    pub const GRP_VOICE_GRANT: u8 = 0x00;
    pub const GRP_VOICE_GRANT_UPDATE: u8 = 0x02;
    pub const GRP_VOICE_GRANT_UPDATE_EXPLICIT: u8 = 0x03;
    pub const IDENTIFIER_UPDATE_VHF_UHF: u8 = 0x34;
    pub const IDENTIFIER_UPDATE: u8 = 0x3D;
}

/// Motorola manufacturer ID — uses a different field layout for grants and is ignored.
const MFRID_MOTOROLA: u8 = 0x90;

/// A voice-channel grant that has enough information to identify a mobile.
#[derive(Debug, Clone)]
pub struct Grant {
    pub tgid: u16,
    pub rid: u32,
    pub channel_id: u16,
    pub dl_hz: u64,
    pub ul_hz: u64,
}

#[derive(Debug, Clone)]
pub enum TsbkEvent {
    Grant(Grant),
    IdentifierUpdate { iden: u8, entry: IdentEntry },
    Other,
}

pub fn parse_tsbk(block: &[u8; 12], freq_table: &FreqTable) -> Option<TsbkEvent> {
    let tsbk = pack_96(block);
    let opcode = ((tsbk >> 88) & 0x3F) as u8;
    let mfrid = ((tsbk >> 80) & 0xFF) as u8;

    match opcode {
        // Group Voice Grant: ch[56..72], ga[40..56], sa[16..40]. Motorola uses a different layout.
        opcode::GRP_VOICE_GRANT if mfrid != MFRID_MOTOROLA => {
            let ch = ((tsbk >> 56) & 0xFFFF) as u16;
            let ga = ((tsbk >> 40) & 0xFFFF) as u16;
            let sa = ((tsbk >> 16) & 0xFF_FFFF) as u32;
            build_grant(ch, ga, sa, freq_table)
        }
        opcode::GRP_VOICE_GRANT_UPDATE if mfrid == 0 => {
            let ch = ((tsbk >> 64) & 0xFFFF) as u16;
            let ga = ((tsbk >> 48) & 0xFFFF) as u16;
            build_grant(ch, ga, 0, freq_table)
        }
        opcode::GRP_VOICE_GRANT_UPDATE_EXPLICIT if mfrid == 0 => {
            let ch = ((tsbk >> 48) & 0xFFFF) as u16;
            let ga = ((tsbk >> 16) & 0xFFFF) as u16;
            build_grant(ch, ga, 0, freq_table)
        }
        // VHF/UHF variant has the same identifier-update layout as the wide-area form.
        opcode::IDENTIFIER_UPDATE | opcode::IDENTIFIER_UPDATE_VHF_UHF => {
            parse_identifier_update(tsbk)
        }
        _ => Some(TsbkEvent::Other),
    }
}

fn build_grant(ch: u16, tgid: u16, rid: u32, ft: &FreqTable) -> Option<TsbkEvent> {
    let (dl_hz, ul_hz) = ft.channel_to_freqs(ch)?;
    Some(TsbkEvent::Grant(Grant { tgid, rid, channel_id: ch, dl_hz, ul_hz }))
}

fn parse_identifier_update(tsbk: u128) -> Option<TsbkEvent> {
    let iden = ((tsbk >> 76) & 0xF) as u8;
    let toff0 = ((tsbk >> 58) & 0x1FF) as u32;
    let spac = ((tsbk >> 48) & 0x3FF) as u32;
    let freq = ((tsbk >> 16) & 0xFFFF_FFFF) as u64;

    let sign_positive = (toff0 >> 8) & 1 != 0;
    let toff_mag = (toff0 & 0xFF) as i64;
    let offset_hz = toff_mag * 250_000 * if sign_positive { 1 } else { -1 };

    Some(TsbkEvent::IdentifierUpdate {
        iden,
        entry: IdentEntry {
            base_hz: freq * 5,
            step_hz: spac * 125,
            offset_hz,
        },
    })
}

fn pack_96(block: &[u8; 12]) -> u128 {
    let mut be = [0u8; 16];
    be[4..].copy_from_slice(block);
    u128::from_be_bytes(be)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_group_voice_grant() {
        // byte 0 : last-block + protected + opcode (0x00)
        // byte 1 : mfrid (0x00 = standard)
        // byte 2 : options (don't care)
        // bytes 3-4 : channel ID
        // bytes 5-6 : group address (TGID)
        // bytes 7-9 : source address (radio ID)
        // bytes 10-11 : CRC (not checked here)
        let ch: u16 = (1u16 << 12) | 0x010;
        let tgid: u16 = 101;
        let rid: u32 = 0x12_3456;
        let mut block = [0u8; 12];
        block[0] = 0x80; // LB=1, P=0, opcode=0x00
        block[1] = 0x00; // mfrid
        block[2] = 0x00; // opts
        block[3] = (ch >> 8) as u8;
        block[4] = ch as u8;
        block[5] = (tgid >> 8) as u8;
        block[6] = tgid as u8;
        block[7] = (rid >> 16) as u8;
        block[8] = (rid >> 8) as u8;
        block[9] = rid as u8;

        let mut ft = FreqTable::new();
        ft.insert(
            1,
            IdentEntry {
                base_hz: 851_000_000,
                step_hz: 12_500,
                offset_hz: -45_000_000,
            },
        );

        match parse_tsbk(&block, &ft).unwrap() {
            TsbkEvent::Grant(g) => {
                assert_eq!(g.tgid, tgid);
                assert_eq!(g.rid, rid);
                assert_eq!(g.channel_id, ch);
                assert_eq!(g.dl_hz, 851_000_000 + 16 * 12_500);
                assert_eq!(g.ul_hz, g.dl_hz - 45_000_000);
            }
            other => panic!("expected Grant, got {other:?}"),
        }
    }

    #[test]
    fn parses_identifier_update() {
        // Opcode 0x3D with negative (sign=0) 45 MHz offset (180 * 250 kHz).
        // Layout: bits [95:90]=opcode, [89:88]=reserved, [87:80]=mfrid,
        //         [79:76]=iden, [75:67]=bw (9b),
        //         [66:58]=toff (9b: sign + 8-bit magnitude),
        //         [57:48]=spac (10b), [47:16]=base*5 Hz (32b), [15:0]=CRC.
        let iden: u32 = 1;
        let bw: u32 = 0b0_0011_1000; // 9 bits; value doesn't affect our test
        let sign_positive: u32 = 0; // uplink below downlink
        let mag: u32 = 180;
        let toff: u32 = (sign_positive << 8) | mag; // 9 bits
        let spac: u32 = 100; // 12.5 kHz (100 * 125 Hz)
        let freq: u32 = 851_000_000 / 5;

        let mut v: u128 = 0;
        v |= 0x3D << 90; // opcode (6b) at [95:90]
        // [89:88] reserved = 0
        // [87:80] mfrid = 0
        v |= (iden as u128) << 76; // 4 bits
        v |= (bw as u128 & 0x1FF) << 67; // 9 bits
        v |= (toff as u128 & 0x1FF) << 58; // 9 bits
        v |= (spac as u128 & 0x3FF) << 48; // 10 bits
        v |= (freq as u128 & 0xFFFF_FFFF) << 16; // 32 bits

        let mut block = [0u8; 12];
        for (i, b) in block.iter_mut().enumerate() {
            *b = ((v >> (88 - (i * 8))) & 0xFF) as u8;
        }

        let ft = FreqTable::new();
        match parse_tsbk(&block, &ft).unwrap() {
            TsbkEvent::IdentifierUpdate { iden: got_iden, entry } => {
                assert_eq!(got_iden, 1);
                assert_eq!(entry.base_hz, 851_000_000);
                assert_eq!(entry.step_hz, 100 * 125);
                assert_eq!(entry.offset_hz, -45_000_000);
            }
            other => panic!("expected IdentifierUpdate, got {other:?}"),
        }
    }
}

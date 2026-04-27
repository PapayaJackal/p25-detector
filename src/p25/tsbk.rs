//! TSBK (Trunking Signaling Block) opcode parsing.
//!
//! Bit-field offsets match OP25 `apps/trunking.py:534-811`.
//! Each TSBK is 96 bits (12 bytes): the last 16 bits are the CRC-16.
//! `tsbk` is the full 96-bit value with the MSB at bit 95.
//! `(tsbk >> 88) & 0x3F` = opcode.

use serde::Serialize;

use crate::p25::freq_table::{FreqTable, IdentEntry};

/// TSBK opcode values (TIA-102.AABF).
mod opcode {
    pub const GRP_VOICE_GRANT: u8 = 0x00;
    pub const GRP_VOICE_GRANT_UPDATE: u8 = 0x02;
    pub const GRP_VOICE_GRANT_UPDATE_EXPLICIT: u8 = 0x03;
    pub const UU_VOICE_GRANT: u8 = 0x04;
    pub const UU_VOICE_GRANT_UPDATE: u8 = 0x06;
    pub const TELE_INT_VOICE_GRANT: u8 = 0x08;
    pub const TELE_INT_VOICE_GRANT_UPDATE: u8 = 0x09;
    pub const IDENTIFIER_UPDATE_VHF_UHF: u8 = 0x34;
    pub const IDENTIFIER_UPDATE: u8 = 0x3D;
}

/// Motorola manufacturer ID — uses a different field layout for grants and is ignored.
const MFRID_MOTOROLA: u8 = 0x90;

/// Which TSBK opcode produced this grant. The `*Update*` variants indicate
/// the call is already in progress (radio is keying when we retune); the
/// non-update variants are fresh allocations where 250–500 ms of
/// PTT-to-key latency typically applies. `UnitToUnit*` are private (radio-
/// to-radio) calls and `TelephoneInterconnect*` are radio-to-PSTN calls;
/// both produce real C4FM voice on the UL just like group voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GrantKind {
    VoiceGrant,
    VoiceGrantUpdate,
    VoiceGrantUpdateExplicit,
    UnitToUnitVoiceGrant,
    UnitToUnitVoiceGrantUpdate,
    TelephoneInterconnectGrant,
    TelephoneInterconnectGrantUpdate,
}

/// A voice-channel grant that has enough information to identify a mobile.
#[derive(Debug, Clone)]
pub struct Grant {
    pub tgid: u16,
    pub rid: u32,
    pub channel_id: u16,
    pub dl_hz: u64,
    pub ul_hz: u64,
    pub kind: GrantKind,
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
            build_grant(ch, ga, sa, freq_table, GrantKind::VoiceGrant)
        }
        opcode::GRP_VOICE_GRANT_UPDATE if mfrid == 0 => {
            let ch = ((tsbk >> 64) & 0xFFFF) as u16;
            let ga = ((tsbk >> 48) & 0xFFFF) as u16;
            build_grant(ch, ga, 0, freq_table, GrantKind::VoiceGrantUpdate)
        }
        opcode::GRP_VOICE_GRANT_UPDATE_EXPLICIT if mfrid == 0 => {
            let ch = ((tsbk >> 48) & 0xFFFF) as u16;
            let ga = ((tsbk >> 16) & 0xFFFF) as u16;
            build_grant(ch, ga, 0, freq_table, GrantKind::VoiceGrantUpdateExplicit)
        }
        // Unit-to-unit voice grant + update share a layout (no service-options
        // byte before the channel field; see sdrtrunk
        // UnitToUnitVoiceChannelGrant{,Update}.java):
        //   ch[16..32], target_addr[32..56], source_addr[56..80].
        // Source is the talker's RID — that's what we report as `rid`. There's
        // no group address; tgid=0 marks "non-group".
        opcode::UU_VOICE_GRANT if mfrid == 0 => {
            let ch = ((tsbk >> 64) & 0xFFFF) as u16;
            let sa = ((tsbk >> 16) & 0xFF_FFFF) as u32;
            build_grant(ch, 0, sa, freq_table, GrantKind::UnitToUnitVoiceGrant)
        }
        opcode::UU_VOICE_GRANT_UPDATE if mfrid == 0 => {
            let ch = ((tsbk >> 64) & 0xFFFF) as u16;
            let sa = ((tsbk >> 16) & 0xFF_FFFF) as u32;
            build_grant(ch, 0, sa, freq_table, GrantKind::UnitToUnitVoiceGrantUpdate)
        }
        // Telephone interconnect: ch[24..40], call_timer[40..56] (skipped),
        // source_addr[56..80]. Per sdrtrunk
        // TelephoneInterconnectVoiceChannelGrant{,Update}.java. The "target"
        // is a PSTN endpoint (not in the TSBK), so tgid stays 0.
        opcode::TELE_INT_VOICE_GRANT if mfrid == 0 => {
            let ch = ((tsbk >> 56) & 0xFFFF) as u16;
            let sa = ((tsbk >> 16) & 0xFF_FFFF) as u32;
            build_grant(ch, 0, sa, freq_table, GrantKind::TelephoneInterconnectGrant)
        }
        opcode::TELE_INT_VOICE_GRANT_UPDATE if mfrid == 0 => {
            let ch = ((tsbk >> 56) & 0xFFFF) as u16;
            let sa = ((tsbk >> 16) & 0xFF_FFFF) as u32;
            build_grant(ch, 0, sa, freq_table, GrantKind::TelephoneInterconnectGrantUpdate)
        }
        // VHF/UHF variant has the same identifier-update layout as the wide-area form.
        opcode::IDENTIFIER_UPDATE | opcode::IDENTIFIER_UPDATE_VHF_UHF => {
            parse_identifier_update(tsbk)
        }
        _ => Some(TsbkEvent::Other),
    }
}

fn build_grant(
    ch: u16,
    tgid: u16,
    rid: u32,
    ft: &FreqTable,
    kind: GrantKind,
) -> Option<TsbkEvent> {
    let (dl_hz, ul_hz) = ft.channel_to_freqs(ch)?;
    Some(TsbkEvent::Grant(Grant { tgid, rid, channel_id: ch, dl_hz, ul_hz, kind }))
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
                assert_eq!(g.kind, GrantKind::VoiceGrant);
            }
            other => panic!("expected Grant, got {other:?}"),
        }
    }

    /// Build a TSBK byte block from sdrtrunk-style bit assignments. `bits`
    /// is `[(start, len, value), ...]` where `start` is the sdrtrunk bit
    /// index (0 = MSB of byte 0). Same ordering used by sdrtrunk's
    /// `BinaryMessage.getInt(int[])` accessors.
    fn build_block(bits: &[(usize, usize, u64)]) -> [u8; 12] {
        let mut block = [0u8; 12];
        for &(start, len, value) in bits {
            for i in 0..len {
                let bit = start + i;
                let bit_val = (value >> (len - 1 - i)) & 1;
                if bit_val != 0 {
                    block[bit / 8] |= 1 << (7 - (bit % 8));
                }
            }
        }
        block
    }

    fn nspac_iden() -> FreqTable {
        let mut ft = FreqTable::new();
        ft.insert(
            1,
            IdentEntry { base_hz: 769_006_250, step_hz: 12_500, offset_hz: 30_000_000 },
        );
        ft
    }

    #[test]
    fn parses_unit_to_unit_voice_grant() {
        // Opcode at bits 2-7, MFID at 8-15, ch at 16-31 (iden=1, channel=0),
        // target at 32-55, source at 56-79.
        let block = build_block(&[
            (0, 2, 0b10),               // LB=1, P=0
            (2, 6, 0x04),               // opcode UU_VOICE_GRANT
            (8, 8, 0),                  // mfid
            (16, 16, (1u64 << 12) | 0), // ch: iden=1, channel=0
            (32, 24, 0xCAFE_42),        // target RID
            (56, 24, 0x12_3456),        // source RID (the talker)
        ]);
        match parse_tsbk(&block, &nspac_iden()).unwrap() {
            TsbkEvent::Grant(g) => {
                assert_eq!(g.kind, GrantKind::UnitToUnitVoiceGrant);
                assert_eq!(g.tgid, 0);
                assert_eq!(g.rid, 0x12_3456);
                assert_eq!(g.dl_hz, 769_006_250);
                assert_eq!(g.ul_hz, 769_006_250 + 30_000_000);
            }
            other => panic!("expected Grant, got {other:?}"),
        }
    }

    #[test]
    fn parses_unit_to_unit_voice_grant_update() {
        let block = build_block(&[
            (0, 2, 0b10),
            (2, 6, 0x06),
            (8, 8, 0),
            (16, 16, (1u64 << 12) | 5), // iden=1, channel=5
            (32, 24, 0),
            (56, 24, 0x77_8899),
        ]);
        match parse_tsbk(&block, &nspac_iden()).unwrap() {
            TsbkEvent::Grant(g) => {
                assert_eq!(g.kind, GrantKind::UnitToUnitVoiceGrantUpdate);
                assert_eq!(g.tgid, 0);
                assert_eq!(g.rid, 0x77_8899);
                assert_eq!(g.dl_hz, 769_006_250 + 5 * 12_500);
            }
            other => panic!("expected Grant, got {other:?}"),
        }
    }

    #[test]
    fn parses_telephone_interconnect_grant() {
        // Tele has 8-bit options at bits 16-23 before the channel field, so
        // ch lives at bits 24-39 (same offset as group voice grant).
        let block = build_block(&[
            (0, 2, 0b10),
            (2, 6, 0x08),
            (8, 8, 0),
            (16, 8, 0),                 // service options (unused)
            (24, 16, (1u64 << 12) | 3), // ch: iden=1, channel=3
            (40, 16, 60),               // call_timer (60 × 100ms = 6 s) — ignored
            (56, 24, 0x42_4242),        // source RID
        ]);
        match parse_tsbk(&block, &nspac_iden()).unwrap() {
            TsbkEvent::Grant(g) => {
                assert_eq!(g.kind, GrantKind::TelephoneInterconnectGrant);
                assert_eq!(g.tgid, 0);
                assert_eq!(g.rid, 0x42_4242);
                assert_eq!(g.dl_hz, 769_006_250 + 3 * 12_500);
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

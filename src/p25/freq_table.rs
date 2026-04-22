//! P25 identifier-update / band-plan table.
//!
//! Fed by TSBK opcode 0x3D (Identifier Update) — see OP25
//! `apps/trunking.py:736-752`. Converts a 16-bit channel ID to a downlink and
//! uplink frequency in Hz.

#[derive(Debug, Clone, Copy)]
pub struct IdentEntry {
    pub base_hz: u64,
    pub step_hz: u32,
    /// Absolute uplink offset in Hz. Positive means uplink is above downlink.
    pub offset_hz: i64,
}

#[derive(Default, Debug, Clone)]
pub struct FreqTable {
    entries: [Option<IdentEntry>; 16],
}

impl FreqTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, iden: u8, entry: IdentEntry) {
        if iden < 16 {
            self.entries[iden as usize] = Some(entry);
        }
    }

    pub fn get(&self, iden: u8) -> Option<&IdentEntry> {
        self.entries.get(iden as usize).and_then(|e| e.as_ref())
    }

    /// Returns (downlink_hz, uplink_hz) for a channel ID, if the table entry exists.
    /// Uplink is clamped to zero if the offset would put it below DC.
    pub fn channel_to_freqs(&self, channel_id: u16) -> Option<(u64, u64)> {
        let iden = ((channel_id >> 12) & 0xF) as u8;
        let channel = (channel_id & 0xFFF) as u64;
        let e = self.get(iden)?;
        let dl = e.base_hz + (e.step_hz as u64) * channel;
        let ul = ((dl as i64) + e.offset_hz).max(0) as u64;
        Some((dl, ul))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_downlink_and_uplink() {
        let mut t = FreqTable::new();
        t.insert(
            1,
            IdentEntry {
                base_hz: 851_000_000,
                step_hz: 12_500,
                offset_hz: -45_000_000,
            },
        );
        let ch = (1u16 << 12) | 0x010; // iden=1, channel=16
        let (dl, ul) = t.channel_to_freqs(ch).unwrap();
        assert_eq!(dl, 851_000_000 + 16 * 12_500);
        assert_eq!(ul, dl - 45_000_000);
    }
}

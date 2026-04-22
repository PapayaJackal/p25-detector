/// 4-level slicer. Emits dibits `0..=3` corresponding to nominal symbol levels
/// `[+3, +1, -1, -3]` in the P25 C4FM convention (dibit 01 = +1, 11 = -1, etc.).
///
/// Tracks a running mean and peak to derive adaptive thresholds.
pub struct FskSlicer {
    mean: f32,
    amp: f32,
    alpha: f32,
    amp_alpha: f32,
}

impl Default for FskSlicer {
    fn default() -> Self {
        Self {
            mean: 0.0,
            amp: 1.0,
            alpha: 0.01,
            amp_alpha: 0.005,
        }
    }
}

impl FskSlicer {
    /// Maps P25 C4FM dibit codes per TIA-102.BAAA.
    #[inline]
    fn encode(level: i8) -> u8 {
        match level {
            3 => 0b01,
            1 => 0b00,
            -1 => 0b10,
            _ => 0b11,
        }
    }

    pub fn slice_one(&mut self, y: f32) -> u8 {
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * y;
        let centered = y - self.mean;
        let abs = centered.abs();
        if abs > self.amp * 0.25 {
            self.amp = (1.0 - self.amp_alpha) * self.amp + self.amp_alpha * (abs / 3.0).max(1e-3);
        }
        let unit = self.amp;
        let level = if centered > 2.0 * unit {
            3
        } else if centered > 0.0 {
            1
        } else if centered > -2.0 * unit {
            -1
        } else {
            -3
        };
        Self::encode(level)
    }

    pub fn slice(&mut self, input: &[f32], out: &mut Vec<u8>) {
        for &y in input {
            out.push(self.slice_one(y));
        }
    }
}

/// 4-level slicer. Emits dibits `0..=3` corresponding to nominal symbol levels
/// `[+3, +1, -1, -3]` in the P25 C4FM convention (dibit 01 = +1, 11 = -1, etc.).
///
/// Tracks a running mean (DC removal) plus independent EMAs of the inner (±1)
/// and outer (±3) rail amplitudes. The decision threshold is the midpoint
/// between the two rails — self-calibrating against any SNR. A single-amp
/// `E[|y|]/K` estimator can never sit at the right threshold across SNRs
/// because noise inflates `E[|y|]` above `2×unit` in a way that depends on
/// `σ/unit`; tracking the two rails separately sidesteps the problem.
pub struct FskSlicer {
    mean: f32,
    /// EMA of `|centered|` for samples classified as inner (±1).
    amp_inner: f32,
    /// EMA of `|centered|` for samples classified as outer (±3). Also has a
    /// floor-ratchet: any sample whose magnitude exceeds the current estimate
    /// pulls it up instantly, which unsticks cold-start from all-zero EMAs.
    amp_outer: f32,
    alpha: f32,
    amp_alpha: f32,
    /// Count of each emitted dibit value since the last `dibit_hist_since_read`.
    /// For uniformly random C4FM data all four buckets should be ≈25%; skew
    /// diagnoses slicer bias (threshold too low → ±3 buckets inflated,
    /// threshold too high → ±1 buckets inflated).
    dibit_hist: [u64; 4],
}

impl Default for FskSlicer {
    fn default() -> Self {
        Self {
            mean: 0.0,
            amp_inner: 0.0,
            amp_outer: 0.0,
            alpha: 0.01,
            amp_alpha: 0.005,
            dibit_hist: [0; 4],
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

        let threshold = 0.5 * (self.amp_inner + self.amp_outer);
        let is_outer = abs > threshold;

        if is_outer {
            self.amp_outer = (1.0 - self.amp_alpha) * self.amp_outer + self.amp_alpha * abs;
        } else {
            self.amp_inner = (1.0 - self.amp_alpha) * self.amp_inner + self.amp_alpha * abs;
        }
        // Cold start: both EMAs are 0 so every sample initially classifies as
        // outer (abs > 0), which grows amp_outer from zero toward E[|y|] ≈ 2u
        // over ~400 ms at amp_alpha=0.005. Once amp_outer is roughly unit, the
        // threshold drops below 1u so some inner samples start falling into
        // amp_inner, which then grows toward 1u and pulls threshold to 2u.
        // Equilibrium settles in ~1 s. A floor-ratchet (`amp_outer = abs` on
        // any peak) used to live here to accelerate that, but it also biased
        // amp_outer toward noise peaks in steady state — inflating threshold
        // and skewing the dibit histogram toward inner. Better to wait.

        let level = if centered > 0.0 {
            if is_outer { 3 } else { 1 }
        } else if is_outer {
            -3
        } else {
            -1
        };
        let dibit = Self::encode(level);
        self.dibit_hist[dibit as usize] = self.dibit_hist[dibit as usize].saturating_add(1);
        dibit
    }

    pub fn slice(&mut self, input: &[f32], out: &mut Vec<u8>) {
        for &y in input {
            out.push(self.slice_one(y));
        }
    }

    pub fn mean(&self) -> f32 {
        self.mean
    }

    pub fn amp_inner(&self) -> f32 {
        self.amp_inner
    }

    pub fn amp_outer(&self) -> f32 {
        self.amp_outer
    }

    /// Returns dibit counts `[d00, d01, d10, d11]` since the last call and
    /// resets the underlying counters. Dibit values per TIA-102.BAAA:
    /// `00=+1`, `01=+3`, `10=-1`, `11=-3`.
    pub fn dibit_hist_since_read(&mut self) -> [u64; 4] {
        std::mem::take(&mut self.dibit_hist)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed equiprobable C4FM symbols plus modest noise and verify the rail
    /// estimates converge near 1×unit and 3×unit, and the dibit histogram is
    /// balanced. Burns the first half of samples for convergence then checks
    /// the steady-state histogram over the back half.
    #[test]
    fn rails_converge_under_noise() {
        let unit: f32 = 0.1;
        let sigma: f32 = 0.03;
        let symbols = [1.0, 3.0, -1.0, -3.0];
        let mut slicer = FskSlicer::default();
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next_rand = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };
        // Warm-up phase: let the EMAs settle.
        for _ in 0..20_000 {
            let idx = ((next_rand() >> 33) & 3) as usize;
            // Uniform noise in [-sigma, sigma] — rough Gaussian stand-in, sum
            // of a few uniforms would be closer but this is enough to seed
            // bootstrap.
            let noise = (next_rand() as f32 / u64::MAX as f32 - 0.5) * 2.0 * sigma;
            slicer.slice_one(symbols[idx] * unit + noise);
        }
        let _ = slicer.dibit_hist_since_read();
        // Steady-state measurement phase.
        for _ in 0..20_000 {
            let idx = ((next_rand() >> 33) & 3) as usize;
            let noise = (next_rand() as f32 / u64::MAX as f32 - 0.5) * 2.0 * sigma;
            slicer.slice_one(symbols[idx] * unit + noise);
        }
        let hist = slicer.dibit_hist_since_read();
        let total: u64 = hist.iter().sum();
        for &h in &hist {
            let frac = h as f32 / total as f32;
            assert!((frac - 0.25).abs() < 0.03, "dibit skew: {:?}", hist);
        }
        assert!(
            (slicer.amp_inner() - 1.0 * unit).abs() < 0.15 * unit,
            "amp_inner={} vs expected {}",
            slicer.amp_inner(),
            unit,
        );
        assert!(
            (slicer.amp_outer() - 3.0 * unit).abs() < 0.15 * unit,
            "amp_outer={} vs expected {}",
            slicer.amp_outer(),
            3.0 * unit,
        );
    }
}

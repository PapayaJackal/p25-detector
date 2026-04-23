use num_complex::Complex32;

/// FM discriminator: y[n] = arg(x[n] * conj(x[n-1])).
/// Output scale: for C4FM at 4800 sym/s, 25 kHz IF, expect symbols at ≈ ±1 and ±3
/// after normalization and matched filtering.
pub struct FmDiscriminator {
    last: Complex32,
}

impl Default for FmDiscriminator {
    fn default() -> Self {
        Self {
            last: Complex32::new(0.0, 0.0),
        }
    }
}

impl FmDiscriminator {
    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<f32>) {
        for &x in input {
            let prod = x * self.last.conj();
            let phase = prod.im.atan2(prod.re);
            out.push(phase);
            self.last = x;
        }
    }
}

/// Single-pole IIR DC blocker: y[n] = x[n] - x[n-1] + α·y[n-1].
/// With α = 0.9995 at 48 kHz the -3 dB corner is ≈ 4 Hz, well below any
/// real modulation content but well above static LO offset.
pub struct DcBlock {
    alpha: f32,
    last_in: f32,
    last_out: f32,
}

impl DcBlock {
    pub fn new(alpha: f32) -> Self {
        Self { alpha, last_in: 0.0, last_out: 0.0 }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        for &x in input {
            let y = x - self.last_in + self.alpha * self.last_out;
            out.push(y);
            self.last_in = x;
            self.last_out = y;
        }
    }
}

/// Real matched-filter for C4FM. `coeffs` is a raised-cosine-shaped low-pass
/// designed for the target samples-per-symbol.
pub struct MatchedFilter {
    taps: Vec<f32>,
    history: Vec<f32>,
    write: usize,
}

impl MatchedFilter {
    pub fn new(taps: Vec<f32>) -> Self {
        let n = taps.len();
        Self {
            taps,
            history: vec![0.0; n.next_power_of_two().max(2)],
            write: 0,
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let mask = self.history.len() - 1;
        for &x in input {
            self.history[self.write] = x;
            self.write = (self.write + 1) & mask;
            let mut acc = 0.0f32;
            let mut idx = (self.write + self.history.len() - 1) & mask;
            for &t in &self.taps {
                acc += self.history[idx] * t;
                idx = (idx + self.history.len() - 1) & mask;
            }
            out.push(acc);
        }
    }
}

/// Normalized raised-cosine pulse, unity gain at DC, `num_taps` long, centered.
/// Roll-off 0.2 is a common value for C4FM matched filters.
pub fn raised_cosine(num_taps: usize, samples_per_symbol: f32, rolloff: f32) -> Vec<f32> {
    use std::f32::consts::PI;
    let mut taps = Vec::with_capacity(num_taps);
    let m = num_taps as f32 - 1.0;
    let mut sum = 0.0f32;
    for n in 0..num_taps {
        let t = (n as f32 - m / 2.0) / samples_per_symbol;
        let denom = 1.0 - (2.0 * rolloff * t).powi(2);
        let v = if denom.abs() < 1e-6 {
            PI / 4.0 * (PI * t).sin() / (PI * t).max(1e-6)
        } else if t.abs() < 1e-6 {
            1.0
        } else {
            (PI * t).sin() / (PI * t) * (PI * rolloff * t).cos() / denom
        };
        taps.push(v);
        sum += v;
    }
    if sum.abs() > 0.0 {
        for t in taps.iter_mut() {
            *t /= sum;
        }
    }
    taps
}

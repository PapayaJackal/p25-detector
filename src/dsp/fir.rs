use num_complex::Complex32;
use std::f32::consts::PI;

/// Polyphase-free FIR with integer decimation. Keeps a delay line internally.
pub struct FirDecimator {
    taps: Vec<f32>,
    history: Vec<Complex32>,
    write: usize,
    decim: usize,
    phase: usize,
}

impl FirDecimator {
    pub fn new(taps: Vec<f32>, decim: usize) -> Self {
        let n = taps.len();
        Self {
            taps,
            history: vec![Complex32::new(0.0, 0.0); n.next_power_of_two().max(2)],
            write: 0,
            decim: decim.max(1),
            phase: 0,
        }
    }

    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        let mask = self.history.len() - 1;
        let n_taps = self.taps.len();
        for &x in input {
            self.history[self.write] = x;
            self.write = (self.write + 1) & mask;
            self.phase += 1;
            if self.phase >= self.decim {
                self.phase = 0;
                let mut acc = Complex32::new(0.0, 0.0);
                let mut idx = (self.write + self.history.len() - 1) & mask;
                for &tap in self.taps.iter().take(n_taps) {
                    acc += self.history[idx] * tap;
                    idx = (idx + self.history.len() - 1) & mask;
                }
                out.push(acc);
            }
        }
    }
}

/// Low-pass FIR via windowed sinc (Hamming). Cutoff is normalized to Nyquist (0.0–1.0).
pub fn design_lowpass(num_taps: usize, cutoff_norm: f32) -> Vec<f32> {
    assert!(num_taps > 0);
    let m = num_taps as f32 - 1.0;
    let mut taps = Vec::with_capacity(num_taps);
    let mut sum = 0.0f32;
    for n in 0..num_taps {
        let x = n as f32 - m / 2.0;
        let sinc = if x.abs() < 1e-6 {
            cutoff_norm
        } else {
            (PI * cutoff_norm * x).sin() / (PI * x)
        };
        let w = 0.54 - 0.46 * (2.0 * PI * n as f32 / m).cos();
        let t = sinc * w;
        taps.push(t);
        sum += t;
    }
    if sum.abs() > 0.0 {
        for t in taps.iter_mut() {
            *t /= sum;
        }
    }
    taps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_passes_dc() {
        let taps = design_lowpass(31, 0.25);
        let mut f = FirDecimator::new(taps, 1);
        let input = vec![Complex32::new(1.0, 0.0); 200];
        let mut out = Vec::new();
        f.process(&input, &mut out);
        let last = out.last().copied().unwrap();
        assert!((last.re - 1.0).abs() < 0.01);
        assert!(last.im.abs() < 0.01);
    }

    #[test]
    fn decimates_correctly() {
        let taps = design_lowpass(11, 0.5);
        let mut f = FirDecimator::new(taps, 4);
        let input = vec![Complex32::new(1.0, 0.0); 100];
        let mut out = Vec::new();
        f.process(&input, &mut out);
        assert_eq!(out.len(), 25);
    }
}

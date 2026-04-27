use num_complex::Complex32;
use std::f32::consts::PI;

/// Polyphase-free FIR with integer decimation. Keeps a delay line internally.
///
/// History is a doubled ring buffer: each incoming sample is written at both
/// `write` and `write + n_taps`, so the convolution window is always the
/// contiguous slice `history[write..write+n_taps]`. That eliminates modular
/// index arithmetic from the inner loop and lets LLVM vectorize the
/// multiply-accumulate over the taps. Taps are stored reversed so that
/// `taps_rev[i]` lines up with `window[i]` under a plain forward zip.
pub struct FirDecimator {
    taps_rev: Vec<f32>,
    history: Vec<Complex32>,
    write: usize,
    n_taps: usize,
    decim: usize,
    phase: usize,
}

impl FirDecimator {
    pub fn new(taps: Vec<f32>, decim: usize) -> Self {
        let n_taps = taps.len().max(1);
        let mut taps_rev = taps;
        taps_rev.reverse();
        Self {
            taps_rev,
            history: vec![Complex32::new(0.0, 0.0); 2 * n_taps],
            write: 0,
            n_taps,
            decim: decim.max(1),
            phase: 0,
        }
    }

    /// Zero the delay line and reset the polyphase counter. Used after a
    /// retune to keep stale samples out of the convolution window.
    pub fn reset(&mut self) {
        for s in self.history.iter_mut() {
            *s = Complex32::new(0.0, 0.0);
        }
        self.write = 0;
        self.phase = 0;
    }

    pub fn process(&mut self, input: &[Complex32], out: &mut Vec<Complex32>) {
        let n = self.n_taps;
        out.reserve(input.len() / self.decim + 1);
        for &x in input {
            self.history[self.write] = x;
            self.history[self.write + n] = x;
            self.write += 1;
            if self.write == n {
                self.write = 0;
            }
            self.phase += 1;
            if self.phase >= self.decim {
                self.phase = 0;
                let window = &self.history[self.write..self.write + n];
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for (&t, s) in self.taps_rev.iter().zip(window.iter()) {
                    re += t * s.re;
                    im += t * s.im;
                }
                out.push(Complex32::new(re, im));
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

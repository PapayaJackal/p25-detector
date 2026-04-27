use num_complex::Complex32;
use rustfft::Fft;

/// Half-bandwidth of the P25 Phase 1 channel kept around the target bin
/// (±6.25 kHz matches the channel raster). Convert to bins at runtime from
/// sample rate.
pub const CHANNEL_HALF_BW_HZ: f32 = 6_250.0;

/// Fraction of the sampled bandwidth on each side of Nyquist excluded from
/// the noise estimate, where the SDR's anti-alias filter rolls off.
pub const EDGE_SKIP_FRAC: f32 = 0.0625;

/// Median-to-mean ratio for an exponentially-distributed RV (`median =
/// ln(2)·mean`). Each FFT bin's `|X[k]|²` is exponential under thermal-noise
/// input, so the spatial median across out-of-channel bins underestimates the
/// mean by this factor. Pass this as `median_to_mean_bias` when `bin_power`
/// is from a single FFT chunk.
pub const MEDIAN_BIAS_EXPONENTIAL: f32 = std::f32::consts::LN_2;

/// Median-to-mean ratio for a Gamma-distributed RV with large shape parameter
/// (Welch periodogram averaged over many FFT chunks: per-bin distribution
/// becomes `Gamma(M, λ/M)`). For `M ≥ 60` the median lands within ~0.5%
/// (~0.02 dB) of the mean, so no correction is applied. Pass this when
/// `bin_power` is an averaged spectrum.
pub const MEDIAN_BIAS_AVERAGED: f32 = 1.0;

/// Converts a linear power ratio to dBFS (full-scale = 1.0 after -1..1 IQ normalization).
pub fn to_dbfs(power: f32) -> f32 {
    const FLOOR: f32 = 1e-12;
    10.0 * power.max(FLOOR).log10()
}

/// Linear-domain SNR in dB from in-channel (S+N) and same-bandwidth noise (N)
/// powers. Returns `None` when the signal isn't above noise — log-domain
/// subtraction would underflow or produce a negative-infinity result.
pub fn compute_snr_db(signal: f32, noise: f32) -> Option<f32> {
    let signal_only = (signal - noise).max(0.0);
    if signal_only > 0.0 && noise > 0.0 {
        Some(10.0 * (signal_only / noise).log10())
    } else {
        None
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ChannelMeasurement {
    /// In-channel power: sum of |X[k]|²/N² over ±half_bw_bins around center_bin.
    pub signal: f32,
    /// Noise reference scaled to the channel bandwidth: median per-bin power
    /// across out-of-channel bins (excluding spectrum edges) × channel_bins.
    pub noise: f32,
}

/// Computes per-bin power |X[k]|²/N² from one FFT-sized chunk and writes into
/// `bin_power_out`. Used by callers that want to feed `measure_channel_from_bins`
/// repeatedly with shared scratch buffers.
pub fn fft_bin_power(
    chunk: &[Complex32],
    fft: &dyn Fft<f32>,
    scratch: &mut [Complex32],
    bin_power_out: &mut [f32],
) {
    let n = scratch.len();
    assert!(n > 0, "fft_bin_power: empty scratch buffer");
    assert_eq!(chunk.len(), n, "fft_bin_power: chunk len must match scratch");
    assert_eq!(bin_power_out.len(), n, "fft_bin_power: out len must match scratch");
    scratch.copy_from_slice(chunk);
    fft.process(scratch);
    let norm = (n as f32).powi(2);
    for (s, p) in scratch.iter().zip(bin_power_out.iter_mut()) {
        *p = s.norm_sqr() / norm;
    }
}

/// Extracts in-channel power and a same-bandwidth noise reference from a
/// pre-computed per-bin power vector (FFT layout: bin 0 = DC, bins N/2±… near
/// Nyquist). Channel sits at `center_bin` with FFT-circular ±`half_bw_bins`
/// either side. Noise is the median of bins outside the channel and outside
/// the outermost `edge_skip_bins` near Nyquist (where the SDR's anti-alias
/// filter rolls off), then divided by `median_to_mean_bias` to convert the
/// median to an unbiased mean estimate, then scaled by `2*half_bw_bins+1`
/// to match channel bandwidth. Median (rather than plain mean) keeps the
/// estimator robust to other transmitters present elsewhere in the baseband.
///
/// `median_to_mean_bias` should be [`MEDIAN_BIAS_EXPONENTIAL`] when
/// `bin_power` is from a single FFT chunk and [`MEDIAN_BIAS_AVERAGED`]
/// when it's a Welch periodogram.
pub fn measure_channel_from_bins(
    bin_power: &[f32],
    center_bin: usize,
    half_bw_bins: usize,
    edge_skip_bins: usize,
    median_to_mean_bias: f32,
    noise_buf: &mut Vec<f32>,
) -> ChannelMeasurement {
    let n = bin_power.len();
    if n < 4 || center_bin >= n {
        return ChannelMeasurement::default();
    }
    let channel_bins = 2 * half_bw_bins + 1;
    if channel_bins >= n {
        return ChannelMeasurement::default();
    }

    let nyq = n / 2;
    let edge_lo = nyq.saturating_sub(edge_skip_bins);
    let edge_hi = (nyq + edge_skip_bins).min(n);

    let mut signal = 0.0f64;
    noise_buf.clear();
    for (i, &p) in bin_power.iter().enumerate() {
        if circular_within(i, center_bin, half_bw_bins, n) {
            signal += p as f64;
        } else if i < edge_lo || i >= edge_hi {
            noise_buf.push(p);
        }
    }

    let median = if noise_buf.is_empty() {
        0.0
    } else {
        let mid = noise_buf.len() / 2;
        noise_buf.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        noise_buf[mid]
    };

    ChannelMeasurement {
        signal: signal as f32,
        noise: median / median_to_mean_bias * channel_bins as f32,
    }
}

fn circular_within(i: usize, center: usize, half: usize, n: usize) -> bool {
    let d = i.abs_diff(center);
    d.min(n - d) <= half
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_noise(n: usize, level: f32) -> Vec<f32> {
        vec![level; n]
    }

    /// Expected `noise` field for a flat-spectrum input: `level × channel_bins`
    /// is the raw median-times-bandwidth result; the 1/ln(2) bias correction
    /// inflates it because for flat input median == mean and the correction
    /// over-counts by a constant factor. Real (exponential) noise is verified
    /// separately in `unbiased_ratio_on_exponential_bin_noise` below.
    fn flat_noise_estimate(level: f32, channel_bins: usize) -> f32 {
        level * channel_bins as f32 / std::f32::consts::LN_2
    }

    #[test]
    fn flat_spectrum_signal_matches_sum_and_noise_picks_up_bias_factor() {
        let n = 256;
        let bp = flat_noise(n, 0.001);
        let mut buf = Vec::new();
        let m = measure_channel_from_bins(&bp, 0, 3, 8, MEDIAN_BIAS_EXPONENTIAL, &mut buf);
        assert!((m.signal - 0.007).abs() < 1e-6);
        assert!((m.noise - flat_noise_estimate(0.001, 7)).abs() < 1e-6);
    }

    #[test]
    fn signal_above_flat_noise_separates() {
        let n = 256;
        let mut bp = flat_noise(n, 0.001);
        // Bump the channel bins (centered at DC, ±3)
        for p in bp.iter_mut().take(4) {
            *p = 0.01;
        }
        for p in bp.iter_mut().rev().take(3) {
            *p = 0.01;
        }
        let mut buf = Vec::new();
        let m = measure_channel_from_bins(&bp, 0, 3, 8, MEDIAN_BIAS_EXPONENTIAL, &mut buf);
        assert!((m.signal - 0.07).abs() < 1e-6);
        assert!((m.noise - flat_noise_estimate(0.001, 7)).abs() < 1e-6);
    }

    #[test]
    fn median_rejects_isolated_other_transmitter() {
        let n = 256;
        let mut bp = flat_noise(n, 0.001);
        // Plant a strong signal far from our channel — should be ignored by the median.
        for p in bp[50..60].iter_mut() {
            *p = 0.5;
        }
        let mut buf = Vec::new();
        let m = measure_channel_from_bins(&bp, 0, 3, 8, MEDIAN_BIAS_EXPONENTIAL, &mut buf);
        assert!((m.noise - flat_noise_estimate(0.001, 7)).abs() < 1e-6);
    }

    /// Pure-noise calibration: under exponential per-bin power (the actual
    /// distribution of `|X[k]|²` for thermal noise after FFT), an unbiased
    /// noise estimator should give `signal / noise ≈ 1.0` over many trials.
    /// This pins the 1/ln(2) bias correction; without it the integrated ratio
    /// sits at ≈ 1.443 (≈ +1.6 dB).
    #[test]
    fn unbiased_ratio_on_exponential_bin_noise() {
        let n = 2048;
        let trials = 500;
        let half_bw = 6;
        let edge_skip = (n as f32 * EDGE_SKIP_FRAC) as usize;
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next_uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Top 32 bits as f32 in (0, 1] — avoid log(0) at the unlucky end.
            ((state >> 32) as f32 + 1.0) / (u32::MAX as f32 + 2.0)
        };
        let mut bp = vec![0.0f32; n];
        let mut buf = Vec::new();
        let mut signal_sum = 0.0f64;
        let mut noise_sum = 0.0f64;
        for _ in 0..trials {
            for p in bp.iter_mut() {
                *p = -next_uniform().ln();
            }
            let m = measure_channel_from_bins(
                &bp,
                0,
                half_bw,
                edge_skip,
                MEDIAN_BIAS_EXPONENTIAL,
                &mut buf,
            );
            signal_sum += m.signal as f64;
            noise_sum += m.noise as f64;
        }
        let ratio = (signal_sum / noise_sum) as f32;
        // Expected ratio = 1.0; CV per trial ≈ 0.28 (gamma(13)/median dominates),
        // so 500 trials gives ≈ ±0.013 at one σ. 0.05 is ~4 σ headroom.
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "ratio={ratio} (want ≈1.0)",
        );
    }

    /// Welch path: averaging M=200 chunks pulls the per-bin distribution to
    /// near-Gaussian (Gamma(M, λ/M)), where median ≈ mean. With
    /// `MEDIAN_BIAS_AVERAGED = 1.0` the integrated ratio should sit at 1.0
    /// without the per-chunk 1/ln(2) correction.
    #[test]
    fn averaged_spectrum_unbiased_with_welch_constant() {
        let n = 2048;
        let chunks = 200;
        let trials = 50;
        let half_bw = 6;
        let edge_skip = (n as f32 * EDGE_SKIP_FRAC) as usize;
        let mut state: u64 = 0xB5297A4D;
        let mut next_uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 32) as f32 + 1.0) / (u32::MAX as f32 + 2.0)
        };
        let mut buf = Vec::new();
        let mut signal_sum = 0.0f64;
        let mut noise_sum = 0.0f64;
        let mut avg = vec![0.0f32; n];
        for _ in 0..trials {
            avg.fill(0.0);
            for _ in 0..chunks {
                for p in avg.iter_mut() {
                    *p += -next_uniform().ln();
                }
            }
            let inv = 1.0 / chunks as f32;
            for p in avg.iter_mut() {
                *p *= inv;
            }
            let m = measure_channel_from_bins(
                &avg,
                0,
                half_bw,
                edge_skip,
                MEDIAN_BIAS_AVERAGED,
                &mut buf,
            );
            signal_sum += m.signal as f64;
            noise_sum += m.noise as f64;
        }
        let ratio = (signal_sum / noise_sum) as f32;
        // For M=200 the Gamma median lands ≈ 0.998·mean (Wilson–Hilferty);
        // expected ratio is essentially 1.0. CV across 50 trials is small.
        assert!(
            (ratio - 1.0).abs() < 0.03,
            "ratio={ratio} (want ≈1.0 within 0.03)",
        );
    }
}

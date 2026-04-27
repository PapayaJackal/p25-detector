use num_complex::Complex32;
use rustfft::Fft;

/// Half-bandwidth of the P25 Phase 1 channel kept around the target bin
/// (±6.25 kHz matches the channel raster). Convert to bins at runtime from
/// sample rate.
pub const CHANNEL_HALF_BW_HZ: f32 = 6_250.0;

/// Fraction of the sampled bandwidth on each side of Nyquist excluded from
/// the noise estimate, where the SDR's anti-alias filter rolls off.
pub const EDGE_SKIP_FRAC: f32 = 0.0625;

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
/// filter rolls off), scaled by `2*half_bw_bins+1` to match channel bandwidth.
/// Median is robust to other transmitters present elsewhere in the baseband.
pub fn measure_channel_from_bins(
    bin_power: &[f32],
    center_bin: usize,
    half_bw_bins: usize,
    edge_skip_bins: usize,
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
        noise: median * channel_bins as f32,
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

    #[test]
    fn noise_floor_matches_flat_spectrum() {
        let n = 256;
        let bp = flat_noise(n, 0.001);
        let mut buf = Vec::new();
        let m = measure_channel_from_bins(&bp, 0, 3, 8, &mut buf);
        // 7 channel bins * 0.001 = 0.007 for each
        assert!((m.signal - 0.007).abs() < 1e-6);
        assert!((m.noise - 0.007).abs() < 1e-6);
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
        let m = measure_channel_from_bins(&bp, 0, 3, 8, &mut buf);
        // Signal: 7 * 0.01 = 0.07; Noise reference: 7 * 0.001 = 0.007
        assert!((m.signal - 0.07).abs() < 1e-6);
        assert!((m.noise - 0.007).abs() < 1e-6);
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
        let m = measure_channel_from_bins(&bp, 0, 3, 8, &mut buf);
        assert!((m.noise - 0.007).abs() < 1e-6);
    }
}

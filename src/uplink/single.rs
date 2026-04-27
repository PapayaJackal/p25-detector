use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::Utc;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use tracing::{info, warn};

use crate::audio::Beeper;
use crate::config::Mode;
use crate::dsp::power::{
    compute_snr_db, fft_bin_power, measure_channel_from_bins, to_dbfs, CHANNEL_HALF_BW_HZ,
    EDGE_SKIP_FRAC,
};
use crate::log::{JsonlLogger, Measurement};
use crate::p25::Grant;
use crate::sdr::{IqSource, RtlSdr, SAMPLE_RATE};
use crate::uplink::UplinkWatcher;

/// How long to drain after a retune before trusting the IQ stream. The
/// rtlsdr_mt reader thread feeds a 16-deep crossbeam channel of ~13.6 ms
/// USB buffers (see `sdr::backend`), so up to ~218 ms of pre-retune samples
/// can already be queued when `set_center_hz` returns. Add ~30 ms of
/// R820T PLL settle on top and round to 250 ms. Anything shorter lets stale
/// samples reach the downstream IIR/EMAs and either retrains them on the
/// wrong frequency (CC return) or biases the FFT power estimate (UL).
const RETUNE_SETTLE_MS: u64 = 250;
/// Window over which we integrate uplink power. P25 voice calls start
/// roughly 250–500 ms after the grant and voice frames are 180 ms apiece, so
/// a 200 ms window frequently misses the key-up entirely. 600 ms catches the
/// first 2–3 voice frames in practice; VAD then picks out the keyed portion.
const MEASURE_MS: u64 = 600;
/// FFT length for in-channel/noise extraction. Bin width = sample_rate /
/// FFT_SIZE; chosen so the bin width stays well below the P25 12.5 kHz
/// channel raster across plausible SDR sample rates.
const FFT_SIZE: usize = 2048;
/// VAD threshold (linear): a chunk is considered keyed if in-channel power
/// exceeds the (bias-corrected) noise reference by this factor. The noise
/// estimator is unbiased after the median→mean correction in
/// [`crate::dsp::power`], so the threshold is the actual antenna-side
/// `(S+N)/N` ratio. 4.0 ≈ +6 dB — comfortably above the per-chunk noise
/// tail (13-bin sum has CV ≈ 0.28, so a noise-only chunk crosses 4× at
/// ~10 σ — astronomically rare).
///
/// The threshold is now only a *presence detector*: it gates whether a
/// grant gets an SNR reported at all. The reported SNR itself is computed
/// over every chunk in the window (keyed or not), which sidesteps the
/// conditional-mean floor a hard threshold would otherwise create
/// (`floor_dB ≈ 10·log10(T-1)`: T=4 → +4.8 dB, T=8 → +8.5 dB) on grants
/// whose UL is just barely above the threshold.
const VAD_THRESHOLD_LINEAR: f32 = 4.0;
/// Once VAD has accepted this many FFT chunks, the in-channel mean has
/// converged within ~0.1 dB (CV ≈ 1/√(N · channel_bins)) and further
/// integration just delays our return to the control channel. The window is
/// still capped at MEASURE_MS for the case where keyup never happens.
const MIN_KEYED_CHUNKS: u64 = 80;
/// Baseband offset at which we place the UL channel of interest after
/// retuning. The LO itself is set to `ul_hz - LO_OFFSET_HZ` so the channel
/// lands at this positive baseband frequency rather than at DC. RTL-SDR's
/// LO self-mixing dumps a constant ~18-dB-above-noise spike at bin 0 and
/// IQ imbalance creates an image a bin or two off DC; both would otherwise
/// fall inside our ±6.25 kHz channel window and inflate the measurement.
/// 300 kHz is far enough off DC (≈256 bins at 2.4 Msps / 2048-pt FFT) that
/// the channel ±6.25 kHz can't overlap the corrupted region, while still
/// well inside the SDR's anti-alias passband (≈±1.2 MHz).
const LO_OFFSET_HZ: u32 = 300_000;

pub struct SingleSdrWatcher {
    sdr: RtlSdr,
    cc_freq_hz: u32,
    min_interval: Duration,
    last_measurement: HashMap<u16, Instant>,
    logger: JsonlLogger,
    mode: Mode,
    iq_buf: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
    fft_scratch: Vec<Complex32>,
    bin_power: Vec<f32>,
    noise_buf: Vec<f32>,
    half_bw_bins: usize,
    edge_skip_bins: usize,
    /// FFT bin where the UL channel center lands after the LO offset.
    /// Computed once from [`LO_OFFSET_HZ`] / bin_width.
    center_bin: usize,
    beeper: Option<Beeper>,
}

struct UplinkSnapshot {
    channel_dbfs: f32,
    noise_dbfs: f32,
    snr_db: Option<f32>,
    /// FFT chunks during the measurement window whose in-channel/noise ratio
    /// exceeded [`VAD_THRESHOLD_LINEAR`].
    keyed_count: u64,
    /// FFT chunks measured (early-exit when [`MIN_KEYED_CHUNKS`] is reached
    /// shortens this; otherwise it's [`MEASURE_MS`] worth of chunks).
    chunk_count: u64,
    /// Largest single-chunk in-channel/noise ratio observed in the window.
    /// A real keyup hits the tens-to-hundreds; a noise-tail VAD trigger
    /// barely clears [`VAD_THRESHOLD_LINEAR`].
    peak_ratio: f32,
}

impl SingleSdrWatcher {
    pub fn new(
        sdr: RtlSdr,
        cc_freq_hz: u32,
        min_measure_interval_ms: u64,
        logger: JsonlLogger,
        mode: Mode,
        beeper: Option<Beeper>,
    ) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
        let half_bw_bins = (CHANNEL_HALF_BW_HZ / bin_width).ceil() as usize;
        let edge_skip_bins = (FFT_SIZE as f32 * EDGE_SKIP_FRAC) as usize;
        let center_bin = (LO_OFFSET_HZ as f32 / bin_width).round() as usize;
        Self {
            sdr,
            cc_freq_hz,
            min_interval: Duration::from_millis(min_measure_interval_ms),
            last_measurement: HashMap::new(),
            logger,
            mode,
            iq_buf: vec![Complex32::new(0.0, 0.0); 1 << 16],
            fft,
            fft_scratch: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            bin_power: vec![0.0f32; FFT_SIZE],
            noise_buf: Vec::with_capacity(FFT_SIZE),
            half_bw_bins,
            edge_skip_bins,
            center_bin,
            beeper,
        }
    }

    pub fn read_cc(&mut self, out: &mut [Complex32]) -> Result<usize> {
        self.sdr.read_iq(out)
    }

    fn should_measure(&self, tgid: u16) -> bool {
        match self.last_measurement.get(&tgid) {
            Some(t) => t.elapsed() >= self.min_interval,
            None => true,
        }
    }

    /// Drain ~`RETUNE_SETTLE_MS` of samples and discard them. Sized to flush
    /// the rtlsdr URB queue plus PLL settle (see [`RETUNE_SETTLE_MS`]); used
    /// after every retune so downstream consumers see only post-tune samples.
    fn drain_post_retune(&mut self) -> Result<()> {
        let samples_to_settle = (SAMPLE_RATE as u64 * RETUNE_SETTLE_MS / 1000) as usize;
        let mut drained = 0usize;
        while drained < samples_to_settle {
            let n = self.sdr.read_iq(&mut self.iq_buf)?;
            if n == 0 {
                break;
            }
            drained += n;
        }
        Ok(())
    }

    fn measure_uplink(&mut self, ul_hz: u32) -> Result<UplinkSnapshot> {
        // Tune the LO below the UL channel by [`LO_OFFSET_HZ`] so the channel
        // lands at +LO_OFFSET in baseband (FFT bin `self.center_bin`),
        // keeping it clear of the RTL-SDR's DC self-mix and IQ-imbalance
        // image around bin 0.
        let lo_hz = ul_hz.saturating_sub(LO_OFFSET_HZ);
        self.sdr.set_center_hz(lo_hz)?;
        self.drain_post_retune()?;

        let total_needed = (SAMPLE_RATE as u64 * MEASURE_MS / 1000) as usize;
        let mut collected = 0usize;
        let mut signal_sum = 0.0f64;
        let mut noise_sum = 0.0f64;
        let mut chunk_count = 0u64;
        let mut keyed_count = 0u64;
        let mut peak_ratio = 0.0f32;

        'outer: while collected < total_needed {
            let n = self.sdr.read_iq(&mut self.iq_buf)?;
            if n == 0 {
                break;
            }
            let mut i = 0;
            while i + FFT_SIZE <= n {
                fft_bin_power(
                    &self.iq_buf[i..i + FFT_SIZE],
                    self.fft.as_ref(),
                    &mut self.fft_scratch,
                    &mut self.bin_power,
                );
                let m = measure_channel_from_bins(
                    &self.bin_power,
                    self.center_bin,
                    self.half_bw_bins,
                    self.edge_skip_bins,
                    &mut self.noise_buf,
                );
                signal_sum += m.signal as f64;
                noise_sum += m.noise as f64;
                chunk_count += 1;
                let ratio = if m.noise > 0.0 { m.signal / m.noise } else { 0.0 };
                if ratio > peak_ratio {
                    peak_ratio = ratio;
                }
                if ratio > VAD_THRESHOLD_LINEAR {
                    keyed_count += 1;
                    if keyed_count >= MIN_KEYED_CHUNKS {
                        break 'outer;
                    }
                }
                i += FFT_SIZE;
            }
            collected += n;
        }

        let mean_noise = if chunk_count > 0 {
            (noise_sum / chunk_count as f64) as f32
        } else {
            0.0
        };
        // Population mean over *every* chunk in the window — keyed or not —
        // to avoid the conditional-mean floor a hard VAD cut would impose.
        // The threshold's only job is gating whether SNR gets reported at
        // all; the SNR value itself is the unbiased average channel power
        // divided by the unbiased average noise. Side-effect: brief calls
        // see SNR scaled by their duty cycle in the window, which is the
        // honest answer (you really did receive less energy).
        let mean_signal = if chunk_count > 0 {
            (signal_sum / chunk_count as f64) as f32
        } else {
            0.0
        };
        let (channel_lin, snr_db) = if keyed_count >= MIN_KEYED_CHUNKS {
            (mean_signal, compute_snr_db(mean_signal, mean_noise))
        } else {
            (mean_noise, None)
        };

        Ok(UplinkSnapshot {
            channel_dbfs: to_dbfs(channel_lin),
            noise_dbfs: to_dbfs(mean_noise),
            snr_db,
            keyed_count,
            chunk_count,
            peak_ratio,
        })
    }
}

impl UplinkWatcher for SingleSdrWatcher {
    fn on_grant(&mut self, grant: Grant) -> bool {
        if grant.ul_hz == 0 || grant.ul_hz > u32::MAX as u64 {
            warn!(tgid = grant.tgid, rid = grant.rid, ul_hz = grant.ul_hz, "skipping grant with invalid uplink frequency");
            return false;
        }
        if !self.should_measure(grant.tgid) {
            return false;
        }

        info!(
            tgid = grant.tgid,
            rid = grant.rid,
            ul_hz = grant.ul_hz,
            kind = ?grant.kind,
            "measuring uplink",
        );
        let snap = match self.measure_uplink(grant.ul_hz as u32) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "uplink measurement failed");
                if let Err(e) = self.sdr.set_center_hz(self.cc_freq_hz) {
                    warn!(error = %e, "failed to retune back to CC");
                }
                if let Err(e) = self.drain_post_retune() {
                    warn!(error = %e, "drain after CC retune failed");
                }
                return true;
            }
        };

        if let Err(e) = self.sdr.set_center_hz(self.cc_freq_hz) {
            warn!(error = %e, "failed to retune back to CC");
        }
        if let Err(e) = self.drain_post_retune() {
            warn!(error = %e, "drain after CC retune failed");
        }

        info!(
            tgid = grant.tgid,
            rid = grant.rid,
            kind = ?grant.kind,
            channel_dbfs = snap.channel_dbfs,
            noise_dbfs = snap.noise_dbfs,
            snr_db = snap.snr_db.unwrap_or(f32::NAN),
            keyed = snap.keyed_count,
            chunks = snap.chunk_count,
            peak_ratio = snap.peak_ratio,
            "uplink measured",
        );

        self.last_measurement.insert(grant.tgid, Instant::now());
        self.logger.log(&Measurement {
            ts: Utc::now(),
            tgid: grant.tgid,
            rid: grant.rid,
            dl_hz: grant.dl_hz as u32,
            ul_hz: grant.ul_hz as u32,
            kind: grant.kind,
            channel_dbfs: snap.channel_dbfs,
            noise_dbfs: snap.noise_dbfs,
            snr_db: snap.snr_db,
            keyed_count: Some(snap.keyed_count),
            chunk_count: Some(snap.chunk_count),
            peak_ratio: Some(snap.peak_ratio),
            mode: self.mode,
        });
        if let (Some(b), Some(snr)) = (&self.beeper, snap.snr_db) {
            b.beep(snr);
        }
        true
    }
}

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
    EDGE_SKIP_FRAC, MEDIAN_BIAS_AVERAGED,
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
/// FFT length for in-channel/noise extraction. Bin width = sample_rate /
/// FFT_SIZE; chosen so the bin width stays well below the P25 12.5 kHz
/// channel raster across plausible SDR sample rates.
const FFT_SIZE: usize = 2048;
/// Block length (in milliseconds) over which we Welch-average bin powers
/// before applying the VAD threshold. 50 ms is the sweet spot: long enough
/// to drop per-bin variance to CV ≈ 0.036 (`1/√60` channel-bin samples),
/// short enough to track P25 voice frames (180 ms apiece) and PTT bursts.
const BLOCK_MS: u64 = 50;
/// Number of FFT chunks per Welch-averaged block. Derived from
/// [`SAMPLE_RATE`] and [`BLOCK_MS`] / `FFT_SIZE`. ≈ 58 at 2.4 Msps.
const BLOCK_CHUNKS: usize = (SAMPLE_RATE as usize) * (BLOCK_MS as usize) / 1000 / FFT_SIZE;
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
    measure_window_ms: u64,
    vad_threshold: f32,
    min_keyed_blocks: u64,
    last_measurement: HashMap<u16, Instant>,
    logger: JsonlLogger,
    mode: Mode,
    iq_buf: Vec<Complex32>,
    fft: Arc<dyn Fft<f32>>,
    fft_scratch: Vec<Complex32>,
    bin_power: Vec<f32>,
    /// Per-block accumulator: sums per-bin power across the `BLOCK_CHUNKS`
    /// chunks of the current block, then divides to form a Welch-averaged
    /// spectrum for that block before VAD thresholding. Reset between blocks.
    block_bin_sum: Vec<f32>,
    /// Welch sum across blocks that passed the VAD threshold. Each summand
    /// is itself a per-block averaged spectrum, so dividing by `keyed_blocks`
    /// at the end produces a low-variance averaged spectrum used for the
    /// final SNR. Reset between measurements.
    keyed_bin_sum: Vec<f32>,
    noise_buf: Vec<f32>,
    half_bw_bins: usize,
    edge_skip_bins: usize,
    /// FFT bin where the UL channel center lands after the LO offset.
    /// Computed once from [`LO_OFFSET_HZ`] / bin_width.
    center_bin: usize,
    beeper: Option<Beeper>,
}

struct UplinkSnapshot {
    /// Mean in-channel power (dBFS) across every block in the window —
    /// keyed or not — so a sub-gate signal still shows up as a small
    /// `channel_dbfs - noise_dbfs` gap.
    channel_dbfs: f32,
    /// Mean same-bandwidth noise reference (dBFS) across every block.
    noise_dbfs: f32,
    /// SNR (dB) computed from the keyed-block Welch-averaged spectrum, i.e.
    /// "how strong was the signal *when present*". `None` when fewer than
    /// `min_keyed_blocks` blocks crossed the VAD threshold.
    snr_db: Option<f32>,
    /// SNR (dB) of the single best block observed in the window. Preserves
    /// brief-PTT-burst sensitivity that the keyed-block-averaged `snr_db`
    /// would lose if only one block crossed and was integrated alone.
    peak_block_snr_db: Option<f32>,
    /// Welch-averaged blocks (50 ms each) whose in-channel/noise ratio
    /// crossed `vad_threshold`.
    keyed_blocks: u64,
    /// Total complete blocks measured in the window.
    block_count: u64,
    /// Largest single-block in-channel/noise ratio observed.
    peak_ratio: f32,
    /// Argmax bin (excluding DC) on the strongest block. For a
    /// correctly-tuned UL the peak should land within `center_bin ±
    /// half_bw_bins`.
    peak_bin: usize,
}

impl SingleSdrWatcher {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sdr: RtlSdr,
        cc_freq_hz: u32,
        min_measure_interval_ms: u64,
        measure_window_ms: u64,
        vad_threshold: f32,
        min_keyed_blocks: u64,
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
            measure_window_ms,
            vad_threshold,
            min_keyed_blocks,
            last_measurement: HashMap::new(),
            logger,
            mode,
            iq_buf: vec![Complex32::new(0.0, 0.0); 1 << 16],
            fft,
            fft_scratch: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            bin_power: vec![0.0f32; FFT_SIZE],
            block_bin_sum: vec![0.0f32; FFT_SIZE],
            keyed_bin_sum: vec![0.0f32; FFT_SIZE],
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

        // Reset per-measurement Welch accumulators.
        self.block_bin_sum.fill(0.0);
        self.keyed_bin_sum.fill(0.0);

        let total_needed = (SAMPLE_RATE as u64 * self.measure_window_ms / 1000) as usize;
        let mut collected = 0usize;
        let mut chunks_in_block = 0usize;
        let mut block_count = 0u64;
        let mut keyed_blocks = 0u64;
        let mut window_signal_sum = 0.0f64;
        let mut window_noise_sum = 0.0f64;
        let mut peak_ratio = 0.0f32;
        let mut peak_block_snr_db: Option<f32> = None;
        let mut peak_bin = self.center_bin;

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
                for (acc, p) in self.block_bin_sum.iter_mut().zip(self.bin_power.iter()) {
                    *acc += *p;
                }
                chunks_in_block += 1;
                i += FFT_SIZE;

                if chunks_in_block >= BLOCK_CHUNKS {
                    // Finalize this block: divide the per-bin sum by the chunk
                    // count to get a Welch-averaged spectrum, then derive
                    // signal/noise from it. The averaged spectrum has per-bin
                    // CV ≈ 1/√BLOCK_CHUNKS (~0.13 at 60 chunks) versus the
                    // single-chunk 1.0, dropping the channel-bin sum's CV
                    // from ~0.28 to ~0.04 — that's why the VAD threshold can
                    // be set close to 1 without false-positives.
                    let inv = 1.0 / chunks_in_block as f32;
                    for p in self.block_bin_sum.iter_mut() {
                        *p *= inv;
                    }
                    let m = measure_channel_from_bins(
                        &self.block_bin_sum,
                        self.center_bin,
                        self.half_bw_bins,
                        self.edge_skip_bins,
                        MEDIAN_BIAS_AVERAGED,
                        &mut self.noise_buf,
                    );
                    let ratio = if m.noise > 0.0 { m.signal / m.noise } else { 0.0 };
                    block_count += 1;
                    window_signal_sum += m.signal as f64;
                    window_noise_sum += m.noise as f64;

                    if ratio > peak_ratio {
                        peak_ratio = ratio;
                        peak_bin = argmax_bin_excl_dc(&self.block_bin_sum);
                        // Only report a peak SNR when the block actually
                        // cleared the VAD gate. Otherwise we'd be reporting
                        // 10·log10(noise_max_ratio − 1) for whatever noise
                        // spike happened to be loudest — meaningless and
                        // visually indistinguishable from a real weak signal.
                        peak_block_snr_db = if ratio > self.vad_threshold {
                            compute_snr_db(m.signal, m.noise)
                        } else {
                            None
                        };
                    }

                    if ratio > self.vad_threshold {
                        keyed_blocks += 1;
                        for (k, b) in self.keyed_bin_sum.iter_mut().zip(self.block_bin_sum.iter())
                        {
                            *k += *b;
                        }
                    }

                    self.block_bin_sum.fill(0.0);
                    chunks_in_block = 0;

                    if keyed_blocks >= self.min_keyed_blocks {
                        break 'outer;
                    }
                }
            }
            collected += n;
        }

        let (mean_signal, mean_noise) = if block_count > 0 {
            let n = block_count as f64;
            (
                (window_signal_sum / n) as f32,
                (window_noise_sum / n) as f32,
            )
        } else {
            (0.0, 0.0)
        };

        // Final SNR: Welch-average the keyed-block spectra (already each a
        // Welch average over BLOCK_CHUNKS chunks → effective shape
        // ≥ BLOCK_CHUNKS·keyed_blocks, well past the regime where median ≈
        // mean) and re-extract signal/noise from the averaged spectrum.
        let snr_db = if keyed_blocks >= self.min_keyed_blocks {
            let inv = 1.0 / keyed_blocks as f32;
            for p in self.keyed_bin_sum.iter_mut() {
                *p *= inv;
            }
            let m = measure_channel_from_bins(
                &self.keyed_bin_sum,
                self.center_bin,
                self.half_bw_bins,
                self.edge_skip_bins,
                MEDIAN_BIAS_AVERAGED,
                &mut self.noise_buf,
            );
            compute_snr_db(m.signal, m.noise)
        } else {
            None
        };

        Ok(UplinkSnapshot {
            channel_dbfs: to_dbfs(mean_signal),
            noise_dbfs: to_dbfs(mean_noise),
            snr_db,
            peak_block_snr_db,
            keyed_blocks,
            block_count,
            peak_ratio,
            peak_bin,
        })
    }
}

impl UplinkWatcher for SingleSdrWatcher {
    fn on_grant(&mut self, grant: Grant) -> bool {
        if grant.ul_hz == 0 || grant.ul_hz > u32::MAX as u64 {
            warn!(tgid = grant.tgid, rid = grant.rid, ul_hz = grant.ul_hz, "skipping grant with invalid uplink frequency");
            return false;
        }
        // Mid-call grant updates (GRP_VOICE_GRANT_UPDATE{,_EXPLICIT}) carry no
        // source RID. The keyup happened on the original grant, so by the time
        // we retune we'd catch only the tail — or nothing if the call has
        // already ended. Skip them entirely; we'll measure on the next fresh
        // grant from this talker.
        if grant.rid == 0 {
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
            peak_block_snr_db = snap.peak_block_snr_db.unwrap_or(f32::NAN),
            keyed_blocks = snap.keyed_blocks,
            blocks = snap.block_count,
            peak_ratio = snap.peak_ratio,
            peak_bin = snap.peak_bin,
            expected_bin = self.center_bin,
            "uplink measured",
        );

        self.last_measurement.insert(grant.tgid, Instant::now());
        self.logger.log(&Measurement {
            ts: Utc::now(),
            tgid: grant.tgid,
            rid: grant.rid,
            iden: (grant.channel_id >> 12) as u8,
            channel: grant.channel_id & 0xFFF,
            dl_hz: grant.dl_hz as u32,
            ul_hz: grant.ul_hz as u32,
            kind: grant.kind,
            channel_dbfs: snap.channel_dbfs,
            noise_dbfs: snap.noise_dbfs,
            snr_db: snap.snr_db,
            peak_block_snr_db: snap.peak_block_snr_db,
            keyed_blocks: Some(snap.keyed_blocks),
            block_count: Some(snap.block_count),
            peak_ratio: Some(snap.peak_ratio),
            peak_bin: Some(snap.peak_bin),
            expected_bin: Some(self.center_bin),
            mode: self.mode,
        });
        if let (Some(b), Some(snr)) = (&self.beeper, snap.snr_db) {
            b.beep(snr);
        }
        true
    }
}

/// Argmax of `bin_power` excluding the DC self-mix region (bins 0..3 and
/// the wrap-around mirror N-3..N). Used as a one-shot alignment check on
/// the block that produced `peak_ratio`: if the loudest non-DC bin sits
/// far from `center_bin`, the SDR didn't tune where we asked.
fn argmax_bin_excl_dc(bin_power: &[f32]) -> usize {
    const DC_GUARD: usize = 3;
    let n = bin_power.len();
    let mut best_idx = DC_GUARD;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &p) in bin_power.iter().enumerate() {
        if i < DC_GUARD || i + DC_GUARD >= n {
            continue;
        }
        if p > best_val {
            best_val = p;
            best_idx = i;
        }
    }
    best_idx
}

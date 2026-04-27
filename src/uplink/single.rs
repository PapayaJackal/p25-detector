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

const RETUNE_SETTLE_MS: u64 = 60;
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
/// exceeds the noise reference by this factor. 2.0 ≈ +3 dB — the boundary
/// where C4FM voice starts to be decodable.
const VAD_THRESHOLD_LINEAR: f32 = 2.0;
/// Once VAD has accepted this many FFT chunks, the in-channel mean has
/// converged within ~0.1 dB (CV ≈ 1/√(N · channel_bins)) and further
/// integration just delays our return to the control channel. The window is
/// still capped at MEASURE_MS for the case where keyup never happens.
const MIN_KEYED_CHUNKS: u64 = 80;

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
    beeper: Option<Beeper>,
}

struct UplinkSnapshot {
    channel_dbfs: f32,
    noise_dbfs: f32,
    snr_db: Option<f32>,
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

    fn measure_uplink(&mut self, ul_hz: u32) -> Result<UplinkSnapshot> {
        self.sdr.set_center_hz(ul_hz)?;
        let samples_to_settle = (SAMPLE_RATE as u64 * RETUNE_SETTLE_MS / 1000) as usize;
        let mut drained = 0usize;
        while drained < samples_to_settle {
            let n = self.sdr.read_iq(&mut self.iq_buf)?;
            if n == 0 {
                break;
            }
            drained += n;
        }

        let total_needed = (SAMPLE_RATE as u64 * MEASURE_MS / 1000) as usize;
        let mut collected = 0usize;
        let mut signal_keyed_sum = 0.0f64;
        let mut signal_keyed_count = 0u64;
        let mut noise_sum = 0.0f64;
        let mut noise_count = 0u64;

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
                    0,
                    self.half_bw_bins,
                    self.edge_skip_bins,
                    &mut self.noise_buf,
                );
                noise_sum += m.noise as f64;
                noise_count += 1;
                if m.signal > m.noise * VAD_THRESHOLD_LINEAR {
                    signal_keyed_sum += m.signal as f64;
                    signal_keyed_count += 1;
                    if signal_keyed_count >= MIN_KEYED_CHUNKS {
                        break 'outer;
                    }
                }
                i += FFT_SIZE;
            }
            collected += n;
        }

        let mean_noise = if noise_count > 0 {
            (noise_sum / noise_count as f64) as f32
        } else {
            0.0
        };
        let (channel_lin, snr_db) = if signal_keyed_count > 0 {
            let s = (signal_keyed_sum / signal_keyed_count as f64) as f32;
            (s, compute_snr_db(s, mean_noise))
        } else {
            (mean_noise, None)
        };

        Ok(UplinkSnapshot {
            channel_dbfs: to_dbfs(channel_lin),
            noise_dbfs: to_dbfs(mean_noise),
            snr_db,
        })
    }
}

impl UplinkWatcher for SingleSdrWatcher {
    fn on_grant(&mut self, grant: Grant) {
        if grant.ul_hz == 0 || grant.ul_hz > u32::MAX as u64 {
            warn!(tgid = grant.tgid, rid = grant.rid, ul_hz = grant.ul_hz, "skipping grant with invalid uplink frequency");
            return;
        }
        if !self.should_measure(grant.tgid) {
            return;
        }

        info!(tgid = grant.tgid, rid = grant.rid, ul_hz = grant.ul_hz, "measuring uplink");
        let snap = match self.measure_uplink(grant.ul_hz as u32) {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "uplink measurement failed");
                if let Err(e) = self.sdr.set_center_hz(self.cc_freq_hz) {
                    warn!(error = %e, "failed to retune back to CC");
                }
                return;
            }
        };

        if let Err(e) = self.sdr.set_center_hz(self.cc_freq_hz) {
            warn!(error = %e, "failed to retune back to CC");
        }

        info!(
            tgid = grant.tgid,
            rid = grant.rid,
            channel_dbfs = snap.channel_dbfs,
            noise_dbfs = snap.noise_dbfs,
            snr_db = snap.snr_db.unwrap_or(f32::NAN),
            "uplink measured",
        );

        self.last_measurement.insert(grant.tgid, Instant::now());
        self.logger.log(&Measurement {
            ts: Utc::now(),
            tgid: grant.tgid,
            rid: grant.rid,
            dl_hz: grant.dl_hz as u32,
            ul_hz: grant.ul_hz as u32,
            channel_dbfs: snap.channel_dbfs,
            noise_dbfs: snap.noise_dbfs,
            snr_db: snap.snr_db,
            mode: self.mode,
        });
        if let (Some(b), Some(snr)) = (&self.beeper, snap.snr_db) {
            b.beep(snr);
        }
    }
}

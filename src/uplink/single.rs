use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::Utc;
use num_complex::Complex32;
use tracing::{info, warn};

use crate::config::Mode;
use crate::dsp::power::{mean_power, to_dbfs};
use crate::log::{JsonlLogger, Measurement};
use crate::p25::Grant;
use crate::sdr::{IqSource, RtlSdr, SAMPLE_RATE};
use crate::uplink::UplinkWatcher;

const RETUNE_SETTLE_MS: u64 = 60;
/// Window over which we integrate uplink power. P25 voice calls start
/// roughly 250–500 ms after the grant and voice frames are 180 ms apiece, so
/// a 200 ms window frequently misses the key-up entirely. 600 ms catches the
/// first 2–3 voice frames in practice; the peak channel also lets us pick
/// out the actual TX level even when only a fraction of the window is keyed.
const MEASURE_MS: u64 = 600;

pub struct SingleSdrWatcher {
    sdr: RtlSdr,
    cc_freq_hz: u32,
    min_interval: Duration,
    last_measurement: HashMap<u16, Instant>,
    logger: JsonlLogger,
    mode: Mode,
    iq_buf: Vec<Complex32>,
}

impl SingleSdrWatcher {
    pub fn new(
        sdr: RtlSdr,
        cc_freq_hz: u32,
        min_measure_interval_ms: u64,
        logger: JsonlLogger,
        mode: Mode,
    ) -> Self {
        Self {
            sdr,
            cc_freq_hz,
            min_interval: Duration::from_millis(min_measure_interval_ms),
            last_measurement: HashMap::new(),
            logger,
            mode,
            iq_buf: vec![Complex32::new(0.0, 0.0); 1 << 16],
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

    /// Returns `(mean_dbfs, peak_dbfs)`. Peak is the max mean-power over the
    /// `read_iq`-sized chunks the driver hands back (~25–30 ms each), which
    /// gives us the actual TX level when a mobile is keyed for only part of
    /// the window. Mean tracks the full-window average, useful for noise
    /// floor comparison.
    fn measure_uplink_rssi(&mut self, ul_hz: u32) -> Result<(f32, f32)> {
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
        let mut total_power_weighted = 0.0f64;
        let mut peak_power = 0.0f32;

        while collected < total_needed {
            let n = self.sdr.read_iq(&mut self.iq_buf)?;
            if n == 0 {
                break;
            }
            let power = mean_power(&self.iq_buf[..n]);
            total_power_weighted += (power as f64) * (n as f64);
            if power > peak_power {
                peak_power = power;
            }
            collected += n;
        }

        let avg = if collected > 0 {
            (total_power_weighted / collected as f64) as f32
        } else {
            0.0
        };
        Ok((to_dbfs(avg), to_dbfs(peak_power)))
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

        info!(tgid = grant.tgid, rid = grant.rid, ul_hz = grant.ul_hz, "measuring uplink RSSI");
        let (rssi_mean, rssi_peak) = match self.measure_uplink_rssi(grant.ul_hz as u32) {
            Ok(r) => r,
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
            rssi_mean_dbfs = rssi_mean,
            rssi_peak_dbfs = rssi_peak,
            "uplink RSSI measured",
        );

        self.last_measurement.insert(grant.tgid, Instant::now());
        self.logger.log(&Measurement {
            ts: Utc::now(),
            tgid: grant.tgid,
            rid: grant.rid,
            dl_hz: grant.dl_hz as u32,
            ul_hz: grant.ul_hz as u32,
            rssi_dbfs: rssi_mean,
            rssi_peak_dbfs: Some(rssi_peak),
            mode: self.mode,
        });
    }
}

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use tracing::warn;

use crate::audio::Beeper;
use crate::config::Mode;
use crate::dsp::power::to_dbfs;
use crate::log::{JsonlLogger, Measurement};
use crate::p25::Grant;
use crate::sdr::{IqSource, RtlSdr};
use crate::uplink::UplinkWatcher;

const FFT_SIZE: usize = 2048;

pub struct DualSdrWatcher {
    sdr: RtlSdr,
    center_hz: u32,
    sample_rate: u32,
    fft: Arc<dyn Fft<f32>>,
    logger: JsonlLogger,
    mode: Mode,
    iq_buf: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    bin_power: Vec<f32>,
    avg_alpha: f32,
    beeper: Option<Beeper>,
}

impl DualSdrWatcher {
    pub fn new(
        sdr: RtlSdr,
        center_hz: u32,
        sample_rate: u32,
        logger: JsonlLogger,
        mode: Mode,
        beeper: Option<Beeper>,
    ) -> Result<Self> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        Ok(Self {
            sdr,
            center_hz,
            sample_rate,
            fft,
            logger,
            mode,
            iq_buf: vec![Complex32::new(0.0, 0.0); 1 << 15],
            fft_scratch: vec![Complex32::new(0.0, 0.0); FFT_SIZE],
            bin_power: vec![0.0f32; FFT_SIZE],
            avg_alpha: 0.1,
            beeper,
        })
    }

    pub fn pump(&mut self) -> Result<()> {
        let n = self.sdr.read_iq(&mut self.iq_buf)?;
        let mut i = 0;
        while i + FFT_SIZE <= n {
            self.fft_scratch
                .copy_from_slice(&self.iq_buf[i..i + FFT_SIZE]);
            self.fft.process(&mut self.fft_scratch);
            let norm = (FFT_SIZE as f32).powi(2);
            for (k, c) in self.fft_scratch.iter().enumerate() {
                let p = c.norm_sqr() / norm;
                self.bin_power[k] = (1.0 - self.avg_alpha) * self.bin_power[k] + self.avg_alpha * p;
            }
            i += FFT_SIZE;
        }
        Ok(())
    }

    fn bin_for_freq(&self, ul_hz: u32) -> Option<usize> {
        let delta = ul_hz as i64 - self.center_hz as i64;
        let bin_width = self.sample_rate as f64 / FFT_SIZE as f64;
        let raw = (delta as f64 / bin_width).round() as i64;
        let half = FFT_SIZE as i64 / 2;
        if raw < -half || raw >= half {
            return None;
        }
        Some(raw.rem_euclid(FFT_SIZE as i64) as usize)
    }
}

impl UplinkWatcher for DualSdrWatcher {
    fn on_grant(&mut self, grant: Grant) {
        if grant.ul_hz == 0 || grant.ul_hz > u32::MAX as u64 {
            warn!(tgid = grant.tgid, "uplink frequency out of range for dual-sdr watcher");
            return;
        }
        let ul_hz = grant.ul_hz as u32;
        let bin = match self.bin_for_freq(ul_hz) {
            Some(b) => b,
            None => {
                warn!(
                    ul_hz,
                    center = self.center_hz,
                    "uplink frequency falls outside wideband capture"
                );
                return;
            }
        };

        let p = self.bin_power[bin];
        let rssi = to_dbfs(p);

        self.logger.log(&Measurement {
            ts: Utc::now(),
            tgid: grant.tgid,
            rid: grant.rid,
            dl_hz: grant.dl_hz as u32,
            ul_hz,
            rssi_dbfs: rssi,
            rssi_peak_dbfs: None,
            mode: self.mode,
        });
        if let Some(b) = &self.beeper {
            b.beep(rssi);
        }
    }
}

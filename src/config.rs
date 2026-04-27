use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    SingleSdr,
    DualSdr,
}

#[derive(Parser, Debug)]
#[command(
    name = "p25-detector",
    about = "Passive P25 Phase 1 mobile-radio proximity detector",
    version
)]
pub struct Cli {
    /// Control-channel frequency, in Hz (supports `851012500` or `851.0125e6`)
    #[arg(long, value_parser = parse_freq)]
    pub cc_freq: f64,

    /// Comma-separated list of TGIDs whose uplink we should measure. If
    /// omitted, every grant on the control channel is measured.
    #[arg(long, value_delimiter = ',')]
    pub watch_tgid: Vec<u16>,

    /// Uplink-watch mode
    #[arg(long, value_enum, default_value_t = Mode::SingleSdr)]
    pub mode: Mode,

    /// RTL-SDR device index for the control channel
    #[arg(long, default_value_t = 0)]
    pub cc_device: u32,

    /// RTL-SDR device index for the uplink (dual-sdr mode only)
    #[arg(long)]
    pub uplink_device: Option<u32>,

    /// Center frequency for the wideband uplink capture (dual-sdr mode only)
    #[arg(long, value_parser = parse_freq)]
    pub uplink_center: Option<f64>,

    /// Tuner gain in tenths of a dB (e.g. 400 = 40.0 dB). Omit for AGC.
    #[arg(long)]
    pub gain: Option<i32>,

    /// Tuner frequency correction in ppm (e.g. -4 if kalibrate reports
    /// -3.6 ppm). Defaults to 0.
    #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
    pub ppm: i32,

    /// Minimum time between RSSI measurements for the same TGID (single-sdr mode)
    #[arg(long, default_value_t = 5000)]
    pub min_measure_interval_ms: u64,

    /// Single-SDR-only: how long to integrate uplink power per grant, in
    /// milliseconds. P25 voice calls start ~250–500 ms after the grant and
    /// voice frames are 180 ms apiece, so the default 600 ms catches the
    /// first 2–3 frames; longer windows catch late keyups and short bursts
    /// at the cost of more time off the control channel.
    #[arg(long, default_value_t = 600)]
    pub measure_window_ms: u64,

    /// Single-SDR-only: per-block linear (S+N)/N ratio above which a 50 ms
    /// Welch-averaged block counts as "keyed". With ~60 chunks per block
    /// the per-block CV ≈ 0.036, so 1.5 (≈ +1.8 dB above noise) is ~14 σ —
    /// false positives are negligible while still catching real signals
    /// down to ~+1 dB antenna SNR. Lower toward 1.1 if you're chasing very
    /// faint transmitters.
    #[arg(long, default_value_t = 1.5)]
    pub vad_threshold: f32,

    /// Single-SDR-only: minimum number of keyed *blocks* (50 ms Welch
    /// averages) required before the keyed-segment `snr_db` is reported.
    /// 4 blocks ≈ 200 ms of activity, giving the keyed-segment Welch sum
    /// ~1500 chunk-equivalents and SNR precision ≈ 0.4 dB. Drop to 1 to
    /// surface single-block PTT bursts (or just consult the always-reported
    /// `peak_block_snr_db` field for the strongest single block).
    #[arg(long, default_value_t = 4)]
    pub min_keyed_blocks: u64,

    /// Output log path. `-` or omitted means stdout.
    #[arg(long)]
    pub log: Option<PathBuf>,

    /// Silence the audio beep that normally plays on each measurement.
    #[arg(long, default_value_t = false)]
    pub no_beep: bool,

    /// Suppress the beep when SNR (dB) is below this value (e.g. `8`). SNR
    /// is gain-invariant, so this threshold doesn't need re-tuning when
    /// `--gain` changes. Default: no threshold.
    #[arg(long, allow_hyphen_values = true)]
    pub beep_snr_min: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub mode: Mode,
    pub cc_freq_hz: u32,
    pub cc_device: u32,
    pub uplink_device: Option<u32>,
    pub uplink_center_hz: Option<u32>,
    pub gain: Option<i32>,
    pub ppm: i32,
    pub watched_tgids: Vec<u16>,
    pub min_measure_interval_ms: u64,
    pub measure_window_ms: u64,
    pub vad_threshold: f32,
    pub min_keyed_blocks: u64,
    pub log_path: Option<PathBuf>,
    pub beep: bool,
    pub beep_snr_min_db: f32,
}

impl Cli {
    pub fn into_runtime(self) -> Result<RuntimeConfig> {
        if self.mode == Mode::DualSdr && self.uplink_center.is_none() {
            bail!("--mode dual-sdr requires --uplink-center");
        }
        let cc_freq_hz = freq_to_u32(self.cc_freq, "cc-freq")?;
        let uplink_center_hz = self
            .uplink_center
            .map(|f| freq_to_u32(f, "uplink-center"))
            .transpose()?;

        let log_path = match self.log {
            Some(p) if p.as_os_str() == "-" => None,
            other => other,
        };

        Ok(RuntimeConfig {
            mode: self.mode,
            cc_freq_hz,
            cc_device: self.cc_device,
            uplink_device: self.uplink_device,
            uplink_center_hz,
            gain: self.gain,
            ppm: self.ppm,
            watched_tgids: self.watch_tgid,
            min_measure_interval_ms: self.min_measure_interval_ms,
            measure_window_ms: self.measure_window_ms,
            vad_threshold: self.vad_threshold,
            min_keyed_blocks: self.min_keyed_blocks,
            log_path,
            beep: !self.no_beep,
            beep_snr_min_db: self.beep_snr_min.unwrap_or(f32::NEG_INFINITY),
        })
    }
}

fn parse_freq(s: &str) -> Result<f64, String> {
    s.parse::<f64>().map_err(|e| format!("bad frequency {s:?}: {e}"))
}

fn freq_to_u32(f: f64, name: &str) -> Result<u32> {
    if !f.is_finite() || !(24e6..=1_766e6).contains(&f) {
        bail!("{name} {f} out of RTL-SDR tuning range (24 MHz – 1766 MHz)");
    }
    Ok(f.round() as u32)
}

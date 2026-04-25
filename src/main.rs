use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use clap::Parser;
use num_complex::Complex32;

mod audio;
mod config;
mod dsp;
mod log;
mod p25;
mod sdr;
mod uplink;

use audio::Beeper;
use config::{Cli, Mode, RuntimeConfig};
use p25::Decoder;
use sdr::{IqSource, RtlSdr, SAMPLE_RATE};
use uplink::{UplinkWatcher, dual::DualSdrWatcher, single::SingleSdrWatcher};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cfg = Cli::parse().into_runtime()?;

    tracing::info!(
        mode = ?cfg.mode,
        cc_freq_hz = cfg.cc_freq_hz,
        watched = ?cfg.watched_tgids,
        "starting p25-detector",
    );

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }

    match cfg.mode {
        Mode::SingleSdr => run_single(cfg, stop),
        Mode::DualSdr => run_dual(cfg, stop),
    }
}

fn make_beeper(enabled: bool, rssi_min_dbfs: f32) -> Option<Beeper> {
    if !enabled {
        return None;
    }
    match Beeper::try_new(rssi_min_dbfs) {
        Ok(b) => Some(b),
        Err(e) => {
            tracing::warn!(error = %e, "audio output unavailable; continuing silently");
            None
        }
    }
}

fn run_single(cfg: RuntimeConfig, stop: Arc<AtomicBool>) -> Result<()> {
    let logger = log::JsonlLogger::open(cfg.log_path.as_deref())?;
    let beeper = make_beeper(cfg.beep, cfg.beep_rssi_min_dbfs);
    let sdr = RtlSdr::open(cfg.cc_device, cfg.cc_freq_hz, SAMPLE_RATE, cfg.gain, cfg.ppm)
        .context("opening RTL-SDR on CC")?;

    let mut decoder = Decoder::new(SAMPLE_RATE, cfg.watched_tgids.clone());
    let mut watcher = SingleSdrWatcher::new(
        sdr,
        cfg.cc_freq_hz,
        cfg.min_measure_interval_ms,
        logger,
        cfg.mode,
        beeper,
    );

    let mut buf = vec![Complex32::new(0.0, 0.0); 1 << 16];
    while !stop.load(Ordering::SeqCst) {
        let n = watcher.read_cc(&mut buf)?;
        for grant in decoder.process(&buf[..n]) {
            watcher.on_grant(grant);
        }
    }
    Ok(())
}

fn run_dual(cfg: RuntimeConfig, stop: Arc<AtomicBool>) -> Result<()> {
    let logger = log::JsonlLogger::open(cfg.log_path.as_deref())?;
    let beeper = make_beeper(cfg.beep, cfg.beep_rssi_min_dbfs);
    let uplink_center = cfg
        .uplink_center_hz
        .context("dual-sdr mode requires --uplink-center")?;
    let uplink_device = cfg.uplink_device.unwrap_or(1);

    let mut cc = RtlSdr::open(cfg.cc_device, cfg.cc_freq_hz, SAMPLE_RATE, cfg.gain, cfg.ppm)?;
    let uplink_sdr =
        RtlSdr::open(uplink_device, uplink_center, SAMPLE_RATE, cfg.gain, cfg.ppm)
            .context("opening RTL-SDR for uplink watcher")?;
    let mut decoder = Decoder::new(SAMPLE_RATE, cfg.watched_tgids.clone());
    let mut watcher = DualSdrWatcher::new(
        uplink_sdr,
        uplink_center,
        SAMPLE_RATE,
        logger,
        cfg.mode,
        beeper,
    )?;

    let mut buf = vec![Complex32::new(0.0, 0.0); 1 << 16];
    while !stop.load(Ordering::SeqCst) {
        let n = cc.read_iq(&mut buf)?;
        for grant in decoder.process(&buf[..n]) {
            watcher.on_grant(grant);
        }
        watcher.pump()?;
    }
    Ok(())
}

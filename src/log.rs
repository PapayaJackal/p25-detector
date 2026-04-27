use std::fs::OpenOptions;
use std::io::{BufWriter, Stdout, Write, stdout};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::config::Mode;
use crate::p25::GrantKind;

#[derive(Debug, Serialize)]
pub struct Measurement {
    pub ts: DateTime<Utc>,
    pub tgid: u16,
    pub rid: u32,
    /// Band-plan identifier (top 4 bits of the TSBK channel ID). Lets you
    /// tell at a glance which iden table — and thus which band — the grant
    /// landed on, without back-deriving from `dl_hz`.
    pub iden: u8,
    /// Channel number within the iden table (low 12 bits of the TSBK channel
    /// ID). With `iden` it identifies the band-plan slot the site actually
    /// transmitted: `dl_hz = base[iden] + channel · step[iden]`.
    pub channel: u16,
    pub dl_hz: u32,
    pub ul_hz: u32,
    /// Which TSBK opcode produced the grant — distinguishes a fresh
    /// allocation from a mid-call update, where keying is already in
    /// progress on retune.
    pub kind: GrantKind,
    /// In-channel power (S+N) in dBFS. For single-SDR mode this is the mean
    /// over every chunk in the measurement window (keyed or not), so
    /// `channel_dbfs - noise_dbfs` is meaningful even when `snr_db` is null.
    /// For dual-SDR mode it is a snapshot from the EMA-smoothed bin powers
    /// at grant time.
    pub channel_dbfs: f32,
    /// Same-bandwidth noise floor reference (N) in dBFS, estimated from the
    /// median power of out-of-channel bins in the captured baseband.
    pub noise_dbfs: f32,
    /// Signal-to-noise ratio in dB: 10·log10((channel - noise) / noise) in
    /// linear units. For single-SDR mode, computed from a Welch average
    /// across all keyed blocks ("how strong was the signal when present",
    /// integrated for variance). `None` when fewer than `min_keyed_blocks`
    /// blocks crossed the VAD threshold. Stable across gain settings,
    /// unlike the raw dBFS fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snr_db: Option<f32>,
    /// Single-SDR-only diagnostic: SNR (dB) of the single strongest 50 ms
    /// block observed in the window. Preserves brief-PTT sensitivity even
    /// when `snr_db` is null because `min_keyed_blocks` wasn't met.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_block_snr_db: Option<f32>,
    /// Single-SDR-only diagnostic: how many 50 ms Welch-averaged blocks
    /// crossed the VAD threshold. A real keyup looks like a high
    /// `keyed_blocks / block_count` ratio; a marginal trigger sits at or
    /// just past `min_keyed_blocks`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyed_blocks: Option<u64>,
    /// Single-SDR-only diagnostic: total complete blocks measured in the
    /// window. Smaller than the configured measure-window block budget
    /// when early-exit fired.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_count: Option<u64>,
    /// Single-SDR-only diagnostic: maximum single-block in-channel/noise
    /// ratio observed. A real keyup hits the tens-to-hundreds; a noise-tail
    /// trigger barely clears the VAD threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_ratio: Option<f32>,
    /// Single-SDR-only diagnostic: argmax FFT bin (excluding DC) on the
    /// strongest block. Compare against `expected_bin`: if they differ by
    /// more than the channel half-bandwidth, the tuner didn't land where
    /// we asked (or a stronger interferer is dominating the band).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_bin: Option<usize>,
    /// Single-SDR-only diagnostic: bin where the UL channel center is
    /// expected to land given `LO_OFFSET_HZ` and the FFT bin width.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_bin: Option<usize>,
    pub mode: Mode,
}

enum Sink {
    Stdout(Stdout),
    File(BufWriter<std::fs::File>),
}

impl Sink {
    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        match self {
            Sink::Stdout(s) => {
                let mut h = s.lock();
                h.write_all(line.as_bytes())?;
                h.write_all(b"\n")
            }
            Sink::File(f) => {
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
                f.flush()
            }
        }
    }
}

#[derive(Clone)]
pub struct JsonlLogger {
    inner: Arc<Mutex<Sink>>,
}

impl JsonlLogger {
    pub fn open(path: Option<&Path>) -> Result<Self> {
        let sink = match path {
            Some(p) => {
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .with_context(|| format!("opening log {p:?}"))?;
                Sink::File(BufWriter::new(f))
            }
            None => Sink::Stdout(stdout()),
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(sink)),
        })
    }

    pub fn log(&self, m: &Measurement) {
        let line = match serde_json::to_string(m) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "serializing measurement");
                return;
            }
        };
        if let Ok(mut guard) = self.inner.lock()
            && let Err(e) = guard.write_line(&line)
        {
            tracing::error!(error = %e, "writing log line");
        }
    }
}

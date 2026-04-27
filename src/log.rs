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
    pub dl_hz: u32,
    pub ul_hz: u32,
    /// Which TSBK opcode produced the grant — distinguishes a fresh
    /// allocation from a mid-call update, where keying is already in
    /// progress on retune.
    pub kind: GrantKind,
    /// In-channel power (S+N) in dBFS. For single-SDR mode this is the mean
    /// over chunks where keyup was detected; for dual-SDR mode it is a
    /// snapshot from the EMA-smoothed bin powers at grant time.
    pub channel_dbfs: f32,
    /// Same-bandwidth noise floor reference (N) in dBFS, estimated from the
    /// median power of out-of-channel bins in the captured baseband.
    pub noise_dbfs: f32,
    /// Signal-to-noise ratio in dB: 10·log10((channel - noise) / noise) in
    /// linear units. `None` when no keyup was detected within the measurement
    /// window. Stable across gain settings, unlike the raw dBFS fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snr_db: Option<f32>,
    /// Single-SDR-only diagnostic: how many of the FFT chunks in the window
    /// crossed the VAD threshold. A real keyup looks like a high
    /// `keyed_count / chunk_count` ratio; a noise-tail VAD trigger sits
    /// just at the [`MIN_KEYED_CHUNKS`](crate::uplink::single) gate value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyed_count: Option<u64>,
    /// Single-SDR-only diagnostic: total FFT chunks measured in the window.
    /// Smaller than the [`MEASURE_MS`](crate::uplink::single) chunk budget
    /// when early-exit fired.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_count: Option<u64>,
    /// Single-SDR-only diagnostic: maximum single-chunk in-channel/noise
    /// ratio observed. A real keyup hits the tens-to-hundreds; a noise-tail
    /// trigger barely clears the VAD threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_ratio: Option<f32>,
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

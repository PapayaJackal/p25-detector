use std::fs::OpenOptions;
use std::io::{BufWriter, Stdout, Write, stdout};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::config::Mode;

#[derive(Debug, Serialize)]
pub struct Measurement {
    pub ts: DateTime<Utc>,
    pub tgid: u16,
    pub rid: u32,
    pub dl_hz: u32,
    pub ul_hz: u32,
    /// Mean power over the measurement window, dBFS. Matches noise floor on
    /// idle channels; actual TX pulls it up only for the fraction of the
    /// window the mobile was keyed.
    pub rssi_dbfs: f32,
    /// Peak chunk power during the measurement window, dBFS. A chunk is
    /// whatever the SDR driver hands back per `read_iq` — on the order of
    /// 25–30 ms. Best estimate of the actual mobile TX strength when the
    /// key-up happens inside our measurement window. Only populated by
    /// single-SDR mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rssi_peak_dbfs: Option<f32>,
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

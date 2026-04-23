//! Audio feedback: short tone on every logged grant.
//!
//! Pitch is a hash of `(tgid, rid)` mapped onto an A-minor pentatonic scale
//! covering A4–A6. That range stays inside the ear's roughly-flat
//! equal-loudness region (~400 Hz – 2 kHz), so perceived volume is dominated
//! by RSSI — not by which (tgid, rid) happened to draw a low tone. Volume
//! itself is linear in dBFS (so perceived loudness tracks RSSI) and clamped.
//!
//! Single voice: a new beep replaces an in-flight one.

use std::f32::consts::TAU;

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, bounded};
use tracing::warn;

const BEEP_MS: u32 = 220;
const ATTACK_MS: u32 = 5;
const RELEASE_MS: u32 = 40;

const RSSI_QUIET_DBFS: f32 = -70.0;
const RSSI_LOUD_DBFS: f32 = -20.0;
const AMP_MIN: f32 = 0.02;
const AMP_MAX: f32 = 0.55;

const PENTATONIC_SEMITONES: [i32; 5] = [0, 3, 5, 7, 10];
const OCTAVES: i32 = 2;
const BASE_HZ: f32 = 440.0;

#[derive(Copy, Clone)]
struct BeepCmd {
    freq_hz: f32,
    amp: f32,
    total_samples: u32,
    attack_samples: u32,
    release_samples: u32,
}

pub struct Beeper {
    tx: Sender<BeepCmd>,
    sample_rate: u32,
    _stream: cpal::Stream,
}

impl Beeper {
    pub fn try_new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default audio output device")?;
        let supported = device
            .default_output_config()
            .context("querying default output config")?;
        let sample_rate = supported.sample_rate();
        let sample_format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();

        let (tx, rx) = bounded::<BeepCmd>(4);
        let sr_f = sample_rate as f32;

        let stream = match sample_format {
            cpal::SampleFormat::F32 => start_stream::<f32>(&device, &config, sr_f, rx)?,
            cpal::SampleFormat::I16 => start_stream::<i16>(&device, &config, sr_f, rx)?,
            cpal::SampleFormat::U16 => start_stream::<u16>(&device, &config, sr_f, rx)?,
            other => bail!("unsupported audio sample format {other:?}"),
        };

        Ok(Self {
            tx,
            sample_rate,
            _stream: stream,
        })
    }

    pub fn beep(&self, tgid: u16, rid: u32, rssi_dbfs: f32) {
        let total = (self.sample_rate * BEEP_MS) / 1000;
        let cmd = BeepCmd {
            freq_hz: tone_for(tgid, rid),
            amp: amp_for(rssi_dbfs),
            total_samples: total,
            attack_samples: (self.sample_rate * ATTACK_MS) / 1000,
            release_samples: (self.sample_rate * RELEASE_MS) / 1000,
        };
        let _ = self.tx.try_send(cmd);
    }
}

fn start_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sr: f32,
    rx: Receiver<BeepCmd>,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32> + Send + 'static,
{
    let channels = config.channels as usize;
    let mut osc = Oscillator::default();
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            while let Ok(cmd) = rx.try_recv() {
                osc.trigger(cmd);
            }
            for frame in data.chunks_mut(channels) {
                let s = T::from_sample(osc.next_sample(sr));
                for slot in frame.iter_mut() {
                    *slot = s;
                }
            }
        },
        |e| warn!(error = %e, "audio stream error"),
        None,
    )?;
    stream.play().context("starting audio stream")?;
    Ok(stream)
}

#[derive(Default)]
struct Oscillator {
    phase: f32,
    freq_hz: f32,
    amp: f32,
    elapsed: u32,
    total_samples: u32,
    attack_samples: u32,
    release_samples: u32,
}

impl Oscillator {
    fn trigger(&mut self, cmd: BeepCmd) {
        self.freq_hz = cmd.freq_hz;
        self.amp = cmd.amp;
        self.phase = 0.0;
        self.elapsed = 0;
        self.total_samples = cmd.total_samples;
        self.attack_samples = cmd.attack_samples.max(1);
        self.release_samples = cmd.release_samples.max(1);
    }

    fn next_sample(&mut self, sr: f32) -> f32 {
        if self.elapsed >= self.total_samples {
            return 0.0;
        }
        let remaining = self.total_samples - self.elapsed;
        let env = if self.elapsed < self.attack_samples {
            self.elapsed as f32 / self.attack_samples as f32
        } else if remaining < self.release_samples {
            remaining as f32 / self.release_samples as f32
        } else {
            1.0
        };
        let s = (self.phase * TAU).sin() * self.amp * env;
        self.phase += self.freq_hz / sr;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        self.elapsed += 1;
        s
    }
}

fn tone_for(tgid: u16, rid: u32) -> f32 {
    let slots = (PENTATONIC_SEMITONES.len() as u32) * (OCTAVES as u32);
    let mix = (tgid as u32)
        .wrapping_mul(2_246_822_519)
        .wrapping_add(rid.wrapping_mul(3_266_489_917));
    let slot = mix % slots;
    let octave = (slot / PENTATONIC_SEMITONES.len() as u32) as i32;
    let step = PENTATONIC_SEMITONES[(slot % PENTATONIC_SEMITONES.len() as u32) as usize];
    let semitones = octave * 12 + step;
    BASE_HZ * 2_f32.powf(semitones as f32 / 12.0)
}

fn amp_for(rssi_dbfs: f32) -> f32 {
    if !rssi_dbfs.is_finite() {
        return AMP_MIN;
    }
    let t = ((rssi_dbfs - RSSI_QUIET_DBFS) / (RSSI_LOUD_DBFS - RSSI_QUIET_DBFS)).clamp(0.0, 1.0);
    AMP_MIN + t * (AMP_MAX - AMP_MIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tones_stay_in_flat_loudness_region() {
        for tgid in [1u16, 42, 1234, 65535] {
            for rid in [1u32, 99, 1_000_000, u32::MAX] {
                let f = tone_for(tgid, rid);
                assert!(
                    (BASE_HZ..=BASE_HZ * 4.0).contains(&f),
                    "tone {f} out of range"
                );
            }
        }
    }

    #[test]
    fn different_ids_generally_give_different_tones() {
        let a = tone_for(100, 500);
        let b = tone_for(101, 500);
        let c = tone_for(100, 501);
        assert!(a != b || a != c, "hash is degenerate");
    }

    #[test]
    fn amp_clamps_and_tracks_rssi() {
        assert!((amp_for(-200.0) - AMP_MIN).abs() < 1e-6);
        assert!((amp_for(0.0) - AMP_MAX).abs() < 1e-6);
        assert!(amp_for(-45.0) > amp_for(-60.0));
        assert!(amp_for(-20.0) > amp_for(-45.0));
    }
}

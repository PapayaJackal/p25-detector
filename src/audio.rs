//! Audio feedback: short tone on every logged grant.
//!
//! Pitch tracks RSSI: louder signal (closer mobile) → higher note. RSSI is
//! mapped linearly onto an A-minor pentatonic scale spanning A3–A7 (four
//! octaves, 20 slots) for good proximity resolution. Amplitude also scales
//! with RSSI, so distant/likely-out-of-range signals are quiet and reinforce
//! the low pitch, while close signals are loud and high.
//!
//! Single voice: a new beep replaces an in-flight one.

use std::f32::consts::TAU;

use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, bounded};
use tracing::warn;

const BEEP_MS: u32 = 260;
const ATTACK_MS: u32 = 25;
const RELEASE_MS: u32 = 120;

const RSSI_QUIET_DBFS: f32 = -70.0;
const RSSI_LOUD_DBFS: f32 = -20.0;
const AMP_MIN: f32 = 0.04;
const AMP_MAX: f32 = 0.22;

const PENTATONIC_SEMITONES: [i32; 5] = [0, 3, 5, 7, 10];
const OCTAVES: i32 = 4;
const BASE_HZ: f32 = 220.0;

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
    rssi_min_dbfs: f32,
    _stream: cpal::Stream,
}

impl Beeper {
    pub fn try_new(rssi_min_dbfs: f32) -> Result<Self> {
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
            rssi_min_dbfs,
            _stream: stream,
        })
    }

    pub fn beep(&self, rssi_dbfs: f32) {
        if !rssi_dbfs.is_finite() || rssi_dbfs < self.rssi_min_dbfs {
            return;
        }
        let t = rssi_t(rssi_dbfs);
        let total = (self.sample_rate * BEEP_MS) / 1000;
        let cmd = BeepCmd {
            freq_hz: tone_for(t),
            amp: AMP_MIN + t * (AMP_MAX - AMP_MIN),
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
        // Raised cosine: no corners at attack/release boundaries, so no soft click.
        let ramp = if self.elapsed < self.attack_samples {
            self.elapsed as f32 / self.attack_samples as f32
        } else if remaining < self.release_samples {
            remaining as f32 / self.release_samples as f32
        } else {
            1.0
        };
        let env = 0.5 - 0.5 * (ramp * std::f32::consts::PI).cos();
        let s = (self.phase * TAU).sin() * self.amp * env;
        self.phase += self.freq_hz / sr;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        self.elapsed += 1;
        s
    }
}

fn rssi_t(rssi_dbfs: f32) -> f32 {
    if !rssi_dbfs.is_finite() {
        return 0.0;
    }
    ((rssi_dbfs - RSSI_QUIET_DBFS) / (RSSI_LOUD_DBFS - RSSI_QUIET_DBFS)).clamp(0.0, 1.0)
}

fn tone_for(t: f32) -> f32 {
    let slots = (PENTATONIC_SEMITONES.len() as i32) * OCTAVES;
    let slot = ((t * slots as f32) as i32).min(slots - 1);
    let octave = slot / PENTATONIC_SEMITONES.len() as i32;
    let step = PENTATONIC_SEMITONES[(slot % PENTATONIC_SEMITONES.len() as i32) as usize];
    let semitones = octave * 12 + step;
    BASE_HZ * 2_f32.powf(semitones as f32 / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tones_stay_within_scale_range() {
        let top = BASE_HZ * 2_f32.powi(OCTAVES);
        for rssi in [-200.0, -70.0, -45.0, -20.0, 0.0, f32::NAN] {
            let f = tone_for(rssi_t(rssi));
            assert!((BASE_HZ..=top).contains(&f), "tone {f} out of range");
        }
    }

    #[test]
    fn louder_rssi_gives_higher_pitch() {
        assert!(tone_for(rssi_t(-20.0)) > tone_for(rssi_t(-45.0)));
        assert!(tone_for(rssi_t(-45.0)) > tone_for(rssi_t(-70.0)));
    }

    #[test]
    fn tone_clamps_at_rails() {
        assert!((tone_for(rssi_t(-200.0)) - BASE_HZ).abs() < 1e-3);
        assert!(tone_for(rssi_t(0.0)) > BASE_HZ * 2_f32.powi(OCTAVES - 1));
    }

    #[test]
    fn weak_rssi_is_quiet_strong_is_loud() {
        assert!(rssi_t(-70.0) < 0.01);
        assert!(rssi_t(-20.0) > 0.99);
        let quiet = AMP_MIN + rssi_t(-60.0) * (AMP_MAX - AMP_MIN);
        let loud = AMP_MIN + rssi_t(-25.0) * (AMP_MAX - AMP_MIN);
        assert!(loud > quiet * 2.0);
    }
}

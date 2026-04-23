use anyhow::Result;
use num_complex::Complex32;

pub const SAMPLE_RATE: u32 = 2_400_000;

pub trait IqSource {
    fn read_iq(&mut self, out: &mut [Complex32]) -> Result<usize>;
    fn set_center_hz(&mut self, hz: u32) -> Result<()>;
}

#[cfg(feature = "rtlsdr")]
pub use backend::RtlSdr;

#[cfg(not(feature = "rtlsdr"))]
pub use stub::RtlSdr;

#[cfg(feature = "rtlsdr")]
mod backend {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::JoinHandle;

    use anyhow::Context;
    use crossbeam_channel::{Receiver, bounded};

    const READER_BUFS: u32 = 8;
    const READER_BUF_LEN: u32 = 1 << 16;

    /// Lookup table mapping the RTL-SDR's u8 [0,255] sample format to f32 in [-1,+1).
    /// Built once in `open()` and consulted per-sample in the hot decode path.
    fn make_iq_lut() -> [f32; 256] {
        let mut t = [0f32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = (i as f32 - 127.5) / 127.5;
        }
        t
    }

    /// Reader callback pushes owned Vec<u8> chunks into this channel; the
    /// main thread drains them into Complex32 pairs.
    pub struct RtlSdr {
        ctl: rtlsdr_mt::Controller,
        rx: Receiver<Vec<u8>>,
        stop: Arc<AtomicBool>,
        pending: Vec<u8>,
        reader_thread: Option<JoinHandle<()>>,
        iq_lut: [f32; 256],
    }

    impl RtlSdr {
        pub fn open(
            index: u32,
            center_hz: u32,
            rate_hz: u32,
            gain: Option<i32>,
            ppm: i32,
        ) -> Result<Self> {
            // rtlsdr_mt's Error type is `()` — there is no underlying message to
            // preserve, so `.ok().context(...)` is the most informative we can be.
            let (ctl, reader) = rtlsdr_mt::open(index)
                .ok()
                .with_context(|| format!("rtlsdr_mt::open({index}) failed"))?;
            let mut ctl = ctl;
            ctl.set_sample_rate(rate_hz).ok().context("set_sample_rate failed")?;
            ctl.set_ppm(ppm).ok().context("set_ppm failed")?;
            // R820T PLL-relock workaround: on some dongles the first tune after
            // power-up prints "[R82XX] PLL not locked!" and leaves the LO in a bad
            // state. Calling set_center_freq twice in a row to the target lets the
            // tuner reprogram the VCO from an already-initialized state on the
            // second call, which usually locks. librtlsdr doesn't surface PLL
            // status to callers (it only prints to stderr and returns OK), so we
            // just do the dance unconditionally.
            ctl.set_center_freq(center_hz).ok().context("set_center_freq failed")?;
            ctl.set_center_freq(center_hz).ok().context("set_center_freq failed")?;
            match gain {
                Some(tenths) => {
                    ctl.disable_agc().ok();
                    ctl.set_tuner_gain(tenths).ok().context("set_tuner_gain failed")?;
                }
                None => {
                    ctl.enable_agc().ok();
                }
            }

            let (tx, rx) = bounded::<Vec<u8>>(16);
            let stop = Arc::new(AtomicBool::new(false));
            let stop_reader = stop.clone();
            let mut reader_mt = reader;

            let handle = std::thread::Builder::new()
                .name(format!("rtlsdr-reader-{index}"))
                .spawn(move || {
                    let _ = reader_mt.read_async(READER_BUFS, READER_BUF_LEN, |bytes| {
                        if stop_reader.load(Ordering::Relaxed) {
                            return;
                        }
                        let _ = tx.try_send(bytes.to_vec());
                    });
                })?;

            Ok(Self {
                ctl,
                rx,
                stop,
                pending: Vec::with_capacity(READER_BUF_LEN as usize),
                reader_thread: Some(handle),
                iq_lut: make_iq_lut(),
            })
        }

        fn refill(&mut self) -> Result<()> {
            if !self.pending.is_empty() {
                return Ok(());
            }
            self.pending = self.rx.recv().context("reader channel closed")?;
            Ok(())
        }
    }

    impl Drop for RtlSdr {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            self.ctl.cancel_async_read();
            if let Some(h) = self.reader_thread.take() {
                let _ = h.join();
            }
        }
    }

    impl IqSource for RtlSdr {
        fn read_iq(&mut self, out: &mut [Complex32]) -> Result<usize> {
            if out.is_empty() {
                return Ok(0);
            }
            let bytes_needed = out.len() * 2;

            // accumulate bytes until we have at least one IQ pair
            while self.pending.len() < 2 {
                self.refill()?;
            }

            let available = self.pending.len().min(bytes_needed);
            let pairs = available / 2;
            let consumed = pairs * 2;
            let lut = &self.iq_lut;
            for (slot, chunk) in out
                .iter_mut()
                .zip(self.pending[..consumed].chunks_exact(2))
            {
                *slot = Complex32::new(lut[chunk[0] as usize], lut[chunk[1] as usize]);
            }
            if consumed == self.pending.len() {
                self.pending.clear();
            } else {
                self.pending.drain(..consumed);
            }
            Ok(pairs)
        }

        fn set_center_hz(&mut self, hz: u32) -> Result<()> {
            self.ctl.set_center_freq(hz).ok().context("set_center_freq failed")
        }
    }
}

#[cfg(not(feature = "rtlsdr"))]
mod stub {
    use super::*;

    pub struct RtlSdr;

    impl RtlSdr {
        pub fn open(
            _index: u32,
            _center_hz: u32,
            _rate_hz: u32,
            _gain: Option<i32>,
            _ppm: i32,
        ) -> Result<Self> {
            Ok(Self)
        }
    }

    impl IqSource for RtlSdr {
        fn read_iq(&mut self, out: &mut [Complex32]) -> Result<usize> {
            for s in out.iter_mut() {
                *s = Complex32::new(0.0, 0.0);
            }
            Ok(out.len())
        }
        fn set_center_hz(&mut self, _hz: u32) -> Result<()> {
            Ok(())
        }
    }
}

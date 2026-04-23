use std::time::{Duration, Instant};

use num_complex::Complex32;

use crate::dsp::c4fm::{DcBlock, FmDiscriminator, MatchedFilter, raised_cosine};
use crate::dsp::fir::{FirDecimator, design_lowpass};
use crate::dsp::fsk4::FskSlicer;
use crate::dsp::symbol_sync::GardnerSync;
use crate::p25::crc::crc16_p25;
use crate::p25::framer::{Framer, Frame};
use crate::p25::freq_table::FreqTable;
use crate::p25::nid::Duid;
use crate::p25::trellis::{TrellisStats, trellis_decode_block};
use crate::p25::tsbk::{Grant, TsbkEvent, parse_tsbk};

const SYMBOL_RATE: u32 = 4800;
/// How often to log a lock-status heartbeat once we've seen our first TSBK.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
/// How often to emit a DEBUG decoder-stats heartbeat. Shorter than
/// [`HEARTBEAT_INTERVAL`] so pre-lock debugging stays responsive.
const STATS_INTERVAL: Duration = Duration::from_secs(5);
/// Target intermediate rate after the first decimator. Must be an integer
/// divisor of the input sample rate that leaves room for the matched filter.
const IF_RATE: u32 = 48_000;
/// Bit length of one trellis-coded block (TSBK / PDU).
const TRELLIS_BLOCK_BITS: usize = 196;
/// Trellis blocks per TSBK frame.
const TSBK_BLOCKS_PER_FRAME: usize = 3;
/// Single-pole DC-block α at [`IF_RATE`]. Corner ≈ 4 Hz.
const DC_BLOCK_ALPHA: f32 = 0.9995;
/// Body dibit positions of status dibits inside a TSBK frame body.
/// Derived from the status-dibit-every-36 rule; see notes in code.
const STATUS_DIBIT_POSITIONS: [usize; 9] = [14, 50, 86, 122, 158, 194, 230, 266, 302];

/// A grant whose channel ID hasn't yet been resolved by an identifier-update.
#[derive(Debug, Clone, Copy)]
struct PendingGrant {
    tgid: u16,
    rid: u32,
    channel_id: u16,
}

pub struct Decoder {
    watched: Vec<u16>,
    decim: FirDecimator,
    disc: FmDiscriminator,
    dc_block: DcBlock,
    matched: MatchedFilter,
    sync: GardnerSync,
    slicer: FskSlicer,
    framer: Framer,
    freq_table: FreqTable,
    pending: Vec<PendingGrant>,
    decim_out: Vec<Complex32>,
    disc_out: Vec<f32>,
    dc_out: Vec<f32>,
    matched_out: Vec<f32>,
    sym_out: Vec<f32>,
    dibits: Vec<u8>,
    frames: Vec<Frame>,
    tsbk_bits: Vec<bool>,
    ever_framed: bool,
    ever_tsbk: bool,
    tsbks_since_report: u32,
    last_heartbeat: Instant,
    stats: DecoderStats,
    last_stats: Instant,
}

/// Rolling counters reset at each DEBUG stats heartbeat.
#[derive(Default, Debug, Clone, Copy)]
struct DecoderStats {
    frames_by_duid: [u32; 16],
    tsbk_blocks_seen: u32,
    tsbk_trellis_fail: u32,
    tsbk_crc_fail: u32,
    tsbk_crc_ok: u32,
    /// Rolling trellis per-step HD histogram + tie count across every block
    /// decode attempt in the window. High hd=0 count means codewords arrive
    /// clean; high hd=1 tells us errors are concentrated where the decoder's
    /// uniqueness check bites (the NEXT_WORDS table has pairwise min distance
    /// 2, so 1-bit errors can't be uniquely resolved greedily).
    trellis: TrellisStats,
    /// Running sum of discriminator output since the last heartbeat;
    /// divided by [`disc_count`](Self::disc_count) to report the FM DC bias.
    disc_sum: f64,
    /// Running sum of squares of discriminator output; combined with
    /// [`disc_sum`] to report stddev. Ideal C4FM σ ≈ 0.17 rad at 48 kHz IF;
    /// pure-noise FM-discriminator σ ≈ π/√3 ≈ 1.8 rad.
    disc_sum_sq: f64,
    disc_count: u64,
    /// Running sum of |IQ|² pre-decimator; combined with [`iq_count`] to
    /// report input RMS. Tells us the analog-frontend signal level.
    iq_power_sum: f64,
    iq_count: u64,
}

impl Decoder {
    pub fn new(sample_rate_hz: u32, watched_tgids: Vec<u16>) -> Self {
        assert!(
            sample_rate_hz.is_multiple_of(IF_RATE),
            "sample rate {sample_rate_hz} must be an integer multiple of IF rate {IF_RATE}"
        );
        let decim = (sample_rate_hz / IF_RATE) as usize;
        let taps = design_lowpass(63, 1.0 / decim as f32);
        let mf_taps = raised_cosine(41, (IF_RATE / SYMBOL_RATE) as f32, 0.2);
        let sps = (IF_RATE / SYMBOL_RATE) as f32;

        Self {
            watched: watched_tgids,
            decim: FirDecimator::new(taps, decim),
            disc: FmDiscriminator::default(),
            dc_block: DcBlock::new(DC_BLOCK_ALPHA),
            matched: MatchedFilter::new(mf_taps),
            sync: GardnerSync::new(sps),
            slicer: FskSlicer::default(),
            framer: Framer::new(),
            freq_table: FreqTable::new(),
            pending: Vec::new(),
            decim_out: Vec::with_capacity(1 << 14),
            disc_out: Vec::with_capacity(1 << 14),
            dc_out: Vec::with_capacity(1 << 14),
            matched_out: Vec::with_capacity(1 << 14),
            sym_out: Vec::with_capacity(1 << 12),
            dibits: Vec::with_capacity(1 << 12),
            frames: Vec::with_capacity(8),
            tsbk_bits: Vec::with_capacity(TRELLIS_BLOCK_BITS * TSBK_BLOCKS_PER_FRAME),
            ever_framed: false,
            ever_tsbk: false,
            tsbks_since_report: 0,
            last_heartbeat: Instant::now(),
            stats: DecoderStats::default(),
            last_stats: Instant::now(),
        }
    }

    pub fn process(&mut self, iq: &[Complex32]) -> Vec<Grant> {
        self.decim_out.clear();
        self.disc_out.clear();
        self.dc_out.clear();
        self.matched_out.clear();
        self.sym_out.clear();
        self.dibits.clear();
        self.frames.clear();

        // Sum |c|² in f32 per-block, then promote to f64 at the block boundary.
        // The f32→f64 cast inside a `.map().sum()` blocks vectorization; a plain
        // f32 accumulator over a ≤64K-sample block has ~1e-3 relative error,
        // which is invisible for statistics-only reporting.
        let mut block_power = 0.0f32;
        for &c in iq {
            block_power += c.re * c.re + c.im * c.im;
        }
        self.stats.iq_power_sum += block_power as f64;
        self.stats.iq_count = self.stats.iq_count.saturating_add(iq.len() as u64);

        self.decim.process(iq, &mut self.decim_out);
        self.disc.process(&self.decim_out, &mut self.disc_out);
        let mut disc_sum = 0.0f64;
        let mut disc_sum_sq = 0.0f64;
        for &v in &self.disc_out {
            let v = v as f64;
            disc_sum += v;
            disc_sum_sq += v * v;
        }
        self.stats.disc_sum += disc_sum;
        self.stats.disc_sum_sq += disc_sum_sq;
        self.stats.disc_count = self
            .stats
            .disc_count
            .saturating_add(self.disc_out.len() as u64);
        self.dc_block.process(&self.disc_out, &mut self.dc_out);
        self.matched.process(&self.dc_out, &mut self.matched_out);
        self.sync.process(&self.matched_out, &mut self.sym_out);
        self.slicer.slice(&self.sym_out, &mut self.dibits);
        self.framer.push_dibits(&self.dibits, &mut self.frames);

        if !self.ever_framed && !self.frames.is_empty() {
            self.ever_framed = true;
            tracing::info!("PHY lock: first P25 frame sync acquired");
        }

        let mut out = Vec::new();
        let mut fresh_tsbks: u32 = 0;
        let mut frames = std::mem::take(&mut self.frames);
        for frame in frames.drain(..) {
            let duid_ix = frame.duid as usize;
            if duid_ix < self.stats.frames_by_duid.len() {
                self.stats.frames_by_duid[duid_ix] =
                    self.stats.frames_by_duid[duid_ix].saturating_add(1);
            }
            tracing::trace!(duid = ?frame.duid, body_dibits = frame.dibits.len(), "p25 frame");
            if frame.duid == Duid::Tsbk {
                let tsbk_stats = handle_tsbk_frame(
                    &frame,
                    &mut self.freq_table,
                    &mut self.pending,
                    &mut self.tsbk_bits,
                    &self.watched,
                    &mut out,
                    &mut self.stats.trellis,
                );
                fresh_tsbks += tsbk_stats.crc_ok;
                self.stats.tsbk_blocks_seen =
                    self.stats.tsbk_blocks_seen.saturating_add(tsbk_stats.blocks_seen);
                self.stats.tsbk_trellis_fail = self
                    .stats
                    .tsbk_trellis_fail
                    .saturating_add(tsbk_stats.trellis_fail);
                self.stats.tsbk_crc_fail =
                    self.stats.tsbk_crc_fail.saturating_add(tsbk_stats.crc_fail);
                self.stats.tsbk_crc_ok =
                    self.stats.tsbk_crc_ok.saturating_add(tsbk_stats.crc_ok);
            }
        }
        self.frames = frames;

        if fresh_tsbks > 0 {
            self.tsbks_since_report = self.tsbks_since_report.saturating_add(fresh_tsbks);
            if !self.ever_tsbk {
                self.ever_tsbk = true;
                tracing::info!("control channel lock: first valid TSBK decoded");
            }
        }

        if self.ever_tsbk && self.last_heartbeat.elapsed() >= HEARTBEAT_INTERVAL {
            let elapsed = self.last_heartbeat.elapsed();
            tracing::info!(
                tsbks = self.tsbks_since_report,
                secs = elapsed.as_secs(),
                "control channel heartbeat",
            );
            self.tsbks_since_report = 0;
            self.last_heartbeat = Instant::now();
        }

        if self.last_stats.elapsed() >= STATS_INTERVAL {
            let s = self.stats;
            let (disc_dc, disc_std) = if s.disc_count > 0 {
                let n = s.disc_count as f64;
                let mean = s.disc_sum / n;
                let var = (s.disc_sum_sq / n - mean * mean).max(0.0);
                (mean, var.sqrt())
            } else {
                (0.0, 0.0)
            };
            let iq_rms = if s.iq_count > 0 {
                (s.iq_power_sum / s.iq_count as f64).sqrt()
            } else {
                0.0
            };
            let (gardner_err_mean, gardner_sat_frac) = self.sync.err_stats_since_read();
            let dh = self.slicer.dibit_hist_since_read();
            let dh_total = (dh[0] + dh[1] + dh[2] + dh[3]).max(1) as f32;
            let dibit_p1 = dh[0] as f32 / dh_total;
            let dibit_p3 = dh[1] as f32 / dh_total;
            let dibit_m1 = dh[2] as f32 / dh_total;
            let dibit_m3 = dh[3] as f32 / dh_total;
            let trellis_metric_avg = if s.trellis.blocks_decoded > 0 {
                s.trellis.metric_sum as f32 / s.trellis.blocks_decoded as f32
            } else {
                0.0
            };
            tracing::debug!(
                hdu = s.frames_by_duid[Duid::Hdu as usize],
                tdu = s.frames_by_duid[Duid::Tdu as usize],
                ldu1 = s.frames_by_duid[Duid::Ldu1 as usize],
                tsbk = s.frames_by_duid[Duid::Tsbk as usize],
                ldu2 = s.frames_by_duid[Duid::Ldu2 as usize],
                pdu = s.frames_by_duid[Duid::Pdu as usize],
                tdu_lc = s.frames_by_duid[Duid::TduLc as usize],
                blocks = s.tsbk_blocks_seen,
                trellis_fail = s.tsbk_trellis_fail,
                crc_fail = s.tsbk_crc_fail,
                crc_ok = s.tsbk_crc_ok,
                iq_rms = iq_rms,
                disc_dc = disc_dc,
                disc_std = disc_std,
                slicer_mean = self.slicer.mean(),
                slicer_inner = self.slicer.amp_inner(),
                slicer_outer = self.slicer.amp_outer(),
                gardner_mu = self.sync.mu(),
                gardner_freq = self.sync.freq(),
                gardner_lock_q = self.sync.lock_q(),
                gardner_wraps = self.sync.wraps_since_read(),
                gardner_err_mean = gardner_err_mean,
                gardner_sat_frac = gardner_sat_frac,
                trellis_hd0 = s.trellis.hd_hist[0],
                trellis_hd1 = s.trellis.hd_hist[1],
                trellis_hd2 = s.trellis.hd_hist[2],
                trellis_hd3 = s.trellis.hd_hist[3],
                trellis_hd4 = s.trellis.hd_hist[4],
                trellis_steps = s.trellis.steps,
                trellis_metric_avg = trellis_metric_avg,
                dibit_p1 = dibit_p1,
                dibit_p3 = dibit_p3,
                dibit_m1 = dibit_m1,
                dibit_m3 = dibit_m3,
                secs = self.last_stats.elapsed().as_secs(),
                "decoder stats",
            );
            self.stats = DecoderStats::default();
            self.last_stats = Instant::now();
        }

        if !self.pending.is_empty() {
            let freq_table = &self.freq_table;
            let watched = &self.watched;
            self.pending.retain(|p| {
                match freq_table.channel_to_freqs(p.channel_id) {
                    Some((dl_hz, ul_hz)) => {
                        let g = Grant {
                            tgid: p.tgid,
                            rid: p.rid,
                            channel_id: p.channel_id,
                            dl_hz,
                            ul_hz,
                        };
                        if watched.is_empty() || watched.contains(&g.tgid) {
                            out.push(g);
                        }
                        false
                    }
                    None => true,
                }
            });
        }

        out
    }
}

/// Per-frame tally of trellis / CRC outcomes, used by the decoder stats
/// heartbeat to make pre-lock failures visible at DEBUG level.
#[derive(Default, Debug, Clone, Copy)]
struct TsbkFrameStats {
    blocks_seen: u32,
    trellis_fail: u32,
    crc_fail: u32,
    crc_ok: u32,
}

fn handle_tsbk_frame(
    frame: &Frame,
    freq_table: &mut FreqTable,
    pending: &mut Vec<PendingGrant>,
    bits: &mut Vec<bool>,
    watched: &[u16],
    out: &mut Vec<Grant>,
    trellis_stats: &mut TrellisStats,
) -> TsbkFrameStats {
    let mut stats = TsbkFrameStats::default();
    bits.clear();
    unpack_tsbk_bits(&frame.dibits, bits);
    if bits.len() < TRELLIS_BLOCK_BITS * TSBK_BLOCKS_PER_FRAME {
        tracing::trace!(
            got = bits.len(),
            want = TRELLIS_BLOCK_BITS * TSBK_BLOCKS_PER_FRAME,
            "tsbk frame body too short",
        );
        return stats;
    }
    for blk in 0..TSBK_BLOCKS_PER_FRAME {
        stats.blocks_seen += 1;
        let start = blk * TRELLIS_BLOCK_BITS;
        let block: &[bool; TRELLIS_BLOCK_BITS] = bits[start..start + TRELLIS_BLOCK_BITS]
            .try_into()
            .expect("slice length matches TRELLIS_BLOCK_BITS");
        let decoded = match trellis_decode_block(block, trellis_stats) {
            Some(b) => b,
            None => {
                stats.trellis_fail += 1;
                tracing::trace!(block = blk, "tsbk trellis decode failed");
                continue;
            }
        };
        if crc16_p25(&decoded) != 0 {
            stats.crc_fail += 1;
            tracing::trace!(
                block = blk,
                opcode = decoded[0] & 0x3F,
                "tsbk crc mismatch",
            );
            continue;
        }
        stats.crc_ok += 1;
        let last_block_flag = decoded[0] >> 7;
        if let Some(evt) = parse_tsbk(&decoded, freq_table) {
            match evt {
                TsbkEvent::Grant(g) => {
                    if g.dl_hz == 0 {
                        pending.push(PendingGrant {
                            tgid: g.tgid,
                            rid: g.rid,
                            channel_id: g.channel_id,
                        });
                    } else if watched.is_empty() || watched.contains(&g.tgid) {
                        out.push(g);
                    }
                }
                TsbkEvent::IdentifierUpdate { iden, entry } => {
                    freq_table.insert(iden, entry);
                }
                TsbkEvent::Other => {}
            }
        }
        if last_block_flag != 0 {
            break;
        }
    }
    stats
}

/// Strips in-frame status dibits and unpacks each remaining dibit into two MSB-first bits.
fn unpack_tsbk_bits(body: &[u8], out: &mut Vec<bool>) {
    let mut next_skip = 0usize;
    for (i, &d) in body.iter().enumerate() {
        if next_skip < STATUS_DIBIT_POSITIONS.len() && STATUS_DIBIT_POSITIONS[next_skip] == i {
            next_skip += 1;
            continue;
        }
        out.push((d >> 1) & 1 != 0);
        out.push(d & 1 != 0);
    }
}

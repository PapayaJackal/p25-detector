use num_complex::Complex32;

use crate::dsp::c4fm::{FmDiscriminator, MatchedFilter, raised_cosine};
use crate::dsp::fir::{FirDecimator, design_lowpass};
use crate::dsp::fsk4::FskSlicer;
use crate::dsp::symbol_sync::GardnerSync;
use crate::p25::crc::crc16_p25;
use crate::p25::framer::{Framer, Frame};
use crate::p25::freq_table::FreqTable;
use crate::p25::nid::Duid;
use crate::p25::trellis::trellis_decode_block;
use crate::p25::tsbk::{Grant, TsbkEvent, parse_tsbk};

const SYMBOL_RATE: u32 = 4800;
/// Target intermediate rate after the first decimator. Must be an integer
/// divisor of the input sample rate that leaves room for the matched filter.
const IF_RATE: u32 = 48_000;
/// Bit length of one trellis-coded block (TSBK / PDU).
const TRELLIS_BLOCK_BITS: usize = 196;
/// Trellis blocks per TSBK frame.
const TSBK_BLOCKS_PER_FRAME: usize = 3;
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
    matched: MatchedFilter,
    sync: GardnerSync,
    slicer: FskSlicer,
    framer: Framer,
    freq_table: FreqTable,
    pending: Vec<PendingGrant>,
    decim_out: Vec<Complex32>,
    disc_out: Vec<f32>,
    matched_out: Vec<f32>,
    sym_out: Vec<f32>,
    dibits: Vec<u8>,
    frames: Vec<Frame>,
    tsbk_bits: Vec<bool>,
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
            matched: MatchedFilter::new(mf_taps),
            sync: GardnerSync::new(sps),
            slicer: FskSlicer::default(),
            framer: Framer::new(),
            freq_table: FreqTable::new(),
            pending: Vec::new(),
            decim_out: Vec::with_capacity(1 << 14),
            disc_out: Vec::with_capacity(1 << 14),
            matched_out: Vec::with_capacity(1 << 14),
            sym_out: Vec::with_capacity(1 << 12),
            dibits: Vec::with_capacity(1 << 12),
            frames: Vec::with_capacity(8),
            tsbk_bits: Vec::with_capacity(TRELLIS_BLOCK_BITS * TSBK_BLOCKS_PER_FRAME),
        }
    }

    pub fn process(&mut self, iq: &[Complex32]) -> Vec<Grant> {
        self.decim_out.clear();
        self.disc_out.clear();
        self.matched_out.clear();
        self.sym_out.clear();
        self.dibits.clear();
        self.frames.clear();

        self.decim.process(iq, &mut self.decim_out);
        self.disc.process(&self.decim_out, &mut self.disc_out);
        self.matched.process(&self.disc_out, &mut self.matched_out);
        self.sync.process(&self.matched_out, &mut self.sym_out);
        self.slicer.slice(&self.sym_out, &mut self.dibits);
        self.framer.push_dibits(&self.dibits, &mut self.frames);

        let mut out = Vec::new();
        let mut frames = std::mem::take(&mut self.frames);
        for frame in frames.drain(..) {
            if frame.duid == Duid::Tsbk {
                handle_tsbk_frame(
                    &frame,
                    &mut self.freq_table,
                    &mut self.pending,
                    &mut self.tsbk_bits,
                    &self.watched,
                    &mut out,
                );
            }
        }
        self.frames = frames;

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

fn handle_tsbk_frame(
    frame: &Frame,
    freq_table: &mut FreqTable,
    pending: &mut Vec<PendingGrant>,
    bits: &mut Vec<bool>,
    watched: &[u16],
    out: &mut Vec<Grant>,
) {
    bits.clear();
    unpack_tsbk_bits(&frame.dibits, bits);
    if bits.len() < TRELLIS_BLOCK_BITS * TSBK_BLOCKS_PER_FRAME {
        return;
    }
    for blk in 0..TSBK_BLOCKS_PER_FRAME {
        let start = blk * TRELLIS_BLOCK_BITS;
        let block: &[bool; TRELLIS_BLOCK_BITS] = bits[start..start + TRELLIS_BLOCK_BITS]
            .try_into()
            .expect("slice length matches TRELLIS_BLOCK_BITS");
        let decoded = match trellis_decode_block(block) {
            Some(b) => b,
            None => continue,
        };
        if crc16_p25(&decoded) != 0 {
            continue;
        }
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

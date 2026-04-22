# p25-detector

Passive P25 Phase 1 mobile-radio proximity detector.

Listens to a P25 trunked-system control channel, parses TSBKs, and for each
grant on a watched talkgroup measures RSSI on the mobile's uplink frequency.
The output is a JSONL stream of `(timestamp, tgid, rid, dl_hz, ul_hz, rssi_dbfs)`
rows — enough to correlate observed uplink signal strength with a mobile's
activity over time. No voice is demodulated or logged.

## How it works

Control-channel path (`src/dsp` + `src/p25`):
RTL-SDR → decimate to 48 kHz IF → FM discriminate → matched filter (RRC) →
Gardner symbol sync → 4FSK slice → NID/Duid frame sync → trellis decode
(1/2-rate) → TSBK CRC-16 → opcode parse. Grants are buffered until an
IDENTIFIER_UPDATE arrives with the matching channel ID, at which point the
uplink frequency is known and the grant is emitted.

Uplink measurement (`src/uplink`):
- **single-sdr** — one RTL-SDR. On a grant, retune to the uplink, settle,
  integrate power over a short window, retune back to the control channel.
  Fast to set up but misses control traffic while measuring.
- **dual-sdr** — a second RTL-SDR capturing a wideband slice of the uplink
  band continuously. Grants are measured by picking out the target bin from
  an FFT average. No control-channel gap.

## Build

```sh
cargo build --release
```

Requires `librtlsdr` headers on the system for the default `rtlsdr` feature.
To build without the RTL-SDR backend (stubbed source, for tests or
cross-compilation):

```sh
cargo build --release --no-default-features
```

## Usage

```sh
p25-detector \
  --cc-freq 851.0125e6 \
  --watch-tgid 1234,5678 \
  --mode single-sdr \
  --log run.jsonl
```

Dual-SDR mode, with a wideband capture centered on the uplink block:

```sh
p25-detector \
  --cc-freq 851.0125e6 \
  --watch-tgid 1234 \
  --mode dual-sdr \
  --cc-device 0 \
  --uplink-device 1 \
  --uplink-center 806.5e6 \
  --gain 400
```

Flags:

| flag | meaning |
| --- | --- |
| `--cc-freq` | Control-channel frequency (Hz; scientific notation OK) |
| `--watch-tgid` | Comma-separated TGIDs to measure |
| `--mode` | `single-sdr` (default) or `dual-sdr` |
| `--cc-device` | RTL-SDR index for the control channel (default `0`) |
| `--uplink-device` | RTL-SDR index for the uplink (dual-sdr; default `1`) |
| `--uplink-center` | Center frequency of the wideband uplink capture (dual-sdr, required) |
| `--gain` | Tuner gain in tenths of a dB (e.g. `400` = 40.0 dB); omit for AGC |
| `--min-measure-interval-ms` | Per-TGID cooldown between measurements (single-sdr; default `5000`) |
| `--log` | Output path, or `-`/omitted for stdout |

Set `RUST_LOG=debug` for per-frame decoder tracing.

## Output

One JSON object per line:

```json
{"ts":"2026-04-22T17:03:11.482Z","tgid":1234,"rid":1715004,"dl_hz":854237500,"ul_hz":809237500,"rssi_dbfs":-42.1,"mode":"single-sdr"}
```

## Scope and non-goals

- P25 Phase 1 only. No Phase 2 (TDMA), no DMR, no NXDN.
- Control-channel parsing is limited to TSBK grants and identifier updates —
  enough to resolve uplink frequency, nothing more.
- No voice audio is decoded. No uplink payload is demodulated; only power.
- No direction finding or geolocation. RSSI only.

## Layout

```
src/
  main.rs           entrypoint + runtime wiring
  config.rs         CLI → RuntimeConfig
  sdr.rs            RTL-SDR driver + IqSource trait
  log.rs            JSONL measurement sink
  dsp/              FIR, C4FM matched filter, Gardner sync, 4FSK slicer, power
  p25/              NID, framer, trellis 1/2, BCH, CRC-16, TSBK parser, freq table
  uplink/           single-sdr and dual-sdr watcher implementations
```

## License

WTFPL.

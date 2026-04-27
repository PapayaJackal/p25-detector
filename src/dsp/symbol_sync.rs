/// Gardner timing-error detector + Farrow-cubic interpolator.
/// Consumes a stream of baseband real samples at `sps` samples/symbol and
/// emits one decision per symbol.
///
/// Loop filter is a PI controller: the proportional term nulls phase error,
/// the integrator tracks long-term sample-rate offset between TX symbol clock
/// and RX-derived clock. Gains and bounds follow OP25's `gardner_cc_impl`:
/// err is clipped to [-1, 1] to survive noise bursts, the integrator is
/// bounded to ±[`FREQ_MAX`] so it can't walk off, and the proportional /
/// integral gains are critically damped (gain_i ≈ gain_p²/4).
pub struct GardnerSync {
    sps: f32,
    mu: f32,
    gain_p: f32,
    gain_i: f32,
    freq: f32,
    hist: [f32; 4],
    last_sym: f32,
    mid_sym: f32,
    counter: f32,
    toggle: bool,
    /// Number of times `mu` wrapped through 0 or 1 since [`wraps_since_read`]
    /// was last called. A healthy lock has wraps only during acquisition.
    wraps: u32,
    /// EMA of the Yair-Linn lock-quality metric `(y² - mid²)/(y² + mid²)`.
    /// ≈ +1 when locked (end-of-symbol hits the peak, mid-symbol near zero
    /// crossing), ≈ 0 when the loop is random-walking on noise.
    lock_q: f32,
    /// Running sum of TED error since the last `err_since_read`. Divided by
    /// `err_count` yields the per-symbol error bias — a nonzero mean while the
    /// integrator is unsaturated points at a systematic bias source.
    err_sum: f64,
    err_count: u64,
    /// Count of end-of-symbol updates where `freq` landed at ±FREQ_MAX after
    /// the integrator update. High ratio vs `err_count` means the loop wants
    /// to drift faster than the rail allows.
    sat_count: u64,
}

/// Integrator bound: ±1% of nominal symbol clock. Raised from OP25's 0.2%
/// default because real captures have been observed pinning the rail — either
/// from large SDR clock offset or from a small systematic error bias. 1% gives
/// the loop room to find its steady state so we can tell the two regimes apart.
const FREQ_MAX: f32 = 1e-2;

impl GardnerSync {
    pub fn new(sps: f32) -> Self {
        let gain_p = 0.025;
        Self {
            sps,
            mu: 0.5,
            gain_p,
            gain_i: gain_p * gain_p / 4.0,
            freq: 0.0,
            hist: [0.0; 4],
            last_sym: 0.0,
            mid_sym: 0.0,
            counter: 0.0,
            toggle: false,
            wraps: 0,
            lock_q: 0.0,
            err_sum: 0.0,
            err_count: 0,
            sat_count: 0,
        }
    }

    /// Restore the loop to its post-`new` state without dropping `sps` or
    /// the gain configuration. Stats counters are zeroed too so the next
    /// heartbeat reports only post-reset behaviour.
    pub fn reset(&mut self) {
        self.mu = 0.5;
        self.freq = 0.0;
        self.hist = [0.0; 4];
        self.last_sym = 0.0;
        self.mid_sym = 0.0;
        self.counter = 0.0;
        self.toggle = false;
        self.wraps = 0;
        self.lock_q = 0.0;
        self.err_sum = 0.0;
        self.err_count = 0;
        self.sat_count = 0;
    }

    pub fn mu(&self) -> f32 {
        self.mu
    }

    pub fn freq(&self) -> f32 {
        self.freq
    }

    pub fn lock_q(&self) -> f32 {
        self.lock_q
    }

    pub fn wraps_since_read(&mut self) -> u32 {
        std::mem::take(&mut self.wraps)
    }

    /// Returns (mean_err, sat_frac) since the last call and resets the
    /// underlying counters. `sat_frac` is the fraction of end-of-symbol
    /// updates where `freq` was at ±FREQ_MAX after the integrator step.
    pub fn err_stats_since_read(&mut self) -> (f32, f32) {
        let n = std::mem::take(&mut self.err_count);
        let sum = std::mem::take(&mut self.err_sum);
        let sat = std::mem::take(&mut self.sat_count);
        if n == 0 {
            (0.0, 0.0)
        } else {
            ((sum / n as f64) as f32, sat as f32 / n as f32)
        }
    }

    fn interp(&self, mu: f32) -> f32 {
        let x0 = self.hist[0];
        let x1 = self.hist[1];
        let x2 = self.hist[2];
        let x3 = self.hist[3];
        let a = -(1.0 / 6.0) * x0 + 0.5 * x1 - 0.5 * x2 + (1.0 / 6.0) * x3;
        let b = 0.5 * x0 - x1 + 0.5 * x2;
        let c = -(1.0 / 3.0) * x0 - 0.5 * x1 + x2 - (1.0 / 6.0) * x3;
        let d = x1;
        ((a * mu + b) * mu + c) * mu + d
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let half_sps = self.sps * 0.5;
        for &x in input {
            self.hist[0] = self.hist[1];
            self.hist[1] = self.hist[2];
            self.hist[2] = self.hist[3];
            self.hist[3] = x;

            self.counter -= 1.0;
            if self.counter < 0.0 {
                self.counter += half_sps;
                let y = self.interp(self.mu);
                if self.toggle {
                    // Standard Gardner TED: err = mid × (prev_end - curr_end).
                    // The inverted form (curr - prev) would push `mu` the wrong
                    // way under timing drift and saturate the integrator — see
                    // OP25 `gardner_cc_impl.cc:169` for the reference sign.
                    let err = ((self.last_sym - y) * self.mid_sym).clamp(-1.0, 1.0);
                    self.freq = (self.freq + self.gain_i * err).clamp(-FREQ_MAX, FREQ_MAX);
                    self.mu += self.gain_p * err + self.freq;
                    self.err_sum += err as f64;
                    self.err_count = self.err_count.saturating_add(1);
                    if self.freq.abs() >= FREQ_MAX - 1e-9 {
                        self.sat_count = self.sat_count.saturating_add(1);
                    }
                    let y2 = y * y;
                    let m2 = self.mid_sym * self.mid_sym;
                    let denom = y2 + m2;
                    if denom > 1e-9 {
                        let q = (y2 - m2) / denom;
                        self.lock_q = 0.995 * self.lock_q + 0.005 * q;
                    }
                    // Gardner's TED has two stable equilibria sps/2 apart: the
                    // correct symbol phase (end samples on peaks, mid on
                    // transitions, lock_q → +1) and the inverted phase (swapped,
                    // lock_q → -1). Once lock_q settles firmly negative we're in
                    // the wrong one — re-label the next trigger as end instead
                    // of mid so the roles swap. The trailing `toggle = !toggle`
                    // below means presetting to false here leaves next trigger
                    // as end, giving two ends in a row and shifting the phase.
                    if self.lock_q < -0.1 {
                        self.toggle = false;
                        self.lock_q = 0.0;
                    }
                    // mu is the fractional interpolation offset into hist[1..2].
                    // When it wraps past 1, the interpolation point has moved one
                    // whole input sample forward, so the history needs to catch
                    // up: delay the next trigger by one sample. OP25's gardner
                    // does the same when d_mu>1: `d_mu--; i++` advances the input
                    // pointer. Getting this sign wrong turns the PI loop into
                    // positive feedback and makes wraps compound instead of
                    // correct.
                    while self.mu >= 1.0 {
                        self.mu -= 1.0;
                        self.counter += 1.0;
                        self.wraps = self.wraps.saturating_add(1);
                    }
                    while self.mu < 0.0 {
                        self.mu += 1.0;
                        self.counter -= 1.0;
                        self.wraps = self.wraps.saturating_add(1);
                    }
                    out.push(y);
                    self.last_sym = y;
                } else {
                    self.mid_sym = y;
                }
                self.toggle = !self.toggle;
            }
        }
    }
}

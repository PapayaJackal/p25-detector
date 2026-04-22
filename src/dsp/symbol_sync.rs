/// Gardner timing-error detector + Farrow-cubic interpolator.
/// Consumes a stream of baseband real samples at `sps` samples/symbol and
/// emits one decision per symbol.
pub struct GardnerSync {
    sps: f32,
    mu: f32,
    gain: f32,
    hist: [f32; 4],
    last_sym: f32,
    mid_sym: f32,
    counter: f32,
    toggle: bool,
}

impl GardnerSync {
    pub fn new(sps: f32) -> Self {
        Self {
            sps,
            mu: 0.5,
            gain: 0.05,
            hist: [0.0; 4],
            last_sym: 0.0,
            mid_sym: 0.0,
            counter: 0.0,
            toggle: false,
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
                    let err = (y - self.last_sym) * self.mid_sym;
                    self.mu += self.gain * err;
                    while self.mu >= 1.0 {
                        self.mu -= 1.0;
                        self.counter -= 1.0;
                    }
                    while self.mu < 0.0 {
                        self.mu += 1.0;
                        self.counter += 1.0;
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

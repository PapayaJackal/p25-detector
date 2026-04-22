use num_complex::Complex32;

/// Returns average power (mean |x|²) of a complex baseband buffer.
pub fn mean_power(buf: &[Complex32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for &x in buf {
        s += (x.re as f64) * (x.re as f64) + (x.im as f64) * (x.im as f64);
    }
    (s / buf.len() as f64) as f32
}

/// Converts a linear power ratio to dBFS (full-scale = 1.0 after -1..1 IQ normalization).
pub fn to_dbfs(power: f32) -> f32 {
    const FLOOR: f32 = 1e-12;
    10.0 * power.max(FLOOR).log10()
}

//! Generated measurements with known complexity, for tests that need to check
//! what was recovered against what was actually there.
//!
//! Everything here is seeded. A test that fails must fail again on the next run
//! with the same data, or it reports noise instead of a defect.

// Each test binary compiles this module separately and uses a different part of
// it, so what is dead in one is live in another.
#![allow(dead_code)]

use big_o::Model;

/// Distinct input sizes in a generated sample. Enough that the models have room
/// to disagree, few enough that a full sweep stays quick.
pub const POINTS: usize = 24;

/// Decades of input size a generated sample spans.
///
/// Models are separated by how fast they grow, so this range is what makes them
/// distinguishable at all — more so than the number of points in it.
pub const DECADES: f64 = 4.0;

/// Shares of the total that a generated curve's constant term may contribute.
///
/// Swept rather than fixed, because the two ends are different problems and a
/// single value in the middle hides both.
///
/// Zero is the pure power law — the shape a clean algorithmic benchmark
/// actually produces, and the one the free-exponent model competes hardest on,
/// since it is fitted in log-log space and has no constant term of its own to
/// spend. Fixing this at a comfortable 0.2 was enough on its own to hide
/// quadratic recovery falling to 90%.
///
/// The upper end is held well below one: a curve whose constant term dominates
/// is flat over the measured range whatever its name, and a sweep built from
/// those measures signal-to-noise rather than the thing under test.
pub const OFFSET_SHARES: [f64; 2] = [0.0, 0.2];

/// Multiplicative timing noise: `y * (1 + e)`, `e ~ N(0, sigma)`.
///
/// Benchmark noise scales with the measurement — a run that is 3% slow is 3%
/// slow whether it took a microsecond or a second — so the noise is a fraction
/// of the value, not a fixed amount added to it.
pub fn noisy(model: Model, trial: u64, sigma: f64) -> Vec<(f64, f64)> {
    with_offset_share(model, trial, sigma, 0.2)
}

/// As [`noisy`], with the curve's constant term sized to `offset_share` of the
/// growth term at the largest input. See [`OFFSET_SHARES`].
pub fn with_offset_share(
    model: Model,
    trial: u64,
    sigma: f64,
    offset_share: f64,
) -> Vec<(f64, f64)> {
    let mut rng = Rng::seeded(model, trial);
    let curve = curve(model, &mut rng, offset_share);

    inputs(model)
        .into_iter()
        .map(|x| (x, curve(x) * (1.0 + sigma * rng.gaussian())))
        .collect()
}

/// The input sizes a model is measured over.
fn inputs(model: Model) -> Vec<f64> {
    let last = (POINTS - 1) as f64;
    (0..POINTS)
        .map(|i| match model {
            // An exponential over four decades of input size overflows long
            // before the last point, so it is measured over a linear range.
            Model::Exponential => 1.0 + i as f64,
            _ => 10f64.powf(DECADES * i as f64 / last),
        })
        .collect()
}

/// Draws a curve of the given shape with random but sane coefficients.
fn curve(model: Model, rng: &mut Rng, offset_share: f64) -> Box<dyn Fn(f64) -> f64> {
    let gain = rng.between(1.0, 100.0);
    let largest = inputs(model).into_iter().fold(0.0, f64::max);

    // Sized against the growth term at the largest input, so that the shape of
    // the curve is what the sample carries.
    let mut offset_of = |growth: f64| rng.between(0.0, offset_share * gain * growth);

    match model {
        Model::Constant => {
            let level = rng.between(10.0, 1000.0);
            Box::new(move |_x| level)
        }
        Model::Logarithmic => {
            let offset = offset_of(largest.ln());
            Box::new(move |x| gain * x.ln() + offset)
        }
        Model::Linear => {
            let offset = offset_of(largest);
            Box::new(move |x| gain * x + offset)
        }
        Model::Linearithmic => {
            let offset = offset_of(largest * largest.ln());
            Box::new(move |x| gain * x * x.ln() + offset)
        }
        Model::Quadratic => {
            let offset = offset_of(largest.powi(2));
            Box::new(move |x| gain * x.powi(2) + offset)
        }
        Model::Cubic => {
            let offset = offset_of(largest.powi(3));
            Box::new(move |x| gain * x.powi(3) + offset)
        }
        Model::Polynomial => {
            // Kept clear of the whole numbers, where the honest answer is the
            // named model rather than the free exponent.
            let power = rng.pick(&[(0.4, 0.8), (1.3, 1.7), (2.3, 2.7), (3.3, 3.7)]);
            Box::new(move |x| gain * x.powf(power))
        }
        Model::Exponential => {
            let base = rng.between(1.2, 2.0);
            let gain = gain / 100.0;
            Box::new(move |x| gain * base.powf(x))
        }
    }
}

/// A xorshift64 generator, seeded per model and trial so that every cell of a
/// sweep draws its own reproducible data.
pub struct Rng(u64);

impl Rng {
    pub fn seeded(model: Model, trial: u64) -> Self {
        let label = model.notation().bytes().fold(0u64, |hash, byte| {
            hash.wrapping_mul(0x100_0000_01b3) ^ byte as u64
        });
        // Any non-zero state will do; xorshift is stuck at zero.
        Self(label ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A value in `[low, high)`.
    pub fn between(&mut self, low: f64, high: f64) -> f64 {
        low + self.unit() * (high - low)
    }

    /// A value drawn from one of the given ranges, chosen at random.
    fn pick(&mut self, ranges: &[(f64, f64)]) -> f64 {
        match ranges.get(self.next_u64() as usize % ranges.len().max(1)) {
            Some(&(low, high)) => self.between(low, high),
            None => 0.0,
        }
    }

    /// A standard normal value, by the Box-Muller transform.
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

//! Choosing between fitted models, and reporting how firm that choice is.

use crate::data::{self, Sample};
use crate::error::Error;
use crate::fit::{self, Fit};
use crate::model::{self, Model};
use crate::warning::Warning;

/// How much worse than the best-scoring model a simpler one may score and still
/// be preferred.
///
/// Lowest error alone cannot choose between these models, because several of
/// them contain each other. A constant is a line of zero slope, and a line is a
/// polynomial of free exponent 1, so the richer model can always match the
/// simpler one's fit and then spend its extra freedom on the noise. Ranking by
/// error alone therefore never returns `O(1)` for flat data, and returns
/// `O(n^2.0004)` for quadratic data that happens to carry 2% of timing noise —
/// right, unreadable, and hiding the answer the user came for.
///
/// So a model that fits within this margin of the best is treated as fitting
/// equally well, and among those the simplest wins.
const SIMPLICITY_TOLERANCE: f64 = 0.05;

/// Further allowance, per free parameter, for a model that spends fewer of them.
///
/// [`corrected_error`] removes the error reduction an extra parameter buys *on
/// average*, but not the spread around that average — and six two-parameter
/// models each get a turn at beating `O(1)` on the same flat data, so the
/// luckiest of them often does. This is set so that a model has to be better by
/// more than chance would plausibly supply, not merely better.
///
/// It applies only where the parameter counts differ. Choosing between a fitted
/// exponent and a named model of the same shape is [`SIMPLICITY_TOLERANCE`]'s
/// job and is unaffected.
const PARAMETER_TOLERANCE: f64 = 0.15;

/// Absolute slack added to the tolerance above.
///
/// On data that fits almost exactly, both errors are at the level of floating
/// point noise and their *ratio* is arbitrary — a named model can score 3x the
/// polynomial's error and still be exact for every practical purpose.
const SCORE_FLOOR: f64 = 1e-8;

/// Distinct input sizes below which the choice rests on very little.
const ADVISED_POINTS: usize = 6;

/// Decades of input size below which the models are hard to tell apart.
const ADVISED_DECADES: f64 = 3.0;

/// Resamples drawn to estimate confidence.
///
/// Enough to resolve the reported fraction to a percent or so, and cheap: the
/// closed-form fit makes each resample a handful of passes over the sample.
const RESAMPLES: usize = 100;

/// Fixed seed for the resampling.
///
/// Inference is a pure function of its input: the same measurements must always
/// produce the same confidence, or a CI job that asserts on it fails at random.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// The outcome of inferring a complexity from measurements.
#[derive(Clone, Debug)]
pub struct Inference {
    /// The model that best describes the measurements.
    pub best: Fit,

    /// Every model that could be fitted, ordered best first.
    pub all: Vec<Fit>,

    /// Fraction of resampled subsets of the measurements that chose the same
    /// model, in `[0, 1]`.
    ///
    /// Low confidence means the choice rests on particular measurements rather
    /// than on the shape of the data — usually too few input sizes, or too
    /// narrow a range of them.
    pub confidence: f64,

    /// Conditions that weaken the inference without invalidating it.
    pub warnings: Vec<Warning>,
}

/// Inference with a restricted set of candidate models.
///
/// # Example
/// ```
/// use big_o::{Analysis, Model};
///
/// let data = [(1., 1.), (2., 4.), (3., 9.), (4., 16.), (5., 25.)];
///
/// // Ask only whether the cost is linear or quadratic.
/// let inference = Analysis::new()
///     .models([Model::Linear, Model::Quadratic])
///     .infer(&data)
///     .unwrap();
///
/// assert_eq!(inference.best.model, Model::Quadratic);
/// ```
#[derive(Clone, Debug)]
pub struct Analysis {
    models: Vec<Model>,
}

impl Default for Analysis {
    fn default() -> Self {
        Self::new()
    }
}

impl Analysis {
    /// An analysis over every model the crate can fit.
    pub fn new() -> Self {
        Self {
            models: model::ALL.to_vec(),
        }
    }

    /// Restricts the analysis to the given models.
    ///
    /// Narrowing the field is how a caller asserts something they already know
    /// — that a sort cannot be worse than `O(n^2)`, say — so that the answer is
    /// the best of the models they consider possible rather than of all of them.
    pub fn models(mut self, models: impl IntoIterator<Item = Model>) -> Self {
        self.models = models.into_iter().collect();
        self
    }

    /// Infers the complexity of `data`.
    ///
    /// # Errors
    /// Returns [`Error::NotEnoughData`] if fewer than three distinct input
    /// sizes survive preparation, and [`Error::NoValidComplexity`] if none of
    /// the candidate models describes what does.
    pub fn infer(&self, data: &[(f64, f64)]) -> Result<Inference, Error> {
        let sample = data::prepare(data);
        let points = sample.points().len();
        if points < data::MIN_POINTS {
            return Err(Error::NotEnoughData {
                needed: data::MIN_POINTS,
                got: points,
            });
        }

        let (all, unfittable) = self.fit_all(&sample);
        let best = select(&all, points).ok_or(Error::NoValidComplexity)?;

        Ok(Inference {
            confidence: self.confidence(&sample, best.model),
            warnings: self.warnings(&sample, &unfittable),
            best,
            all,
        })
    }

    /// Fits every candidate model to the sample.
    ///
    /// Returns the fits that can compete, best first, and the models that could
    /// not be fitted at all — as opposed to those that were fitted and then
    /// judged implausible, which had their say and lost.
    fn fit_all(&self, sample: &Sample) -> (Vec<Fit>, Vec<Model>) {
        let points = sample.points().len();
        let mut fits: Vec<Fit> = Vec::with_capacity(self.models.len());
        let mut unfittable: Vec<Model> = Vec::new();

        for &model in &self.models {
            match fit::fit(model, sample) {
                Some(fit) if is_plausible(&fit) => fits.push(fit),
                Some(_) => {}
                None => unfittable.push(model),
            }
        }
        fits.sort_by(|a, b| cmp(corrected_error(a, points), corrected_error(b, points)));
        (fits, unfittable)
    }

    /// Estimates how much the choice of model depends on which measurements
    /// happened to be taken.
    ///
    /// Refits repeated resamples of the measurements, drawn with replacement,
    /// and reports the fraction that chose the same model. A choice driven by
    /// the shape of the data survives having a few points swapped for
    /// duplicates of others; a choice driven by one lucky measurement does not.
    fn confidence(&self, sample: &Sample, best: Model) -> f64 {
        let points = sample.points();
        if points.is_empty() {
            return 0.0;
        }

        let mut rng = Rng::new(SEED);
        let mut agreed = 0usize;
        let mut compared = 0usize;
        for _ in 0..RESAMPLES {
            let drawn: Vec<(f64, f64)> = (0..points.len())
                .filter_map(|_| points.get(rng.below(points.len())).copied())
                .collect();
            // Drawing with replacement duplicates input sizes, which collapse
            // again on preparation; a draw left too thin proves nothing either
            // way, so it is not counted rather than counted as disagreement.
            let resample = data::prepare(&drawn);
            if resample.points().len() < data::MIN_POINTS {
                continue;
            }
            compared += 1;
            let (refitted, _) = self.fit_all(&resample);
            if select(&refitted, resample.points().len()).is_some_and(|fit| fit.model == best) {
                agreed += 1;
            }
        }

        match compared {
            0 => 0.0,
            _ => agreed as f64 / compared as f64,
        }
    }

    /// Collects everything about the sample that weakens the inference.
    fn warnings(&self, sample: &Sample, unfittable: &[Model]) -> Vec<Warning> {
        let mut warnings = Vec::new();

        let points = sample.points().len();
        if points < ADVISED_POINTS {
            warnings.push(Warning::TooFewPoints {
                got: points,
                advised: ADVISED_POINTS,
            });
        }

        let decades = sample.decades();
        if decades < ADVISED_DECADES {
            warnings.push(Warning::NarrowRange {
                decades,
                advised: ADVISED_DECADES,
            });
        }

        if sample.is_non_monotonic() {
            warnings.push(Warning::NonMonotonic);
        }
        if sample.is_decreasing() {
            warnings.push(Warning::DecreasingCost);
        }

        if !unfittable.is_empty() {
            warnings.push(Warning::ModelsSkipped(unfittable.to_vec()));
        }

        warnings
    }
}

/// How close a fitted base may come to one before the exponential it belongs to
/// is really a constant wearing an exponential's name.
const DEGENERATE_BASE: f64 = 1e-3;

/// Whether a fit describes a cost that could actually have been measured, under
/// the name it was fitted with.
///
/// Two ways it might not. A negative multiplier means the fitted curve goes
/// negative, which no cost does; the model is describing a trend it cannot
/// represent. And `1^n` is not exponential growth, it is a constant — reporting
/// it as `O(1^n)` is both wrong and unreadable, and `O(1)` is already competing.
///
/// A negative *exponent* is neither of those and is allowed: it is how a
/// falling cost is described.
fn is_plausible(fit: &Fit) -> bool {
    use crate::fit::ModelParams::*;
    match fit.params {
        Constant { offset } => offset >= 0.0,
        Exponential { gain, base } => gain >= 0.0 && (base - 1.0).abs() > DEGENERATE_BASE,
        Logarithmic { gain, .. }
        | Linear { gain, .. }
        | Linearithmic { gain, .. }
        | Quadratic { gain, .. }
        | Cubic { gain, .. }
        | Polynomial { gain, .. } => gain >= 0.0,
    }
}

/// Free parameters the model estimates from the data.
fn parameters(model: Model) -> usize {
    match model {
        Model::Constant => 1,
        _ => 2,
    }
}

/// The fit's error, corrected for how much of it the model was able to absorb.
///
/// Every free parameter is one the model can spend on fitting the noise rather
/// than the signal, which lowers its error whether or not the extra freedom was
/// warranted. Dividing by the degrees of freedom left over is the standard
/// correction for that, and without it a model with a spare parameter beats the
/// right one on flat data every time.
fn corrected_error(fit: &Fit, points: usize) -> f64 {
    let spent = parameters(fit.model);
    match points > spent {
        true => fit.relative_error * (points as f64 / (points - spent) as f64).sqrt(),
        false => fit.relative_error,
    }
}

/// How much freedom a model has to bend towards the data.
///
/// A constant has a level and no shape. The named curves have a level and a
/// scale, but their shape is fixed by their name. A free-exponent polynomial
/// has the shape as a parameter too, and can imitate any of them.
fn flexibility(model: Model) -> u8 {
    match model {
        Model::Constant => 0,
        _ if model.has_free_exponent() => 2,
        _ => 1,
    }
}

/// Total ordering over scores, treating any incomparable pair as equal.
fn cmp(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Whether `candidate` describes the data as well as `best` does, once the
/// freedom each of them had to bend towards the noise is allowed for.
fn fits_as_well_as(candidate: &Fit, best: &Fit, points: usize) -> bool {
    let saved = parameters(best.model).saturating_sub(parameters(candidate.model));
    let allowance = 1.0 + SIMPLICITY_TOLERANCE + PARAMETER_TOLERANCE * saved as f64;

    corrected_error(candidate, points) <= corrected_error(best, points) * allowance + SCORE_FLOOR
}

/// Picks the model to report from fits ordered best-scoring first.
///
/// The lowest error does not settle it: see [`SIMPLICITY_TOLERANCE`] for why the
/// simplest model that fits about as well is the one worth reporting.
fn select(fitted: &[Fit], points: usize) -> Option<Fit> {
    let best = fitted.first().copied()?;

    fitted
        .iter()
        .filter(|fit| fits_as_well_as(fit, &best, points))
        .min_by(|a, b| {
            // Among models that fit equally well, the one with less freedom to
            // have fitted the noise: `O(n^2)` over `O(n^2.0004)`. Between two
            // models with the same freedom there is no simplicity argument to
            // make — neither is the simpler explanation — so the better fit
            // wins. Preferring the slower-growing one there merely reports
            // `O(n)` for data that is `O(n log n)` with a large constant term.
            flexibility(a.model)
                .cmp(&flexibility(b.model))
                .then(cmp(corrected_error(a, points), corrected_error(b, points)))
        })
        .copied()
        .or(Some(best))
}

/// A xorshift64 generator.
///
/// Seeded from a constant so that inference stays a pure function of its input;
/// resampling needs values that are spread out, not values that are secret.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value in `0..n`, or `0` if `n` is zero.
    fn below(&mut self, n: usize) -> usize {
        match n {
            0 => 0,
            n => (self.next_u64() % n as u64) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::ModelParams;

    /// A sample size typical of a real sweep, for the degrees-of-freedom
    /// correction the selection applies.
    const POINTS: usize = 24;

    fn fit_of(model: Model, params: ModelParams, relative_error: f64) -> Fit {
        Fit {
            model,
            params,
            r_squared: 1.0 - relative_error,
            relative_error,
        }
    }

    fn quadratic(relative_error: f64) -> Fit {
        fit_of(
            Model::Quadratic,
            ModelParams::Quadratic {
                gain: 1.0,
                offset: 0.0,
            },
            relative_error,
        )
    }

    fn polynomial(power: f64, relative_error: f64) -> Fit {
        fit_of(
            Model::Polynomial,
            ModelParams::Polynomial { gain: 1.0, power },
            relative_error,
        )
    }

    fn constant(relative_error: f64) -> Fit {
        fit_of(
            Model::Constant,
            ModelParams::Constant { offset: 1.0 },
            relative_error,
        )
    }

    fn linear(relative_error: f64) -> Fit {
        fit_of(
            Model::Linear,
            ModelParams::Linear {
                gain: 1.0,
                offset: 0.0,
            },
            relative_error,
        )
    }

    #[test]
    fn prefers_a_named_model_that_ties_with_a_free_exponent() {
        let chosen = select(&[polynomial(2.0004, 0.0200), quadratic(0.0201)], POINTS);

        assert_eq!(chosen.map(|fit| fit.model), Some(Model::Quadratic));
    }

    #[test]
    fn keeps_a_free_exponent_that_wins_by_more_than_the_margin() {
        let chosen = select(&[polynomial(2.5, 0.001), quadratic(0.15)], POINTS);

        assert_eq!(chosen.map(|fit| fit.model), Some(Model::Polynomial));
    }

    #[test]
    fn prefers_a_named_model_when_both_fit_exactly() {
        // Both errors are floating point noise, so their ratio says nothing.
        let chosen = select(&[polynomial(2.0, 1e-17), quadratic(3e-16)], POINTS);

        assert_eq!(chosen.map(|fit| fit.model), Some(Model::Quadratic));
    }

    #[test]
    fn prefers_a_constant_to_a_line_that_only_tilts_towards_the_noise() {
        // A line can always match a constant and then spend its slope on noise,
        // so ranking by error alone would never report O(1) for flat data.
        let chosen = select(&[linear(0.0495), constant(0.0500)], POINTS);

        assert_eq!(chosen.map(|fit| fit.model), Some(Model::Constant));
    }

    #[test]
    fn keeps_a_line_that_beats_a_constant_by_more_than_the_margin() {
        let chosen = select(&[linear(0.01), constant(0.40)], POINTS);

        assert_eq!(chosen.map(|fit| fit.model), Some(Model::Linear));
    }

    #[test]
    fn leaves_a_named_winner_alone() {
        let chosen = select(&[quadratic(0.01), polynomial(2.5, 0.30)], POINTS);

        assert_eq!(chosen.map(|fit| fit.model), Some(Model::Quadratic));
    }

    #[test]
    fn selects_nothing_from_nothing() {
        assert_eq!(select(&[], POINTS), None);
    }

    #[test]
    fn rejects_a_curve_that_would_go_negative() {
        assert!(!is_plausible(&fit_of(
            Model::Linear,
            ModelParams::Linear {
                gain: -1.0,
                offset: 0.0
            },
            0.0
        )));
        assert!(is_plausible(&polynomial(-1.0, 0.0)), "a falling cost");
    }

    #[test]
    fn resampling_is_reproducible() {
        let mut a = Rng::new(SEED);
        let mut b = Rng::new(SEED);

        let drawn: Vec<usize> = (0..32).map(|_| a.below(10)).collect();

        assert!(drawn.iter().all(|&i| i < 10));
        assert_eq!(drawn, (0..32).map(|_| b.below(10)).collect::<Vec<_>>());
        assert!(drawn.windows(2).any(|pair| pair[0] != pair[1]), "not stuck");
        assert_eq!(Rng::new(SEED).below(0), 0);
    }
}

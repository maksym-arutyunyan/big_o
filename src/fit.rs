//! Fitting one model to a sample, and scoring how well it describes it.

use crate::data::{self, Sample};
use crate::linalg::{self, Line};
use crate::model::Model;
use std::cmp::Ordering;
use std::fmt;

/// Coefficients of a fitted approximation function.
///
/// Each variant carries exactly the parameters its shape has, so a fitted
/// polynomial always has an exponent and a fitted line never carries an unused
/// one. There is no representable fit with a parameter missing.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ModelParams {
    /// `f(x) = offset`
    Constant {
        /// Value the cost settles at.
        offset: f64,
    },
    /// `f(x) = gain * ln(x) + offset`
    Logarithmic {
        /// Multiplier of `ln(x)`.
        gain: f64,
        /// Cost at `x = 1`.
        offset: f64,
    },
    /// `f(x) = gain * x + offset`
    Linear {
        /// Cost per unit of input.
        gain: f64,
        /// Cost at `x = 0`.
        offset: f64,
    },
    /// `f(x) = gain * x * ln(x) + offset`
    Linearithmic {
        /// Multiplier of `x * ln(x)`.
        gain: f64,
        /// Cost at `x = 1`.
        offset: f64,
    },
    /// `f(x) = gain * x^2 + offset`
    Quadratic {
        /// Multiplier of `x^2`.
        gain: f64,
        /// Cost at `x = 0`.
        offset: f64,
    },
    /// `f(x) = gain * x^3 + offset`
    Cubic {
        /// Multiplier of `x^3`.
        gain: f64,
        /// Cost at `x = 0`.
        offset: f64,
    },
    /// `f(x) = gain * x^power`
    Polynomial {
        /// Cost at `x = 1`.
        gain: f64,
        /// Fitted exponent. Negative when cost falls as the input grows.
        power: f64,
    },
    /// `f(x) = gain * base^x`
    Exponential {
        /// Cost at `x = 0`.
        gain: f64,
        /// Fitted base.
        base: f64,
    },
}

impl ModelParams {
    /// Evaluates the fitted function at `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        match *self {
            ModelParams::Constant { offset } => offset,
            ModelParams::Logarithmic { gain, offset } => gain * x.ln() + offset,
            ModelParams::Linear { gain, offset } => gain * x + offset,
            ModelParams::Linearithmic { gain, offset } => gain * x * x.ln() + offset,
            ModelParams::Quadratic { gain, offset } => gain * x.powi(2) + offset,
            ModelParams::Cubic { gain, offset } => gain * x.powi(3) + offset,
            ModelParams::Polynomial { gain, power } => gain * x.powf(power),
            ModelParams::Exponential { gain, base } => gain * base.powf(x),
        }
    }

    /// Whether every coefficient is a finite number.
    fn is_finite(&self) -> bool {
        match *self {
            ModelParams::Constant { offset } => offset.is_finite(),
            ModelParams::Polynomial { gain, power } => gain.is_finite() && power.is_finite(),
            ModelParams::Exponential { gain, base } => gain.is_finite() && base.is_finite(),
            ModelParams::Logarithmic { gain, offset }
            | ModelParams::Linear { gain, offset }
            | ModelParams::Linearithmic { gain, offset }
            | ModelParams::Quadratic { gain, offset }
            | ModelParams::Cubic { gain, offset } => gain.is_finite() && offset.is_finite(),
        }
    }
}

/// One model fitted to a set of measurements.
///
/// # Example
/// ```
/// let data = [(1., 1.), (2., 4.), (3., 9.), (4., 16.), (5., 25.)];
///
/// let inference = big_o::infer_complexity(&data).unwrap();
///
/// assert_eq!(inference.best.model, big_o::Model::Quadratic);
/// assert!(inference.best.is_at_most(big_o::Model::Cubic));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fit {
    /// The model class that was fitted.
    pub model: Model,

    /// Coefficients recovered from the data.
    pub params: ModelParams,

    /// Fraction of the variation in cost this fit explains, in `(-inf, 1]`.
    /// One is exact; zero is no better than predicting the mean.
    pub r_squared: f64,

    /// Typical error as a fraction of the measurement it was made against.
    ///
    /// Each residual is divided by its own measurement before the errors are
    /// combined, so being 3% out at the smallest input counts as much as being
    /// 3% out at the largest. Weighting by the absolute residual instead would
    /// let the last few points decide everything, which is why `O(n log n)`
    /// data used to be reported as `O(n)`: over the top decade the two curves
    /// differ by almost nothing in absolute terms.
    ///
    /// Scale-free, so fits to data in nanoseconds and in seconds compare
    /// directly — which raw residual sums do not.
    pub relative_error: f64,
}

impl Fit {
    /// Evaluates the fitted function at `x`.
    pub fn evaluate(&self, x: f64) -> f64 {
        self.params.evaluate(x)
    }

    /// Whether this fit grows no faster than `model`.
    ///
    /// ```
    /// let data = [(1., 2.), (2., 4.), (3., 6.), (4., 8.), (5., 10.)];
    /// let inference = big_o::infer_complexity(&data).unwrap();
    ///
    /// assert!(inference.best.is_at_most(big_o::Model::Linear));
    /// assert!(inference.best.is_at_most(big_o::Model::Quadratic));
    /// ```
    pub fn is_at_most(&self, model: Model) -> bool {
        self.degree() <= model.upper_degree()
    }

    /// Whether this fit grows strictly slower than `model`.
    pub fn is_faster_than(&self, model: Model) -> bool {
        self.degree() < model.lower_degree()
    }

    /// Where this fit sits on the polynomial-degree scale.
    ///
    /// Signed, so a cost that falls as the input grows lands below `O(1)`
    /// rather than wrapping around to the fastest-looking rank.
    fn degree(&self) -> f64 {
        match self.params {
            ModelParams::Polynomial { power, .. } => power,
            _ => self.model.upper_degree(),
        }
    }
}

/// Ordering by growth rate, with fitted exponents interleaved among the named
/// models: `O(n) < O(n^1.5) < O(n^2)`.
///
/// Partial in both the directions the name suggests. Growth rate says nothing
/// about two fits that grow alike — two quadratics with different constants are
/// not ordered, and reporting them equal would contradict `PartialEq`, which
/// std requires to agree with this.
impl PartialOrd for Fit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.degree().partial_cmp(&other.degree())? {
            Ordering::Equal if self != other => None,
            ordering => Some(ordering),
        }
    }
}

/// Renders the notation with fitted values substituted where the model has a
/// parameter to substitute: `O(n^2.03)`, `O(1.98^n)`. Models whose shape is
/// fully determined by their name keep that name.
impl fmt::Display for Fit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.params {
            ModelParams::Polynomial { power, .. } => write!(f, "O(n^{})", exponent(power)),
            ModelParams::Exponential { base, .. } => write!(f, "O({}^n)", trim(base)),
            _ => write!(f, "{}", self.model.notation()),
        }
    }
}

/// Renders a fitted exponent so that it cannot be read as a named model.
///
/// Two decimals is the right precision to read, but an exponent of 2.004 would
/// render as `O(n^2)` — indistinguishable from a fitted `Quadratic`, which is a
/// different and stronger claim about the data. Where rounding would erase that
/// distinction, more precision is shown instead.
fn exponent(power: f64) -> String {
    let rounded = trim(power);
    match rounded.contains('.') || power.fract() == 0.0 {
        true => rounded,
        false => format!("{power:.3}"),
    }
}

/// Formats a fitted parameter to two decimals without trailing zeros, so an
/// exponent reads as `2.03` and `2` rather than `2.03` and `2.00`.
fn trim(value: f64) -> String {
    let text = format!("{value:.2}");
    match text.trim_end_matches('0').trim_end_matches('.') {
        "" | "-" => "0".to_string(),
        trimmed => trimmed.to_string(),
    }
}

/// Transforms a measurement into the space where `model` is a straight line.
///
/// Returns `None` when the point has no image there: the logarithmic models are
/// undefined at `x = 0`, and the two models fitted in log-`y` space are
/// undefined for costs that are zero or negative. Such a point is dropped for
/// that model alone, so the models that *can* describe it still compete.
fn linearize(model: Model, x: f64, y: f64) -> Option<(f64, f64)> {
    let point = match model {
        Model::Constant => (0.0, y),
        Model::Logarithmic => (x.ln(), y),
        Model::Linear => (x, y),
        Model::Linearithmic => (x * x.ln(), y),
        Model::Quadratic => (x.powi(2), y),
        Model::Cubic => (x.powi(3), y),
        Model::Polynomial => (x.ln(), y.ln()),
        Model::Exponential => (x, y.ln()),
    };
    (point.0.is_finite() && point.1.is_finite()).then_some(point)
}

/// How much a point's error counts when fitting `model`.
///
/// The models fitted against cost directly are fitted to minimize the *relative*
/// error, because that is what they are then judged on and because timing noise
/// is proportional to the measurement. Least squares does that when each point
/// is weighted by `1 / y^2`. Without it a model is chosen on one criterion
/// after being fitted to a different one, and loses comparisons it should win:
/// `O(n log n)` data was reported as `O(n)` because an evenly weighted line
/// through the largest few measurements is an excellent line and a poor curve.
///
/// The two models fitted in log-`y` space need no weight — a residual in the
/// logarithm is already a relative one.
fn weight(model: Model, y: f64, floor: f64) -> f64 {
    match model {
        Model::Polynomial | Model::Exponential => 1.0,
        _ => {
            let scale = y.abs().max(floor);
            match scale > 0.0 {
                true => 1.0 / (scale * scale),
                false => 1.0,
            }
        }
    }
}

/// Converts a line fitted in linearized space back into the model's own
/// coefficients.
fn delinearize(model: Model, line: Line) -> ModelParams {
    let Line { gain, offset } = line;
    match model {
        Model::Constant => ModelParams::Constant { offset },
        Model::Logarithmic => ModelParams::Logarithmic { gain, offset },
        Model::Linear => ModelParams::Linear { gain, offset },
        Model::Linearithmic => ModelParams::Linearithmic { gain, offset },
        Model::Quadratic => ModelParams::Quadratic { gain, offset },
        Model::Cubic => ModelParams::Cubic { gain, offset },
        // Both were fitted in log-`y` space, where the multiplier became an
        // additive offset and the exponent became the slope.
        Model::Polynomial => ModelParams::Polynomial {
            gain: offset.exp(),
            power: gain,
        },
        Model::Exponential => ModelParams::Exponential {
            gain: offset.exp(),
            base: gain.exp(),
        },
    }
}

/// Smallest measurement, as a share of the average one, that is still divided
/// by itself when the relative error is formed.
///
/// A measurement of zero would otherwise make every model infinitely wrong.
/// Set low enough that it binds only on values which are effectively zero, and
/// not on the genuinely small measurements at the start of an exponential.
const SMALLEST_MEANINGFUL_SHARE: f64 = 1e-6;

/// Scores `params` against the measurements they were fitted to.
///
/// Both scores are computed in the original space rather than the linearized
/// one, so models reached through different transforms stay comparable: a fit
/// that looks tight in log-`y` space can be badly wrong in seconds.
fn score(params: &ModelParams, data: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mean = linalg::mean(data.iter().map(|&(_, y)| y))?;
    let magnitude = linalg::mean(data.iter().map(|&(_, y)| y.abs()))?;
    let floor = magnitude * SMALLEST_MEANINGFUL_SHARE;

    let mut sum_squared_error = 0.0;
    let mut sum_squared_total = 0.0;
    let mut sum_squared_relative = 0.0;
    for &(x, y) in data {
        let error = y - params.evaluate(x);
        if !error.is_finite() {
            return None;
        }
        sum_squared_error += error * error;
        sum_squared_total += (y - mean) * (y - mean);

        let scale = y.abs().max(floor);
        if scale > 0.0 {
            let relative = error / scale;
            sum_squared_relative += relative * relative;
        } else if error != 0.0 {
            // Nothing was measured here and the model predicts something.
            return None;
        }
    }
    if !sum_squared_error.is_finite()
        || !sum_squared_total.is_finite()
        || !sum_squared_relative.is_finite()
    {
        return None;
    }

    // Data with no variation in cost is explained exactly by any fit that
    // reproduces it, and not at all by one that does not.
    let r_squared = match sum_squared_total > 0.0 {
        true => 1.0 - sum_squared_error / sum_squared_total,
        false => (sum_squared_error == 0.0) as u8 as f64,
    };
    let relative_error = (sum_squared_relative / data.len() as f64).sqrt();

    (r_squared.is_finite() && relative_error.is_finite()).then_some((r_squared, relative_error))
}

/// Fits `model` to `sample`.
///
/// Returns `None` when this model cannot describe this data — too few points
/// survive linearization, or the coefficients or scores are not finite. The
/// caller skips the model rather than failing, because data one model cannot
/// consume is usually still described by the others.
pub(crate) fn fit(model: Model, sample: &Sample) -> Option<Fit> {
    let data = sample.points();
    let magnitude = linalg::mean(data.iter().map(|&(_, y)| y.abs()))?;
    let floor = magnitude * SMALLEST_MEANINGFUL_SHARE;

    let linearized: Vec<(f64, f64, f64)> = data
        .iter()
        .filter_map(|&(x, y)| {
            let (u, v) = linearize(model, x, y)?;
            Some((u, v, weight(model, y, floor)))
        })
        .collect();
    if linearized.len() < data::MIN_POINTS {
        return None;
    }

    // A constant has no slope to estimate: every point linearizes to the same
    // `x`, so the least-squares line is undetermined and the fit is the mean.
    let line = match model {
        Model::Constant => Line {
            gain: 0.0,
            offset: linalg::weighted_mean(linearized.iter().map(|&(_, y, w)| (y, w)))?,
        },
        _other => linalg::fit_line(&linearized)?,
    };

    let params = delinearize(model, line);
    if !params.is_finite() {
        return None;
    }
    let (r_squared, relative_error) = score(&params, data)?;

    Some(Fit {
        model,
        params,
        r_squared,
        relative_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fitted(model: Model, data: &[(f64, f64)]) -> Fit {
        fit(model, &data::prepare(data)).expect("this model fits this data")
    }

    fn quadratic_data() -> Vec<(f64, f64)> {
        (1..=10).map(|n| (n as f64, (n * n) as f64)).collect()
    }

    #[test]
    fn recovers_coefficients_in_the_original_space() {
        let fit = fitted(Model::Quadratic, &quadratic_data());

        assert_eq!(
            fit.params,
            ModelParams::Quadratic {
                gain: 1.0,
                offset: 0.0
            }
        );
        assert!((fit.r_squared - 1.0).abs() < 1e-9);
        assert!(fit.relative_error < 1e-9);
    }

    #[test]
    fn fits_a_constant_to_flat_data_exactly() {
        let fit = fitted(Model::Constant, &[(1., 7.), (2., 7.), (3., 7.)]);

        assert_eq!(fit.params, ModelParams::Constant { offset: 7.0 });
        assert!(fit.relative_error < 1e-12);
    }

    #[test]
    fn a_constant_sits_where_the_relative_error_is_least() {
        // Not the arithmetic mean of 20: being 10 out at a cost of 10 is a
        // hundred percent wrong, and being 10 out at 30 is a third of that.
        let fit = fitted(Model::Constant, &[(1., 10.), (2., 20.), (3., 30.)]);

        match fit.params {
            ModelParams::Constant { offset } => {
                assert!(
                    offset < 20.0,
                    "should be pulled below the mean, got {offset}"
                );
                assert!(offset > 10.0);
            }
            other => panic!("expected a constant, got {other:?}"),
        }
    }

    #[test]
    fn scores_are_scale_free() {
        let nanoseconds = quadratic_data();
        let seconds: Vec<(f64, f64)> = nanoseconds.iter().map(|&(x, y)| (x, y / 1e9)).collect();

        let a = fitted(Model::Linear, &nanoseconds);
        let b = fitted(Model::Linear, &seconds);

        assert!((a.relative_error - b.relative_error).abs() < 1e-9);
        assert!((a.r_squared - b.r_squared).abs() < 1e-9);
    }

    #[test]
    fn a_point_without_an_image_is_dropped_for_that_model_only() {
        assert_eq!(linearize(Model::Logarithmic, 0.0, 1.0), None);
        assert_eq!(linearize(Model::Polynomial, 0.0, 1.0), None);
        assert_eq!(linearize(Model::Linear, 0.0, 1.0), Some((0.0, 1.0)));

        assert_eq!(linearize(Model::Exponential, 1.0, 0.0), None);
        assert_eq!(linearize(Model::Polynomial, 1.0, -1.0), None);
        assert_eq!(linearize(Model::Quadratic, 1.0, -1.0), Some((1.0, -1.0)));
    }

    #[test]
    fn skips_a_model_that_cannot_consume_the_data() {
        let sample = data::prepare(&[(0., 1.), (1., 2.), (2., 3.)]);

        assert!(fit(Model::Logarithmic, &sample).is_none());
        assert!(fit(Model::Linear, &sample).is_some());
    }

    #[test]
    fn orders_fitted_exponents_among_the_named_models() {
        let quadratic = fitted(Model::Quadratic, &quadratic_data());
        let free: Vec<(f64, f64)> = (1..=10).map(|n| (n as f64, (n as f64).powf(1.5))).collect();
        let polynomial = fitted(Model::Polynomial, &free);
        let linear = fitted(Model::Linear, &[(1., 1.), (2., 2.), (3., 3.), (4., 4.)]);

        assert!(linear < polynomial);
        assert!(polynomial < quadratic);
    }

    #[test]
    fn substitutes_fitted_values_into_the_notation() {
        let fit = Fit {
            model: Model::Polynomial,
            params: ModelParams::Polynomial {
                gain: 1.0,
                power: 2.031,
            },
            r_squared: 1.0,
            relative_error: 0.0,
        };
        assert_eq!(fit.to_string(), "O(n^2.03)");

        let fit = Fit {
            model: Model::Exponential,
            params: ModelParams::Exponential {
                gain: 1.0,
                base: 1.981,
            },
            r_squared: 1.0,
            relative_error: 0.0,
        };
        assert_eq!(fit.to_string(), "O(1.98^n)");
    }

    #[test]
    fn named_models_keep_their_canonical_notation() {
        assert_eq!(
            fitted(Model::Quadratic, &quadratic_data()).to_string(),
            "O(n^2)"
        );

        let linearithmic: Vec<(f64, f64)> = (1..=10)
            .map(|n| (n as f64, n as f64 * (n as f64).ln()))
            .collect();
        assert_eq!(
            fitted(Model::Linearithmic, &linearithmic).to_string(),
            "O(n log n)"
        );
    }

    #[test]
    fn a_fitted_exponent_never_masquerades_as_a_named_model() {
        let almost_square = Fit {
            model: Model::Polynomial,
            params: ModelParams::Polynomial {
                gain: 1.0,
                power: 2.004,
            },
            r_squared: 1.0,
            relative_error: 0.0,
        };

        assert_eq!(almost_square.to_string(), "O(n^2.004)");
        assert_eq!(exponent(2.0), "2", "a whole exponent still reads plainly");
        assert_eq!(exponent(1.5), "1.5");
        assert_eq!(exponent(-1.0), "-1");
    }

    #[test]
    fn fits_that_grow_alike_are_unordered_rather_than_equal() {
        let a = fitted(Model::Quadratic, &quadratic_data());
        let scaled: Vec<(f64, f64)> = quadratic_data()
            .iter()
            .map(|&(x, y)| (x, 3.0 * y))
            .collect();
        let b = fitted(Model::Quadratic, &scaled);

        assert_ne!(a, b);
        assert_eq!(a.partial_cmp(&b), None, "must not contradict PartialEq");
        assert_eq!(a.partial_cmp(&a), Some(Ordering::Equal));
    }

    #[test]
    fn trims_trailing_zeros_from_fitted_values() {
        assert_eq!(trim(2.0), "2");
        assert_eq!(trim(2.5), "2.5");
        assert_eq!(trim(2.031), "2.03");
        assert_eq!(trim(-1.0), "-1");
        assert_eq!(trim(0.001), "0");
    }
}

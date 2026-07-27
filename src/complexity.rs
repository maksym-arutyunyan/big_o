use crate::data::Sample;
use crate::error::Error;
use crate::linalg::{self, Line};
use crate::name;
use crate::name::Name;
use crate::params::Params;

/// Result of fitting an asymptotic computational complexity to a set of
/// measurements.
///
/// `Complexity` instances are typically created via [`infer_complexity`] or
/// the [`complexity`] helper and can be compared using their [`rank`] field.
///
/// # Example
/// ```
/// # use big_o::Name;
/// let linear = big_o::complexity("O(n)").unwrap();
/// let quadratic = big_o::complexity("O(n^2)").unwrap();
/// assert_eq!(linear.name, Name::Linear);
/// assert!(linear.rank < quadratic.rank);
/// ```
#[derive(Clone, Debug)]
pub struct Complexity {
    /// Human-readable name of the complexity model.
    pub name: Name,

    /// String representation in Big&nbsp;O notation.
    pub notation: &'static str,

    /// Parameters of the fitted approximation function.
    pub params: Params,

    /// Relative rank used to compare complexities.
    /// A lower value means a better asymptotic behaviour.
    pub rank: u32,
}

/// Returns a calculated approximation function `f(x)`
fn get_function(name: Name, params: Params) -> Result<Box<dyn Fn(f64) -> f64>, Error> {
    if let (Some(a), Some(b)) = match name {
        Name::Polynomial => (params.gain, params.power),
        Name::Exponential => (params.gain, params.base),
        _other => (params.gain, params.offset),
    } {
        let f: Box<dyn Fn(f64) -> f64> = match name {
            Name::Constant => Box::new(move |_x| b),
            Name::Logarithmic => Box::new(move |x| a * x.ln() + b),
            Name::Linear => Box::new(move |x| a * x + b),
            Name::Linearithmic => Box::new(move |x| a * x * x.ln() + b),
            Name::Quadratic => Box::new(move |x| a * x.powi(2) + b),
            Name::Cubic => Box::new(move |x| a * x.powi(3) + b),
            Name::Polynomial => Box::new(move |x| a * x.powf(b)),
            Name::Exponential => Box::new(move |x| a * b.powf(x)),
        };
        Ok(f)
    } else {
        Err(Error::MissingFunctionCoeffs)
    }
}

/// Computes values of `f(x)` given `x`
#[allow(dead_code)]
fn compute_f(name: Name, params: Params, x: &[f64]) -> Result<Vec<f64>, Error> {
    let f = get_function(name, params)?;
    let y = x.iter().copied().map(f).collect();
    Ok(y)
}

pub struct ComplexityBuilder {
    name: Name,
    params: Option<Params>,
}

impl ComplexityBuilder {
    pub fn new(name: Name) -> Self {
        Self { name, params: None }
    }

    #[allow(dead_code)] // Used in tests.
    pub fn power(&mut self, x: f64) -> &mut Self {
        self.params = Some(Params::new().power(x).build());
        self
    }

    pub fn build(&self) -> Result<Complexity, Error> {
        let mut params = Params::new().build();
        if let Some(p) = &self.params {
            params = p.clone();
        }
        let rank = rank(self.name, params.clone())?;
        Ok(Complexity {
            name: self.name,
            notation: name::notation(self.name),
            params,
            rank,
        })
    }
}

/// Transforms a measurement into the space where `name` is a straight line.
///
/// Returns `None` when the point has no image in that space: the logarithmic
/// models are undefined at `x = 0`, and the two models fitted in log-`y` space
/// are undefined for costs that are zero or negative. Such a point is dropped
/// for that model alone, so the models that *can* describe it still compete.
fn linearize(name: Name, x: f64, y: f64) -> Option<(f64, f64)> {
    let point = match name {
        Name::Constant => (0.0, y),
        Name::Logarithmic => (x.ln(), y),
        Name::Linear => (x, y),
        Name::Linearithmic => (x * x.ln(), y),
        Name::Quadratic => (x.powi(2), y),
        Name::Cubic => (x.powi(3), y),
        Name::Polynomial => (x.ln(), y.ln()),
        Name::Exponential => (x, y.ln()),
    };
    (point.0.is_finite() && point.1.is_finite()).then_some(point)
}

/// Converts linear coeffs `gain` and `offset` to corresponding complexity params.
fn delinearize(name: Name, line: Line) -> Params {
    let Line { gain, offset } = line;
    // Delinearize coeffs.
    let (a, b) = match name {
        Name::Polynomial => (offset.exp(), gain),
        Name::Exponential => (offset.exp(), gain.exp()),
        _other => (gain, offset),
    };
    // Convert coeffs to params.
    match name {
        Name::Polynomial => Params::new().gain(a).power(b).build(),
        Name::Exponential => Params::new().gain(a).base(b).build(),
        _other => Params::new().gain(a).offset(b).build(),
    }
}

fn calculate_residuals(name: Name, params: Params, data: &[(f64, f64)]) -> Option<f64> {
    let f = get_function(name, params).ok()?;
    let residuals: f64 = data.iter().map(|(x, y)| (*y - f(*x)).abs()).sum();

    residuals.is_finite().then_some(residuals)
}

fn rank(name: Name, params: Params) -> Result<u32, Error> {
    // Rank is similar to a degree of a corresponding polynomial:
    // - constant: 0, f(x) = x ^ 0.000
    // - logarithmic: 130, empirical value k for a big x in f(x) = x ^ k
    //     base 1_000_000 log of 6 is 0.130
    //     approx. f(x) = x ^ 0.130
    // - linear: 1_000, f(x) = x ^ 1.000
    // - linearithmic: 1_130, approx. f(x) = x ^ 1.130
    // - quadratic: 2_000, f(x) = x ^ 2.000
    // - cubic: 3_000, f(x) = x ^ 3.000
    // - polynomial: depends on polynomial degree
    // - exponential: 1_000_000, practically there is no sense in polynomial degree > 1_000.000
    match name {
        Name::Constant => Ok(0),
        Name::Logarithmic => Ok(130),
        Name::Linear => Ok(1_000),
        Name::Linearithmic => Ok(1_130),
        Name::Quadratic => Ok(2_000),
        Name::Cubic => Ok(3_000),
        Name::Polynomial => match params.power {
            Some(power) => Ok(std::cmp::min((1_000.0 * power) as u32, 1_000_000)),
            None => Err(Error::MissingPolynomialPower),
        },
        Name::Exponential => Ok(1_000_000),
    }
}

/// Fits a function of the given complexity to a prepared sample.
///
/// Returns `None` when this model cannot describe this data — too few points
/// survive linearization, or the fitted coefficients are not finite. The caller
/// skips the model rather than failing, because the data a model cannot consume
/// is usually still described by the others.
pub(crate) fn fit(name: Name, sample: &Sample) -> Option<Complexity> {
    let data = sample.points();
    let linearized: Vec<(f64, f64)> = data
        .iter()
        .filter_map(|&(x, y)| linearize(name, x, y))
        .collect();
    if linearized.len() < crate::data::MIN_POINTS {
        return None;
    }

    // A constant has no slope to estimate: every point linearizes to the same
    // `x`, so the least-squares line is undetermined and the fit is the mean.
    let line = match name {
        Name::Constant => Line {
            gain: 0.0,
            offset: linalg::mean(linearized.iter().map(|&(_, y)| y))?,
        },
        _other => linalg::fit_line(&linearized)?,
    };

    let params = delinearize(name, line);
    // Residuals are measured in the original space, not the linearized one, so
    // that models linearized by different transforms stay comparable.
    let residuals = calculate_residuals(name, params.clone(), data)?;
    let params = Params {
        residuals: Some(residuals),
        ..params
    };
    let rank = rank(name, params.clone()).ok()?;

    Some(Complexity {
        name,
        notation: name::notation(name),
        params,
        rank,
    })
}

/// Creates `Complexity` from string.
///
/// # Example
/// ```
/// use big_o::{complexity, Name::*};
///
/// let linear = complexity("O(n)").unwrap();
/// assert_eq!(linear.name, Linear);
///
/// let cubic = complexity("O(n^3)").unwrap();
/// assert_eq!(cubic.name, Cubic);
///
/// assert!(linear.rank < cubic.rank);
/// ```
pub fn complexity(string: &str) -> Result<Complexity, Error> {
    let name: Name = string.try_into()?;
    crate::complexity::ComplexityBuilder::new(name).build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    fn constant() -> Complexity {
        ComplexityBuilder::new(Name::Constant).build().unwrap()
    }

    fn logarithmic() -> Complexity {
        ComplexityBuilder::new(Name::Logarithmic).build().unwrap()
    }

    fn linear() -> Complexity {
        ComplexityBuilder::new(Name::Linear).build().unwrap()
    }

    fn linearithmic() -> Complexity {
        ComplexityBuilder::new(Name::Linearithmic).build().unwrap()
    }

    fn quadratic() -> Complexity {
        ComplexityBuilder::new(Name::Quadratic).build().unwrap()
    }

    fn cubic() -> Complexity {
        ComplexityBuilder::new(Name::Cubic).build().unwrap()
    }

    fn exponential() -> Complexity {
        ComplexityBuilder::new(Name::Exponential).build().unwrap()
    }

    fn polynomial(power: f64) -> Complexity {
        ComplexityBuilder::new(Name::Polynomial)
            .power(power)
            .build()
            .unwrap()
    }

    #[test]
    fn polynomial_missing_power_error() {
        let err = ComplexityBuilder::new(Name::Polynomial)
            .build()
            .unwrap_err();
        assert!(matches!(err, Error::MissingPolynomialPower));
    }

    #[test]
    fn test_complecity_rank() {
        // O(1) < ... < O(n)
        assert!(constant().rank < logarithmic().rank);
        assert!(logarithmic().rank < polynomial(0.5).rank);
        assert!(polynomial(0.5).rank < linear().rank);

        // O(n) < ... < O(n^2)
        assert!(linear().rank < linearithmic().rank);
        assert!(linearithmic().rank < polynomial(1.5).rank);
        assert!(polynomial(1.5).rank < quadratic().rank);

        // O(n^2) < ... < O(n^3)
        assert!(quadratic().rank < polynomial(2.5).rank);
        assert!(polynomial(2.5).rank < cubic().rank);

        // O(n^3) < ... < O(c^n)
        assert!(cubic().rank < polynomial(3.5).rank);
        assert!(polynomial(3.5).rank < exponential().rank);
    }

    #[test]
    fn a_point_without_an_image_is_dropped_for_that_model_only() {
        // ln(0) has no image, so x = 0 leaves the log-space models one point short.
        assert_eq!(linearize(Name::Logarithmic, 0.0, 1.0), None);
        assert_eq!(linearize(Name::Polynomial, 0.0, 1.0), None);
        assert_eq!(linearize(Name::Linear, 0.0, 1.0), Some((0.0, 1.0)));

        // A cost of zero or less has no image in log-y space.
        assert_eq!(linearize(Name::Exponential, 1.0, 0.0), None);
        assert_eq!(linearize(Name::Polynomial, 1.0, -1.0), None);
        assert_eq!(linearize(Name::Quadratic, 1.0, -1.0), Some((1.0, -1.0)));
    }

    #[test]
    fn fits_a_constant_as_the_mean_cost() {
        let sample = crate::data::prepare(&[(1., 10.), (2., 20.), (3., 30.)]);

        let fitted = fit(Name::Constant, &sample).expect("a constant always fits");

        assert_eq!(fitted.params.offset, Some(20.0));
        assert_eq!(fitted.params.gain, Some(0.0));
    }

    #[test]
    fn skips_a_model_that_cannot_consume_the_data() {
        // Only two points survive ln(x), one short of the minimum.
        let sample = crate::data::prepare(&[(0., 1.), (1., 2.), (2., 3.)]);

        assert!(fit(Name::Logarithmic, &sample).is_none());
        assert!(fit(Name::Linear, &sample).is_some());
    }
}

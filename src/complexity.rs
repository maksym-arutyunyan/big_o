use crate::error::Error;
use crate::linalg;
use crate::name;
use crate::name::Name;
use crate::params::Params;

/// A structure to describe asymptotic computational complexity
#[derive(Clone, Debug)]
pub struct Complexity {
    /// Human-readable name
    pub name: Name,

    /// Big O notation
    pub notation: &'static str,

    /// Approximation function parameters
    pub params: Params,

    /// Relative rank to compare complexities
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
        Err(Error::MissingFunctionCoeffsError)
    }
}

/// Computes values of `f(x)` given `x`
#[allow(dead_code)]
fn compute_f(name: Name, params: Params, x: Vec<f64>) -> Result<Vec<f64>, Error> {
    let f = get_function(name, params)?;
    let y = x.into_iter().map(f).collect();
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

    pub fn build(&self) -> Complexity {
        let mut params = Params::new().build();
        if let Some(p) = &self.params {
            params = p.clone();
        }
        let rank = rank(self.name, params.clone());
        Complexity {
            name: self.name,
            notation: name::notation(self.name),
            params,
            rank,
        }
    }
}

/// Transforms input data into linear complexity.
fn linearize(name: Name, x: f64, y: f64) -> (f64, f64) {
    match name {
        Name::Constant => (0.0, y),
        Name::Logarithmic => (x.ln(), y),
        Name::Linear => (x, y),
        Name::Linearithmic => (x * x.ln(), y),
        Name::Quadratic => (x.powi(2), y),
        Name::Cubic => (x.powi(3), y),
        Name::Polynomial => (x.ln(), y.ln()),
        Name::Exponential => (x, y.ln()),
    }
}

/// Converts linear coeffs `gain` and `offset` to corresponding complexity params.
fn delinearize(name: Name, gain: f64, offset: f64) -> Params {
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

fn calculate_residuals(name: Name, params: Params, data: Vec<(f64, f64)>) -> Result<f64, Error> {
    let f = get_function(name, params)?;
    let residuals = data.into_iter().map(|(x, y)| (y - f(x)).abs()).sum();

    Ok(residuals)
}

fn rank(name: Name, params: Params) -> u32 {
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
        Name::Constant => 0,
        Name::Logarithmic => 130,
        Name::Linear => 1_000,
        Name::Linearithmic => 1_130,
        Name::Quadratic => 2_000,
        Name::Cubic => 3_000,
        Name::Polynomial => match params.power {
            Some(power) => std::cmp::min((1_000.0 * power) as u32, 1_000_000),
            None => panic!("Polynomial is missing its power parameter"),
        },
        Name::Exponential => 1_000_000,
    }
}

/// Fits a function of given complexity into input data.
pub fn fit(name: Name, data: Vec<(f64, f64)>) -> Result<Complexity, Error> {
    let linearized = data
        .clone()
        .into_iter()
        .map(|(x, y)| linearize(name, x, y))
        .collect();

    let (gain, offset, _residuals) = linalg::fit_line(linearized)?;
    let params = delinearize(name, gain, offset);
    // Calculate delinearized residuals.
    let residuals = calculate_residuals(name, params.clone(), data)?;
    let params = Params {
        residuals: Some(residuals),
        ..params
    };
    let rank = rank(name, params.clone());

    Ok(Complexity {
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
    Ok(crate::complexity::ComplexityBuilder::new(name).build())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant() -> Complexity {
        ComplexityBuilder::new(Name::Constant).build()
    }

    fn logarithmic() -> Complexity {
        ComplexityBuilder::new(Name::Logarithmic).build()
    }

    fn linear() -> Complexity {
        ComplexityBuilder::new(Name::Linear).build()
    }

    fn linearithmic() -> Complexity {
        ComplexityBuilder::new(Name::Linearithmic).build()
    }

    fn quadratic() -> Complexity {
        ComplexityBuilder::new(Name::Quadratic).build()
    }

    fn cubic() -> Complexity {
        ComplexityBuilder::new(Name::Cubic).build()
    }

    fn exponential() -> Complexity {
        ComplexityBuilder::new(Name::Exponential).build()
    }

    fn polynomial(power: f64) -> Complexity {
        ComplexityBuilder::new(Name::Polynomial)
            .power(power)
            .build()
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
}

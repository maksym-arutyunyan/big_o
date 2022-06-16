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
}

impl Complexity {
    /// Returns a calculated approximation function `f(x)`
    pub fn get_function(&self) -> Result<Box<dyn Fn(f64) -> f64>, &'static str> {
        let p = &self.params;
        if let (Some(a), Some(b)) = match self.name {
            Name::Polynomial => (p.gain, p.power),
            Name::Exponential => (p.gain, p.base),
            _other => (p.gain, p.offset),
        } {
            let f: Box<dyn Fn(f64) -> f64> = match self.name {
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
            Err("No cofficients to compute f(x)")
        }
    }

    /// Computes values of `f(x)` given `x`
    pub fn compute_f(&self, x: Vec<f64>) -> Result<Vec<f64>, &'static str> {
        let f = self.get_function()?;
        let y = x.into_iter().map(f).collect();
        Ok(y)
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
fn delinearize(name: Name, gain: f64, offset: f64, residuals: f64) -> Params {
    // Delinearize coeffs.
    let (a, b) = match name {
        Name::Polynomial => (offset.exp(), gain),
        Name::Exponential => (offset.exp(), gain.exp()),
        _other => (gain, offset),
    };
    // Convert coeffs to params.
    match name {
        Name::Polynomial => Params::new().gain(a).power(b).residuals(residuals).build(),
        Name::Exponential => Params::new().gain(a).base(b).residuals(residuals).build(),
        _other => Params::new().gain(a).offset(b).residuals(residuals).build(),
    }
}

/// Fits a function of given complexity into input data.
pub fn fit(name: Name, data: Vec<(f64, f64)>) -> Result<Complexity, &'static str> {
    let linearized = data
        .into_iter()
        .map(|(x, y)| linearize(name, x, y))
        .collect();

    let (gain, offset, residuals) = linalg::fit_line(linearized)?;

    Ok(Complexity {
        name,
        notation: name::notation(name),
        params: delinearize(name, gain, offset, residuals),
    })
}

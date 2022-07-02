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

pub struct ComplexityBuilder {
    name: Name,
    params: Option<Params>,
}

impl ComplexityBuilder {
    pub fn new(name: Name) -> Self {
        Self { name, params: None }
    }

    pub fn power(&mut self, x: f64) -> &mut Self {
        self.params = Some(Params::new().power(x).build());
        self
    }

    pub fn build(&self) -> Complexity {
        let mut params = Params::new().build();
        if let Some(p) = &self.params {
            params = p.clone();
        }
        Complexity {
            name: self.name,
            notation: name::notation(self.name),
            params,
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

fn rank(complexity: Complexity) -> u32 {
    match complexity.name {
        Name::Constant => 0,
        Name::Logarithmic => 500,
        Name::Linear => 1_000,
        Name::Linearithmic => 1_500,
        Name::Quadratic => 2_000,
        Name::Cubic => 3_000,
        Name::Polynomial => {
            match complexity.params.power {
                Some(power) => {
                    if power < 1.0 {
                        500
                    } else if power < 2.0 {
                        1_500
                    } else if power < 3.0 {
                        2_500
                    } else {
                        10_000
                    }
                }
                None => 0, // TODO: fix error
            }
        }
        Name::Exponential => 1_000_000,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_obvious_cases() {
        let constant = ComplexityBuilder::new(Name::Constant).build();
        let logarithmic = ComplexityBuilder::new(Name::Logarithmic).build();
        let linear = ComplexityBuilder::new(Name::Linear).build();
        let linearithmic = ComplexityBuilder::new(Name::Linearithmic).build();
        let quadratic = ComplexityBuilder::new(Name::Quadratic).build();
        let cubic = ComplexityBuilder::new(Name::Cubic).build();
        let exponential = ComplexityBuilder::new(Name::Exponential).build();

        assert!(rank(constant) < rank(logarithmic.clone()));
        assert!(rank(logarithmic) < rank(linear.clone()));
        assert!(rank(linear) < rank(linearithmic.clone()));
        assert!(rank(linearithmic) < rank(quadratic.clone()));
        assert!(rank(quadratic) < rank(cubic.clone()));
        assert!(rank(cubic) < rank(exponential.clone()));
    }

    #[test]
    fn rank_polynomial() {
        let constant = ComplexityBuilder::new(Name::Constant).build();
        let linear = ComplexityBuilder::new(Name::Linear).build();
        let quadratic = ComplexityBuilder::new(Name::Quadratic).build();
        let cubic = ComplexityBuilder::new(Name::Cubic).build();
        let exponential = ComplexityBuilder::new(Name::Exponential).build();

        let p05 = ComplexityBuilder::new(Name::Polynomial).power(0.5).build();
        let p15 = ComplexityBuilder::new(Name::Polynomial).power(1.5).build();
        let p25 = ComplexityBuilder::new(Name::Polynomial).power(2.5).build();
        let p35 = ComplexityBuilder::new(Name::Polynomial).power(3.5).build();

        // O(1) < O(n^0.5) < O(n)
        assert!(rank(constant) < rank(p05.clone()));
        assert!(rank(p05) < rank(linear.clone()));

        // O(n) < O(n^1.5) < O(n^2)
        assert!(rank(linear) < rank(p15.clone()));
        assert!(rank(p15) < rank(quadratic.clone()));

        // O(n^2) < O(n^2.5) < O(n^3)
        assert!(rank(quadratic) < rank(p25.clone()));
        assert!(rank(p25) < rank(cubic.clone()));

        // O(n^3) < O(n^3.5) < O(c^n)
        assert!(rank(cubic) < rank(p35.clone()));
        assert!(rank(p35) < rank(exponential.clone()));
    }

}

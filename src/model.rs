use crate::error::Error;
use std::fmt;
use std::str::FromStr;

/// A class of asymptotic computational complexity.
///
/// A `Model` names a shape, not a fitted curve: `Polynomial` stands for
/// `O(n^m)` with the exponent left free, and the exponent only exists once the
/// model has been fitted to data. See [`Fit`](crate::Fit) for the result of
/// that fitting.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Model {
    /// `O(1)`
    Constant,
    /// `O(log n)`
    Logarithmic,
    /// `O(n)`
    Linear,
    /// `O(n log n)`
    Linearithmic,
    /// `O(n^2)`
    Quadratic,
    /// `O(n^3)`
    Cubic,
    /// `O(n^m)`, exponent fitted from the data.
    Polynomial,
    /// `O(c^n)`, base fitted from the data.
    Exponential,
}

/// Every model the crate can fit, in ascending order of growth.
pub(crate) const ALL: [Model; 8] = [
    Model::Constant,
    Model::Logarithmic,
    Model::Linear,
    Model::Linearithmic,
    Model::Quadratic,
    Model::Cubic,
    Model::Polynomial,
    Model::Exponential,
];

/// Where a logarithm sits on the polynomial-degree scale.
///
/// `log n` is not a power of `n`, but over the input sizes benchmarks actually
/// reach it behaves like a very small one: across six decades, `log n` grows by
/// the same factor as `n^0.13`. That is what makes `O(log n)` and `O(n^0.5)`
/// comparable at all, and the constant is only ever used for ordering.
const LOG_DEGREE: f64 = 0.13;

impl Model {
    /// The notation this model is written in, with any fitted parameter left
    /// symbolic. [`Fit`](crate::Fit)'s `Display` substitutes the fitted value.
    pub fn notation(&self) -> &'static str {
        match self {
            Model::Constant => "O(1)",
            Model::Logarithmic => "O(log n)",
            Model::Linear => "O(n)",
            Model::Linearithmic => "O(n log n)",
            Model::Quadratic => "O(n^2)",
            Model::Cubic => "O(n^3)",
            Model::Polynomial => "O(n^m)",
            Model::Exponential => "O(c^n)",
        }
    }

    /// Whether this model's shape has a parameter fitted from the data, which
    /// is what lets one model class contain several growth rates.
    pub(crate) fn has_free_exponent(&self) -> bool {
        matches!(self, Model::Polynomial)
    }

    /// The largest growth rate this model class admits, on the degree scale.
    ///
    /// Named models pin a single degree. `Polynomial` admits any finite one,
    /// which is what makes "is this at most polynomial?" a real question with
    /// `Exponential` as the only answer of no.
    pub(crate) fn upper_degree(&self) -> f64 {
        match self {
            Model::Polynomial => f64::MAX,
            other => other.nominal_degree(),
        }
    }

    /// The smallest growth rate this model class admits, on the degree scale.
    pub(crate) fn lower_degree(&self) -> f64 {
        match self {
            // A constant is the slowest-growing polynomial, so nothing is
            // strictly faster than "some polynomial".
            Model::Polynomial => 0.0,
            other => other.nominal_degree(),
        }
    }

    /// This model's degree when it has exactly one.
    fn nominal_degree(&self) -> f64 {
        match self {
            Model::Constant => 0.0,
            Model::Logarithmic => LOG_DEGREE,
            Model::Linear => 1.0,
            Model::Linearithmic => 1.0 + LOG_DEGREE,
            Model::Quadratic => 2.0,
            Model::Cubic => 3.0,
            // Never reached for these two, which have no single degree; the
            // values keep the function total and correctly ordered anyway.
            Model::Polynomial => 1.0,
            Model::Exponential => f64::INFINITY,
        }
    }
}

impl From<Model> for &str {
    fn from(model: Model) -> &'static str {
        model.notation()
    }
}

impl TryFrom<&str> for Model {
    type Error = Error;

    fn try_from(string: &str) -> Result<Self, Self::Error> {
        match &string.to_lowercase()[..] {
            "o(1)" | "constant" => Ok(Model::Constant),
            "o(log n)" | "logarithmic" => Ok(Model::Logarithmic),
            "o(n)" | "linear" => Ok(Model::Linear),
            "o(n log n)" | "linearithmic" => Ok(Model::Linearithmic),
            "o(n^2)" | "quadratic" => Ok(Model::Quadratic),
            "o(n^3)" | "cubic" => Ok(Model::Cubic),
            "o(n^m)" | "polynomial" => Ok(Model::Polynomial),
            "o(c^n)" | "exponential" => Ok(Model::Exponential),
            _ => Err(Error::ParseNotation),
        }
    }
}

impl FromStr for Model {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.try_into()
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.notation())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOTATION_TEST_CASES: [(&str, Model); 8] = [
        ("O(1)", Model::Constant),
        ("O(log n)", Model::Logarithmic),
        ("O(n)", Model::Linear),
        ("O(n log n)", Model::Linearithmic),
        ("O(n^2)", Model::Quadratic),
        ("O(n^3)", Model::Cubic),
        ("O(n^m)", Model::Polynomial),
        ("O(c^n)", Model::Exponential),
    ];

    const NAMED_TEST_CASES: [(&str, Model); 8] = [
        ("Constant", Model::Constant),
        ("Logarithmic", Model::Logarithmic),
        ("Linear", Model::Linear),
        ("Linearithmic", Model::Linearithmic),
        ("Quadratic", Model::Quadratic),
        ("Cubic", Model::Cubic),
        ("Polynomial", Model::Polynomial),
        ("Exponential", Model::Exponential),
    ];

    #[test]
    fn model_into_string() {
        for (string, model) in NOTATION_TEST_CASES {
            let converted: &str = model.into();
            assert_eq!(converted, string);
        }
    }

    #[test]
    fn model_to_string() {
        for (string, model) in NOTATION_TEST_CASES {
            assert_eq!(model.to_string(), string);
        }
    }

    #[test]
    fn parses_notation_and_name() {
        for (string, model) in [NOTATION_TEST_CASES, NAMED_TEST_CASES].concat() {
            assert_eq!(Model::try_from(string), Ok(model));
            assert_eq!(string.parse::<Model>(), Ok(model));
            let into: Model = string.try_into().expect("a known notation");
            assert_eq!(into, model);
        }
    }

    #[test]
    fn rejects_unknown_notation() {
        assert_eq!(
            Model::try_from("irrelevant text"),
            Err(Error::ParseNotation)
        );
        assert_eq!(
            "irrelevant text".parse::<Model>(),
            Err(Error::ParseNotation)
        );
    }

    #[test]
    fn named_models_are_ordered_by_growth() {
        let degrees: Vec<f64> = ALL
            .iter()
            .filter(|m| !m.has_free_exponent())
            .map(|m| m.nominal_degree())
            .collect();

        assert!(
            degrees.windows(2).all(|pair| pair[0] < pair[1]),
            "ALL should list models in ascending order of growth, got {degrees:?}"
        );
    }

    #[test]
    fn polynomial_spans_every_finite_degree() {
        assert!(Model::Cubic.upper_degree() < Model::Polynomial.upper_degree());
        assert!(Model::Polynomial.upper_degree() < Model::Exponential.upper_degree());
        assert_eq!(
            Model::Polynomial.lower_degree(),
            Model::Constant.lower_degree()
        );
    }
}

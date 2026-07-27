use std::fmt;

/// Reasons a complexity could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Returned when the sample holds too few distinct input sizes to tell the
    /// models apart. Repeated measurements of one input size collapse to a
    /// single point, so they count once.
    NotEnoughData {
        /// Distinct input sizes required.
        needed: usize,
        /// Distinct input sizes supplied.
        got: usize,
    },
    /// Returned when a string cannot be parsed into a [`Name`](crate::name::Name).
    ParseNotation,
    /// Returned when not all coefficients for the approximation function are provided.
    MissingFunctionCoeffs,
    /// Returned when a polynomial complexity lacks a power parameter.
    MissingPolynomialPower,
    /// Returned when no complexity model fits the input data.
    NoValidComplexity,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::NotEnoughData { needed, got } => write!(
                f,
                "Need at least {needed} distinct input sizes to infer a complexity, got {got}"
            ),
            Error::ParseNotation => write!(f, "Can't convert string to Name"),
            Error::MissingFunctionCoeffs => write!(f, "No coefficients to compute f(x)"),
            Error::MissingPolynomialPower => write!(f, "Polynomial power parameter is missing"),
            Error::NoValidComplexity => write!(f, "No valid complexity could be inferred"),
        }
    }
}

impl std::error::Error for Error {}

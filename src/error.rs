use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// Returned when the underlying least-squares solver fails.
    LSTSQError(String),
    /// Returned when a string cannot be parsed into a [`Name`](crate::name::Name).
    ParseNotationError,
    /// Returned when not all coefficients for the approximation function are provided.
    MissingFunctionCoeffsError,
    /// Returned when a polynomial complexity lacks a power parameter.
    MissingPolynomialPower,
    /// Returned when input data contains invalid values.
    InvalidInput(String),
    /// Returned when no complexity model fits the input data.
    NoValidComplexity,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::LSTSQError(msg) => write!(f, "LSTSQ failed: {msg}"),
            Error::ParseNotationError => write!(f, "Can't convert string to Name"),
            Error::MissingFunctionCoeffsError => write!(f, "No coefficients to compute f(x)"),
            Error::MissingPolynomialPower => write!(f, "Polynomial power parameter is missing"),
            Error::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            Error::NoValidComplexity => write!(f, "No valid complexity could be inferred"),
        }
    }
}

impl std::error::Error for Error {}

use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// Returned when the underlying least-squares solver fails.
    LSTSQError(String),
    /// Returned when a string cannot be parsed into a [`Name`](crate::name::Name).
    ParseNotationError,
    /// Returned when not all coefficients for the approximation function are provided.
    MissingFunctionCoeffsError,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::LSTSQError(msg) => write!(f, "LSTSQ failed: {msg}"),
            Error::ParseNotationError => write!(f, "Can't convert string to Name"),
            Error::MissingFunctionCoeffsError => write!(f, "No coefficients to compute f(x)"),
        }
    }
}

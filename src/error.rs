use std::fmt;

#[derive(Debug)]
pub enum Error {
    LSTSQError(String),
    ParseNotationError,
    MissingFunctionCoeffsError,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::LSTSQError(msg) => write!(f, "LSTQS failed: {msg}"),
            Error::ParseNotationError => write!(f, "Can't convert string to Name"),
            Error::MissingFunctionCoeffsError => write!(f, "No cofficients to compute f(x)"),
        }
    }
}

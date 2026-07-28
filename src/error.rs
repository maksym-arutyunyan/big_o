use std::fmt;

/// Reasons a complexity could not be produced.
///
/// Non-exhaustive: match with a `_` arm, because a later release may fail for a
/// reason this one has no name for.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Returned when a string cannot be parsed into a [`Model`](crate::Model).
    ParseNotation,
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
            Error::ParseNotation => write!(f, "Can't convert string to a complexity model"),
            Error::NoValidComplexity => write!(f, "No valid complexity could be inferred"),
        }
    }
}

impl std::error::Error for Error {}

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![warn(missing_docs)]

mod analysis;
mod data;
mod error;
mod fit;
mod linalg;
mod model;
mod warning;

pub use crate::analysis::{Analysis, Inference};
pub use crate::error::Error;
pub use crate::fit::{Fit, ModelParams};
pub use crate::model::Model;
pub use crate::warning::Warning;

/// Infers the asymptotic complexity of measured `(input size, cost)` pairs.
///
/// Repeated measurements of the same input size are collapsed to their median
/// before fitting, so an occasional descheduled run does not steer the result.
/// Non-finite measurements are dropped. A model that cannot describe what
/// remains — a logarithmic one given an input size of zero, say — is skipped
/// and listed in [`Inference::warnings`], and the models that can still compete.
///
/// Use [`Analysis`] to infer over a restricted set of models.
///
/// # Errors
/// Returns [`Error::NotEnoughData`] if fewer than three distinct input sizes
/// survive that preparation, and [`Error::NoValidComplexity`] if no model
/// describes what does.
///
/// # Example
/// ```
/// // Cost measured over growing inputs, with a few percent of timing noise.
/// let data = [
///     (100., 10_180.),
///     (200., 39_800.),
///     (400., 161_440.),
///     (800., 637_440.),
///     (1600., 2_570_240.),
///     (3200., 10_352_640.),
///     (6400., 40_673_280.),
///     (12800., 164_167_680.),
/// ];
///
/// let inference = big_o::infer_complexity(&data).unwrap();
///
/// assert_eq!(inference.best.model, big_o::Model::Quadratic);
/// assert_eq!(inference.best.to_string(), "O(n^2)");
/// assert!(inference.best.is_at_most(big_o::Model::Quadratic));
/// assert!(inference.confidence > 0.9);
/// ```
pub fn infer_complexity(data: &[(f64, f64)]) -> Result<Inference, Error> {
    Analysis::new().infer(data)
}

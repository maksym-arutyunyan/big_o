#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

mod complexity;
mod data;
mod error;
mod linalg;
mod name;
mod params;
mod validate;

pub use crate::complexity::complexity;
pub use crate::complexity::Complexity;
pub use crate::error::Error;
pub use crate::name::Name;
pub use crate::params::Params;

/// Infers complexity of given data points, returns the best and all the fitted complexities.
///
/// Repeated measurements of the same input size are collapsed to their median
/// before fitting, so an occasional descheduled run does not steer the result.
/// Non-finite measurements are dropped. A model that cannot describe the
/// remaining data — a logarithmic one given `x = 0`, say — is skipped, and the
/// models that can still compete.
///
/// # Errors
/// Returns [`Error::NotEnoughData`] if fewer than three distinct input sizes
/// survive that preparation, and [`Error::NoValidComplexity`] if no model fits
/// what does.
///
/// # Example
/// ```
/// # use assert_approx_eq::assert_approx_eq;
/// // f(x) = gain * x ^ 2 + offset
/// let data = vec![(1., 1.), (2., 4.), (3., 9.), (4., 16.)];
///
/// let (complexity, _all) = big_o::infer_complexity(&data).unwrap();
///
/// assert_eq!(complexity.name, big_o::Name::Quadratic);
/// assert_eq!(complexity.notation, "O(n^2)");
/// assert_approx_eq!(complexity.params.gain.unwrap(), 1.0, 1e-6);
/// assert_approx_eq!(complexity.params.offset.unwrap(), 0.0, 1e-6);
/// assert!(complexity.rank < big_o::complexity("O(n^3)").unwrap().rank);
/// ```
pub fn infer_complexity(data: &[(f64, f64)]) -> Result<(Complexity, Vec<Complexity>), Error> {
    let sample = data::prepare(data);
    if sample.points().len() < data::MIN_POINTS {
        return Err(Error::NotEnoughData {
            needed: data::MIN_POINTS,
            got: sample.points().len(),
        });
    }

    let mut all_fitted: Vec<Complexity> = name::all_names()
        .into_iter()
        .filter_map(|name| complexity::fit(name, &sample))
        .filter(validate::is_valid)
        .collect();

    all_fitted.sort_by(|a, b| {
        let (a, b) = (a.params.residuals, b.params.residuals);
        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let best_complexity = all_fitted
        .first()
        .cloned()
        .ok_or(Error::NoValidComplexity)?;

    Ok((best_complexity, all_fitted))
}

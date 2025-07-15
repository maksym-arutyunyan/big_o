#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

mod complexity;
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
    if data.is_empty() || data.iter().all(|(x, y)| *x == 0.0 && *y == 0.0) {
        return Err(Error::NoValidComplexity);
    }
    let mut all_fitted: Vec<Complexity> = Vec::new();
    for name in name::all_names() {
        let complexity = complexity::fit(name, data)?;
        if validate::is_valid(&complexity) {
            all_fitted.push(complexity);
        }
    }
    if all_fitted.is_empty() {
        return Err(Error::NoValidComplexity);
    }
    all_fitted.sort_by(|a, b| a.params.residuals.partial_cmp(&b.params.residuals).unwrap());
    let best_complexity = all_fitted[0].clone();

    Ok((best_complexity, all_fitted))
}

//! Infers asymptotic computational complexity.
//!
//! `big_o` helps to estimate computational complexity of algorithms by inspecting measurement data
//! (eg. execution time, memory consumption, etc). Users are expected to provide measurement data,
//! `big_o` will try to fit a set of complexity models and return the best fit.
//!
//! # Example
//! ```
//! // f(x) = gain * x ^ 2 + offset
//! let data = vec![(1., 1.), (2., 4.), (3., 9.), (4., 16.)];
//!
//! let (complexity, _all) = big_o::infer_complexity(data).unwrap();
//!
//! assert_eq!(complexity.name, big_o::Name::Quadratic);
//! assert_eq!(complexity.notation, "O(n^2)");
//! ```

mod complexity;
mod linalg;
mod name;
mod params;
mod validate;

pub use crate::complexity::Complexity;
pub use crate::name::Name;
pub use crate::params::Params;

/// Infers complexity of given data points, returns the best and all the fitted complexities.
///
/// # Example
/// ```
/// // f(x) = gain * x ^ 2 + offset
/// let data = vec![(1., 1.), (2., 4.), (3., 9.), (4., 16.)];
///
/// let (complexity, _all) = big_o::infer_complexity(data).unwrap();
///
/// assert_eq!(complexity.name, big_o::Name::Quadratic);
/// assert_eq!(complexity.notation, "O(n^2)");
/// ```
pub fn infer_complexity(
    data: Vec<(f64, f64)>,
) -> Result<(Complexity, Vec<Complexity>), &'static str> {
    let mut all_fitted: Vec<Complexity> = Vec::new();
    for name in name::all_names() {
        let complexity = complexity::fit(name, data.clone())?;
        if validate::is_valid(&complexity) {
            all_fitted.push(complexity);
        }
    }
    all_fitted.sort_by(|a, b| a.params.residuals.partial_cmp(&b.params.residuals).unwrap());
    let best_complexity = all_fitted[0].clone();

    Ok((best_complexity, all_fitted))
}

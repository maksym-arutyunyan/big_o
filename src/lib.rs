//! Infers asymptotic computational complexity.
//!
//! `big_o` helps to estimate computational complexity of algorithms by inspecting measurement data
//! (eg. execution time, memory consumption, etc). Users are expected to provide measurement data,
//! `big_o` will try to fit a set of complexity models and return the best fit.

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
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
///
/// assert_eq!(best.name, big_o::Name::Quadratic);
/// assert_eq!(best.notation, "O(n^2)");
/// assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
/// assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 0.0, 1e-6);
/// ```
pub fn infer_complexity(
    data: Vec<(f64, f64)>,
) -> Result<(Complexity, Vec<Complexity>), &'static str> {
    let mut fitted: Vec<Complexity> = Vec::new();
    for name in name::all_names() {
        let complexity = complexity::fit(name, data.clone())?;
        if validate::is_valid(&complexity) {
            fitted.push(complexity);
        }
    }
    fitted.sort_by(|a, b| a.params.residuals.partial_cmp(&b.params.residuals).unwrap());
    let best = fitted[0].clone();

    Ok((best, fitted))
}

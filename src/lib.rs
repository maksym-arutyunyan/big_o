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
/// # Examples
/// ```
/// // f(x) = offset
/// let data = vec![(1., 11.), (2., 11.), (3., 11.), (4., 11.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Constant);
/// assert_eq!(best.notation, "O(1)");
/// assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 0.0, 1e-6);
/// assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 11.0, 1e-6);
///
/// // f(x) = gain * log(x) + offset
/// let data = vec![(10., 1.), (100., 2.), (1_000., 3.), (10_000., 4.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Logarithmic);
/// assert_eq!(best.notation, "O(log n)");
///
/// // f(x) = gain * x + offset
/// let data = vec![(1., 17.), (2., 27.), (3., 37.), (4., 47.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Linear);
/// assert_eq!(best.notation, "O(n)");
/// assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 10.0, 1e-6);
/// assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 7.0, 1e-6);
///
/// // f(x) = gain * x * log(x) + offset
/// let data = vec![(10., 10.), (100., 200.), (1_000., 3_000.), (10_000., 40_000.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Linearithmic);
/// assert_eq!(best.notation, "O(n log n)");
///
/// // f(x) = gain * x ^ 2 + offset
/// let data = vec![(1., 1.), (2., 4.), (3., 9.), (4., 16.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Quadratic);
/// assert_eq!(best.notation, "O(n^2)");
/// assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
/// assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 0.0, 1e-6);
///
/// // f(x) = gain * x ^ 3 + offset
/// let data = vec![(1., 1.), (2., 8.), (3., 27.), (4., 64.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Cubic);
/// assert_eq!(best.notation, "O(n^3)");
/// assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
/// assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 0.0, 1e-6);
///
/// // f(x) = gain * x ^ power
/// let data = vec![(1., 1.), (2., 16.), (3., 81.), (4., 256.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Polynomial);
/// assert_eq!(best.notation, "O(n^m)");
/// assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
/// assert_approx_eq::assert_approx_eq!(best.params.power.unwrap(), 4.0, 1e-6);
///
/// // f(x) = gain * base ^ x
/// let data = vec![(1., 2.), (2., 4.), (3., 8.), (4., 16.)];
/// let (best, _all) = big_o::infer_complexity(data).unwrap();
/// assert_eq!(best.name, big_o::Name::Exponential);
/// assert_eq!(best.notation, "O(c^n)");
/// assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
/// assert_approx_eq::assert_approx_eq!(best.params.base.unwrap(), 2.0, 1e-6);
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

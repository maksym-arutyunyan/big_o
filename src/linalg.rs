//! Closed-form ordinary least squares for a single predictor.
//!
//! Simple linear regression has an exact solution in terms of the centered
//! second moments of the sample, so fitting needs no iterative solver and no
//! matrix decomposition. That matters here beyond speed: a decomposition can
//! fail at runtime on degenerate input, whereas the closed form has exactly one
//! degenerate case — no spread in `x` — which is reported rather than raised.

/// Coefficients of the line `f(x) = gain * x + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Line {
    /// Slope of the fitted line.
    pub gain: f64,

    /// Value of the fitted line at `x = 0`.
    pub offset: f64,
}

/// Fits `f(x) = gain * x + offset` to weighted `(x, y, weight)` points by least
/// squares.
///
/// A weight is how much that point's error counts. Weighting every point alike
/// minimizes the absolute error, which lets the largest measurements decide the
/// whole fit; weighting by `1 / y^2` minimizes the *relative* error, which is
/// what data carrying proportional noise calls for.
///
/// Returns `None` when the coefficients are not determined by the sample:
/// fewer than two points, a non-finite value, no weighted spread in `x`, or an
/// intermediate sum that overflows to infinity.
pub(crate) fn fit_line(data: &[(f64, f64, f64)]) -> Option<Line> {
    if data.len() < 2 {
        return None;
    }

    let mut total = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &(x, y, w) in data {
        if !x.is_finite() || !y.is_finite() || !w.is_finite() || w < 0.0 {
            return None;
        }
        total += w;
        sum_x += w * x;
        sum_y += w * y;
    }
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mean_x = sum_x / total;
    let mean_y = sum_y / total;
    if !mean_x.is_finite() || !mean_y.is_finite() {
        return None;
    }

    // Centering before accumulating keeps the second moments well scaled: the
    // uncentered form subtracts two large, nearly equal quantities and loses
    // most of its significant digits on data spanning several decades.
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for &(x, y, w) in data {
        let dx = x - mean_x;
        sxx += w * dx * dx;
        sxy += w * dx * (y - mean_y);
    }
    if !sxx.is_finite() || sxx <= 0.0 || !sxy.is_finite() {
        return None;
    }

    let gain = sxy / sxx;
    let offset = mean_y - gain * mean_x;
    if !gain.is_finite() || !offset.is_finite() {
        return None;
    }
    Some(Line { gain, offset })
}

/// Returns the weighted mean of `(value, weight)` pairs, or `None` if there is
/// no weight to average over.
pub(crate) fn weighted_mean(values: impl IntoIterator<Item = (f64, f64)>) -> Option<f64> {
    let mut sum = 0.0;
    let mut total = 0.0;
    for (value, weight) in values {
        if !value.is_finite() || !weight.is_finite() || weight < 0.0 {
            return None;
        }
        sum += value * weight;
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mean = sum / total;
    mean.is_finite().then_some(mean)
}

/// Returns the mean of `values`, or `None` if it is empty or non-finite.
pub(crate) fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if !value.is_finite() {
            return None;
        }
        sum += value;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let mean = sum / count as f64;
    mean.is_finite().then_some(mean)
}

/// Returns the median of `values`, consuming and reordering the slice.
///
/// For an even count this averages the two central elements. Returns `None` for
/// an empty slice; non-finite values are assumed to have been filtered out by
/// the caller and sort to the end.
pub(crate) fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    let median = match values.len() % 2 {
        0 => (values[mid - 1] + values[mid]) / 2.0,
        _ => values[mid],
    };
    median.is_finite().then_some(median)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    const EPSILON: f64 = 1e-12;

    /// Fits with every point counting the same.
    fn fitted(data: &[(f64, f64)]) -> Line {
        let weighted: Vec<(f64, f64, f64)> = data.iter().map(|&(x, y)| (x, y, 1.0)).collect();
        fit_line(&weighted).expect("line is determined by this data")
    }

    fn unweighted(data: &[(f64, f64)]) -> Option<Line> {
        let weighted: Vec<(f64, f64, f64)> = data.iter().map(|&(x, y)| (x, y, 1.0)).collect();
        fit_line(&weighted)
    }

    #[test]
    fn fits_a_line_through_the_origin() {
        let line = fitted(&[(0., 0.), (1., 1.), (2., 2.), (3., 3.)]);

        assert_approx_eq!(line.gain, 1., EPSILON);
        assert_approx_eq!(line.offset, 0., EPSILON);
    }

    #[test]
    fn fits_a_line_with_an_offset() {
        let line = fitted(&[(0., 7.), (1., 17.), (2., 27.), (3., 37.)]);

        assert_approx_eq!(line.gain, 10., EPSILON);
        assert_approx_eq!(line.offset, 7., EPSILON);
    }

    #[test]
    fn fits_a_line_through_noisy_data() {
        // Symmetric residuals about y = 2x + 1, so the fit recovers it exactly.
        let line = fitted(&[(1., 4.), (1., 2.), (2., 6.), (2., 4.), (3., 8.), (3., 6.)]);

        assert_approx_eq!(line.gain, 2., 1e-9);
        assert_approx_eq!(line.offset, 1., 1e-9);
    }

    #[test]
    fn stays_accurate_across_many_decades() {
        // Where the uncentered normal equations lose their significant digits.
        let data: Vec<(f64, f64)> = (0..6)
            .map(|k| {
                let x = 10f64.powi(k);
                (x, 3.0 * x + 5.0)
            })
            .collect();

        let line = fitted(&data);

        assert_approx_eq!(line.gain, 3., 1e-9);
        assert_approx_eq!(line.offset, 5., 1e-6);
    }

    #[test]
    fn reports_undetermined_fits_instead_of_failing() {
        assert_eq!(unweighted(&[]), None);
        assert_eq!(unweighted(&[(1., 1.)]), None, "a single point");
        assert_eq!(unweighted(&[(1., 1.), (1., 2.)]), None, "no spread in x");
        assert_eq!(unweighted(&[(1., 1.), (2., f64::NAN)]), None, "NaN");
        assert_eq!(
            unweighted(&[(1., 1.), (f64::INFINITY, 2.)]),
            None,
            "infinity"
        );
        assert_eq!(fit_line(&[(1., 1., 0.0), (2., 2., 0.0)]), None, "no weight");
    }

    #[test]
    fn weights_decide_which_points_the_line_follows() {
        // Three points on y = x, and one far outlier at the far end.
        let data = [(1., 1., 1.0), (2., 2., 1.0), (3., 3., 1.0), (4., 40., 1.0)];
        let pulled = fit_line(&data).expect("determined");

        let mut discounted = data;
        discounted[3].2 = 1e-6;
        let ignored = fit_line(&discounted).expect("determined");

        assert!(pulled.gain > 5.0, "the outlier drags an even-weighted fit");
        assert_approx_eq!(ignored.gain, 1., 1e-3);
        assert_approx_eq!(ignored.offset, 0., 1e-3);
    }

    #[test]
    fn weighted_mean_of_values() {
        assert_eq!(weighted_mean([(1., 1.), (3., 1.)]), Some(2.));
        assert_eq!(weighted_mean([(1., 3.), (5., 1.)]), Some(2.));
        assert_eq!(weighted_mean([]), None);
        assert_eq!(weighted_mean([(1., 0.)]), None, "no weight to average over");
    }

    #[test]
    fn mean_of_values() {
        assert_eq!(mean([1., 2., 3.]), Some(2.));
        assert_eq!(mean([]), None);
        assert_eq!(mean([1., f64::NAN]), None);
    }

    #[test]
    fn median_of_values() {
        assert_eq!(median(&mut [3., 1., 2.]), Some(2.), "odd count");
        assert_eq!(median(&mut [4., 1., 3., 2.]), Some(2.5), "even count");
        assert_eq!(median(&mut [7.]), Some(7.));
        assert_eq!(median(&mut []), None);
    }
}

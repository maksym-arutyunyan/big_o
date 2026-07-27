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

/// Fits `f(x) = gain * x + offset` to `data` by ordinary least squares.
///
/// Returns `None` when the coefficients are not determined by the sample:
/// fewer than two points, a non-finite coordinate, no spread in `x`, or an
/// intermediate sum that overflows to infinity.
pub(crate) fn fit_line(data: &[(f64, f64)]) -> Option<Line> {
    if data.len() < 2 {
        return None;
    }
    let n = data.len() as f64;

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &(x, y) in data {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        sum_x += x;
        sum_y += y;
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    if !mean_x.is_finite() || !mean_y.is_finite() {
        return None;
    }

    // Centering before accumulating keeps the second moments well scaled: the
    // uncentered form subtracts two large, nearly equal quantities and loses
    // most of its significant digits on data spanning several decades.
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for &(x, y) in data {
        let dx = x - mean_x;
        sxx += dx * dx;
        sxy += dx * (y - mean_y);
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

    fn fitted(data: &[(f64, f64)]) -> Line {
        fit_line(data).expect("line is determined by this data")
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
        assert_eq!(fit_line(&[]), None);
        assert_eq!(fit_line(&[(1., 1.)]), None, "a single point");
        assert_eq!(fit_line(&[(1., 1.), (1., 2.)]), None, "no spread in x");
        assert_eq!(fit_line(&[(1., 1.), (2., f64::NAN)]), None, "NaN");
        assert_eq!(fit_line(&[(1., 1.), (f64::INFINITY, 2.)]), None, "infinity");
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

use crate::error::Error;

/// Fits a line `f(x) = gain * x + offset` into input `data` points.
///
/// Returns linear coeffs `gain`, `offset` and `residuals`.
pub fn fit_line(data: &[(f64, f64)]) -> Result<(f64, f64, f64), Error> {
    use nalgebra::{Dynamic, OMatrix, OVector, U2};

    let (xs, ys): (Vec<f64>, Vec<f64>) = data.iter().cloned().unzip();
    let mut xs_flat_matrix = Vec::with_capacity(2 * xs.len());
    for x in xs {
        xs_flat_matrix.push(x);
        xs_flat_matrix.push(1.0);
    }
    let a = OMatrix::<f64, Dynamic, U2>::from_row_slice(&xs_flat_matrix);
    let b = OVector::<f64, Dynamic>::from_row_slice(&ys);

    let epsilon = 1e-10;
    let results =
        lstsq::lstsq(&a, &b, epsilon).map_err(|msg| Error::LSTSQError(msg.to_string()))?;

    let gain = results.solution[0];
    let offset = results.solution[1];
    let residuals = results.residuals;

    Ok((gain, offset, residuals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    const EPSILON: f64 = 1e-12;

    #[test]
    fn test_fit_line_1() {
        let data = vec![(0., 0.), (1., 1.), (2., 2.), (3., 3.)];

        let (gain, offset, residuals) = fit_line(&data).unwrap();

        assert_approx_eq!(gain, 1., EPSILON);
        assert_approx_eq!(offset, 0., EPSILON);
        assert_approx_eq!(residuals, 0., EPSILON);
    }

    #[test]
    fn test_fit_line_2() {
        let data = vec![(0., 7.), (1., 17.), (2., 27.), (3., 37.)];

        let (gain, offset, residuals) = fit_line(&data).unwrap();

        assert_approx_eq!(gain, 10., EPSILON);
        assert_approx_eq!(offset, 7., EPSILON);
        assert_approx_eq!(residuals, 0., EPSILON);
    }
}

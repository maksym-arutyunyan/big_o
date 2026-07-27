//! Preparation of raw measurements into a sample that can be fitted.
//!
//! Benchmark output is rarely one `y` per `x`: the same input size is usually
//! measured several times, and a run that was descheduled shows up as a single
//! large outlier. Collapsing each input size to its median absorbs those
//! outliers without discarding the repetition, and it leaves the fitters with
//! the one shape they can rely on — strictly increasing, finite `x`.

use crate::linalg;

/// Smallest sample a fit can be meaningfully inferred from.
///
/// Two points determine a line exactly, so any two-point sample fits every
/// two-parameter model perfectly and carries no evidence about which one is
/// right. Three is the first size at which the models can disagree.
pub(crate) const MIN_POINTS: usize = 3;

/// Measurements grouped by input size, one median `y` per distinct `x`,
/// sorted by ascending `x`.
pub(crate) struct Sample {
    points: Vec<(f64, f64)>,
}

impl Sample {
    /// The prepared points: finite, sorted, one per distinct `x`.
    pub(crate) fn points(&self) -> &[(f64, f64)] {
        &self.points
    }
}

/// Prepares raw measurements for fitting.
///
/// Drops non-finite points, collapses repeated `x` to their median `y`, and
/// sorts by ascending `x`.
pub(crate) fn prepare(data: &[(f64, f64)]) -> Sample {
    let mut finite: Vec<(f64, f64)> = data
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
    finite.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut points: Vec<(f64, f64)> = Vec::with_capacity(finite.len());
    let mut group: Vec<f64> = Vec::new();
    let mut current: Option<f64> = None;
    for (x, y) in finite {
        match current {
            Some(seen) if seen == x => group.push(y),
            _ => {
                if let (Some(seen), Some(median)) = (current, linalg::median(&mut group)) {
                    points.push((seen, median));
                }
                group.clear();
                group.push(y);
                current = Some(x);
            }
        }
    }
    if let (Some(seen), Some(median)) = (current, linalg::median(&mut group)) {
        points.push((seen, median));
    }

    Sample { points }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
        prepare(data).points().to_vec()
    }

    #[test]
    fn sorts_by_input_size() {
        assert_eq!(
            points(&[(3., 30.), (1., 10.), (2., 20.)]),
            vec![(1., 10.), (2., 20.), (3., 30.)]
        );
    }

    #[test]
    fn collapses_repeated_input_sizes_to_their_median() {
        // The 900.0 is a descheduled run; the median ignores it, a mean would not.
        assert_eq!(
            points(&[(1., 10.), (1., 12.), (1., 900.), (2., 20.)]),
            vec![(1., 12.), (2., 20.)]
        );
    }

    #[test]
    fn averages_the_two_central_values_of_an_even_group() {
        assert_eq!(points(&[(1., 10.), (1., 20.)]), vec![(1., 15.)]);
    }

    #[test]
    fn drops_non_finite_measurements() {
        let sample = prepare(&[(1., 10.), (2., f64::NAN), (f64::INFINITY, 30.), (3., 30.)]);

        assert_eq!(sample.points(), &[(1., 10.), (3., 30.)]);
    }

    #[test]
    fn handles_empty_input() {
        let sample = prepare(&[]);

        assert!(sample.points().is_empty());
    }
}

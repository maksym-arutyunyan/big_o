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

    /// Whether cost falls overall as the input grows.
    ///
    /// Compares the means of the first and last third rather than adjacent
    /// points, so ordinary measurement noise does not register as a trend.
    pub(crate) fn is_decreasing(&self) -> bool {
        let third = self.points.len() / 3;
        if third == 0 {
            return false;
        }
        let head = linalg::mean(self.points[..third].iter().map(|&(_, y)| y));
        let tail = linalg::mean(
            self.points[self.points.len() - third..]
                .iter()
                .map(|&(_, y)| y),
        );
        match (head, tail) {
            (Some(head), Some(tail)) => tail < head,
            _ => false,
        }
    }

    /// Whether cost reverses direction more often than noise alone explains.
    ///
    /// Counts sign changes in the successive differences: a monotone curve has
    /// none, and unbiased noise on a monotone curve produces them only where
    /// consecutive steps are small. More than a third of the steps reversing
    /// means no monotone model describes the data.
    pub(crate) fn is_non_monotonic(&self) -> bool {
        let deltas: Vec<f64> = self
            .points
            .windows(2)
            .map(|pair| pair[1].1 - pair[0].1)
            .collect();
        let reversals = deltas
            .windows(2)
            .filter(|pair| pair[0] * pair[1] < 0.0)
            .count();
        !deltas.is_empty() && reversals * 3 > deltas.len()
    }

    /// How many decades of input size the sample spans.
    pub(crate) fn decades(&self) -> f64 {
        let (Some(&(first, _)), Some(&(last, _))) = (self.points.first(), self.points.last())
        else {
            return 0.0;
        };
        if first <= 0.0 || last <= 0.0 {
            return 0.0;
        }
        (last / first).log10().max(0.0)
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

    #[test]
    fn detects_a_falling_cost() {
        let rising = prepare(&[(1., 1.), (2., 2.), (3., 3.), (4., 4.), (5., 5.), (6., 6.)]);
        let falling = prepare(&[(1., 6.), (2., 5.), (3., 4.), (4., 3.), (5., 2.), (6., 1.)]);

        assert!(!rising.is_decreasing());
        assert!(falling.is_decreasing());
    }

    #[test]
    fn tolerates_noise_but_not_a_sawtooth() {
        let noisy = prepare(&[
            (1., 1.),
            (2., 2.1),
            (3., 2.9),
            (4., 4.),
            (5., 5.1),
            (6., 5.9),
        ]);
        let sawtooth = prepare(&[(1., 1.), (2., 9.), (3., 2.), (4., 8.), (5., 3.), (6., 7.)]);

        assert!(!noisy.is_non_monotonic());
        assert!(sawtooth.is_non_monotonic());
    }

    #[test]
    fn measures_span_in_decades() {
        let narrow = prepare(&[(1000., 1.), (1050., 2.), (1100., 3.)]);
        let wide = prepare(&[(1., 1.), (100., 2.), (10000., 3.)]);

        assert!(narrow.decades() < 0.1);
        assert!((wide.decades() - 4.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_sample_reports_nothing() {
        let sample = prepare(&[]);

        assert_eq!(sample.decades(), 0.0);
        assert!(!sample.is_decreasing());
        assert!(!sample.is_non_monotonic());
    }
}

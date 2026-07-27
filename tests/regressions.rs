//! Inputs that once produced a wrong answer, a nonsensical one, or a crash.
//!
//! Each test names what went wrong and what the answer must be now. New cases
//! belong here rather than in a file of their own, so that the whole history of
//! things that have gone wrong stays one file to read.

use big_o::{Error, Fit, Model, ModelParams, Warning};

/// A dataset containing an input size of zero used to abort the process.
///
/// The logarithmic models linearize through `ln(x)`, which has no value at
/// zero; the resulting `-inf` reached the least-squares solver, which raised
/// "Singular value was NaN" from inside its decomposition. Those models are now
/// skipped for such a dataset and the rest still answer.
#[test]
fn an_input_size_of_zero_no_longer_aborts_the_process() {
    let data: Vec<(f64, f64)> = (0..20).map(|n| (n as f64, 3.0 * n as f64 + 1.0)).collect();

    let inference = big_o::infer_complexity(&data).expect("linear data is inferable");

    assert_eq!(inference.best.model, Model::Linear);
    let skipped = inference.warnings.iter().any(
        |w| matches!(w, Warning::ModelsSkipped(models) if models.contains(&Model::Logarithmic)),
    );
    assert!(skipped, "warnings should name the models that were skipped");
}

/// Quadratic data with a few percent of timing noise reported `O(n^2.000382)`.
///
/// The free exponent could bend towards the noise where a fixed one could not,
/// so it always scored fractionally better, and users almost never saw a named
/// complexity.
#[test]
fn noisy_quadratic_data_reports_a_named_complexity() {
    let mut state: u64 = 12345;
    let data: Vec<(f64, f64)> = (1..=100)
        .map(|n| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (state >> 33) as f64 / (1u64 << 31) as f64;
            let x = n as f64;
            // About +/-2% of deterministic noise, no generator dependency.
            (x, x * x * (1.0 + 0.04 * (unit - 0.5)))
        })
        .collect();

    let inference = big_o::infer_complexity(&data).expect("noisy quadratic data is inferable");

    assert_eq!(inference.best.model, Model::Quadratic);
    assert_eq!(inference.best.to_string(), "O(n^2)");
}

/// A single measurement reported `O(n)` with near-zero residuals.
///
/// One point fits every model exactly, so the answer was an artefact of the
/// order the models happened to be tried in.
#[test]
fn a_single_measurement_is_not_enough_to_infer_from() {
    let err = big_o::infer_complexity(&[(1.0, 5.0)]).unwrap_err();

    assert_eq!(err, Error::NotEnoughData { needed: 3, got: 1 });
}

/// Repeated measurements of one input size reported `O(n log n)`.
///
/// They carry no information about growth — every measurement is at the same
/// input size — but the fitters saw a column of points and answered anyway.
#[test]
fn repeated_measurements_of_one_input_size_carry_no_growth() {
    let data: Vec<(f64, f64)> = (0..50).map(|i| (7.0, 100.0 + i as f64)).collect();

    let err = big_o::infer_complexity(&data).unwrap_err();

    assert_eq!(err, Error::NotEnoughData { needed: 3, got: 1 });
}

/// Cost that falls as the input grows fitted `O(n^-1)` and then ranked it as
/// the *fastest* possible complexity.
///
/// The rank was an unsigned integer, so multiplying a negative exponent by a
/// thousand and casting it produced zero — the rank of `O(1)`. Ordering is now
/// signed, and a falling cost is reported as one.
#[test]
fn a_falling_cost_is_reported_rather_than_ranked_as_constant() {
    let data: Vec<(f64, f64)> = (1..=50).map(|n| (n as f64, 100.0 / n as f64)).collect();

    let inference = big_o::infer_complexity(&data).expect("a falling cost is still a fit");

    assert!(
        matches!(inference.best.params, ModelParams::Polynomial { power, .. } if (power + 1.0).abs() < 0.01),
        "expected a polynomial of power about -1, got {:?}",
        inference.best.params
    );
    assert!(inference.warnings.contains(&Warning::DecreasingCost));
    assert!(
        inference.best < constant_fit(),
        "a falling cost must order below O(1), not wrap around above it"
    );
}

/// A fitted `O(1)` to compare orderings against.
fn constant_fit() -> Fit {
    let data: Vec<(f64, f64)> = (1..=10).map(|n| (n as f64, 5.0)).collect();
    big_o::infer_complexity(&data)
        .expect("constant data is inferable")
        .best
}

/// A non-finite measurement must never become a fit.
#[test]
fn a_non_finite_measurement_never_becomes_a_fit() {
    let mut data: Vec<(f64, f64)> = (1..=20).map(|n| (n as f64, 2.0 * n as f64)).collect();
    data.push((21.0, f64::NAN));

    let inference = big_o::infer_complexity(&data).expect("the finite points still fit");

    assert_eq!(inference.best.model, Model::Linear);
    assert!(inference.best.r_squared.is_finite());
    assert!(inference.best.relative_error.is_finite());

    // With nothing finite left there is nothing to infer from.
    let err = big_o::infer_complexity(&[(1.0, f64::NAN), (2.0, f64::NAN)]).unwrap_err();
    assert_eq!(err, Error::NotEnoughData { needed: 3, got: 0 });
}

/// Measurements from a linear workload, reported as linear.
#[test]
fn regression_0001() {
    let data: Vec<(f64, f64)> = (1..=49)
        .map(|n| (n as f64, 45384.6 * n as f64 + 0.4))
        .collect();

    let inference = big_o::infer_complexity(&data).expect("linear data is inferable");

    assert_eq!(
        inference.best.model,
        Model::Linear,
        "best={:?}, all={:?}",
        inference.best,
        inference.all
    );
}

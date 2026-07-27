//! The public surface, exercised the way a caller would use it.

mod synthetic;

use big_o::{Analysis, Error, Fit, Model, ModelParams, Warning};

/// Clean measurements of a known curve, over a range wide enough to identify it.
fn clean(model: Model) -> Vec<(f64, f64)> {
    synthetic::noisy(model, 0, 0.0)
}

fn infer(model: Model) -> Fit {
    big_o::infer_complexity(&clean(model))
        .unwrap_or_else(|e| panic!("{model} data should be inferable: {e}"))
        .best
}

#[test]
fn recovers_every_model_from_its_own_curve() {
    for model in [
        Model::Constant,
        Model::Logarithmic,
        Model::Linear,
        Model::Linearithmic,
        Model::Quadratic,
        Model::Cubic,
        Model::Polynomial,
        Model::Exponential,
    ] {
        assert_eq!(infer(model).model, model);
    }
}

#[test]
fn recovers_the_coefficients_it_was_given() {
    let data = [(1., 5.), (2., 9.), (3., 13.), (4., 17.), (5., 21.)];

    let fit = big_o::infer_complexity(&data).expect("linear data").best;

    match fit.params {
        ModelParams::Linear { gain, offset } => {
            assert!((gain - 4.0).abs() < 1e-9, "gain {gain}");
            assert!((offset - 1.0).abs() < 1e-9, "offset {offset}");
        }
        other => panic!("expected a linear fit, got {other:?}"),
    }
    assert!((fit.r_squared - 1.0).abs() < 1e-9);
    assert!(fit.relative_error < 1e-9);
    assert!((fit.evaluate(6.0) - 25.0).abs() < 1e-9);
}

#[test]
fn ranks_every_model_that_could_be_fitted() {
    let inference = big_o::infer_complexity(&clean(Model::Quadratic)).expect("quadratic data");

    assert!(inference.all.len() > 1, "all candidates should be reported");
    assert!(inference.all.contains(&inference.best));
    let errors: Vec<f64> = inference.all.iter().map(|f| f.relative_error).collect();
    assert!(
        errors.windows(2).all(|pair| pair[0] <= pair[1]),
        "candidates should be ordered best first, got {errors:?}"
    );
}

#[test]
fn narrowing_the_candidates_narrows_the_answer() {
    let data = clean(Model::Quadratic);

    let inference = Analysis::new()
        .models([Model::Linear, Model::Cubic])
        .infer(&data)
        .expect("one of the two still fits best");

    assert!(inference
        .all
        .iter()
        .all(|fit| fit.model != Model::Quadratic));
    assert!([Model::Linear, Model::Cubic].contains(&inference.best.model));
}

#[test]
fn an_empty_candidate_set_infers_nothing() {
    let err = Analysis::new()
        .models([])
        .infer(&clean(Model::Linear))
        .unwrap_err();

    assert_eq!(err, Error::NoValidComplexity);
}

#[test]
fn compares_against_a_named_bound() {
    let linear = infer(Model::Linear);

    assert!(linear.is_at_most(Model::Linear));
    assert!(linear.is_at_most(Model::Quadratic));
    assert!(linear.is_at_most(Model::Polynomial), "n is a polynomial");
    assert!(!linear.is_at_most(Model::Logarithmic));

    assert!(linear.is_faster_than(Model::Quadratic));
    assert!(!linear.is_faster_than(Model::Linear));
    assert!(!linear.is_faster_than(Model::Constant));
}

#[test]
fn an_exponential_is_the_one_thing_that_is_not_polynomial() {
    let exponential = infer(Model::Exponential);

    assert!(!exponential.is_at_most(Model::Polynomial));
    assert!(!exponential.is_at_most(Model::Cubic));
    assert!(exponential.is_at_most(Model::Exponential));
}

/// The relative comparison that motivated the ordering: a fitted exponent has
/// to fall between the named models it sits between, not beside them.
#[test]
fn orders_fitted_exponents_among_the_named_models() {
    let ordered = [
        infer(Model::Constant),
        infer(Model::Logarithmic),
        infer(Model::Linear),
        infer(Model::Linearithmic),
        infer(Model::Quadratic),
        infer(Model::Cubic),
        infer(Model::Exponential),
    ];

    for pair in ordered.windows(2) {
        match pair {
            [slower, faster] => assert!(slower < faster, "{slower} should order below {faster}"),
            _ => unreachable!("windows(2) yields pairs"),
        }
    }

    // A fitted O(n^1.5) belongs between O(n) and O(n^2).
    let data: Vec<(f64, f64)> = (1..=40).map(|n| (n as f64, (n as f64).powf(1.5))).collect();
    let free = big_o::infer_complexity(&data)
        .expect("a free exponent")
        .best;

    assert!(infer(Model::Linear) < free);
    assert!(free < infer(Model::Quadratic));
}

#[test]
fn substitutes_fitted_values_into_the_notation() {
    let data: Vec<(f64, f64)> = (1..=40).map(|n| (n as f64, (n as f64).powf(1.5))).collect();
    let free = big_o::infer_complexity(&data)
        .expect("a free exponent")
        .best;
    assert_eq!(free.to_string(), "O(n^1.5)");

    let data: Vec<(f64, f64)> = (1..=24).map(|n| (n as f64, 2f64.powi(n))).collect();
    let exponential = big_o::infer_complexity(&data).expect("an exponential").best;
    assert_eq!(exponential.to_string(), "O(2^n)");

    // A model whose shape its name already fixes keeps that name.
    assert_eq!(infer(Model::Linearithmic).to_string(), "O(n log n)");
    assert_eq!(infer(Model::Constant).to_string(), "O(1)");
}

#[test]
fn parses_a_model_from_its_notation_or_its_name() {
    assert_eq!("O(n^2)".parse::<Model>(), Ok(Model::Quadratic));
    assert_eq!("quadratic".parse::<Model>(), Ok(Model::Quadratic));
    assert_eq!(Model::Quadratic.to_string(), "O(n^2)");
    assert_eq!("O(n^2.5)".parse::<Model>(), Err(Error::ParseNotation));
}

#[test]
fn reports_confidence_that_is_the_same_every_run() {
    let data = clean(Model::Quadratic);

    let first = big_o::infer_complexity(&data).expect("quadratic data");
    let second = big_o::infer_complexity(&data).expect("quadratic data");

    assert_eq!(first.confidence.to_bits(), second.confidence.to_bits());
    assert!((0.0..=1.0).contains(&first.confidence));
    assert!(
        first.confidence > 0.9,
        "clean data should be confident, got {}",
        first.confidence
    );
}

#[test]
fn reports_low_confidence_when_the_data_barely_decides() {
    let barely = [
        (1000., 1000.),
        (1010., 1011.),
        (1020., 1019.),
        (1030., 1032.),
    ];

    let inference = big_o::infer_complexity(&barely).expect("something fits");

    assert!(
        inference.confidence < 0.9,
        "a narrow noisy range should not be confident, got {}",
        inference.confidence
    );
}

#[test]
fn warns_about_a_range_too_narrow_to_separate_the_models() {
    let narrow = [(1000., 1.), (1050., 2.), (1100., 3.), (1150., 4.)];

    let inference = big_o::infer_complexity(&narrow).expect("something fits");

    assert!(inference
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::NarrowRange { .. })));
    assert!(inference
        .warnings
        .iter()
        .any(|w| matches!(w, Warning::TooFewPoints { .. })));
}

#[test]
fn warns_when_cost_rises_and_falls() {
    let sawtooth: Vec<(f64, f64)> = (1..=12)
        .map(|n| (n as f64, if n % 2 == 0 { 100.0 } else { 10.0 }))
        .collect();

    let inference = big_o::infer_complexity(&sawtooth).expect("something fits");

    assert!(inference.warnings.contains(&Warning::NonMonotonic));
}

#[test]
fn a_clean_wide_sample_warns_about_nothing() {
    let inference = big_o::infer_complexity(&clean(Model::Quadratic)).expect("quadratic data");

    assert!(
        inference.warnings.is_empty(),
        "expected no warnings, got {:?}",
        inference.warnings
    );
}

#[test]
fn errors_read_as_sentences() {
    let err = big_o::infer_complexity(&[(1.0, 5.0)]).unwrap_err();

    assert_eq!(
        err.to_string(),
        "Need at least 3 distinct input sizes to infer a complexity, got 1"
    );
    let _: &dyn std::error::Error = &err;
}

//! Properties that must hold for every input, not just the ones anyone thought
//! to write down.
//!
//! The crate's promise is that it answers or explains itself, and never takes
//! the process down with it. That promise is about *all* inputs, so it is
//! checked against generated ones rather than a list.

use big_o::{Model, ModelParams};
use proptest::prelude::*;

/// Finite measurements, including the awkward ones: zero, negative, enormous.
fn measurements() -> impl Strategy<Value = Vec<(f64, f64)>> {
    let coordinate = prop_oneof![
        1 => Just(0.0f64),
        4 => -1e12f64..1e12f64,
        1 => -1e300f64..1e300f64,
    ];
    prop::collection::vec((coordinate.clone(), coordinate), 0..64)
}

/// Any measurements at all, finite or not.
fn any_measurements() -> impl Strategy<Value = Vec<(f64, f64)>> {
    let coordinate = prop_oneof![
        3 => -1e12f64..1e12f64,
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
    ];
    prop::collection::vec((coordinate.clone(), coordinate), 0..64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// Never panics, and never reports a number that is not a number.
    #[test]
    fn answers_or_explains_itself(data in measurements()) {
        if let Ok(inference) = big_o::infer_complexity(&data) {
            prop_assert!(inference.best.r_squared.is_finite());
            prop_assert!(inference.best.relative_error.is_finite());
            prop_assert!(inference.best.relative_error >= 0.0);
            prop_assert!((0.0..=1.0).contains(&inference.confidence));
            prop_assert!(inference.all.contains(&inference.best));
            prop_assert!(coefficients_are_finite(&inference.best.params));
        }
    }

    /// Non-finite measurements are dropped, never fitted.
    #[test]
    fn survives_measurements_that_are_not_numbers(data in any_measurements()) {
        if let Ok(inference) = big_o::infer_complexity(&data) {
            prop_assert!(coefficients_are_finite(&inference.best.params));
            prop_assert!(inference.confidence.is_finite());
        }
    }

    /// Inference is a pure function of its input.
    #[test]
    fn the_same_measurements_always_give_the_same_answer(data in measurements()) {
        let first = big_o::infer_complexity(&data);
        let second = big_o::infer_complexity(&data);

        match (first, second) {
            (Ok(a), Ok(b)) => {
                prop_assert_eq!(a.best.model, b.best.model);
                prop_assert_eq!(a.best.relative_error.to_bits(), b.best.relative_error.to_bits());
                prop_assert_eq!(a.confidence.to_bits(), b.confidence.to_bits());
                prop_assert_eq!(a.warnings, b.warnings);
            }
            (Err(a), Err(b)) => prop_assert_eq!(a, b),
            (a, b) => prop_assert!(false, "one call succeeded and the other did not: {:?} {:?}", a, b),
        }
    }

    /// The order of the measurements is not information about growth.
    #[test]
    fn the_order_measurements_arrive_in_does_not_matter(data in measurements()) {
        let mut reversed = data.clone();
        reversed.reverse();

        let forwards = big_o::infer_complexity(&data).map(|i| i.best.model);
        let backwards = big_o::infer_complexity(&reversed).map(|i| i.best.model);

        prop_assert_eq!(forwards, backwards);
    }

    /// Scaling every cost by a positive factor is a change of units, and a
    /// complexity is not measured in any unit.
    #[test]
    fn scaling_the_cost_does_not_change_the_complexity(
        data in measurements(),
        factor in 1e-6f64..1e6f64,
    ) {
        let scaled: Vec<(f64, f64)> = data.iter().map(|&(x, y)| (x, y * factor)).collect();

        let plain = big_o::infer_complexity(&data).map(|i| i.best.model);
        let rescaled = big_o::infer_complexity(&scaled).map(|i| i.best.model);

        if let (Ok(plain), Ok(rescaled)) = (plain, rescaled) {
            prop_assert_eq!(plain, rescaled);
        }
    }

    /// Restricting the candidates can only ever return one of them.
    #[test]
    fn a_restricted_analysis_answers_from_its_own_candidates(data in measurements()) {
        let candidates = [Model::Linear, Model::Quadratic];

        if let Ok(inference) = big_o::Analysis::new().models(candidates).infer(&data) {
            prop_assert!(candidates.contains(&inference.best.model));
        }
    }
}

fn coefficients_are_finite(params: &ModelParams) -> bool {
    match *params {
        ModelParams::Constant { offset } => offset.is_finite(),
        ModelParams::Polynomial { gain, power } => gain.is_finite() && power.is_finite(),
        ModelParams::Exponential { gain, base } => gain.is_finite() && base.is_finite(),
        ModelParams::Logarithmic { gain, offset }
        | ModelParams::Linear { gain, offset }
        | ModelParams::Linearithmic { gain, offset }
        | ModelParams::Quadratic { gain, offset }
        | ModelParams::Cubic { gain, offset } => gain.is_finite() && offset.is_finite(),
    }
}

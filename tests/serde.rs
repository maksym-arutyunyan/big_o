//! The `serde` feature: an inference survives a serialization round trip.
//!
//! The use case is persistence — write a verdict down, read it back later,
//! compare it against a fresh run (a committed CI baseline, a saved report).
//! So the test asserts the round trip preserves the fields such a comparison
//! reads, not any particular wire format.

#![cfg(feature = "serde")]

fn quadratic() -> big_o::Inference {
    let data = [
        (100., 10_180.),
        (200., 39_800.),
        (400., 161_440.),
        (800., 637_440.),
        (1600., 2_570_240.),
        (3200., 10_352_640.),
    ];
    big_o::infer_complexity(&data).expect("quadratic data infers")
}

#[test]
fn inference_round_trips() {
    let before = quadratic();
    let json = serde_json::to_string(&before).expect("inference serializes");
    let after: big_o::Inference = serde_json::from_str(&json).expect("inference deserializes");

    assert_eq!(after.best, before.best);
    assert_eq!(after.all, before.all);
    assert_eq!(after.warnings, before.warnings);
    assert!((after.confidence - before.confidence).abs() < f64::EPSILON);
}

#[test]
fn model_name_is_stable_in_json() {
    let json = serde_json::to_string(&big_o::Model::Linearithmic).expect("model serializes");
    // The variant name is the wire format; a rename would break stored
    // baselines, so it is pinned here.
    assert_eq!(json, "\"Linearithmic\"");
}

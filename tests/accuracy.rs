//! How often the right model is recovered from data that is known to have come
//! from it.
//!
//! This is the measurement the selection rule exists to satisfy, and the only
//! one that can show it works. Any single example can be made to pass by
//! tuning; a sweep across every model at a realistic noise level cannot.

mod synthetic;

use big_o::Model;
use std::collections::BTreeMap;

/// Trials per model at each noise level in the checked-in sweep.
///
/// Sized so the suite stays quick. The wider sweep below carries the real
/// evidence and runs on a schedule.
const TRIALS: u64 = 25;

/// Trials per model in the wider sweep.
const DEEP_TRIALS: u64 = 200;

/// Timing noise a benchmark on a shared machine can be expected to carry.
const NOISE: f64 = 0.05;

/// Fraction of trials that must recover the model they were generated from.
///
/// Applied per model, not averaged: an average hides one model failing
/// completely behind seven that succeed.
const REQUIRED: f64 = 0.95;

/// Runs `trials` per model at noise level `sigma` and returns, per model, the
/// fraction of trials that recovered it.
fn recovery_rates(trials: u64, sigma: f64) -> BTreeMap<&'static str, (f64, Vec<String>)> {
    let mut rates = BTreeMap::new();
    for model in MODELS {
        let mut recovered = 0u64;
        let mut missed: Vec<String> = Vec::new();
        for trial in 0..trials {
            let data = synthetic::noisy(model, trial, sigma);
            match big_o::infer_complexity(&data) {
                Ok(inference) if inference.best.model == model => recovered += 1,
                Ok(inference) => missed.push(format!("trial {trial}: {}", inference.best)),
                Err(e) => missed.push(format!("trial {trial}: {e}")),
            }
        }
        missed.truncate(3);
        rates.insert(model.notation(), (recovered as f64 / trials as f64, missed));
    }
    rates
}

const MODELS: [Model; 8] = [
    Model::Constant,
    Model::Logarithmic,
    Model::Linear,
    Model::Linearithmic,
    Model::Quadratic,
    Model::Cubic,
    Model::Polynomial,
    Model::Exponential,
];

fn assert_recovers(trials: u64, sigma: f64, required: f64) {
    let rates = recovery_rates(trials, sigma);

    let failures: Vec<String> = rates
        .iter()
        .filter(|(_, (rate, _))| *rate < required)
        .map(|(model, (rate, missed))| {
            format!("  {model}: {:.0}% recovered, e.g. {missed:?}", rate * 100.0)
        })
        .collect();

    assert!(
        failures.is_empty(),
        "at {:.0}% noise, {} of 8 models fell below {:.0}% over {trials} trials:\n{}",
        sigma * 100.0,
        failures.len(),
        required * 100.0,
        failures.join("\n")
    );
}

#[test]
fn recovers_every_model_from_clean_data() {
    assert_recovers(TRIALS, 0.0, 1.0);
}

#[test]
fn recovers_every_model_through_realistic_noise() {
    assert_recovers(TRIALS, NOISE, REQUIRED);
}

/// The wider sweep. Excluded from the default run for time; `--ignored` runs it.
#[test]
#[ignore = "wider sweep, run on a schedule rather than per commit"]
fn recovers_every_model_over_many_trials() {
    assert_recovers(DEEP_TRIALS, 0.0, 1.0);
    assert_recovers(DEEP_TRIALS, NOISE, REQUIRED);
}

/// Prints the full confusion matrix. Not an assertion — a way to see *how* a
/// sweep failed when one of the tests above says that it did.
#[test]
#[ignore = "diagnostic, prints the confusion matrix"]
fn report_confusion_matrix() {
    for sigma in [0.0, NOISE] {
        println!(
            "\nnoise {:.0}%, {DEEP_TRIALS} trials per model",
            sigma * 100.0
        );
        for (model, (rate, missed)) in recovery_rates(DEEP_TRIALS, sigma) {
            println!("  {model:<12} {:>5.1}%  {missed:?}", rate * 100.0);
        }
    }
}

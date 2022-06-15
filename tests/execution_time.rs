use std::time::Instant;
use std::{thread, time};

#[allow(dead_code)]
fn write_csv(data: &Vec<(f64, f64)>, path: &str) {
    let mut text = String::new();
    for (x, y) in data {
        text.push_str(&format!("{},{}\n", *x, *y));
    }
    std::fs::write(path, text).expect("Unable to write file");
}

fn measure_execution_time(x: &Vec<u64>, f: fn(u64), runs: usize) -> Vec<(f64, f64)> {
    let mut data = Vec::new();
    for _ in 0..runs {
        for n in x {
            let start = Instant::now();
            f(*n);
            let t = start.elapsed().as_secs_f64();
            data.push((*n as f64, t));
        }
    }
    data
}

fn dummy_constant(_n: u64) {
    thread::sleep(time::Duration::from_millis(100));
}

fn dummy_logarithmic(n: u64) {
    let x = n as f64;
    let n = x.ln() as u64;
    thread::sleep(time::Duration::from_millis(10 * n));
}

fn dummy_linear(n: u64) {
    thread::sleep(time::Duration::from_millis(n));
}

fn dummy_linearithmic(n: u64) {
    let x = n as f64;
    let n = (x * x.ln()) as u64;
    thread::sleep(time::Duration::from_millis(n));
}

fn dummy_quadratic(n: u64) {
    let x = n as f64;
    let n = x.powi(2) as u64;
    thread::sleep(time::Duration::from_millis(n / 3));
}

fn dummy_cubic(n: u64) {
    let x = n as f64;
    let n = x.powi(3) as u64;
    thread::sleep(time::Duration::from_millis(n / 5));
}

fn dummy_polynomial(n: u64) {
    let x = n as f64;
    let n = x.powi(4) as u64;
    thread::sleep(time::Duration::from_millis(n / 7));
}

fn dummy_exponential(n: u64) {
    let x = n as f64;
    let n = 2_f64.powf(x) as u64;
    thread::sleep(time::Duration::from_millis(n));
}

#[test]
#[ignore] // Noisy test, switches between constant and logarithmic.
fn time_infer_complexity_constant() {
    let runs = 3;
    let n: Vec<u64> = vec![1, 1_000, 1_000_000, 1_000_000_000, 1_000_000_000_000];
    let data = measure_execution_time(&n, dummy_constant, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Constant);
}

#[test]
fn time_infer_complexity_logarithmic() {
    let runs = 10;
    let n: Vec<u64> = vec![1, 10, 100, 1_000, 10_000, 100_000];
    let data = measure_execution_time(&n, dummy_logarithmic, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Logarithmic);
}

#[test]
fn time_infer_complexity_linear() {
    let runs = 10;
    let n: Vec<u64> = vec![1, 2, 5, 10, 50, 100, 500];
    let data = measure_execution_time(&n, dummy_linear, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Linear);
}

#[test]
fn time_infer_complexity_linearithmic() {
    let runs = 10;
    let n: Vec<u64> = vec![1, 2, 5, 10, 20, 50, 100];
    let data = measure_execution_time(&n, dummy_linearithmic, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Linearithmic);
}

#[test]
fn time_infer_complexity_quadratic() {
    let runs = 10;
    let n: Vec<u64> = vec![1, 2, 5, 10, 20, 50];
    let data = measure_execution_time(&n, dummy_quadratic, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Quadratic);
}

#[test]
fn time_infer_complexity_cubic() {
    let runs = 5;
    let n: Vec<u64> = vec![1, 2, 5, 10, 20];
    let data = measure_execution_time(&n, dummy_cubic, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Cubic);
}

#[test]
fn time_infer_complexity_polynomial() {
    let runs = 1;
    let n: Vec<u64> = vec![3, 5, 7, 9, 11, 13];
    let data = measure_execution_time(&n, dummy_polynomial, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Polynomial);
}

#[test]
fn time_infer_complexity_exponential() {
    let runs = 1;
    let n: Vec<u64> = vec![1, 3, 5, 7, 9, 11];
    let data = measure_execution_time(&n, dummy_exponential, runs);
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Exponential);
}

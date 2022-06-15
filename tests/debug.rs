use assert_approx_eq::assert_approx_eq;

const EPSILON: f64 = 1e-9;

#[test]
#[ignore]
fn infer_exponential_stress_1() {
    let (name, notation) = (big_o::Name::Exponential, "O(c^n)");
    let (gain, base): (f64, f64) = (530.7409528224476, 0.000013222324524164719);
    let f = Box::new(|x: f64| gain * base.powf(x));

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (best, _all) = big_o::infer_complexity(data).unwrap();

    println!("{:?}", best.params);

    assert_eq!(best.name, name);
    assert_eq!(best.notation, notation);
    assert_approx_eq!(best.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(best.params.base.unwrap(), base, EPSILON);
}

#[test]
#[ignore]
fn infer_exponential_stress_2() {
    let (name, notation) = (big_o::Name::Exponential, "O(c^n)");
    let (gain, base): (f64, f64) = (17.7379380563778, 0.0005420564306968245);
    let f = Box::new(|x: f64| gain * base.powf(x));

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (best, _all) = big_o::infer_complexity(data).unwrap();

    println!("{:?}", best.params);

    assert_eq!(best.name, name);
    assert_eq!(best.notation, notation);
    assert_approx_eq!(best.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(best.params.base.unwrap(), base, EPSILON);
}

#[test]
#[ignore]
fn infer_logarithmic_stress_1() {
    let (name, notation) = (big_o::Name::Logarithmic, "O(log n)");
    let (gain, offset): (f64, f64) = (0.00016209916053888662, 753.1476064361);
    let f = Box::new(|x: f64| gain * x.ln() + offset);

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (best, _all) = big_o::infer_complexity(data).unwrap();

    println!("{:?}", best.params);

    assert_eq!(best.name, name);
    assert_eq!(best.notation, notation);
    assert_approx_eq!(best.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(best.params.offset.unwrap(), offset, EPSILON);
}

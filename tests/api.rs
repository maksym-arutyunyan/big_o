use assert_approx_eq::assert_approx_eq;

const EPSILON: f64 = 1e-9;

#[test]
fn infer_each() {
    type Func = Box<dyn Fn(f64) -> f64>;
    type Notation = &'static str;

    let test_cases: Vec<(Func, big_o::Name, Notation, big_o::Params)> = vec![
        (
            Box::new(|x| 0.0 * x + 1.5),
            big_o::Name::Constant,
            "O(1)",
            big_o::Params::new().gain(0.0).offset(1.5).build(),
        ),
        (
            Box::new(|x| 8.0 * x.ln() + 9.0),
            big_o::Name::Logarithmic,
            "O(log n)",
            big_o::Params::new().gain(8.0).offset(9.0).build(),
        ),
        (
            Box::new(|x| 2.0 * x + 3.0),
            big_o::Name::Linear,
            "O(n)",
            big_o::Params::new().gain(2.0).offset(3.0).build(),
        ),
        (
            Box::new(|x| 1.5 * x * x.ln() + 2.0),
            big_o::Name::Linearithmic,
            "O(n log n)",
            big_o::Params::new().gain(1.5).offset(2.0).build(),
        ),
        (
            Box::new(|x| 4.0 * x.powi(2) + 5.0),
            big_o::Name::Quadratic,
            "O(n^2)",
            big_o::Params::new().gain(4.0).offset(5.0).build(),
        ),
        (
            Box::new(|x| 6.0 * x.powi(3) + 7.0),
            big_o::Name::Cubic,
            "O(n^3)",
            big_o::Params::new().gain(6.0).offset(7.0).build(),
        ),
        (
            Box::new(|x| 3.0 * x.powi(4)),
            big_o::Name::Polynomial,
            "O(n^m)",
            big_o::Params::new().gain(3.0).power(4.0).build(),
        ),
        (
            Box::new(|x| 9.0 * 5.0_f64.powf(x)),
            big_o::Name::Exponential,
            "O(c^n)",
            big_o::Params::new().gain(9.0).base(5.0).build(),
        ),
    ];

    for (f, name, notation, params) in test_cases {
        let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
        let (complexity, _all) = big_o::infer_complexity(data).unwrap();
        assert_eq!(complexity.name, name);
        assert_eq!(complexity.notation, notation);
        assert_approx_eq!(
            complexity.params.gain.unwrap_or(0.),
            params.gain.unwrap_or(0.),
            EPSILON
        );
        assert_approx_eq!(
            complexity.params.offset.unwrap_or(0.),
            params.offset.unwrap_or(0.),
            EPSILON
        );
        assert_approx_eq!(
            complexity.params.power.unwrap_or(0.),
            params.power.unwrap_or(0.),
            EPSILON
        );
        assert_approx_eq!(
            complexity.params.base.unwrap_or(0.),
            params.base.unwrap_or(0.),
            EPSILON
        );
    }
}

#[test]
fn infer_constant() {
    let (name, notation) = (big_o::Name::Constant, "O(1)");
    let gain = 0.0;
    let offset = 1.5;
    let f = Box::new(|x: f64| gain * x + offset);

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.offset.unwrap(), offset, EPSILON);
    assert!(complexity.rank <= big_o::complexity("O(1)").unwrap().rank);
}

#[test]
fn infer_logarithmic() {
    let (name, notation) = (big_o::Name::Logarithmic, "O(log n)");
    let gain = 8.0;
    let offset = 9.0;
    let f = Box::new(|x: f64| gain * x.ln() + offset);

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.offset.unwrap(), offset, EPSILON);
    assert!(complexity.rank <= big_o::complexity("O(log n)").unwrap().rank);
}

#[test]
fn infer_linear() {
    let (name, notation) = (big_o::Name::Linear, "O(n)");
    let gain = 2.0;
    let offset = 3.0;
    let f = Box::new(|x: f64| gain * x + offset);

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.offset.unwrap(), offset, EPSILON);
    assert!(complexity.rank <= big_o::complexity("O(n)").unwrap().rank);
}

#[test]
fn infer_linearithmic() {
    let (name, notation) = (big_o::Name::Linearithmic, "O(n log n)");
    let gain = 1.5;
    let offset = 2.0;
    let f = Box::new(|x: f64| gain * x * x.ln() + offset);

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.offset.unwrap(), offset, EPSILON);
    assert!(complexity.rank <= big_o::complexity("O(n log n)").unwrap().rank);
}

#[test]
fn infer_quadratic() {
    let (name, notation) = (big_o::Name::Quadratic, "O(n^2)");
    let gain = 4.0;
    let offset = 5.0;
    let f = Box::new(|x: f64| gain * x.powi(2) + offset);

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.offset.unwrap(), offset, EPSILON);
    assert!(complexity.rank <= big_o::complexity("O(n^2)").unwrap().rank);
}

#[test]
fn infer_cubic() {
    let (name, notation) = (big_o::Name::Cubic, "O(n^3)");
    let gain = 6.0;
    let offset = 7.0;
    let f = Box::new(|x: f64| gain * x.powi(3) + offset);

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.offset.unwrap(), offset, EPSILON);
    assert!(complexity.rank <= big_o::complexity("O(n^3)").unwrap().rank);
}

#[test]
fn infer_polynomial() {
    let (name, notation) = (big_o::Name::Polynomial, "O(n^m)");
    let gain = 3.0;
    let power = 4.0;
    let f = Box::new(|x: f64| gain * x.powf(power));

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.power.unwrap(), power, EPSILON);
    // Note: impossible to create a generic complexity O(n^m) without providing its degree.
    assert!(complexity.rank < big_o::complexity("O(c^n)").unwrap().rank);
}

#[test]
fn infer_exponential() {
    let (name, notation) = (big_o::Name::Exponential, "O(c^n)");
    let gain = 9.0;
    let base: f64 = 5.0;
    let f = Box::new(|x: f64| gain * base.powf(x));

    let data: Vec<(f64, f64)> = (1..100).map(|i| i as f64).map(|x| (x, f(x))).collect();
    let (complexity, _all) = big_o::infer_complexity(data).unwrap();

    assert_eq!(complexity.name, name);
    assert_eq!(complexity.notation, notation);
    assert_approx_eq!(complexity.params.gain.unwrap(), gain, EPSILON);
    assert_approx_eq!(complexity.params.base.unwrap(), base, EPSILON);
    assert!(complexity.rank <= big_o::complexity("O(c^n)").unwrap().rank);
}

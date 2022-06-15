# big_o

Infers asymptotic computational complexity.

`big_o` helps to estimate computational complexity of algorithms by inspecting measurement data (eg. execution time, memory consumption, etc). Users are expected to provide measurement data, `big_o` will try to fit a set of complexity models and return the best fit.

## Examples

```rs
    // f(x) = offset
    let data = vec![(1., 11.), (2., 11.), (3., 11.), (4., 11.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Constant);
    assert_eq!(best.notation, "O(1)");
    assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 0.0, 1e-6);
    assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 11.0, 1e-6);

    // f(x) = gain * log(x) + offset
    let data = vec![(10., 1.), (100., 2.), (1_000., 3.), (10_000., 4.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Logarithmic);
    assert_eq!(best.notation, "O(log n)");

    // f(x) = gain * x + offset
    let data = vec![(1., 17.), (2., 27.), (3., 37.), (4., 47.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Linear);
    assert_eq!(best.notation, "O(n)");
    assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 10.0, 1e-6);
    assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 7.0, 1e-6);

    // f(x) = gain * x * log(x) + offset
    let data = vec![(10., 10.), (100., 200.), (1_000., 3_000.), (10_000., 40_000.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Linearithmic);
    assert_eq!(best.notation, "O(n log n)");

    // f(x) = gain * x ^ 2 + offset
    let data = vec![(1., 1.), (2., 4.), (3., 9.), (4., 16.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Quadratic);
    assert_eq!(best.notation, "O(n^2)");
    assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
    assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 0.0, 1e-6);

    // f(x) = gain * x ^ 3 + offset
    let data = vec![(1., 1.), (2., 8.), (3., 27.), (4., 64.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Cubic);
    assert_eq!(best.notation, "O(n^3)");
    assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
    assert_approx_eq::assert_approx_eq!(best.params.offset.unwrap(), 0.0, 1e-6);

    // f(x) = gain * x ^ power
    let data = vec![(1., 1.), (2., 16.), (3., 81.), (4., 256.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Polynomial);
    assert_eq!(best.notation, "O(n^m)");
    assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
    assert_approx_eq::assert_approx_eq!(best.params.power.unwrap(), 4.0, 1e-6);

    // f(x) = gain * base ^ x
    let data = vec![(1., 2.), (2., 4.), (3., 8.), (4., 16.)];
    let (best, _all) = big_o::infer_complexity(data).unwrap();
    assert_eq!(best.name, big_o::Name::Exponential);
    assert_eq!(best.notation, "O(c^n)");
    assert_approx_eq::assert_approx_eq!(best.params.gain.unwrap(), 1.0, 1e-6);
    assert_approx_eq::assert_approx_eq!(best.params.base.unwrap(), 2.0, 1e-6);
```

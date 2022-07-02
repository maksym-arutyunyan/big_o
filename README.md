# big_o

Infers asymptotic computational complexity.

`big_o` helps to estimate computational complexity of algorithms by inspecting measurement data (eg. execution time, memory consumption, etc). Users are expected to provide measurement data, `big_o` will try to fit a set of complexity models and return the best fit.

## Example

```rust
use assert_approx_eq::assert_approx_eq;

// f(x) = gain * x ^ 2 + offset
let data = vec![(1., 1.), (2., 4.), (3., 9.), (4., 16.)];

let (complexity, _all) = big_o::infer_complexity(data).unwrap();

assert_eq!(complexity.name, big_o::Name::Quadratic);
assert_eq!(complexity.notation, "O(n^2)");
assert_approx_eq!(complexity.params.gain.unwrap(), 1.0, 1e-6);
assert_approx_eq!(complexity.params.offset.unwrap(), 0.0, 1e-6);
```

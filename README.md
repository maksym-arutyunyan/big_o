# big_o

[gh-image]: https://github.com/maksym-arutyunyan/big_o/workflows/CI/badge.svg
[gh-checks]: https://github.com/maksym-arutyunyan/big_o/actions/workflows/pre-merge-checks.yaml
[cratesio-image]: https://img.shields.io/crates/v/big_o.svg
[cratesio]: https://crates.io/crates/big_o
[docsrs-image]: https://docs.rs/big_o/badge.svg
[docsrs]: https://docs.rs/big_o

[![big_o GitHub Actions][gh-image]][gh-checks]
[![big_o on crates.io][cratesio-image]][cratesio]
[![big_o on docs.rs][docsrs-image]][docsrs]

Infers asymptotic computational complexity.

Measure how long your algorithm takes over a range of input sizes, hand the
measurements to `big_o`, and it fits every complexity model it knows and reports
the one that best describes them — along with how firmly the data supports that
choice.

No dependencies.

## Example

Real measurements, so a few percent of timing noise:

```rust
use big_o::Model;

let measurements = [
    (100., 10_180.),
    (200., 39_800.),
    (400., 161_440.),
    (800., 637_440.),
    (1600., 2_570_240.),
    (3200., 10_352_640.),
    (6400., 40_673_280.),
    (12800., 164_167_680.),
];

let inference = big_o::infer_complexity(&measurements).unwrap();

assert_eq!(inference.best.model, Model::Quadratic);
assert_eq!(inference.best.to_string(), "O(n^2)");

// Assert a bound rather than an exact match, the way you would in a test.
assert!(inference.best.is_at_most(Model::Quadratic));
assert!(inference.best.is_faster_than(Model::Cubic));

// How much of the answer is the data, and how much is the noise.
assert!(inference.confidence > 0.9);
```

Noise does not cost you the named answer: the measurements above are not exactly
quadratic, and `O(n^2)` is still what you get rather than `O(n^2.0004)`.

## Reading the result

`infer_complexity` returns an `Inference`:

- `best` — the model that describes the data best, with its fitted
  coefficients, its `r_squared`, and its `relative_error`.
- `all` — every model that could be fitted, best first, if you want to see what
  came close.
- `confidence` — the fraction of resampled subsets of your measurements that
  chose the same model. Deterministic: the same input always gives the same
  number.
- `warnings` — conditions that weaken the result without invalidating it: too
  few input sizes, too narrow a range of them, cost that falls or that rises and
  falls, and models that could not be fitted at all.

A warning is worth reading before trusting a result. Complexity models are
separated by how fast they grow, so what identifies them is the *range* of input
sizes you measured, not the number of points in it — ten sizes between 1000 and
1100 look linear whatever produced them, and `NarrowRange` says so.

To weigh only the models you consider possible:

```rust
use big_o::{Analysis, Model};

let measurements = [(1., 1.), (2., 4.), (3., 9.), (4., 16.), (5., 25.)];

let inference = Analysis::new()
    .models([Model::Linear, Model::Quadratic])
    .infer(&measurements)
    .unwrap();

assert_eq!(inference.best.model, Model::Quadratic);
```

## Errors

- `NotEnoughData` — fewer than three distinct input sizes. Repeated measurements
  of one size collapse to their median first, so they count once between them.
- `NoValidComplexity` — nothing among the candidate models describes the data.

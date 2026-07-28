# Changelog

## 0.2.0

A rewrite. The crate now aims to be usable on real, noisy benchmark data: it
does not crash, it reports a named complexity where there is one, and it says
how much of the answer is the data rather than the noise.

Breaking: every public type changed. See the README for the current shape.

### Fixed

- **Crash on an input size of zero.** The logarithmic models linearize through
  `ln(x)`, and the resulting `-inf` reached the least-squares solver, which
  raised "Singular value was NaN" from inside its matrix decomposition. Fitting
  is now closed-form, and a model that cannot describe the data is skipped
  rather than taking the call — or the process — down with it.
- **Noisy data almost never got a named answer.** Quadratic measurements with
  2% of timing noise reported `O(n^2.000382)`. Models were ranked by their
  absolute residuals, which cannot choose between models that contain each
  other: the free exponent could bend towards the noise where a fixed one could
  not, so it always won by a hair. Flat data could not report `O(1)` at all,
  for the same reason.
- **One point inferred a complexity.** A single measurement fits every model
  exactly and reported `O(n)`; repeated measurements of one input size reported
  `O(n log n)`. Both now report `NotEnoughData`.
- **A falling cost ranked as the fastest complexity.** The rank was unsigned, so
  a fitted exponent of -1 cast to the rank of `O(1)`. Ordering is now signed.
- **`O(n log n)` data was reported as `O(n)`.** Fitting minimized absolute error
  while selection judged relative error; over the top decade the two curves
  differ by almost nothing in absolute terms.

### Added

- `Inference` carries `confidence`, the fraction of resampled subsets of the
  measurements that chose the same model. Deterministic — the same input always
  gives the same number.
- `warnings` for conditions that weaken a result without invalidating it: too
  few input sizes, too narrow a range of them, a falling cost, a cost that rises
  and falls, and models that could not be fitted.
- `is_at_most` and `is_faster_than` compare a fit against a named bound.
- `Analysis` restricts inference to a chosen set of models.
- `Display for Fit` substitutes fitted values: `O(n^1.5)`, `O(2^n)`.
- `Analysis::accept_range` declares the ladder a caller can afford, so the
  range warnings fire only below it — an acknowledged short ladder stops
  repeating itself, and the warnings that matter stay loud.
- A `serde` feature (off by default) derives `Serialize`/`Deserialize` for the
  result types, so a verdict can be persisted and compared against a later run.
- Repeated measurements of one input size are collapsed to their median, so a
  descheduled run does not steer the result.
- `rust-version` is declared and built in CI.

### Changed

- `Name` is now `Model` and names a class rather than a fitted curve.
- `ModelParams` carries exactly the coefficients each shape has, so a fit with a
  missing parameter cannot be constructed.
- The numeric rank is no longer public.
- Errors are named consistently; `NotEnoughData` reports what it needed.

### Removed

- All runtime dependencies — `nalgebra`, `lstsq` and `float-cmp`.
- Sleep-based tests, which spent about ten seconds of every CI run asleep.

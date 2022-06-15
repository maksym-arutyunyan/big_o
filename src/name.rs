/// Names of asymptotic computational complexities.
#[derive(PartialEq, Debug, Copy, Clone)]
pub enum Name {
    Constant,
    Logarithmic,
    Linear,
    Linearithmic,
    Quadratic,
    Cubic,
    Polynomial,
    Exponential,
}

/// Returns all supported complexity names.
pub fn all_names() -> Vec<Name> {
    vec![
        Name::Constant,
        Name::Logarithmic,
        Name::Linear,
        Name::Linearithmic,
        Name::Quadratic,
        Name::Cubic,
        Name::Polynomial,
        Name::Exponential,
    ]
}

/// Returns complexity notation.
pub fn notation(name: Name) -> &'static str {
    match name {
        Name::Constant => "O(1)",
        Name::Logarithmic => "O(log n)",
        Name::Linear => "O(n)",
        Name::Linearithmic => "O(n log n)",
        Name::Quadratic => "O(n^2)",
        Name::Cubic => "O(n^3)",
        Name::Polynomial => "O(n^m)",
        Name::Exponential => "O(c^n)",
    }
}

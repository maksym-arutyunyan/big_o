use std::fmt;
use std::str::FromStr;

/// Names of asymptotic computational complexities.
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
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

impl From<Name> for &str {
    fn from(name: Name) -> &'static str {
        notation(name)
    }
}

impl TryFrom<&str> for Name {
    type Error = &'static str;

    fn try_from(string: &str) -> Result<Self, Self::Error> {
        match &string.to_lowercase()[..] {
            "o(1)" | "constant" => Ok(Name::Constant),
            "o(log n)" | "logarithmic" => Ok(Name::Logarithmic),
            "o(n)" | "linear" => Ok(Name::Linear),
            "o(n log n)" | "linearithmic" => Ok(Name::Linearithmic),
            "o(n^2)" | "quadratic" => Ok(Name::Quadratic),
            "o(n^3)" | "cubic" => Ok(Name::Cubic),
            "o(n^m)" | "polynomial" => Ok(Name::Polynomial),
            "o(c^n)" | "exponential" => Ok(Name::Exponential),
            _ => Err("Can't convert string to Name"),
        }
    }
}

impl FromStr for Name {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.try_into()
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", notation(*self))
    }
}


#[cfg(test)]
mod tests{
    use super::*;

    const NOTATION_TEST_CASES: [(&str, Name); 8] = [
        ("O(1)", Name::Constant),
        ("O(log n)", Name::Logarithmic),
        ("O(n)", Name::Linear),
        ("O(n log n)", Name::Linearithmic),
        ("O(n^2)", Name::Quadratic),
        ("O(n^3)", Name::Cubic),
        ("O(n^m)", Name::Polynomial),
        ("O(c^n)", Name::Exponential),
    ];

    const NAMED_TEST_CASES: [(&str, Name); 8] = [
        ("Constant", Name::Constant),
        ("Logarithmic", Name::Logarithmic),
        ("Linear", Name::Linear),
        ("Linearithmic", Name::Linearithmic),
        ("Quadratic", Name::Quadratic),
        ("Cubic", Name::Cubic),
        ("Polynomial", Name::Polynomial),
        ("Exponential", Name::Exponential),
    ];

    #[test]
    fn name_into_string() {
        for (string, name) in NOTATION_TEST_CASES {
            let converted: &str = name.into();
            assert_eq!(converted, string);
            assert_eq!(Into::<&str>::into(name), string);
        }
    }

    #[test]
    fn name_to_string() {
        for (string, name) in NOTATION_TEST_CASES {
            assert_eq!(name.to_string(), string);
        }
    }

    #[test]
    fn string_try_from() {
        let test_cases = [NOTATION_TEST_CASES, NAMED_TEST_CASES].concat();
        for (string, name) in test_cases {
            assert_eq!(Name::try_from(string).unwrap(), name);
        }
    }

    #[test]
    #[should_panic]
    fn string_try_from_fails() {
        Name::try_from("irrlevant text").unwrap();
    }

    #[test]
    fn string_try_into() {
        let test_cases = [NOTATION_TEST_CASES, NAMED_TEST_CASES].concat();
        for (string, name) in test_cases {
            let into: Name = string.try_into().unwrap();
            assert_eq!(into, name);
        }
    }

    #[test]
    #[should_panic]
    fn string_try_into_fails() {
        let _: Name = "irrlevant text".try_into().unwrap();
    }

    #[test]
    fn string_parse() {
        let test_cases = [NOTATION_TEST_CASES, NAMED_TEST_CASES].concat();
        for (string, name) in test_cases {
            let parse: Name = string.parse().unwrap();
            assert_eq!(parse, name);
        }
    }

    #[test]
    #[should_panic]
    fn string_parse_fails() {
        let _: Name = "irrlevant text".parse().unwrap();
    }
}

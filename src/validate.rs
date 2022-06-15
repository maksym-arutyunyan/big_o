use crate::complexity::Complexity;
use crate::name::Name;
use crate::params::Params;
use float_cmp::approx_eq;

const EPSILON: f64 = 1e-5;

/// Checks if provided complexity is degraded to a lower degree.
fn is_degraded(name: Name, p: &Params) -> Option<Name> {
    // Any-to-Constant if gain is 0
    // f(x) = 0 * q(x) + offset = offset
    if name != Name::Constant {
        if let Some(value) = p.gain {
            if approx_eq!(f64, value, 0.0, epsilon = EPSILON) {
                return Some(Name::Constant);
            }
        }
    }
    // Polynomial
    if let (Name::Polynomial, Some(power)) = (name, p.power) {
        let cases: Vec<(f64, Name)> = vec![
            (0.0, Name::Constant),  // f(x) = gain * x ^ 0 = gain
            (1.0, Name::Linear),    // f(x) = gain * x ^ 1 = gain * x
            (2.0, Name::Quadratic), // f(x) = gain * x ^ 2
            (3.0, Name::Cubic),     // f(x) = gain * x ^ 3
        ];
        for (expected_power, degraded_to) in cases {
            if approx_eq!(f64, power, expected_power, epsilon = EPSILON) {
                return Some(degraded_to);
            }
        }
    }
    // Exponential
    if let (Name::Exponential, Some(base)) = (name, p.base) {
        let cases: Vec<(f64, Name)> = vec![
            (0.0, Name::Constant), // f(x) = gain * 0 ^ x = 0
            (1.0, Name::Constant), // f(x) = gain * 1 ^ x = gain
        ];
        for (expected_base, degraded_to) in cases {
            if approx_eq!(f64, base, expected_base, epsilon = EPSILON) {
                return Some(degraded_to);
            }
        }
    }

    None
}

pub fn is_valid(complexity: &Complexity) -> bool {
    let p = &complexity.params;
    // Missing residuals.
    if !p.residuals.unwrap_or(std::f64::NAN).is_finite() {
        return false;
    }
    // Negative gain.
    if p.gain.unwrap_or(-1.0) < 0.0 {
        return false;
    }
    // Degraded complexity.
    if is_degraded(complexity.name, p).is_some() {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_degraded_to_constant() {
        // f(x) = 0 * q(x) + offset = offset
        for name in crate::name::all_names() {
            let result = is_degraded(name, &Params::new().gain(0.0).build());
            if name == Name::Constant {
                assert_eq!(result, None);
            } else {
                assert_eq!(result, Some(Name::Constant));
            }
        }
    }

    #[test]
    fn polynomial_degraded_to_constant() {
        // f(x) = gain * x ^ 0 = gain
        assert_eq!(
            is_degraded(Name::Polynomial, &Params::new().power(0.0).build()),
            Some(Name::Constant)
        );
    }

    #[test]
    fn polynomial_degraded_to_linear() {
        // f(x) = gain * x ^ 1
        assert_eq!(
            is_degraded(Name::Polynomial, &Params::new().power(1.0).build()),
            Some(Name::Linear)
        );
    }

    #[test]
    fn polynomial_degraded_to_quadratic() {
        // f(x) = gain * x ^ 2
        assert_eq!(
            is_degraded(Name::Polynomial, &Params::new().power(2.0).build()),
            Some(Name::Quadratic)
        );
    }

    #[test]
    fn polynomial_degraded_to_cubic() {
        // f(x) = gain * x ^ 3
        assert_eq!(
            is_degraded(Name::Polynomial, &Params::new().power(3.0).build()),
            Some(Name::Cubic)
        );
    }

    #[test]
    fn exponential_degraded_to_constant_base_0() {
        // f(x) = gain * 0 ^ x = 0
        assert_eq!(
            is_degraded(Name::Exponential, &Params::new().base(0.0).build()),
            Some(Name::Constant)
        );
    }

    #[test]
    fn exponential_degraded_to_constant_base_1() {
        // f(x) = gain * 1 ^ x = gain
        assert_eq!(
            is_degraded(Name::Exponential, &Params::new().base(0.0).build()),
            Some(Name::Constant)
        );
    }
}

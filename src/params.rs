/// A structure to hold function parameters
///
/// Function examples:
/// - `f(x) = gain * x + offset`
/// - `f(x) = gain * x ^ power`
/// - `f(x) = gain * base ^ x`
///
/// # Example
/// ```
/// let params = big_o::Params::new().gain(2.0).offset(3.0).build();
///
/// assert_eq!(params.gain, Some(2.0));
/// assert_eq!(params.offset, Some(3.0));
/// assert_eq!(params.power, None);
/// ```
#[derive(Clone, Debug)]
pub struct Params {
    /// Multiplicative coefficient of the approximation function.
    ///
    /// Examples:
    /// - `f(x) = gain * x + offset`
    /// - `f(x) = gain * base.powf(x)`
    pub gain: Option<f64>,

    /// Constant offset of the approximation function.
    ///
    /// Example: in `f(x) = gain * x + offset` this is the `offset` term.
    pub offset: Option<f64>,

    /// Sum of absolute residuals between the measured data and the
    /// approximation.
    pub residuals: Option<f64>,

    /// Exponent of a polynomial function.
    ///
    /// Example: `f(x) = gain * x.powf(power)`
    pub power: Option<f64>,

    /// Base of an exponential function.
    ///
    /// Example: `f(x) = gain * base.powf(x)`
    pub base: Option<f64>,
}

/// Params builder
impl Params {
    pub fn new() -> Self {
        Self {
            gain: None,
            offset: None,
            residuals: None,
            power: None,
            base: None,
        }
    }

    pub fn gain(&mut self, value: f64) -> &mut Self {
        self.gain = Some(value);
        self
    }

    pub fn offset(&mut self, value: f64) -> &mut Self {
        self.offset = Some(value);
        self
    }

    pub fn residuals(&mut self, value: f64) -> &mut Self {
        self.residuals = Some(value);
        self
    }

    pub fn power(&mut self, value: f64) -> &mut Self {
        self.power = Some(value);
        self
    }

    pub fn base(&mut self, value: f64) -> &mut Self {
        self.base = Some(value);
        self
    }

    pub fn build(&mut self) -> Params {
        Params {
            gain: self.gain,
            offset: self.offset,
            residuals: self.residuals,
            power: self.power,
            base: self.base,
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        Self::new()
    }
}

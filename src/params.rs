/// A structure to hold function parameters:
///
/// - `f(x) = gain * x + offset`
/// - `f(x) = gain * x ^ power`
/// - `f(x) = gain * base ^ x`
///
/// # Example
/// ```
/// let p = big_o::Params::new().gain(2.0).offset(3.0).build();
///
/// assert_eq!(p.gain, Some(2.0));
/// assert_eq!(p.offset, Some(3.0));
/// assert_eq!(p.power, None);
/// ```
#[derive(Clone, Debug)]
pub struct Params {
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    pub residuals: Option<f64>,
    pub power: Option<f64>,
    pub base: Option<f64>,
}

/// Params builder.
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

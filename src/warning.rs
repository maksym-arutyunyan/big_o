use crate::model::Model;
use std::fmt;

/// Something about the measurements that weakens the inference without
/// invalidating it.
///
/// A warning never changes which model was chosen. It reports the conditions
/// under which that choice was made, so a result that happens to be right for
/// the wrong reasons can be told apart from one that is supported by the data.
///
/// Non-exhaustive: match with a `_` arm, or just `Display` it. Learning to spot
/// a new way for measurements to mislead is the point of the type, and adding
/// one should not break the callers who already print whatever it finds.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Warning {
    /// The sample has few distinct input sizes. The fit is determined, but
    /// there is little room for the models to disagree, so the winner is
    /// sensitive to a single measurement.
    TooFewPoints {
        /// Distinct input sizes measured.
        got: usize,
        /// Distinct input sizes below which this warning is raised.
        advised: usize,
    },

    /// The input sizes span too narrow a range to separate the models.
    ///
    /// Models are told apart by how fast they grow, so the evidence lives in
    /// the range of input sizes, not the number of points in it. Ten sizes
    /// between 1000 and 1100 look linear whatever produced them.
    NarrowRange {
        /// Decades of input size the sample spans.
        decades: f64,
        /// Decades below which this warning is raised.
        advised: f64,
    },

    /// Cost rises and falls repeatedly across the sample. No monotone model
    /// describes that, so the fit is reporting a trend through noise.
    NonMonotonic,

    /// Cost falls as the input grows. Real for a benchmark whose early
    /// iterations pay a warm-up cost, but it is not asymptotic growth.
    DecreasingCost,

    /// These models could not consume the data and took no part in the
    /// comparison — typically the log-space models given an input size of
    /// zero, or a cost of zero.
    ModelsSkipped(Vec<Model>),
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Warning::TooFewPoints { got, advised } => write!(
                f,
                "only {got} distinct input sizes measured, {advised} or more advised"
            ),
            Warning::NarrowRange { decades, advised } => write!(
                f,
                "input sizes span {decades:.1} decades, {advised:.0} or more advised to separate the models"
            ),
            Warning::NonMonotonic => write!(f, "cost rises and falls across the sample"),
            Warning::DecreasingCost => write!(f, "cost falls as the input grows"),
            Warning::ModelsSkipped(models) => {
                write!(f, "models that could not consume the data: ")?;
                for (i, model) in models.iter().enumerate() {
                    match i {
                        0 => write!(f, "{model}")?,
                        _ => write!(f, ", {model}")?,
                    }
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_as_a_sentence() {
        assert_eq!(
            Warning::TooFewPoints { got: 4, advised: 6 }.to_string(),
            "only 4 distinct input sizes measured, 6 or more advised"
        );
        assert_eq!(
            Warning::NarrowRange {
                decades: 0.4,
                advised: 3.0
            }
            .to_string(),
            "input sizes span 0.4 decades, 3 or more advised to separate the models"
        );
        assert_eq!(
            Warning::ModelsSkipped(vec![Model::Logarithmic, Model::Polynomial]).to_string(),
            "models that could not consume the data: O(log n), O(n^m)"
        );
    }
}

use std::num::NonZeroUsize;

use crate::distributions::t::T_SCORE_TABLE;

#[derive(Debug, Clone, Copy)]
pub struct NumericalSample {
    pub mean: f64,
    pub deviation: f64,
    pub n: usize,
}

impl NumericalSample {
    pub fn standard_error_square(&self) -> f64 {
        self.deviation / (self.n as f64)
    }
}

pub fn one_sample_mean(sample: NumericalSample, mean_0: f64) -> f64 {
    let standard_error = sample.standard_error_square().sqrt();
    let t = (sample.mean - mean_0) / standard_error;
    let df = NonZeroUsize::new(sample.n - 1).unwrap();
    T_SCORE_TABLE.p_value_two_sided(df, t)
}

pub fn difference_of_two_means(
    sample_1: NumericalSample,
    sample_2: NumericalSample,
    mean_0: f64,
) -> f64 {
    let standard_error =
        (sample_1.standard_error_square() + sample_2.standard_error_square()).sqrt();
    let t = (sample_1.mean - sample_2.mean - mean_0) / standard_error;
    let df = usize::min(sample_1.n, sample_2.n) - 1;
    let df = NonZeroUsize::new(df).unwrap();
    T_SCORE_TABLE.p_value_two_sided(df, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_sample_mean() {
        assert!(
            one_sample_mean(
                NumericalSample {
                    mean: 97.32,
                    deviation: 16.98_f64.powi(2),
                    n: 100,
                },
                93.29
            ) < 0.05
        );
    }

    #[test]
    fn test_difference_of_two_means() {
        assert!(
            difference_of_two_means(
                NumericalSample {
                    mean: 7.18,
                    deviation: 1.60_f64.powi(2),
                    n: 100,
                },
                NumericalSample {
                    mean: 6.78,
                    deviation: 1.43_f64.powi(2),
                    n: 50,
                },
                0.
            ) >= 0.05
        );
    }
}

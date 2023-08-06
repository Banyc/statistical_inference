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

pub struct FStatistic {
    pub f: f64,
    pub df_1: usize,
    pub df_2: usize,
}

/// Null hypothesis: all means are equal.
pub fn anova(groups: &[NumericalSample], total: NumericalSample) -> FStatistic {
    let msg = mean_square_between_groups(groups, total);
    let mse = mean_square_error(groups, total);
    let f = msg / mse;
    FStatistic {
        f,
        df_1: groups.len() - 1,
        df_2: total.n - groups.len(),
    }
}

fn mean_square_between_groups(groups: &[NumericalSample], total: NumericalSample) -> f64 {
    let df_g = groups.len() - 1;
    let ssg = sum_of_squares_between_groups(groups, total);
    ssg / df_g as f64
}

fn sum_of_squares_between_groups(groups: &[NumericalSample], total: NumericalSample) -> f64 {
    groups
        .iter()
        .map(|group| {
            let difference = group.mean - total.mean;
            group.n as f64 * difference.powi(2)
        })
        .sum()
}

fn mean_square_error(groups: &[NumericalSample], total: NumericalSample) -> f64 {
    let df_e = total.n - groups.len();
    let sse = sum_of_squared_errors(groups, total);
    sse / df_e as f64
}

fn sum_of_squared_errors(groups: &[NumericalSample], total: NumericalSample) -> f64 {
    let sst = sum_of_squares_total(total);
    let ssg = sum_of_squares_between_groups(groups, total);
    sst - ssg
}

fn sum_of_squares_total(total: NumericalSample) -> f64 {
    total.deviation * (total.n - 1) as f64
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

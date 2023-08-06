use std::num::NonZeroUsize;

use crate::distributions::t::T_SCORE_TABLE;

#[derive(Debug, Clone, Copy)]
pub struct NumericalSample {
    pub mean: f64,
    pub deviation: f64,
    pub n: usize,
}

impl NumericalSample {
    pub fn standard_error_squared(&self) -> f64 {
        self.deviation / (self.n as f64)
    }
}

pub fn one_sample_mean(sample: NumericalSample, mean_0: f64) -> f64 {
    let standard_error = sample.standard_error_squared().sqrt();
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
        (sample_1.standard_error_squared() + sample_2.standard_error_squared()).sqrt();
    let t = (sample_1.mean - sample_2.mean - mean_0) / standard_error;
    let df = usize::min(sample_1.n, sample_2.n) - 1;
    let df = NonZeroUsize::new(df).unwrap();
    T_SCORE_TABLE.p_value_two_sided(df, t)
}

#[derive(Debug, Clone, Copy)]
pub struct FStatistic {
    pub f: f64,
    pub df_1: usize,
    pub df_2: usize,
}

/// Null hypothesis: all means are equal.
pub fn anova(groups: &[NumericalSample]) -> FStatistic {
    let total_n = groups.iter().map(|group| group.n).sum::<usize>();

    let df_g = groups.len() - 1;
    let msg = mean_square_between_groups(groups, total_n, df_g);

    let df_e = total_n - groups.len();
    let mse = mean_square_error(groups, df_e);
    let f = msg / mse;
    FStatistic {
        f,
        df_1: df_g,
        df_2: df_e,
    }
}

fn mean_square_between_groups(groups: &[NumericalSample], total_n: usize, df_g: usize) -> f64 {
    let ssg = sum_of_squares_between_groups(groups, total_n);
    ssg / df_g as f64
}

fn sum_of_squares_between_groups(groups: &[NumericalSample], total_n: usize) -> f64 {
    let total_sum = groups
        .iter()
        .map(|group| group.mean * group.n as f64)
        .sum::<f64>();
    let total_mean = total_sum / total_n as f64;
    groups
        .iter()
        .map(|group| {
            let difference = group.mean - total_mean;
            group.n as f64 * difference.powi(2)
        })
        .sum()
}

fn mean_square_error(groups: &[NumericalSample], df_e: usize) -> f64 {
    let sse = sum_of_squared_errors(groups);
    sse / df_e as f64
}

fn sum_of_squared_errors(groups: &[NumericalSample]) -> f64 {
    groups
        .iter()
        .map(|group| (group.n - 1) as f64 * group.deviation)
        .sum()
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

    #[test]
    fn test_anova() {
        let groups = [
            NumericalSample {
                mean: 85.75,
                deviation: 28.25,
                n: 4,
            },
            NumericalSample {
                mean: 84.,
                deviation: 13.00,
                n: 3,
            },
            NumericalSample {
                mean: 90.2,
                deviation: 15.70,
                n: 5,
            },
        ];
        let f = anova(&groups);
        assert_eq!(f.df_1, 2);
        assert_eq!(f.df_2, 9);
        assert!((f.f - 2.1811).abs() < 0.05);
    }
}

use std::num::NonZeroUsize;

use strict_num::{FiniteF64, NormalizedF64, PositiveF64};

use crate::distributions::{
    f::{FParams, F_CDF},
    normal::Z_SCORE_TABLE,
    t::T_SCORE_TABLE,
};

#[derive(Debug, Clone, Copy)]
pub struct NumericalSample {
    pub mean: FiniteF64,
    pub variance: PositiveF64,
    pub count: NonZeroUsize,
}
impl NumericalSample {
    pub fn standard_error_squared(&self) -> f64 {
        self.variance.get() / (self.count.get() as f64)
    }
}

pub fn one_sample_mean(sample: NumericalSample, mean_0: FiniteF64) -> NormalizedF64 {
    let standard_error = standard_error(&[sample]);
    let t = (sample.mean.get() - mean_0.get()) / standard_error;
    let t = FiniteF64::new(t).unwrap();
    let df = NonZeroUsize::new(sample.count.get() - 1).unwrap();
    T_SCORE_TABLE.p_value_two_sided(df, t)
}

pub fn difference_of_two_means(
    sample_1: NumericalSample,
    sample_2: NumericalSample,
    mean_0: FiniteF64,
) -> NormalizedF64 {
    let standard_error = standard_error(&[sample_1, sample_2]);
    let t = (sample_1.mean.get() - sample_2.mean.get() - mean_0.get()) / standard_error;
    let t = FiniteF64::new(t).unwrap();
    let df = sample_1.count.min(sample_2.count).get() - 1;
    let df = NonZeroUsize::new(df).unwrap();
    T_SCORE_TABLE.p_value_two_sided(df, t)
}

fn standard_error(samples: &[NumericalSample]) -> f64 {
    let standard_error_squared = samples
        .iter()
        .map(|x| x.standard_error_squared())
        .sum::<f64>();
    standard_error_squared.sqrt()
}

/// Determine a proper sample size given the null mean is zero.
///
/// `power`: probability that the alternative hypothesis is not confused as a null hypothesis
///
/// - usually in
///   ```math
///   [0.8, 0.9]
///   ```
pub fn min_count_of_each_of_two_samples(
    mean_a: FiniteF64,
    power: NormalizedF64,
    max_p_value: NormalizedF64,
    deviation_1: PositiveF64,
    deviation_2: PositiveF64,
) -> usize {
    let one_sided_p_value = max_p_value.get() / 2.;
    let one_sided_p_value = NormalizedF64::new(one_sided_p_value).unwrap();
    let power_region_extension = Z_SCORE_TABLE.z(power);
    let reject_region_extension = Z_SCORE_TABLE.z(one_sided_p_value);
    let region = reject_region_extension.get() - power_region_extension.get();
    let count = (deviation_1.get() + deviation_2.get()) / (mean_a.get() / region).powi(2);
    count.ceil() as usize
}

/// Null hypothesis: all means are equal.
pub fn anova(groups: &[NumericalSample]) -> (FParams, NormalizedF64) {
    let total_n = groups.iter().map(|group| group.count.get()).sum::<usize>();

    let df_g = groups.len().checked_sub(1).unwrap();
    let df_g = NonZeroUsize::new(df_g).unwrap();
    let msg = mean_square_between_groups(groups, total_n, df_g);

    let df_e = total_n - groups.len();
    let df_e = NonZeroUsize::new(df_e).unwrap();
    let mse = mean_square_error(groups, df_e);
    let x = msg / mse;
    let f_params = FParams {
        x: PositiveF64::new(x).unwrap(),
        df_1: df_g,
        df_2: df_e,
    };
    (f_params, F_CDF.p_value(f_params))
}

fn mean_square_between_groups(
    groups: &[NumericalSample],
    total_n: usize,
    df_g: NonZeroUsize,
) -> f64 {
    let ssg = sum_of_squares_between_groups(groups, total_n);
    ssg / df_g.get() as f64
}

fn sum_of_squares_between_groups(groups: &[NumericalSample], total_n: usize) -> f64 {
    let total_sum = groups
        .iter()
        .map(|group| group.mean.get() * group.count.get() as f64)
        .sum::<f64>();
    let total_mean = total_sum / total_n as f64;
    groups
        .iter()
        .map(|group| {
            let difference = group.mean.get() - total_mean;
            group.count.get() as f64 * difference.powi(2)
        })
        .sum()
}

fn mean_square_error(groups: &[NumericalSample], df_e: NonZeroUsize) -> f64 {
    let sse = sum_of_squared_errors(groups);
    sse / df_e.get() as f64
}

fn sum_of_squared_errors(groups: &[NumericalSample]) -> f64 {
    groups
        .iter()
        .map(|group| (group.count.get() - 1) as f64 * group.variance.get())
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
                    mean: FiniteF64::new(97.32).unwrap(),
                    variance: PositiveF64::new(16.98_f64.powi(2)).unwrap(),
                    count: NonZeroUsize::new(100).unwrap(),
                },
                FiniteF64::new(93.29).unwrap()
            )
            .get()
                < 0.05
        );
    }

    #[test]
    fn test_difference_of_two_means() {
        assert!(
            difference_of_two_means(
                NumericalSample {
                    mean: FiniteF64::new(7.18).unwrap(),
                    variance: PositiveF64::new(1.60_f64.powi(2)).unwrap(),
                    count: NonZeroUsize::new(100).unwrap(),
                },
                NumericalSample {
                    mean: FiniteF64::new(6.78).unwrap(),
                    variance: PositiveF64::new(1.43_f64.powi(2)).unwrap(),
                    count: NonZeroUsize::new(50).unwrap(),
                },
                FiniteF64::new(0.).unwrap()
            )
            .get()
                >= 0.05
        );
    }

    #[test]
    fn test_proper_sample_size() {
        let power = NormalizedF64::new(0.8).unwrap();
        let mean_a = FiniteF64::new(-3.).unwrap();
        let variance = PositiveF64::new((12.0_f64).powi(2)).unwrap();
        let max_p_value = NormalizedF64::new(0.05).unwrap();
        let count =
            min_count_of_each_of_two_samples(mean_a, power, max_p_value, variance, variance);
        assert_eq!(count, 251);
    }

    #[test]
    fn test_anova() {
        let groups = [
            NumericalSample {
                mean: FiniteF64::new(85.75).unwrap(),
                variance: PositiveF64::new(28.25).unwrap(),
                count: NonZeroUsize::new(4).unwrap(),
            },
            NumericalSample {
                mean: FiniteF64::new(84.).unwrap(),
                variance: PositiveF64::new(13.00).unwrap(),
                count: NonZeroUsize::new(3).unwrap(),
            },
            NumericalSample {
                mean: FiniteF64::new(90.2).unwrap(),
                variance: PositiveF64::new(15.70).unwrap(),
                count: NonZeroUsize::new(5).unwrap(),
            },
        ];
        let (f, p) = anova(&groups);
        assert_eq!(f.df_1.get(), 2);
        assert_eq!(f.df_2.get(), 9);
        assert!((f.x.get() - 2.1811).abs() < 0.05);
        assert!((p.get() - 0.1689).abs() < 0.05);
    }
}

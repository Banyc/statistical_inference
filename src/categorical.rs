use std::num::NonZeroUsize;

use crate::{
    distributions::{chi_square::CHI_SQUARE_TABLE, normal::Z_SCORE_TABLE},
    NonNegR, UnitR, R,
};

#[derive(Debug, Copy, Clone)]
pub struct CountAndProportion {
    pub count: usize,
    pub proportion: UnitR<f64>,
}
impl CountAndProportion {
    pub fn is_normally_distributed_enough(&self) -> bool {
        let a = 10. <= self.count as f64 * self.proportion.get();
        let b = 10. <= self.count as f64 * (1. - self.proportion.get());
        a && b
    }

    pub fn standard_error_squared(&self) -> f64 {
        self.proportion.get() * (1. - self.proportion.get()) / self.count as f64
    }
}

pub fn one_proportion(sample: CountAndProportion, p_0: UnitR<f64>) -> UnitR<f64> {
    // Normality check
    assert!(sample.is_normally_distributed_enough());

    let standard_error = standard_error(&[CountAndProportion {
        count: sample.count,
        proportion: p_0,
    }]);
    let z = (sample.proportion.get() - p_0.get()) / standard_error;
    let z = R::new(z).unwrap();
    Z_SCORE_TABLE.p_value_two_sided(z)
}

pub fn difference_of_two_proportions(
    sample_1: CountAndProportion,
    sample_2: CountAndProportion,
    p_0: UnitR<f64>,
) -> UnitR<f64> {
    // Normality check
    assert!(sample_1.is_normally_distributed_enough());
    assert!(sample_2.is_normally_distributed_enough());

    let standard_error = standard_error(&[sample_1, sample_2]);
    let z = ((sample_1.proportion.get() - sample_2.proportion.get()) - p_0.get()) / standard_error;
    let z = R::new(z).unwrap();
    Z_SCORE_TABLE.p_value_two_sided(z)
}

fn standard_error(samples: &[CountAndProportion]) -> f64 {
    let standard_error_squared = samples
        .iter()
        .map(|x| x.standard_error_squared())
        .sum::<f64>();
    standard_error_squared.sqrt()
}

/// Determine a proper sample size given the null proportion is zero.
///
/// `power`: probability that the alternative hypothesis is not confused as a null hypothesis
///
/// - usually in
///   ```math
///   [0.8, 0.9]
///   ```
pub fn min_count_of_each_of_two_samples(
    proportion_1: UnitR<f64>,
    proportion_2: UnitR<f64>,
    p_0: UnitR<f64>,
    power: UnitR<f64>,
    max_p_value: UnitR<f64>,
) -> usize {
    let one_sided_p_value = max_p_value.get() / 2.;
    let one_sided_p_value = UnitR::new(one_sided_p_value).unwrap();
    let power_region_extension = Z_SCORE_TABLE.z(power);
    let reject_region_extension = Z_SCORE_TABLE.z(one_sided_p_value);
    let region = reject_region_extension.get() - power_region_extension.get();
    let error = proportion_1.get() * (1. - proportion_1.get())
        + proportion_2.get() * (1. - proportion_2.get());
    let diff = proportion_1.get() - proportion_2.get() - p_0.get();
    let count = error / (diff / region).powi(2);
    count.ceil() as usize
}

#[derive(Debug, Copy, Clone)]
pub struct CountAndExpect {
    pub count: usize,
    pub expect: NonNegR<f64>,
}
impl CountAndExpect {
    pub fn z_squared(&self) -> f64 {
        let standard_error_squared = self.expect.get();
        (self.count as f64 - self.expect.get()).powi(2) / standard_error_squared
    }
}

/// Null hypothesis: counts from each column is equal to their expected counts respectively
pub fn fitness(catagories: &[CountAndExpect]) -> UnitR<f64> {
    let df = NonZeroUsize::new(catagories.len() - 1).unwrap();

    // Normality check
    catagories
        .iter()
        .for_each(|bin| assert!(bin.expect.get() >= 5.));

    let chi_square = catagories.iter().map(|bin| bin.z_squared()).sum();
    CHI_SQUARE_TABLE.p_value(df, chi_square)
}

/// Null hypothesis: the two variables are independent of each other
pub fn two_way_table_independence<const R: usize, const C: usize>(
    matrix: &[[usize; C]; R],
) -> UnitR<f64> {
    assert!(R >= 2);
    assert!(C >= 2);

    let mut row_total = [0; R];
    let mut col_total = [0; C];
    let mut table_total = 0;
    (0..R).for_each(|r| {
        (0..C).for_each(|c| {
            let cell = matrix[r][c];
            row_total[r] += cell;
            col_total[c] += cell;
            table_total += cell;
        });
    });

    let mut expect = [[NonNegR::new(0.).unwrap(); C]; R];
    (0..R).for_each(|r| {
        (0..C).for_each(|c| {
            let cell_expect = (row_total[r] * col_total[c]) as f64 / table_total as f64;

            // Normality check
            assert!(cell_expect >= 5.);

            expect[r][c] = NonNegR::new(cell_expect).unwrap();
        });
    });

    let df = NonZeroUsize::new((R - 1) * (C - 1)).unwrap();

    let mut chi_square = 0.;
    (0..R).for_each(|r| {
        (0..C).for_each(|c| {
            let bin = CountAndExpect {
                count: matrix[r][c],
                expect: expect[r][c],
            };
            chi_square += bin.z_squared();
        });
    });

    CHI_SQUARE_TABLE.p_value(df, chi_square)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_proportion() {
        let sample = CountAndProportion {
            count: 1000,
            proportion: UnitR::new(0.37).unwrap(),
        };
        let p_0 = UnitR::new(0.5).unwrap();
        assert!(one_proportion(sample, p_0).get() < 0.05);
    }

    #[test]
    fn test_difference_of_two_proportions() {
        let sample_1 = CountAndProportion {
            count: 500 + 44425,
            proportion: UnitR::new(500. / (500 + 44425) as f64).unwrap(),
        };
        let sample_2 = CountAndProportion {
            count: 505 + 44405,
            proportion: UnitR::new(505. / (505 + 44405) as f64).unwrap(),
        };
        let p_0 = UnitR::new(0.).unwrap();
        assert!(difference_of_two_proportions(sample_1, sample_2, p_0).get() > 0.05);

        let sample_1 = CountAndProportion {
            count: 1000,
            proportion: UnitR::new(0.958).unwrap(),
        };
        let sample_2 = CountAndProportion {
            count: 1000,
            proportion: UnitR::new(0.899).unwrap(),
        };
        let p_0 = UnitR::new(0.03).unwrap();
        assert!(difference_of_two_proportions(sample_1, sample_2, p_0).get() < 0.05);
    }

    #[test]
    fn test_proper_sample_size() {
        let proportion_1 = 500. / (500 + 44425) as f64;
        let proportion_2 = 505. / (505 + 44405) as f64;
        let proportion_1 = UnitR::new(proportion_1).unwrap();
        let proportion_2 = UnitR::new(proportion_2).unwrap();
        let p_0 = 0.;
        let p_0 = UnitR::new(p_0).unwrap();
        let power = UnitR::new(0.8).unwrap();
        let max_p_value = UnitR::new(0.05).unwrap();
        let count =
            min_count_of_each_of_two_samples(proportion_1, proportion_2, p_0, power, max_p_value);
        println!("{count}");

        let proportion_1 = 0.958;
        let proportion_2 = 0.899;
        let proportion_1 = UnitR::new(proportion_1).unwrap();
        let proportion_2 = UnitR::new(proportion_2).unwrap();
        let p_0 = 0.03;
        let p_0 = UnitR::new(p_0).unwrap();
        let power = UnitR::new(0.8).unwrap();
        let max_p_value = UnitR::new(0.05).unwrap();
        let count =
            min_count_of_each_of_two_samples(proportion_1, proportion_2, p_0, power, max_p_value);
        println!("{count}");
    }

    #[test]
    fn test_fitness() {
        let bins = [
            CountAndExpect {
                count: 205,
                expect: NonNegR::new(198.).unwrap(),
            },
            CountAndExpect {
                count: 26,
                expect: NonNegR::new(19.25).unwrap(),
            },
            CountAndExpect {
                count: 25,
                expect: NonNegR::new(33.).unwrap(),
            },
            CountAndExpect {
                count: 19,
                expect: NonNegR::new(24.75).unwrap(),
            },
        ];
        assert!(fitness(&bins).get() > 0.05);
    }

    #[test]
    fn test_two_way_table_independence() {
        let matrix = [
            [2, 23, 36],  //
            [71, 50, 37], //
        ];
        assert!(two_way_table_independence(&matrix).get() < 0.05);
    }
}

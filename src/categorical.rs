use std::num::NonZeroUsize;

use strict_num::{NormalizedF64, PositiveF64};

use crate::distributions::{chi_square::CHI_SQUARE_TABLE, normal::Z_SCORE_TABLE};

#[derive(Debug, Copy, Clone)]
pub struct CountAndProportion {
    pub count: usize,
    pub proportion: NormalizedF64,
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

pub fn one_proportion(sample: CountAndProportion, p_0: NormalizedF64) -> NormalizedF64 {
    // Normality check
    assert!(sample.is_normally_distributed_enough());

    let standard_error = standard_error(&[CountAndProportion {
        count: sample.count,
        proportion: p_0,
    }]);
    let z = (sample.proportion.get() - p_0.get()) / standard_error;
    Z_SCORE_TABLE.p_value_two_sided(z)
}

pub fn difference_of_two_proportions(
    sample_1: CountAndProportion,
    sample_2: CountAndProportion,
    p_0: NormalizedF64,
) -> NormalizedF64 {
    // Normality check
    assert!(sample_1.is_normally_distributed_enough());
    assert!(sample_2.is_normally_distributed_enough());

    let standard_error = standard_error(&[sample_1, sample_2]);
    let z = ((sample_1.proportion.get() - sample_2.proportion.get()) - p_0.get()) / standard_error;
    Z_SCORE_TABLE.p_value_two_sided(z)
}

fn standard_error(samples: &[CountAndProportion]) -> f64 {
    let standard_error_squared = samples
        .iter()
        .map(|x| x.standard_error_squared())
        .sum::<f64>();
    standard_error_squared.sqrt()
}

#[derive(Debug, Copy, Clone)]
pub struct CountAndExpect {
    pub count: usize,
    pub expect: PositiveF64,
}
impl CountAndExpect {
    pub fn z_squared(&self) -> f64 {
        let standard_error_squared = self.expect.get();
        (self.count as f64 - self.expect.get()).powi(2) / standard_error_squared
    }
}

/// Null hypothesis: counts from each column is equal to their expected counts respectively
pub fn fitness(catagories: &[CountAndExpect]) -> NormalizedF64 {
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
) -> NormalizedF64 {
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

    let mut expect = [[PositiveF64::new(0.).unwrap(); C]; R];
    (0..R).for_each(|r| {
        (0..C).for_each(|c| {
            let cell_expect = (row_total[r] * col_total[c]) as f64 / table_total as f64;

            // Normality check
            assert!(cell_expect >= 5.);

            expect[r][c] = PositiveF64::new(cell_expect).unwrap();
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
            proportion: NormalizedF64::new(0.37).unwrap(),
        };
        let p_0 = NormalizedF64::new(0.5).unwrap();
        assert!(one_proportion(sample, p_0).get() < 0.05);
    }

    #[test]
    fn test_difference_of_two_proportions() {
        let sample_1 = CountAndProportion {
            count: 500 + 44425,
            proportion: NormalizedF64::new(500. / (500 + 44425) as f64).unwrap(),
        };
        let sample_2 = CountAndProportion {
            count: 505 + 44405,
            proportion: NormalizedF64::new(505. / (505 + 44405) as f64).unwrap(),
        };
        let p_0 = NormalizedF64::new(0.).unwrap();
        assert!(difference_of_two_proportions(sample_1, sample_2, p_0).get() > 0.05);

        let sample_1 = CountAndProportion {
            count: 1000,
            proportion: NormalizedF64::new(0.958).unwrap(),
        };
        let sample_2 = CountAndProportion {
            count: 1000,
            proportion: NormalizedF64::new(0.899).unwrap(),
        };
        let p_0 = NormalizedF64::new(0.03).unwrap();
        assert!(difference_of_two_proportions(sample_1, sample_2, p_0).get() < 0.05);
    }

    #[test]
    fn test_fitness() {
        let bins = [
            CountAndExpect {
                count: 205,
                expect: PositiveF64::new(198.).unwrap(),
            },
            CountAndExpect {
                count: 26,
                expect: PositiveF64::new(19.25).unwrap(),
            },
            CountAndExpect {
                count: 25,
                expect: PositiveF64::new(33.).unwrap(),
            },
            CountAndExpect {
                count: 19,
                expect: PositiveF64::new(24.75).unwrap(),
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

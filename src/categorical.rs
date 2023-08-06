use std::num::NonZeroUsize;

use crate::distributions::{chi_square::CHI_SQUARE_TABLE, normal::Z_SCORE_TABLE};

pub fn one_proportion(p_hat: f64, n: usize, p_0: f64) -> f64 {
    assert!(p_hat <= 1.);
    assert!(p_hat >= 0.);
    assert!(p_0 <= 1.);
    assert!(p_0 >= 0.);

    // Normality check
    assert!(p_hat * n as f64 >= 10.);
    assert!((1. - p_hat) * n as f64 >= 10.);

    let standard_error = ((p_0 * (1. - p_0)) / n as f64).sqrt();
    let z = (p_hat - p_0) / standard_error;
    Z_SCORE_TABLE.p_value_two_sided(z)
}

pub fn difference_of_two_proportions(
    p_hat_1: f64,
    n_1: usize,
    p_hat_2: f64,
    n_2: usize,
    p_0: f64,
) -> f64 {
    assert!(p_hat_1 <= 1.);
    assert!(p_hat_1 >= 0.);
    assert!(p_hat_2 <= 1.);
    assert!(p_hat_2 >= 0.);
    assert!(p_0 <= 1.);
    assert!(p_0 >= 0.);

    // Normality check
    assert!(p_hat_1 * n_1 as f64 >= 10.);
    assert!((1. - p_hat_1) * n_1 as f64 >= 10.);
    assert!(p_hat_2 * n_2 as f64 >= 10.);
    assert!((1. - p_hat_2) * n_2 as f64 >= 10.);

    let standard_error = (((p_hat_1 * (1. - p_hat_1)) / n_1 as f64)
        + ((p_hat_2 * (1. - p_hat_2)) / n_2 as f64))
        .sqrt();
    let z = ((p_hat_1 - p_hat_2) - p_0) / standard_error;
    Z_SCORE_TABLE.p_value_two_sided(z)
}

#[derive(Debug, Copy, Clone)]
pub struct CountAndExpect {
    pub count: usize,
    pub expect: f64,
}

/// Null hypothesis: counts from each column is equal to their expected counts respectively
pub fn fitness(catagories: &[CountAndExpect]) -> f64 {
    let df = NonZeroUsize::new(catagories.len() - 1).unwrap();

    // Normality check
    catagories.iter().for_each(|bin| assert!(bin.expect >= 5.));

    let chi_square = catagories
        .iter()
        .map(|bin| {
            let standard_error_square = bin.expect;
            let z_square = (bin.count as f64 - bin.expect).powi(2) / standard_error_square;
            #[allow(clippy::let_and_return)]
            z_square
        })
        .sum();
    CHI_SQUARE_TABLE.p_value(df, chi_square)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_proportion() {
        assert!(one_proportion(0.37, 1000, 0.5) < 0.05);
    }

    #[test]
    fn test_difference_of_two_proportions() {
        assert!(
            difference_of_two_proportions(
                500. / (500 + 44425) as f64,
                500 + 44425,
                505. / (505 + 44405) as f64,
                505 + 44405,
                0.
            ) > 0.05
        );
        assert!(difference_of_two_proportions(0.958, 1000, 0.899, 1000, 0.03) < 0.05);
    }

    #[test]
    fn test_fitness() {
        let bins = [
            CountAndExpect {
                count: 205,
                expect: 198.,
            },
            CountAndExpect {
                count: 26,
                expect: 19.25,
            },
            CountAndExpect {
                count: 25,
                expect: 33.,
            },
            CountAndExpect {
                count: 19,
                expect: 24.75,
            },
        ];
        assert!(fitness(&bins) > 0.05);
    }
}

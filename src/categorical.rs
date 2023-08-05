use crate::distributions::normal::Z_SCORE_TABLE;

pub fn one_proportion(p_hat: f64, n: usize, p_0: f64) -> f64 {
    assert!(p_hat <= 1.);
    assert!(p_hat >= 0.);
    assert!(p_0 <= 1.);
    assert!(p_0 >= 0.);
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
}

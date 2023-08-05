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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_proportion() {
        assert!(one_proportion(0.37, 1000, 0.5) < 0.05);
    }
}

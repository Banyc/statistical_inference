use std::num::NonZeroUsize;

use crate::distributions::t::T_SCORE_TABLE;

pub fn one_sample_mean(mean: f64, standard_deviation: f64, n: usize, mean_0: f64) -> f64 {
    let standard_error = standard_deviation / (n as f64).sqrt();
    let t = (mean - mean_0) / standard_error;
    let df = NonZeroUsize::new(n - 1).unwrap();
    T_SCORE_TABLE.p_value_two_sided(df, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_sample_mean() {
        assert!(one_sample_mean(97.32, 16.98, 100, 93.29) < 0.05);
    }
}

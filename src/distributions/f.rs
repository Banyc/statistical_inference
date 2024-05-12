use std::rc::Rc;

use once_cell::sync::Lazy;
use reikna::func;
use reikna::func::Function;
use reikna::integral::integrate_wp;

pub static F_CDF: Lazy<FCdf> = Lazy::new(Default::default);

pub struct FCdf {}

impl FCdf {
    pub fn new() -> Self {
        Self {}
    }

    pub fn p_value(&self, df_1: usize, df_2: usize, f: f64) -> f64 {
        // ref:
        // - <https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm>
        // - <https://en.wikipedia.org/wiki/F-distribution>
        // - <https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function>
        let df_1 = df_1 as f64;
        let df_2 = df_2 as f64;
        let x = (df_1 * f) / (df_1 * f + df_2);
        let i = incomplete_beta_function(x, df_1 / 2., df_2 / 2.);
        1. - i
    }
}

impl Default for FCdf {
    fn default() -> Self {
        Self::new()
    }
}

fn incomplete_beta_function(x: f64, a: f64, b: f64) -> f64 {
    let f = func!(move |t: f64| t.powf(a - 1.) * (1. - t).powf(b - 1.));

    let numerator = integrate_wp(&f, 0., x, 10);
    let denominator = integrate_wp(&f, 0., 1., 10);
    numerator / denominator
}

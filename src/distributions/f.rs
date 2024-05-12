use std::{num::NonZeroUsize, rc::Rc};

use once_cell::sync::Lazy;
use reikna::func;
use reikna::func::Function;
use reikna::integral::integrate_wp;
use strict_num::{NormalizedF64, PositiveF64};

pub static F_CDF: Lazy<FCdf> = Lazy::new(Default::default);

pub struct FCdf {}
impl FCdf {
    pub fn new() -> Self {
        Self {}
    }

    pub fn p_value(&self, params: FParams) -> NormalizedF64 {
        // ref:
        // - <https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm>
        // - <https://en.wikipedia.org/wiki/F-distribution>
        // - <https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function>
        let df_1 = params.df_1.get() as f64;
        let df_2 = params.df_2.get() as f64;
        let x = params.x.get();
        let x = (df_1 * x) / (df_1 * x + df_2);
        let x = NormalizedF64::new(x).unwrap();
        let i = incomplete_beta_function(x, df_1 / 2., df_2 / 2.);
        NormalizedF64::new(1. - i.get()).unwrap()
    }
}
impl Default for FCdf {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FParams {
    pub x: PositiveF64,
    pub df_1: NonZeroUsize,
    pub df_2: NonZeroUsize,
}

fn incomplete_beta_function(x: NormalizedF64, a: f64, b: f64) -> NormalizedF64 {
    let f = func!(move |t: f64| t.powf(a - 1.) * (1. - t).powf(b - 1.));

    let numerator = integrate_wp(&f, 0., x.get(), 10);
    let denominator = integrate_wp(&f, 0., 1., 10);
    NormalizedF64::new(numerator / denominator).unwrap()
}

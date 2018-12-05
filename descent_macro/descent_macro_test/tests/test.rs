#![feature(proc_macro_hygiene)]

use descent_macro::expr;
use descent::expr::{Var, Store};

// Could wrap 3 lambdas into one big function. Compilier might be able to
// optimise some common instructions.
// Might just want the 2 functions: f and combined.

// Provide a separate calculation for constant entries, or at the minimum
// indicate if the entire first or second derivative is constant. If have
// one mega-function then might not be much benefit to keeping track if
// something is constant or not.

// Could either have one struct with boxed closures, or distinct type for
// each expression and trait to interface. Second can potentially avoid some
// dereferencing... If have single mega-function then probably best to box
// instead as possible benfit is reduced.

struct ExprStatic<F, D1, D2, A>
where F: Fn(&[f64], &[f64]) -> f64,
      D1: Fn(&[f64], &[f64], &mut[f64]),
      D2: Fn(&[f64], &[f64], &mut[f64]),
      A: Fn(&[f64], &[f64], &mut[f64], &mut[f64]) -> f64,
{
    f: F,
    d1: D1,
    d2: D2,
    all: A,
    d1_sparsity: Vec<Var>,
    d2_sparsity: Vec<(Var, Var)>,
}

trait Usable {
    fn eval(&self, v: &[f64], p: &[f64]) -> f64;
    fn deriv1(&self, v: &[f64], p: &[f64], d: &mut[f64]);
    fn deriv2(&self, v: &[f64], p: &[f64], d: &mut[f64]);
    fn combined(&self, v: &[f64], p: &[f64], d1: &mut[f64], d2: &mut[f64]) -> f64;
}

impl<F, D1, D2, A> Usable for ExprStatic<F, D1, D2, A>
where F: Fn(&[f64], &[f64]) -> f64,
      D1: Fn(&[f64], &[f64], &mut[f64]),
      D2: Fn(&[f64], &[f64], &mut[f64]),
      A: Fn(&[f64], &[f64], &mut[f64], &mut[f64]) -> f64,
{
    fn eval(&self, v: &[f64], p: &[f64]) -> f64 {
        (self.f)(v, p)
    }

    fn deriv1(&self, v: &[f64], p: &[f64], d: &mut[f64]) {
        (self.d1)(v, p, d);
    }

    fn deriv2(&self, v: &[f64], p: &[f64], d: &mut[f64]) {
        (self.d2)(v, p, d);
    }

    fn combined(&self, v: &[f64], p: &[f64], d1: &mut[f64], d2: &mut[f64]) -> f64 {
        (self.all)(v, p, d1, d2)
    }
}

#[test]
fn it_works() {
    let b = 10.0;

    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(5.0);
    let k = s.add_par(20.0);

    let vars = vec![x, y];
    let ex = expr!(x - y * b; x = vars[1], y = vars[0];); // switching vars
    assert!(ex.d1_sparsity[0] == y);
    assert!(ex.d1_sparsity[1] == x);

    let ex = expr!((x * x - y) * b + k; x, y; k);

    assert!(ex.eval(s.vars.as_slice(), s.pars.as_slice()) == -20.0);

    let mut d1_out = vec![0.0, 0.0];

    ex.deriv1(s.vars.as_slice(), s.pars.as_slice(), &mut d1_out);
    assert!(d1_out[0] == 20.0);
    assert!(d1_out[1] == -10.0);
    assert!(ex.d1_sparsity.len() == 2);
    assert!(ex.d1_sparsity[0] == x);
    assert!(ex.d1_sparsity[1] == y);

    let mut d2_out = vec![0.0];

    ex.deriv2(s.vars.as_slice(), s.pars.as_slice(), &mut d2_out);
    assert!(d2_out[0] == 20.0);
    assert!(ex.d2_sparsity.len() == 1);
    assert!(ex.d2_sparsity[0] == (x, x));

    let mut d1_out = vec![0.0, 0.0];
    let mut d2_out = vec![0.0];

    let v = ex.combined(s.vars.as_slice(), s.pars.as_slice(), &mut d1_out, &mut d2_out);
    assert!(v == -20.0);
    assert!(d1_out[0] == 20.0);
    assert!(d1_out[1] == -10.0);
    assert!(d2_out[0] == 20.0);
}

#![feature(proc_macro_hygiene)]

use descent_macro::expr;
use descent::expr::{Var, Store};

// Hrmm either need to pass through v and p to closures, or could capture them
// directly if simple identify and then do a x.0 to directly embed value in
// closure (Var and Par get copied in).
// I think on implementation this is basically the same as closures get impl
// as a struct, but good side is don't have to have extra array to v and p to
// index into for the second case (but might capture too much if not using a
// basic ident (work around with something like let x = xs[i] outside;).
// Actually these lets are not so bad, should perhaps remove x = x[i] syntax as
// part of macro... Or actually could implement this inside a block internally!

//struct ExprExpr {
//    f: F: Rc<Fn(&[f64], &[f64]) -> f64>,
//    d1: D1: Rc<Fn(&[f64], &[f64], &mut[f64])>,
//    v: Vec<Var>,
//    p: Vec<Par>,
//}
//
//// Enable constructing multiple from generator
//struct ExprGenerator {
//    f: F: Rc<Fn(&[f64], &[f64]) -> f64>,
//    d1: D1: Rc<Fn(&[f64], &[f64], &mut[f64])>,
//}
//
//impl ExprGenerator {
//    fn bind(&self, v: Vec<Var>, p: Vec<Par>) -> ExprExpr {
//        ExprExpr { f: self.f.clone(), d1: self.d1.clone(), v, p }
//    }
//}

// Could either have one struct with boxed closures, or distinct type for
// each expression and trait to interface. Second can potentially avoid some
// dereferencing...

struct ExprFuncs<F, D1>
where F: Fn(&[f64], &[f64]) -> f64,
      D1: Fn(&[f64], &[f64], &mut[f64]),
{
    f: F,
    d1: D1,
    //d1_sparsity: Vec<Var>,
}

trait Usable {
    fn eval(&self, v: &[f64], p: &[f64]) -> f64;
    fn deriv1(&self, v: &[f64], p: &[f64], d: &mut[f64]);
}

impl<F, D1> Usable for ExprFuncs<F, D1>
where F: Fn(&[f64], &[f64]) -> f64,
      D1: Fn(&[f64], &[f64], &mut[f64]),
{
    fn eval(&self, v: &[f64], p: &[f64]) -> f64 {
        (self.f)(v, p)
    }

    fn deriv1(&self, v: &[f64], p: &[f64], d: &mut[f64]) {
        (self.d1)(v, p, d);
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
    let _v = expr!(x - y * b; x = vars[0], y = vars[1];);

    let ex = expr!((x * x - y) * b + k; x, y; k);

    let mut d1_out = vec![0.0, 0.0];

    assert!(ex.eval(s.vars.as_slice(), s.pars.as_slice()) == -20.0);
    ex.deriv1(s.vars.as_slice(), s.pars.as_slice(), &mut d1_out);
    assert!(d1_out[0] == 20.0);
    assert!(d1_out[1] == -10.0);

    //let gen = expr_gen!(y + t + k * 50.0; y, t; k);
    //let mut cons = Vec::new();
    //cons.push(gen.bind(vec![y[0], s[0]], vec![p[0]]));
    //cons.push(gen.bind(vec![y[1], s[1]], vec![p[1]]));
    //cons.push(gen.bind(vec![y[2], s[2]], vec![p[2]]));
    //cons.push(gen.bind(vec![y[3], s[3]], vec![p[3]]));
}

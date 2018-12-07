#![feature(proc_macro_hygiene)]

use descent::expr::Store;
use descent_macro::expr;

#[test]
fn it_works() {
    let b = 10.0;

    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(5.0);
    let k = s.add_par(20.0);

    let vars = vec![x, y];
    let ex = expr!(x - y * b; x = vars[1], y = vars[0]); // switching vars
    assert!(ex.d1_sparsity[0] == y);
    assert!(ex.d1_sparsity[1] == x);

    let ex = expr!((x * x - y) * b + k; x, y; k);

    assert!((ex.f)(s.vars.as_slice(), s.pars.as_slice()) == -20.0);

    let mut d1_out = vec![0.0, 0.0];

    (ex.d1)(s.vars.as_slice(), s.pars.as_slice(), &mut d1_out);
    assert!(d1_out[0] == 20.0);
    assert!(d1_out[1] == -10.0);
    assert!(ex.d1_sparsity.len() == 2);
    assert!(ex.d1_sparsity[0] == x);
    assert!(ex.d1_sparsity[1] == y);

    let mut d2_out = vec![0.0];

    (ex.d2)(s.vars.as_slice(), s.pars.as_slice(), &mut d2_out);
    assert!(d2_out[0] == 20.0);
    assert!(ex.d2_sparsity.len() == 1);
    assert!(ex.d2_sparsity[0] == (x, x));

    let mut d1_out = vec![0.0, 0.0];
    let mut d2_out = vec![0.0];

    let v = (ex.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == -20.0);
    assert!(d1_out[0] == 20.0);
    assert!(d1_out[1] == -10.0);
    assert!(d2_out[0] == 20.0);
}

#[test]
fn cos() {
    let mut s = Store::new();
    let x = s.add_var(0.0);

    let ex = expr!(x.cos(); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (ex.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 1.0);
    assert!(d1_out[0] == 0.0);
    assert!(d2_out[0] == -1.0);
}

#[test]
fn sin() {
    let mut s = Store::new();
    let x = s.add_var(0.0);

    let ex = expr!(x.sin(); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (ex.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 0.0);
    assert!(d1_out[0] == 1.0);
    assert!(d2_out[0] == 0.0);
}

// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(proc_macro_hygiene)]

use descent::expr::Store;
use descent_macro::expr;

#[test]
fn scoped_declarations() {
    let b = 10.0;

    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(5.0);
    let k = s.add_par(20.0);

    let vars = vec![x, y];
    let e = expr!(x - y * b; x = vars[1], y = vars[0]); // switching vars
    assert!(e.d1_sparsity[0] == y);
    assert!(e.d1_sparsity[1] == x);

    let e = expr!((x * x - y) * b + k; x, y; k);

    let mut d1_out = vec![0.0, 0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == -20.0);
    assert!(d1_out[0] == 20.0);
    assert!(d1_out[1] == -10.0);
    assert!(d2_out[0] == 20.0);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d1_sparsity[0] == x);
    assert!(e.d1_sparsity[1] == y);
    assert!(e.d2_sparsity.len() == 1);
    assert!(e.d2_sparsity[0] == (x, x));
}

#[test]
fn cos() {
    let mut s = Store::new();
    let x = s.add_var(0.0);

    let e = expr!(x.cos(); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
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

    let e = expr!(x.sin(); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 0.0);
    assert!(d1_out[0] == 1.0);
    assert!(d2_out[0] == 0.0);
}

#[test]
fn powi() {
    let mut s = Store::new();
    let x = s.add_var(5.0);

    let e = expr!(x.powi(0); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 1.0);
    assert!(d1_out[0] == 0.0);
    assert!(d2_out[0] == 0.0);

    let e = expr!(x.powi(1); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 5.0);
    assert!(d1_out[0] == 1.0);
    assert!(d2_out[0] == 0.0);

    let e = expr!(x.powi(2); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 25.0);
    assert!(d1_out[0] == 10.0);
    assert!(d2_out[0] == 2.0);

    let e = expr!(x.powi(3); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 125.0);
    assert!(d1_out[0] == 75.0);
    assert!(d2_out[0] == 30.0);

    let e = expr!(x.powi(-1); x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 1.0 / 5.0);
    assert!(d1_out[0] == -1.0 / 25.0);
    assert!(d2_out[0] == 2.0 / 125.0);
}

#[test]
fn neg() {
    let mut s = Store::new();
    let x = s.add_var(5.0);
    let e = expr!(-x; x);

    let mut d1_out = vec![0.0];
    let mut d2_out = vec![];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == -5.0);
    assert!(d1_out[0] == -1.0);
}

#[test]
fn add() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(2.0);

    let e = expr!(x + y; x, y);

    let mut d1_out = vec![0.0, 0.0];
    let mut d2_out = vec![];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 3.0);
    assert!(d1_out[0] == 1.0);
    assert!(d1_out[1] == 1.0);
}

#[test]
fn mul() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(2.0);

    let e = expr!(x * y; x, y);

    let mut d1_out = vec![0.0, 0.0];
    let mut d2_out = vec![0.0];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == 2.0);
    assert!(d1_out[0] == 2.0);
    assert!(d1_out[1] == 1.0);
    assert!(d2_out[0] == 1.0);
}

#[test]
fn sub() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(2.0);

    let e = expr!(x - y; x, y);

    let mut d1_out = vec![0.0, 0.0];
    let mut d2_out = vec![];

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert!(v == -1.0);
    assert!(d1_out[0] == 1.0);
    assert!(d1_out[1] == -1.0);
}


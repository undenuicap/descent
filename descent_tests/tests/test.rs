// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(proc_macro_hygiene)]

macro_rules! assert_near {
    ( $a:expr, $b:expr, $c:expr ) => {
        assert!(($a - $b).abs() <= $c);
    };
}

use descent::expr::Store;
use descent_macro::expr;

fn prepare_outputs(e: &descent::expr::fixed::ExprFix) -> (Vec<f64>, Vec<f64>) {
    let mut d1_out = Vec::new();
    let mut d2_out = Vec::new();
    d1_out.resize(e.d1_sparsity.len(), 0.0);
    d2_out.resize(e.d2_sparsity.len(), 0.0);
    (d1_out, d2_out)
}

#[test]
fn cos() {
    let mut s = Store::new();
    let x = s.add_var(0.0);

    let e = expr!(x.cos(); x);
    assert!(e.d1_sparsity.len() == 1);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 1.0, 1e-8);
    assert_near!(d1_out[0], 0.0, 1e-8);
    assert_near!(d2_out[0], -1.0, 1e-8);
}

#[test]
fn sin() {
    let mut s = Store::new();
    let x = s.add_var(0.0);

    let e = expr!(x.sin(); x);
    assert!(e.d1_sparsity.len() == 1);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 0.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d2_out[0], 0.0, 1e-8);
}

#[test]
fn powi() {
    let mut s = Store::new();
    let x = s.add_var(5.0);

    let e = expr!(x.powi(0); x);
    assert!(e.d1_sparsity.len() == 0);
    assert!(e.d2_sparsity.len() == 0);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 1.0, 1e-8);

    let e = expr!(x.powi(1); x);
    assert!(e.d1_sparsity.len() == 1);
    assert!(e.d2_sparsity.len() == 0);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 5.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);

    let e = expr!(x.powi(2); x);
    assert!(e.d1_sparsity.len() == 1);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 25.0, 1e-8);
    assert_near!(d1_out[0], 10.0, 1e-8);
    assert_near!(d2_out[0], 2.0, 1e-8);

    let e = expr!(x.powi(3); x);
    assert!(e.d1_sparsity.len() == 1);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 125.0, 1e-8);
    assert_near!(d1_out[0], 75.0, 1e-8);
    assert_near!(d2_out[0], 30.0, 1e-8);

    let e = expr!(x.powi(-1); x);
    assert!(e.d1_sparsity.len() == 1);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 1.0 / 5.0, 1e-8);
    assert_near!(d1_out[0], -1.0 / 25.0, 1e-8);
    assert_near!(d2_out[0], 2.0 / 125.0, 1e-8);
}

#[test]
fn neg() {
    let mut s = Store::new();
    let x = s.add_var(5.0);
    let e = expr!(-x; x);
    assert!(e.d1_sparsity.len() == 1);
    assert!(e.d2_sparsity.len() == 0);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, -5.0, 1e-8);
    assert_near!(d1_out[0], -1.0, 1e-8);
}

#[test]
fn add() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(2.0);

    let e = expr!(x + y; x, y);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 0);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 3.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d1_out[1], 1.0, 1e-8);
}

#[test]
fn mul() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(2.0);

    let e = expr!(x * y; x, y);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 2.0, 1e-8);
    assert_near!(d1_out[0], 2.0, 1e-8);
    assert_near!(d1_out[1], 1.0, 1e-8);
    assert_near!(d2_out[0], 1.0, 1e-8);
}

#[test]
fn sub() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(2.0);

    let e = expr!(x - y; x, y);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, -1.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d1_out[1], -1.0, 1e-8);
}

#[test]
fn scoped_declarations() {
    let b = 10.0;

    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(5.0);
    let k = s.add_par(20.0);

    let vars = vec![x, y];
    let e = expr!(x - y * b; x = vars[1], y = vars[0]); // switching vars
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 0);
    assert!(e.d1_sparsity[0] == y);
    assert!(e.d1_sparsity[1] == x);

    let e = expr!((x * x - y) * b + k; x, y; k);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, -20.0, 1e-8);
    assert_near!(d1_out[0], 20.0, 1e-8);
    assert_near!(d1_out[1], -10.0, 1e-8);
    assert_near!(d2_out[0], 20.0, 1e-8);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d1_sparsity[0] == x);
    assert!(e.d1_sparsity[1] == y);
    assert!(e.d2_sparsity.len() == 1);
    assert!(e.d2_sparsity[0] == (x, x));
}

#[test]
fn priorities() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(3.0);

    let e = expr!(x + - 2.0 * y; x, y);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 0);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, -5.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d1_out[1], -2.0, 1e-8);

    let e = expr!(-x + 2.0 * y; x, y);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 0);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 5.0, 1e-8);
    assert_near!(d1_out[0], -1.0, 1e-8);
    assert_near!(d1_out[1], 2.0, 1e-8);

    let e = expr!(-x.cos() + 2.0 * y; x, y);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 6.0 - 1.0_f64.cos(), 1e-8);
    assert_near!(d1_out[0], 1.0_f64.sin(), 1e-8);
    assert_near!(d1_out[1], 2.0, 1e-8);
    assert_near!(d2_out[0], 1.0_f64.cos(), 1e-8);

    let e = expr!(x - y * y; x, y);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, -8.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d1_out[1], -6.0, 1e-8);
    assert_near!(d2_out[0], -2.0, 1e-8);

    let e = expr!(x + - y * y; x, y);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 1);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, -8.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d1_out[1], -6.0, 1e-8);
    assert_near!(d2_out[0], -2.0, 1e-8);
}

#[test]
fn parameter() {
    let mut s = Store::new();
    let x = s.add_var(1.0);
    let y = s.add_var(3.0);
    let p = s.add_par(5.0);

    let e = expr!(x + p * y; x, y; p);
    assert!(e.d1_sparsity.len() == 2);
    assert!(e.d2_sparsity.len() == 0);

    let (mut d1_out, mut d2_out) = prepare_outputs(&e);

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 16.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d1_out[1], 5.0, 1e-8);

    s.pars[0] = 10.0;

    let v = (e.all)(
        s.vars.as_slice(),
        s.pars.as_slice(),
        &mut d1_out,
        &mut d2_out,
    );
    assert_near!(v, 31.0, 1e-8);
    assert_near!(d1_out[0], 1.0, 1e-8);
    assert_near!(d1_out[1], 10.0, 1e-8);
}

// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NumOps trait required to be in scope to use cos and powi
#![feature(proc_macro_hygiene)]

use descent::expr::ExprStaticSum;
use descent::model::Model;
use descent_ipopt::IpoptModel;
use descent_macro::expr;

fn main() {
    // Below values are: Iter, T in IPOPT, T in NLP (this), T in NLP (mo)
    //let n: usize = 100000; // 8, 2.1, 0.15, 0.568 
    //let n: usize = 10000; // 8, 0.098, 0.016, 0.056
    //let n: usize = 5000; // 8, 0.047, 0.007, 0.021
    let n: usize = 1000; // 75, 0.080, 0.010, 0.035

    let mut m = IpoptModel::new();
    let mut xs = Vec::new();
    for _i in 0..n {
        xs.push(m.add_var(-1.5, 0.0, -0.5));
    }
    println!("Building objective");
    let mut obj = ExprStaticSum::new();
    for &x in xs.iter() {
        obj = obj + expr!((x - 1.0) * (x - 1.0); x);
    }
    println!("Setting objective");
    m.set_obj(obj);
    println!("Building constraints");
    for i in 0..(n - 2) {
        let a = ((i + 2) as f64) / (n as f64);
        // NEED to implement cos
        let e = expr!((y * y + 1.5 * y - a) * z.cos() - x; x = xs[i], y = xs[i + 1], z = xs[i + 2]);
        //let e = expr!((y * y + 1.5 * y - a) * z - x; x = xs[i], y = xs[i + 1], z = xs[i + 2]);
        m.add_con(e, 0.0, 0.0);
    }
    println!("Solving");
    //m.silence();
    m.solve();

    // Make sure debug is on in release (set in cargo)
    //cargo build --release --example problem
    //valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./problem
}

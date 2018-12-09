// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// NumOps trait required to be in scope to use cos and powi
use descent::expr::dynam::{ExprDynSum, NumOps};
use descent::model::Model;
use descent_ipopt::IpoptModel;

fn main() {
    // Below values are: Iter, T in IPOPT, T in NLP (this), T in NLP (mo)
    //let n: usize = 100000; // 8, 2.1, 0.88, 0.568
    //let n: usize = 10000; // 8, 0.098, 0.088, 0.056
    //let n: usize = 5000; // 8, 0.047, 0.043, 0.026
    let n: usize = 1000; // 75, 0.080, 0.080, 0.050

    let mut m = IpoptModel::new();
    let mut xs = Vec::new();
    for _i in 0..n {
        xs.push(m.add_var(-1.5, 0.0, -0.5));
    }
    println!("Building objective");
    // For such large sums, want to use a ExprDynSum. Can make one explicity:
    let mut obj = ExprDynSum::new();
    // or could just use regular expression (automatically gets converted to
    // ExprDynSum when added to model):
    //let mut obj = Expr::from(0.0);
    for &x in &xs {
        obj = obj + (x - 1.0).powi(2);
    }
    println!("Setting objective");
    m.set_obj(obj);
    println!("Building constraints");
    for i in 0..(n - 2) {
        let a = ((i + 2) as f64) / (n as f64);
        let e = (xs[i + 1].powi(2) + 1.5 * xs[i + 1] - a) * xs[i + 2].cos() - xs[i];
        //let e = (xs[i + 1].powi(2) + 1.5 * xs[i + 1] - a) * xs[i + 2] - xs[i];
        m.add_con(e, 0.0, 0.0);
    }
    println!("Solving");
    //m.silence();
    m.solve();

    // Make sure debug is on in release (set in cargo)
    //cargo build --release --example problem
    //valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./problem
}

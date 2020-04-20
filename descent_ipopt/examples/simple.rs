// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate descent;

use descent::model::Model;
use descent_ipopt::IpoptModel;

use std::f64::{INFINITY as INF, NEG_INFINITY as NINF};

fn main() {
    // We want to solve the problem:
    //
    // min 2y
    // s.t. x*x - x <= y
    //      x in [-10, 10]

    // First create a new model to store the variables and constriants:
    let mut m = IpoptModel::new();

    // Create the variables with bounds and an initial value for the solver:
    let x = m.add_var(-10.0, 10.0, 0.0);
    let y = m.add_var(NINF, INF, 0.0);

    // A parameter is added (for illustration purposes), which can be used
    // to adjust the model without having to completely reconstruct the model.
    // This can be useful in live or iterative applications.
    let p = m.add_par(2.0);

    // Set the objective value by passing an expression (in this case making
    // use of the parameter):
    m.set_obj(p * y);

    // Constraints are added to the model by bringing the variables onto one
    // side of the inequality / equality, and setting constant lower and upper
    // bounds for that expression. Infinity (negative infinity) is used if the
    // expresessing is not bounded above (or below), and the bounds can be set
    // to the same value if it is an equality constraint.
    //
    // Here x*x - x <= y becomes 0 <= y - x*x + x <= infinity:
    m.add_con(y - x * x + x, 0.0, INF);

    // Solve it:
    let (stat, sol) = m.solve().unwrap();

    println!("{:?}", stat);
    println!("x: {} and y: {}", sol.var(x), sol.var(y));
    println!("Objective: {}", sol.obj_val);

    // Let's solve it again after adjusting the parameter value, and warm start
    // the solver with the previous solution. We also silence the output from
    // Ipopt:
    m.set_par(p, 4.0);
    m.silence();
    let (stat, sol) = m.warm_solve(sol).unwrap();

    println!("{:?}", stat);
    println!("Resolve objective: {}", sol.obj_val);

    // Build and run this example like so:
    // cargo build --release --example simple
    // ./target/release/examples/simple
}

extern crate descent;

use std::f64;
use descent::ipopt_model::{IpoptModel};
use descent::model::{Model};

fn main() {
    // Create new model
    let mut m = IpoptModel::new();
    // Create variables with bounds and initial value
    let x = m.add_var(-10.0, 10.0, 0.0);
    let y = m.add_var(f64::NEG_INFINITY, f64::INFINITY, 0.0);
    // Parameters can be added to enable the model to be adjusted without
    // being completely reconstructed
    let p = m.add_par(2.0);
    // Set the objective value
    m.set_obj(p*y);
    // Add a constraint to the model with lower and upper bounds
    m.add_con(y - x*x + x, 0.0, f64::INFINITY);
    // Solve it
    //m.silence();
    let (stat, sol) = m.solve();

    println!("{:?}", stat);
    if let Some(ref s) = sol {
        println!("x: {} and y: {}", s.var(x), s.var(y));
        println!("Objective: {}", s.obj_val);
    }

    // Solve again after adjusting parameter, warm starting from previous sol
    m.set_par(p, 4.0);
    m.silence();
    let (stat, sol) = match sol {
        Some(s) => m.warm_solve(s),
        None => m.solve(),
    };
    println!("{:?}", stat);
    if let Some(ref s) = sol {
        println!("Resolve objective: {}", s.obj_val);
    }
    // cargo build --release --example simple 
    // ./target/release/examples/simple
}

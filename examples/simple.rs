extern crate descent;

use std::f64;
use descent::ipopt_model::{IpoptModel};
use descent::model::{Model};

fn main() {
    let mut m = IpoptModel::new();
    let x = m.add_var(-10.0, 10.0, 0.0);
    let y = m.add_var(f64::NEG_INFINITY, f64::INFINITY, 0.0);
    m.set_obj(2.0*y);
    m.add_con(y - x*x + x, 0.0, f64::INFINITY);
    //m.silence();
    let (stat, sol) = m.solve();
    if let Some(ref s) = sol {
        println!("{:?}", stat);
        println!("x: {} and y: {}", s.var(x), s.var(y));
        println!("objective: {}", s.obj_val);
    }
    // cargo build --release --example simple 
    // ./target/release/examples/simple
}

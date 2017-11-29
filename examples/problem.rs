extern crate descent;

use descent::expression::{Film, NumOpsF};
use descent::ipopt_model::{IpoptModel};
use descent::model::{Model};

fn main() {
    // Large expression way worse than lots of small expressions.
    // Some exponential growth
    // Here in particular the objective is the problem
    // Can handle this somewhat when under certain conditions can perform
    // reverse method.  Note this doesn't solve issue of slow start up
    // (when using forward for constant derivatives).
    // Below values are: Iter, T in IPOPT, T in NLP (this), T in NLP (mo)
    //let n: usize = 100000;
    //let n: usize = 10000; // uses too much memory before solving
    //let n: usize = 5000; // 8, 0.054, 0.021, 0.021
    let n: usize = 1000; // 75, 0.080, 0.045, 0.035
    //let n: usize = 100; // 22, 0.006, 0.001, 0.001
    //let n: usize = 10;
    
    let mut m = IpoptModel::new();
    let mut xs = Vec::new();
    for _i in 0..n {
        xs.push(m.add_var(-1.5, 0.0, -0.5));
    }
    let mut obj = Film::from(0.0);
    for &x in &xs {
        obj = obj + (x - 1.0).powi(2);
    }
    //obj = (xs[0] - 1.0).powi(2);
    m.set_obj(obj);
    for i in 0..(n-2) {
        let a = ((i + 2) as f64)/(n as f64);
        let e = (xs[i + 1].powi(2) + 1.5*xs[i + 1] - a)*xs[i + 2].cos() - xs[i];
        //println!("{:?}", e);
        m.add_con(e, 0.0, 0.0);
    }
    //let a = f64::from(0 + 2)/(n as f64);
    //let e = (xs[0 + 1].powi(2) + 1.5*xs[0 + 1] - a)*xs[0 + 2].cos() - xs[0];
    //m.add_con(e, 0.0, 0.0);
    //m.silence();
    m.solve();

    // Make sure debug is on in release (set in cargo)
    //cargo build --release --example problem
    //valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./problem
}

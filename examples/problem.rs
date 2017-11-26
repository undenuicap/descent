extern crate descent;

use descent::expression::{Film, NumOpsF};
use descent::ipopt_model::{IpoptModel};
use descent::model::{Model};

fn main() {
    //let n = 100000;
    let n = 1000; // 75 iters IPOPT 0.150, NLP: 1.807 vs 0.079
    //let n = 100; // 22 iters IPOPT 0.007, NLP: 0.004 vs 0.001
    //let n = 10;
    //
    let mut m = IpoptModel::new();
    let mut xs = Vec::new();
    for _i in 0..n {
        xs.push(m.add_var(-1.5, 0.0, -0.5));
    }
    let mut obj = Film::from(0.0);
    for x in &xs {
        obj = obj + (*x - 1.0).powi(2);
    }
    m.set_obj(obj);
    for i in 0..(n-2) {
        let a = ((i + 2) as f64)/(n as f64);
        let e = ((xs[i + 1]).powi(2)
                 + 1.5*(xs[i + 1]) - a)*(xs[i + 2]).cos() - xs[i];
        //println!("{:?}", e);
        m.add_con(e, 0.0, 0.0);
    }
    m.silence();
    m.solve();
}

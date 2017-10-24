use expression::{Expr, Evaluate, Retrieve, ID};
use model::{Model, VarType};
use ipopt;
use std::slice;

struct Variable {
    id: usize,
    lb: f64,
    ub: f64,
    init: f64,
}

struct Constraint {
    id: usize,
    expr: Expr,
    lb: f64,
    ub: f64,
}

struct IpoptModel {
    vars: Vec<Variable>,
    cons: Vec<Constraint>,
    obj: Expr,
}

impl IpoptModel {
    fn new() -> IpoptModel {
        IpoptModel {
            vars: Vec::new(),
            cons: Vec::new(),
            obj: Expr::Integer(0),
        }
    }
}

impl Model for IpoptModel {
    fn add_var(&mut self, lb: f64, ub: f64) -> Expr {
        let id = self.vars.len();
        self.vars.push(Variable { id: id, lb: lb, ub: ub, init: 0.0 });
        Expr::Variable(id)
    }

    fn add_con(&mut self, expr: Expr, lb: f64, ub: f64) {
        let id = self.cons.len();
        self.cons.push(Constraint { id: id, expr: expr, lb: lb, ub: ub });
    }

    fn set_obj(&mut self, expr: Expr) {
        self.obj = expr;
    }

    fn solve(&self) {
        let mut x_lb: Vec<f64> = Vec::new();
        let mut x_ub: Vec<f64> = Vec::new();
        let mut g_lb: Vec<f64> = Vec::new();
        let mut g_ub: Vec<f64> = Vec::new();
        let mut nele_jac: usize = 0; // need to set
        let mut nele_hess: usize = 0; // need to set
        for v in &self.vars {
            x_lb.push(v.lb);
            x_ub.push(v.ub);
        }
        // Probably store ordered variables for each constraint
        // Store mapping between higher order pairs to vector of relevant
        // constraints
        // Could assume obj needs to be called on each mapped pair (vector is
        // empty if just objective)
        {
            let obj_deg = self.obj.degree();
            nele_hess += obj_deg.higher.len();
            for c in &self.cons {
                g_lb.push(c.lb);
                g_ub.push(c.ub);
                nele_jac += c.expr.variables().len();
                // Think should hen subtract obj entries from each constraint
                // This might be easier once work out what doing with maps
                //nele_hess += c.expr.degree().add(&obj_deg).higher.len();
                nele_hess += c.expr.degree().higher.len();
            }
        }
        // All rest should be handled in callbacks
        //let prob = CreateIpoptProblem(self.vars.len(),
        //                              x_lb.as_ptr(),
        //                              x_ub.as_ptr(),
        //                              self.cons.len(),
        //                              g_lb.as_ptr(),
        //                              g_ub.as_ptr(),
        //                              nele_jac,
        //                              nele_hess,
        //                              0, // C-style indexing
        //                              f,
        //                              g,
        //                              f_grad,
        //                              g_jac,
        //                              l_hess);
    }
}

struct Store<'a> {
    vars: &'a [ipopt::Number],
    pars: &'a Vec<f64>,
}

impl<'a> Retrieve for Store<'a> {
    fn get_var(&self, vid: ID) -> f64 {
        self.vars[vid]
    }

    fn get_par(&self, pid: ID) -> f64 {
        self.pars[pid]
    }
}

#[allow(non_snake_case)]
extern fn f(
        n: ipopt::Index,
        x: *const ipopt::Number,
        new_x: ipopt::Bool,
        obj_value: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {

    let model: &IpoptModel = unsafe { &*(user_data as *const IpoptModel) };

    let pars: Vec<f64> = Vec::new(); // temporary, should link to model pars
    // should check n is what we expect, maybe panic if not
    let store = Store {
        vars: unsafe { slice::from_raw_parts(x, n as usize) },
        pars: &pars,
    };
    unsafe {
        *obj_value = model.obj.value(&store);
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn easy_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0);
        m.set_obj(&x*&x);
        m.solve();
    }
}

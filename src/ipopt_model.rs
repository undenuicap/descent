use expression::{Expr, Evaluate, Retrieve, ID};
use model::Model;
use ipopt;
use std::slice;
use std::collections::HashMap;

struct Variable {
    id: usize,
    lb: f64,
    ub: f64,
    init: f64,
}

struct Parameter {
    id: usize,
    val: f64,
}

struct Constraint {
    id: usize,
    expr: Expr,
    lb: f64,
    ub: f64,
}

struct Objective {
    expr: Expr,
}

struct IpoptModel {
    vars: Vec<Variable>,
    pars: Vec<Parameter>,
    cons: Vec<Constraint>,
    obj: Objective,
}

impl IpoptModel {
    fn new() -> IpoptModel {
        IpoptModel {
            vars: Vec::new(),
            pars: Vec::new(),
            cons: Vec::new(),
            obj: Objective { expr: Expr::Integer(0) },
        }
    }
}

struct HesEntry {
    id: usize, // index into sparse matrix
    obj: bool, // whether objective participates in entry
    cons: Vec<usize>, // constraints that participate in entry
}

struct HesSparsity {
    sp: HashMap<(ID, ID), HesEntry>,
}

impl HesSparsity {
    fn new() -> HesSparsity {
        HesSparsity { sp: HashMap::new() }
    }

    fn get_entry(&mut self, eid: (ID, ID)) -> &mut HesEntry {
        if !self.sp.contains_key(&eid) {
            let id = self.sp.len();
            self.sp.insert(eid, HesEntry { 
                id: id,
                obj: false,
                cons: Vec::new(),
            });
        }
        self.sp.get_mut(&eid).unwrap()
    }

    fn add_con(&mut self, eid: (ID, ID), cid: usize) {
        let ent = self.get_entry(eid);
        ent.cons.push(cid);
    }

    fn add_obj(&mut self, eid: (ID, ID)) {
        let ent = self.get_entry(eid);
        ent.obj = true;
    }

    fn n_ele(&self) -> usize {
        self.sp.len()
    }
}

struct IpoptCBData<'a> {
    model: &'a IpoptModel,
    j_sparsity: &'a Vec<Vec<ID>>,
    h_sparsity: &'a HesSparsity,
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

    fn add_par(&mut self, val: f64) -> Expr {
        let id = self.pars.len();
        self.pars.push(Parameter { id: id, val: val });
        Expr::Parameter(id)
    }

    fn set_obj(&mut self, expr: Expr) {
        self.obj = Objective { expr: expr };
    }

    fn solve(&self) {
        let mut x_lb: Vec<f64> = Vec::new();
        let mut x_ub: Vec<f64> = Vec::new();
        let mut g_lb: Vec<f64> = Vec::new();
        let mut g_ub: Vec<f64> = Vec::new();
        let mut nele_jac: usize = 0; // need to set
        let mut nele_hes: usize = 0; // need to set
        for v in &self.vars {
            x_lb.push(v.lb);
            x_ub.push(v.ub);
        }
        // Variable IDs for each constraint
        let mut j_sparsity: Vec<Vec<ID>> = Vec::new();
        let mut h_sparsity = HesSparsity::new();

        for h in self.obj.expr.degree().higher {
            h_sparsity.add_obj(h);
        }

        for c in &self.cons {
            g_lb.push(c.lb);
            g_ub.push(c.ub);
            // Not sorted, but might not matter
            let var_vec: Vec<ID> = c.expr.variables().into_iter().collect();
            nele_jac += var_vec.len();
            j_sparsity.push(var_vec);
            for h in c.expr.degree().higher {
                h_sparsity.add_con(h, c.id);
            }
        }

        nele_hes = h_sparsity.n_ele();

        let cb_data = IpoptCBData {
            model: &self,
            j_sparsity: &j_sparsity,
            h_sparsity: &h_sparsity,
        };
        // All rest should be handled in callbacks
        //let prob = CreateIpoptProblem(self.vars.len(),
        //                              x_lb.as_ptr(),
        //                              x_ub.as_ptr(),
        //                              self.cons.len(),
        //                              g_lb.as_ptr(),
        //                              g_ub.as_ptr(),
        //                              nele_jac,
        //                              nele_hes,
        //                              0, // C-style indexing
        //                              f,
        //                              g,
        //                              f_grad,
        //                              g_jac,
        //                              l_hess);
        //fn IpoptSolve(
        //        ipopt_problem: IpoptProblem,
        //        x: *const Number,
        //        g: *mut Number,
        //        obj_val: *mut Number,
        //        mult_g: *const Number,
        //        mult_x_L: *const Number,
        //        mult_x_U: *const Number,
        //        user_data: UserDataPtr) -> ApplicationReturnStatus;
        //fn FreeIpoptProblem(ipopt_problem: IpoptProblem);
    }
}

struct Store<'a> {
    vars: &'a [ipopt::Number],
    pars: &'a Vec<Parameter>,
}

impl<'a> Retrieve for Store<'a> {
    fn get_var(&self, vid: ID) -> f64 {
        self.vars[vid]
    }

    fn get_par(&self, pid: ID) -> f64 {
        self.pars[pid].val
    }
}

//#[allow(non_snake_case)]
extern fn f(
        n: ipopt::Index,
        x: *const ipopt::Number,
        new_x: ipopt::Bool,
        obj_value: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {

    let cb_data: &IpoptCBData = unsafe { &*(user_data as *const IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0
    }

    let store = Store {
        vars: unsafe { slice::from_raw_parts(x, n as usize) },
        pars: &cb_data.model.pars,
    };
    unsafe {
        *obj_value = cb_data.model.obj.expr.value(&store);
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

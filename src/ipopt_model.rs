use expression::{Expr, Evaluate, Retrieve, ID};
use model::Model;
use ipopt;
use std::slice;
use std::collections::HashMap;
use std::ptr;
use std::ffi::CString;

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

struct ModelData {
    vars: Vec<Variable>,
    pars: Vec<Parameter>,
    cons: Vec<Constraint>,
    obj: Objective,
}

struct ModelCache {
    j_sparsity: Vec<(usize, ID)>, // jacobian sparsity
    h_sparsity: HesSparsity, // hessian sparsity
    v_obj: Vec<ID>, // variables in objective
}

// Make sure don't implement copy or clone for this otherwise risk of double
// free
struct IpoptProblem {
    prob: ipopt::IpoptProblem,
}

impl Drop for IpoptProblem {
    fn drop(&mut self) {
        unsafe { ipopt::FreeIpoptProblem(self.prob) };
    }
}

struct IpoptModel {
    model: ModelData,
    cache: Option<ModelCache>,
    prob: Option<IpoptProblem>,
    prepared: bool, // problem prepared
}

impl IpoptModel {
    fn new() -> IpoptModel {
        IpoptModel {
            model: ModelData {
                vars: Vec::new(),
                pars: Vec::new(),
                cons: Vec::new(),
                obj: Objective { expr: Expr::Integer(0) },
            },
            cache: None,
            prob: None,
            prepared: false,
        }
    }

    fn prepare(&mut self) {
        if self.prepared && self.cache.is_some() || self.prob.is_some() {
            return; // If still valid don't prepare again
        }
        let mut x_lb: Vec<f64> = Vec::new();
        let mut x_ub: Vec<f64> = Vec::new();
        for v in &self.model.vars {
            x_lb.push(v.lb);
            x_ub.push(v.ub);
        }

        let mut g_lb: Vec<f64> = Vec::new();
        let mut g_ub: Vec<f64> = Vec::new();
        let mut j_sparsity: Vec<(usize, ID)> = Vec::new();
        let mut h_sparsity = HesSparsity::new();

        for h in self.model.obj.expr.degree().higher {
            h_sparsity.add_obj(h);
        }

        for c in &self.model.cons {
            g_lb.push(c.lb);
            g_ub.push(c.ub);
            // Not sorted, but might not matter
            for vid in c.expr.variables() {
                j_sparsity.push((c.id, vid));
            }
            //let var_vec: Vec<ID> = c.expr.variables().into_iter().collect();
            //j_sparsity.push(var_vec);
            for h in c.expr.degree().higher {
                h_sparsity.add_con(h, c.id);
            }
        }

        let nele_hes = h_sparsity.len();
        let nele_jac = j_sparsity.len();
        // Not sorted, don't suppose it matters
        let v_obj: Vec<ID> = self.model.obj.expr.variables()
            .into_iter().collect();

        // x_lb, x_ub, g_lb, g_ub are copied internally so don't need to keep
        let prob = unsafe {
            ipopt::CreateIpoptProblem(self.model.vars.len() as i32,
                                      x_lb.as_ptr(),
                                      x_ub.as_ptr(),
                                      self.model.cons.len() as i32,
                                      g_lb.as_ptr(),
                                      g_ub.as_ptr(),
                                      nele_jac as i32,
                                      nele_hes as i32,
                                      0, // C-style indexing
                                      f,
                                      g,
                                      f_grad,
                                      g_jac,
                                      l_hess)
        };
        self.prob = Some(IpoptProblem { prob: prob });
        self.cache = Some(ModelCache {
            j_sparsity: j_sparsity,
            h_sparsity: h_sparsity,
            v_obj: v_obj,
        });
        self.prepared = true;
    }

    // Options can only be set once model is prepared.
    // They will be lost if the model is modified.
    fn set_str_option(&mut self, key: &str, val: &str) -> bool {
        self.prepare();
        if let Some(ref mut prob) = self.prob {
            let key_c = CString::new(key).unwrap();
            let val_c = CString::new(val).unwrap();
            unsafe {
                ipopt::AddIpoptStrOption(prob.prob, key_c.as_ptr(),
                    val_c.as_ptr()) != 0 // convert to bool
            }
        } else {
            false
        }
    }

    fn set_num_option(&mut self, key: &str, val: f64) -> bool {
        self.prepare();
        if let Some(ref mut prob) = self.prob {
            let key_c = CString::new(key).unwrap();
            unsafe {
                ipopt::AddIpoptNumOption(prob.prob, key_c.as_ptr(),
                    val) != 0 // convert to bool
            }
        } else {
            false
        }
    }

    fn set_int_option(&mut self, key: &str, val: i32) -> bool {
        self.prepare();
        if let Some(ref mut prob) = self.prob {
            let key_c = CString::new(key).unwrap();
            unsafe {
                ipopt::AddIpoptIntOption(prob.prob, key_c.as_ptr(),
                    val) != 0 // convert to bool
            }
        } else {
            false
        }
    }

    // As it uses options above, will only last as long as model stays prepared
    fn silence(&mut self) -> {
        self.set_str_option("sb", "yes")
            && self.set_int_option("print_level", 0)
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

    fn len(&self) -> usize {
        self.sp.len()
    }
}

struct IpoptCBData<'a> {
    model: &'a ModelData,
    cache: &'a ModelCache,
}

impl Model for IpoptModel {
    fn add_var(&mut self, lb: f64, ub: f64) -> Expr {
        self.prepared = false;
        let id = self.model.vars.len();
        self.model.vars.push(Variable { id: id, lb: lb, ub: ub, init: 0.0 });
        Expr::Variable(id)
    }

    fn add_con(&mut self, expr: Expr, lb: f64, ub: f64) {
        self.prepared = false;
        let id = self.model.cons.len();
        self.model.cons.push(Constraint { id: id, expr: expr, lb: lb, ub: ub });
    }

    fn add_par(&mut self, val: f64) -> Expr {
        self.prepared = false;
        let id = self.model.pars.len();
        self.model.pars.push(Parameter { id: id, val: val });
        Expr::Parameter(id)
    }

    fn set_obj(&mut self, expr: Expr) {
        self.prepared = false;
        self.model.obj = Objective { expr: expr };
    }

    fn solve(&mut self) {
        self.prepare();
        if let (&Some(ref cache), &Some(ref prob)) = (&self.cache, &self.prob) {
            let mut x: Vec<f64> = Vec::new(); // will contain solution also
            for v in &self.model.vars {
                x.push(v.init);
            }
            let mut obj_val = 0.0;
            let mut cb_data = IpoptCBData {
                model: &self.model, cache: cache
            };
            let cb_data_ptr = &mut cb_data as *mut _ as ipopt::UserDataPtr;
            let status = unsafe {
                ipopt::IpoptSolve(prob.prob,
                                  x.as_mut_ptr(),
                                  ptr::null_mut(),
                                  &mut obj_val,
                                  ptr::null_mut(),
                                  ptr::null_mut(),
                                  ptr::null_mut(),
                                  cb_data_ptr);
            };
            println!("Result: {}", x[0]);
        } else {
        }
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

extern fn f(
        n: ipopt::Index,
        x: *const ipopt::Number,
        _new_x: ipopt::Bool,
        obj_value: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &IpoptCBData = unsafe { &*(user_data as *const IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }

    let store = Store {
        vars: unsafe { slice::from_raw_parts(x, n as usize) },
        pars: &cb_data.model.pars,
    };

    let value = unsafe { &mut *obj_value };
    *value = cb_data.model.obj.expr.value(&store);
    1
}

extern fn f_grad(
        n: ipopt::Index,
        x: *const ipopt::Number,
        _new_x: ipopt::Bool,
        grad_f: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &IpoptCBData = unsafe { &*(user_data as *const IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }

    let store = Store {
        vars: unsafe { slice::from_raw_parts(x, n as usize) },
        pars: &cb_data.model.pars,
    };

    let values = unsafe { slice::from_raw_parts_mut(grad_f, n as usize) };
    // Might need to zero out memory for other entries
    for vid in &cb_data.cache.v_obj {
        values[*vid] = cb_data.model.obj.expr.deriv(&store, *vid).1;
    }
    1
}

extern fn g(
        n: ipopt::Index,
        x: *const ipopt::Number,
        _new_x: ipopt::Bool,
        m: ipopt::Index,
        g: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &IpoptCBData = unsafe { &*(user_data as *const IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }
    if m != cb_data.model.cons.len() as i32 {
        return 0;
    }

    let store = Store {
        vars: unsafe { slice::from_raw_parts(x, n as usize) },
        pars: &cb_data.model.pars,
    };

    let values = unsafe { slice::from_raw_parts_mut(g, m as usize) };

    for c in &cb_data.model.cons {
        values[c.id] = c.expr.value(&store);
    }
    1
}

extern fn g_jac(
        n: ipopt::Index,
        x: *const ipopt::Number,
        _new_x: ipopt::Bool,
        m: ipopt::Index,
        nele_jac: ipopt::Index,
        i_row: *mut ipopt::Index,
        j_col: *mut ipopt::Index,
        vals: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &IpoptCBData = unsafe { &*(user_data as *const IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }
    if m != cb_data.model.cons.len() as i32 {
        return 0;
    }
    if nele_jac != cb_data.cache.j_sparsity.len() as i32 {
        return 0;
    }

    if vals == ptr::null_mut() {
        // Set sparsity
        let row = unsafe {
            slice::from_raw_parts_mut(i_row, nele_jac as usize)
        };
        let col = unsafe {
            slice::from_raw_parts_mut(j_col, nele_jac as usize)
        };
        for (i, &(cid, vid)) in cb_data.cache.j_sparsity.iter().enumerate() {
            row[i] = cid as i32;
            col[i] = vid as i32;
        }
    } else {
        // Set values
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: &cb_data.model.pars,
        };

        let values = unsafe {
            slice::from_raw_parts_mut(vals, nele_jac as usize)
        };
        for (i, &(cid, vid)) in cb_data.cache.j_sparsity.iter().enumerate() {
            values[i] = cb_data.model.cons[cid].expr.deriv(&store, vid).1;
        }
    }
    1
}

extern fn l_hess(
        n: ipopt::Index,
        x: *const ipopt::Number,
        _new_x: ipopt::Bool,
        obj_factor: ipopt::Number,
        m: ipopt::Index,
        lambda: *const ipopt::Number,
        _new_lambda: ipopt::Bool,
        nele_hes: ipopt::Index,
        i_row: *mut ipopt::Index,
        j_col: *mut ipopt::Index,
        vals: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &IpoptCBData = unsafe { &*(user_data as *const IpoptCBData) };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }
    if m != cb_data.model.cons.len() as i32 {
        return 0;
    }
    if nele_hes != cb_data.cache.h_sparsity.len() as i32 {
        return 0;
    }

    if vals == ptr::null_mut() {
        // Set sparsity
        let row = unsafe {
            slice::from_raw_parts_mut(i_row, nele_hes as usize)
        };
        let col = unsafe {
            slice::from_raw_parts_mut(j_col, nele_hes as usize)
        };
        for (vids, ent) in &cb_data.cache.h_sparsity.sp {
            row[ent.id] = vids.0 as i32;
            col[ent.id] = vids.1 as i32;
        }
    } else {
        // Set values
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: &cb_data.model.pars,
        };

        let lam = unsafe { slice::from_raw_parts(lambda, m as usize) };
        let values = unsafe {
            slice::from_raw_parts_mut(vals, nele_hes as usize)
        };
        for (vids, ent) in &cb_data.cache.h_sparsity.sp {
            let mut v = 0.0;
            if ent.obj {
                v += obj_factor*cb_data.model.obj.expr
                    .deriv2(&store, vids.0, vids.1).3;
            }
            for cid in &ent.cons {
                v += lam[*cid]*cb_data.model.cons[*cid].expr
                    .deriv2(&store, vids.0, vids.1).3;
            }
            values[ent.id] = v;
        }
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
        assert!(m.set_str_option("sb", "yes"));
        assert!(m.set_int_option("print_level", 0));
        m.solve();
    }
}

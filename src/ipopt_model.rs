extern crate fnv;

use expression::{Var, Par, ID};
use expression::{Film, FilmInfo, Retrieve, WorkSpace, Column};
use model::{Model, Solution, SolutionStatus, Con};
use ipopt;
use std::slice;
//use std::collections::HashMap;
use self::fnv::FnvHashMap;
use std::ptr;
use std::ffi::CString;
use std::f64;

struct Variable {
    lb: f64,
    ub: f64,
    init: f64,
}

struct Parameter {
    val: f64,
}

struct Constraint {
    film: Film,
    info: FilmInfo,
    lb: f64,
    ub: f64,
}

struct Objective {
    film: Film,
    info: FilmInfo,
}

struct ModelData {
    vars: Vec<Variable>,
    pars: Vec<Parameter>,
    cons: Vec<Constraint>,
    obj: Objective,
}

#[derive(Debug, Default)]
struct ModelCache {
    j_sparsity: Vec<(usize, ID)>, // jacobian sparsity
    h_sparsity: HesSparsity, // hessian sparsity
    ws: WorkSpace,
    con_sta: Vec<Column>,
    obj_sta: Column,
    con_dyn: Vec<Column>,
    obj_dyn: Column,
    con_ready: bool,
    obj_ready: bool,
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
                obj: Objective { film: Film::from(0.0), info: FilmInfo::new() },
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
        let mut j_sparsity: Vec<(ID, ID)> = Vec::new();
        let mut h_sparsity = HesSparsity::new();

        h_sparsity.add_obj(&self.model.obj.info);

        for (cid, c) in self.model.cons.iter().enumerate() {
            g_lb.push(c.lb);
            g_ub.push(c.ub);
            for v in &c.info.lin {
                j_sparsity.push((cid, v));
            }
            for v in &c.info.nlin {
                j_sparsity.push((cid, v));
            }
            h_sparsity.add_con(&c.info);
        }

        let nele_hes = h_sparsity.len();
        let nele_jac = j_sparsity.len();

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
            con_ready: false,
            obj_ready: false,
            ..Default::default(),
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
    fn silence(&mut self) -> bool {
        self.set_str_option("sb", "yes")
            && self.set_int_option("print_level", 0)
    }

    fn form_init_solution(&self, sol: &mut Solution) {
        // If no missing initial values, pull from variable
        let nvar_store = sol.store.vars.len();
        sol.store.vars.extend(self.model.vars.iter()
                              .skip(nvar_store)
                              .map(|x| x.init));
        sol.store.vars.resize(self.model.vars.len(), 0.0); // if need to shrink
        // Always redo parameters
        sol.store.pars.clear();
        for p in &self.model.pars {
            sol.store.pars.push(p.val);
        }
        // Buffer rest with zeros
        sol.con_mult.resize(self.model.cons.len(), 0.0);
        sol.var_lb_mult.resize(self.model.vars.len(), 0.0);
        sol.var_ub_mult.resize(self.model.vars.len(), 0.0);
    }

    // Should only be called after prepare
    fn ipopt_solve(&mut self, mut sol: Solution) ->
            (SolutionStatus, Option<Solution>) {
        if let (&Some(ref mut cache), &Some(ref prob)) = (&self.cache, &self.prob) {
            self.form_init_solution(&mut sol);
            let ipopt_status;
            {
                // Should do static calcs on cache
                // Should put this is a function that returns a Column
                cache.con_sta.clear();
                for c in &self.model.cons {
                    let mut col = Column::new();
                    // Maybe use a NAN or zero store
                    c.film.ad(&c.info.lin, Vec::new(), &zstore, &mut cache.ws);
                    col.der1 = cache.ws.last().unwrap().der1.clone());
                    c.film.ad(&c.info.nlin, &c.info.quad, &zstore, &mut cache.ws);
                    col.der2 = cache.ws.last().unwrap().der2.clone());
                    cache.con_sta.push(col);
                }

                let mut cb_data = IpoptCBData {
                    model: &self.model,
                    cache: cache,
                    pars: &sol.store.pars,
                };
                let cb_data_ptr = &mut cb_data as *mut _ as ipopt::UserDataPtr;
                ipopt_status = unsafe {
                    ipopt::IpoptSolve(prob.prob,
                                      sol.store.vars.as_mut_ptr(),
                                      ptr::null_mut(), // can calc ourselves
                                      &mut sol.obj_val,
                                      sol.con_mult.as_mut_ptr(),
                                      sol.var_lb_mult.as_mut_ptr(),
                                      sol.var_ub_mult.as_mut_ptr(),
                                      cb_data_ptr)
                };
            }
            // Should probably save ipopt_status to self
            use ipopt::ApplicationReturnStatus as ARS;
            use model::SolutionStatus as SS;
            let status = match ipopt_status {
                ARS::SolveSucceeded | ARS::SolvedToAcceptableLevel =>
                    SS::Solved,
                ARS::InfeasibleProblemDetected => SS::Infeasible,
                _ => SS::Other,
            };
            match ipopt_status {
                ARS::SolveSucceeded | ARS::SolvedToAcceptableLevel =>
                    (status, Some(sol)),
                _ => (status, None),
            }
        } else {
            (SolutionStatus::Error, None)
        }
    }
}

#[derive(Debug, Default)]
struct HesSparsity {
    sp: FnvHashMap<(ID, ID), usize>,
    con_inds: Vec<Vec<usize>>,
    obj_inds: Vec<usize>,
}

impl HesSparsity {
    fn new() -> HesSparsity {
        HesSparsity::default()
    }

    fn get_index(&mut self, eid: (ID, ID)) -> usize {
        let id = self.sp.len(); // incase need to create new
        self.sp.entry(eid).or_insert(id)
    }

    fn add_con(&mut self, info: &FilmInfo) {
        let mut v = Vec::new();
        for &(li1, li2) in &info.quad {
            // get the non-local variable ids
            let p = (info.nlin[li1], info.nlin[li2]);
            v.push(self.get_index(p));
        }
        for &(li1, li2) in &info.nquad {
            // get the non-local variable ids
            let p = (info.nlin[li1], info.nlin[li2]);
            v.push(self.get_index(p));
        }
        self.con_inds.push(v);
    }

    fn add_obj(&mut self, info: &FilmInfo) {
        for &(li1, li2) in &info.quad {
            // get the non-local variable ids
            let p = (info.nlin[li1], info.nlin[li2]);
            self.obj_inds.push(self.get_index(p));
        }
        for &(li1, li2) in &info.nquad {
            // get the non-local variable ids
            let p = (info.nlin[li1], info.nlin[li2]);
            self.obj_inds.push(self.get_index(p));
        }
    }

    fn len(&self) -> usize {
        self.sp.len()
    }
}

struct IpoptCBData<'a> {
    model: &'a ModelData,
    cache: &'a ModelCache,
    pars: &'a Vec<f64>,
}

impl Model for IpoptModel {
    fn add_var(&mut self, lb: f64, ub: f64, init: f64) -> Var {
        self.prepared = false;
        let id = self.model.vars.len();
        self.model.vars.push(Variable { lb: lb, ub: ub, init: init });
        Var(id)
    }

    fn add_par(&mut self, val: f64) -> Par {
        self.prepared = false;
        let id = self.model.pars.len();
        self.model.pars.push(Parameter { val: val });
        Par(id)
    }

    fn add_con(&mut self, film: Film, lb: f64, ub: f64) -> Con {
        self.prepared = false;
        let id = self.model.cons.len();
        let finfo = film.get_info();
        self.model.cons.push(Constraint { film: film, info: finfo,
            lb: lb, ub: ub });
        Con(id)
    }

    fn set_obj(&mut self, film: Film) {
        self.prepared = false;
        let finfo = film.get_info();
        self.model.obj = Objective { film: film, info: finfo };
    }

    fn solve(&mut self) -> (SolutionStatus, Option<Solution>) {
        self.prepare();
        let sol = Solution::new();
        self.ipopt_solve(sol)
    }

    fn warm_solve(&mut self, sol: Solution) ->
            (SolutionStatus, Option<Solution>) {
        self.prepare();
        // Should set up warm start stuff
        self.ipopt_solve(sol)
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
        pars: cb_data.pars,
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
        pars: cb_data.pars,
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
        pars: cb_data.pars,
    };

    let values = unsafe { slice::from_raw_parts_mut(g, m as usize) };

    for (cid, c) in cb_data.model.cons.iter().enumerate() {
        values[cid] = c.expr.value(&store);
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

    if vals.is_null() {
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
            pars: cb_data.pars,
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

    if vals.is_null() {
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
            pars: cb_data.pars,
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
    extern crate test;
    use expression::NumOps;
    use super::*;
    #[test]
    fn univar_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        m.set_obj(&x*&x);
        assert!(m.set_str_option("sb", "yes"));
        assert!(m.set_int_option("print_level", 0));
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.value(&x) - 1.0).abs() < 1e-6);
            assert!((s.obj_val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn multivar_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(&x*&x + &y*&y + &x*&y);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.value(&x) - 1.0).abs() < 1e-6);
            assert!((s.value(&y) + 0.5).abs() < 1e-6);
            assert!((s.obj_val - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn equality_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(&x*&x + &y*&y + &x*&y);
        m.add_con(&x + &y, 0.75, 0.75);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.value(&x) - 1.0).abs() < 1e-6);
            assert!((s.value(&y) + 0.25).abs() < 1e-6);
            assert!((s.obj_val - 0.8125).abs() < 1e-6);
        }
    }

    #[test]
    fn inequality_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(&x*&x + &y*&y + &x*&y);
        m.add_con(&x + &y, 0.25, 0.40);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.value(&x) - 1.0).abs() < 1e-6);
            assert!((s.value(&y) + 0.6).abs() < 1e-6);
            assert!((s.obj_val - 0.76).abs() < 1e-6);
        }
    }

    #[test]
    fn quad_constraint_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(-10.0, 10.0, 0.0);
        let y = m.add_var(f64::NEG_INFINITY, f64::INFINITY, 0.0);
        m.set_obj(2*&y);
        m.add_con(&y - &x*&x + &x, 0.0, f64::INFINITY);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.value(&x) - 0.5).abs() < 1e-5);
            assert!((s.value(&y) + 0.25).abs() < 1e-5);
            assert!((s.obj_val + 0.5).abs() < 1e-4);
        }
    }

    #[bench]
    fn large_problem(b: &mut test::Bencher) {
        //let n = 100000;
        let n = 10;
        b.iter(|| {
            let mut m = IpoptModel::new();
            let mut xs = Vec::new();
            for _i in 0..n {
                xs.push(m.add_var(-1.5, 0.0, -0.5));
            }
            let mut obj = Expr::Integer(0);
            for x in &xs {
                obj = obj + (x - 1).powi(2);
            }
            m.set_obj(obj);
            for i in 0..(n-2) {
                let a = ((i + 2) as f64)/(n as f64);
                m.add_con(((&xs[i + 1]).powi(2) + 1.5*(&xs[i + 1]) - a)
                          *(&xs[i + 2]).cos() - &xs[i], 0.0, 0.0);
            }
            m.silence();
            m.solve();
            //let (stat, sol) = m.solve();
        });
    }
}

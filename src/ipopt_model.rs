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
    cons_const: Vec<Column>,
    obj_const: Column,
    cons: Vec<Column>,
    obj: Column,
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

pub struct IpoptModel {
    model: ModelData,
    cache: Option<ModelCache>,
    prob: Option<IpoptProblem>,
    prepared: bool, // problem prepared
}

fn const_col(film: &Film, info: &FilmInfo, store: &Retrieve,
             ws: &mut WorkSpace) -> Column {
    let mut col = Column::new();
    //col.der1 = film.full_fwd(&info.lin, &Vec::new(), store, ws).der1.clone();
    film.eval(store, &mut ws.ns);
    col.der1 = film.der1_rev(&info.lin, store, &ws.ns, &mut ws.nas,
                             &mut ws.ids);
    col.der2 = film.full_fwd(&info.nlin, &info.quad, store,
                             &mut ws.cols).der2.clone();
    col
}

fn dynam_col(film: &Film, info: &FilmInfo, store: &Retrieve,
             ws: &mut WorkSpace, col: &mut Column) {
    if info.nlin.is_empty() {
        col.val = film.eval(store, &mut ws.ns);
    } else if info.nquad.is_empty() {
        col.val = film.eval(store, &mut ws.ns);
        col.der1 = film.der1_rev(&info.nlin, store, &ws.ns, &mut ws.nas,
                              &mut ws.ids);
    } else {
        *col = film.full_fwd(&info.nlin, &info.nquad, store,
                             &mut ws.cols).clone();
    }
}

impl IpoptModel {
    pub fn new() -> IpoptModel {
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
        // Have a problem if Film is empty.  Don't know how to easy enforce
        // a non-empty Film.  Could verify them, but then makes interface
        // clumbsy.  Could panic late like here.
        // Hrmm should possibly verify at the prepare phase.  Then return solve
        // error if things go bad.
        // Other option is to check as added to model (so before FilmInfo is
        // called).  Ideally should design interface/operations on FilmInfo
        // so that an empty/invalid value is not easily created/possible.
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
            for &v in &c.info.lin {
                j_sparsity.push((cid, v));
            }
            for &v in &c.info.nlin {
                j_sparsity.push((cid, v));
            }
            h_sparsity.add_con(&c.info);
        }

        let nvars = self.model.vars.len();
        let ncons = self.model.cons.len();
        let nele_hes = h_sparsity.len();
        let nele_jac = j_sparsity.len();

        // x_lb, x_ub, g_lb, g_ub are copied internally so don't need to keep
        let prob = unsafe {
            ipopt::CreateIpoptProblem(nvars as i32,
                                      x_lb.as_ptr(),
                                      x_ub.as_ptr(),
                                      ncons as i32,
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

        // From code always returns true
        // For some reason getting incorrect/corrupt callback data
        // Don't need anymore because using new_x
        //unsafe { ipopt::SetIntermediateCallback(prob, intermediate) };

        let mut cache = ModelCache {
                j_sparsity: j_sparsity,
                h_sparsity: h_sparsity,
                ..Default::default()
            };
        cache.cons.resize(ncons, Column::new());

        self.prob = Some(IpoptProblem { prob: prob });
        self.cache = Some(cache);
        self.prepared = true;
    }

    // Options can only be set once model is prepared.
    // They will be lost if the model is modified.
    pub fn set_str_option(&mut self, key: &str, val: &str) -> bool {
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

    pub fn set_num_option(&mut self, key: &str, val: f64) -> bool {
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

    pub fn set_int_option(&mut self, key: &str, val: i32) -> bool {
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
    pub fn silence(&mut self) -> bool {
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
        self.form_init_solution(&mut sol);

        if let (&mut Some(ref mut cache), &Some(ref prob)) =
                (&mut self.cache, &self.prob) {
            let ipopt_status;
            {
                // Just passing it the solution store (in theory var values
                // should not affect the values).
                cache.cons_const.clear();
                for c in &self.model.cons {
                    cache.cons_const.push(const_col(&c.film, &c.info,
                                                   &sol.store, &mut cache.ws));
                }

                cache.obj_const = const_col(&self.model.obj.film,
                                            &self.model.obj.info,
                                            &sol.store, &mut cache.ws);

                let mut cb_data = IpoptCBData {
                    model: &self.model,
                    cache: cache,
                    pars: &sol.store.pars,
                };

                let cb_data_ptr = &mut cb_data as *mut _ as ipopt::UserDataPtr;

                // This and others might throw and exception.  How would we
                // catch?
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
    cons_inds: Vec<Vec<usize>>,
    obj_inds: Vec<usize>,
}

impl HesSparsity {
    fn new() -> HesSparsity {
        HesSparsity::default()
    }

    fn get_index(&mut self, eid: (ID, ID)) -> usize {
        let id = self.sp.len(); // incase need to create new
        *self.sp.entry(eid).or_insert(id)
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
        self.cons_inds.push(v);
    }

    fn add_obj(&mut self, info: &FilmInfo) {
        for &(li1, li2) in &info.quad {
            // get the non-local variable ids
            let ind = self.get_index((info.nlin[li1], info.nlin[li2]));
            self.obj_inds.push(ind);
        }
        for &(li1, li2) in &info.nquad {
            // get the non-local variable ids
            let ind = self.get_index((info.nlin[li1], info.nlin[li2]));
            self.obj_inds.push(ind);
        }
    }

    fn len(&self) -> usize {
        self.sp.len()
    }
}

struct IpoptCBData<'a> {
    model: &'a ModelData,
    cache: &'a mut ModelCache,
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
        //println!("{:?}", finfo);
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

fn solve_obj(cb_data: &mut IpoptCBData, store: &Store) {
    dynam_col(&cb_data.model.obj.film,
              &cb_data.model.obj.info,
              store,
              &mut cb_data.cache.ws,
              &mut cb_data.cache.obj);
}

fn solve_cons(cb_data: &mut IpoptCBData, store: &Store) {
    for (i, c) in cb_data.model.cons.iter().enumerate() {
        dynam_col(&c.film,
                  &c.info,
                  store,
                  &mut cb_data.cache.ws,
                  &mut cb_data.cache.cons[i]);
    }
}

extern fn f(
        n: ipopt::Index,
        x: *const ipopt::Number,
        new_x: ipopt::Bool,
        obj_value: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe {
        &mut *(user_data as *mut IpoptCBData)
    };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }

    if new_x == 1 {
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: cb_data.pars,
        };

        solve_obj(cb_data, &store);
        solve_cons(cb_data, &store);
    }

    let value = unsafe { &mut *obj_value };
    *value = cb_data.cache.obj.val;
    1
}

extern fn f_grad(
        n: ipopt::Index,
        x: *const ipopt::Number,
        new_x: ipopt::Bool,
        grad_f: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe {
        &mut *(user_data as *mut IpoptCBData)
    };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }

    if new_x == 1 {
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: cb_data.pars,
        };

        solve_obj(cb_data, &store);
        solve_cons(cb_data, &store);
    }

    let values = unsafe { slice::from_raw_parts_mut(grad_f, n as usize) };
    // Should check if need to zero other entries in grad_f
    // Should check if we only need to upload constant values once (for jac at
    // least).
    // Should check if we can calculate on demand, ie use new_x to trigger other
    // states.
    // For f_grad looks like one first call values are not saved, but after
    // that they are.
    // g_grad doesn't have same issue
    // l_hess is completely over the place, (scaling going on?)
    // For large sums and multiplications, should consider adding new operator
    // that might include a constant factor.
    //println!("bef: {:?}", values);

    // Not sure if we need to clear, doing it anyway
    for v in values.iter_mut() {
        *v = 0.0;
    }

    // f_grad expects variables in order
    for (i, &v) in cb_data.model.obj.info.lin.iter().enumerate() {
        values[v] = cb_data.cache.obj_const.der1[i];
    }
    for (i, &v) in cb_data.model.obj.info.nlin.iter().enumerate() {
        values[v] = cb_data.cache.obj.der1[i];
    }
    //println!("aft: {:?}", values);
    1
}

extern fn g(
        n: ipopt::Index,
        x: *const ipopt::Number,
        new_x: ipopt::Bool,
        m: ipopt::Index,
        g: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe {
        &mut *(user_data as *mut IpoptCBData)
    };

    if n != cb_data.model.vars.len() as i32 {
        return 0;
    }
    if m != cb_data.model.cons.len() as i32 {
        return 0;
    }

    if new_x == 1 {
        let store = Store {
            vars: unsafe { slice::from_raw_parts(x, n as usize) },
            pars: cb_data.pars,
        };

        solve_obj(cb_data, &store);
        solve_cons(cb_data, &store);
    }

    let values = unsafe { slice::from_raw_parts_mut(g, m as usize) };

    for (i, col) in cb_data.cache.cons.iter().enumerate() {
        values[i] = col.val;
    }
    1
}

extern fn g_jac(
        n: ipopt::Index,
        x: *const ipopt::Number,
        new_x: ipopt::Bool,
        m: ipopt::Index,
        nele_jac: ipopt::Index,
        i_row: *mut ipopt::Index,
        j_col: *mut ipopt::Index,
        vals: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe {
        &mut *(user_data as *mut IpoptCBData)
    };

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
        if new_x == 1 {
            let store = Store {
                vars: unsafe { slice::from_raw_parts(x, n as usize) },
                pars: cb_data.pars,
            };

            solve_obj(cb_data, &store);
            solve_cons(cb_data, &store);
        }

        let values = unsafe {
            slice::from_raw_parts_mut(vals, nele_jac as usize)
        };

        //println!("bef: {:?}", values);
        // Could have put all the constant derivatives in one great big vector,
        // and then just copy that chunk over.  Would require different
        // sparsity ordering.
        let mut vind = 0_usize;
        for (i, col) in cb_data.cache.cons.iter().enumerate() {
            let sp = vind + cb_data.cache.cons_const[i].der1.len();
            let ed = sp + col.der1.len();
            values[vind..sp].copy_from_slice(&cb_data.cache.cons_const[i].der1);
            values[sp..ed].copy_from_slice(&col.der1);
            vind = ed;
        }
        //println!("aft: {:?}", values);
    }
    1
}

extern fn l_hess(
        n: ipopt::Index,
        x: *const ipopt::Number,
        new_x: ipopt::Bool,
        obj_factor: ipopt::Number,
        m: ipopt::Index,
        lambda: *const ipopt::Number,
        _new_lambda: ipopt::Bool,
        nele_hes: ipopt::Index,
        i_row: *mut ipopt::Index,
        j_col: *mut ipopt::Index,
        vals: *mut ipopt::Number,
        user_data: ipopt::UserDataPtr) -> ipopt::Bool {
    let cb_data: &mut IpoptCBData = unsafe {
        &mut *(user_data as *mut IpoptCBData)
    };

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
        for (vids, &ind) in &cb_data.cache.h_sparsity.sp {
            row[ind] = vids.0 as i32;
            col[ind] = vids.1 as i32;
        }
    } else {
        // Set values
        if new_x == 1 {
            let store = Store {
                vars: unsafe { slice::from_raw_parts(x, n as usize) },
                pars: cb_data.pars,
            };

            solve_obj(cb_data, &store);
            solve_cons(cb_data, &store);
        }

        let lam = unsafe { slice::from_raw_parts(lambda, m as usize) };
        let values = unsafe {
            slice::from_raw_parts_mut(vals, nele_hes as usize)
        };

        //println!("bef: {:?}", values);
        // Looks like this is required as values are non-zero and all over the
        // place on next callback (maybe scaled?)
        for v in values.iter_mut() {
            *v = 0.0;
        }

        let mut ind_pos = 0;
        for v in &cb_data.cache.obj_const.der2 {
            let vind = cb_data.cache.h_sparsity.obj_inds[ind_pos];
            values[vind] += obj_factor*v;
            ind_pos += 1;
        }

        for v in &cb_data.cache.obj.der2 {
            let vind = cb_data.cache.h_sparsity.obj_inds[ind_pos];
            values[vind] += obj_factor*v;
            ind_pos += 1;
        }

        for i in 0..(m as usize) {
            ind_pos = 0;

            for v in &cb_data.cache.cons_const[i].der2 {
                let vind = cb_data.cache.h_sparsity.cons_inds[i][ind_pos];
                values[vind] += lam[i]*v;
                ind_pos += 1;
            }

            for v in &cb_data.cache.cons[i].der2 {
                let vind = cb_data.cache.h_sparsity.cons_inds[i][ind_pos];
                values[vind] += lam[i]*v;
                ind_pos += 1;
            }
        }
        //println!("aft: {:?}", values);
    }
    1
}

#[cfg(test)]
mod tests {
    extern crate test;
    use expression::NumOpsF;
    use super::*;
    #[test]
    fn univar_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        m.set_obj(x*x);
        assert!(m.silence());
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.var(x) - 1.0).abs() < 1e-6);
            assert!((s.obj_val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn multivar_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(x*x + y*y + x*y);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.var(x) - 1.0).abs() < 1e-6);
            assert!((s.var(y) + 0.5).abs() < 1e-6);
            assert!((s.obj_val - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn equality_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(x*x + y*y + x*y);
        m.add_con(x + y, 0.75, 0.75);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.var(x) - 1.0).abs() < 1e-6);
            assert!((s.var(y) + 0.25).abs() < 1e-6);
            assert!((s.obj_val - 0.8125).abs() < 1e-6);
        }
    }

    #[test]
    fn inequality_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(1.0, 5.0, 0.0);
        let y = m.add_var(-1.0, 1.0, 0.0);
        m.set_obj(x*x + y*y + x*y);
        m.add_con(x + y, 0.25, 0.40);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.var(x) - 1.0).abs() < 1e-6);
            assert!((s.var(y) + 0.6).abs() < 1e-6);
            assert!((s.obj_val - 0.76).abs() < 1e-6);
        }
    }

    #[test]
    fn quad_constraint_problem() {
        let mut m = IpoptModel::new();
        let x = m.add_var(-10.0, 10.0, 0.0);
        let y = m.add_var(f64::NEG_INFINITY, f64::INFINITY, 0.0);
        m.set_obj(2.0*y);
        m.add_con(y - x*x + x, 0.0, f64::INFINITY);
        m.silence();
        let (stat, sol) = m.solve();
        assert_eq!(stat, SolutionStatus::Solved);
        assert!(sol.is_some());
        if let Some(ref s) = sol {
            assert!((s.var(x) - 0.5).abs() < 1e-5);
            assert!((s.var(y) + 0.25).abs() < 1e-5);
            assert!((s.obj_val + 0.5).abs() < 1e-4);
        }
    }

    #[bench]
    fn large_problem(b: &mut test::Bencher) {
        let n = 10;
        b.iter(|| {
            let mut m = IpoptModel::new();
            let mut xs = Vec::new();
            for _i in 0..n {
                xs.push(m.add_var(-1.5, 0.0, -0.5));
            }
            let mut obj = Film::from(0.0);
            for &x in &xs {
                obj = obj + (x - 1.0).powi(2);
            }
            m.set_obj(obj);
            for i in 0..(n-2) {
                let a = ((i + 2) as f64)/(n as f64);
                let e = (xs[i + 1].powi(2) + 1.5*xs[i + 1] - a)*xs[i + 2].cos()
                    - xs[i];
                m.add_con(e, 0.0, 0.0);
            }
            m.silence();
            m.solve();
        });
    }
}

use expression::{Film, Store, ID, Var, Par, WorkSpace, Retrieve};

//pub enum VarType {
//    Continuous,
//    Integer,
//    Binary,
//}

#[derive(Debug, Clone, Copy)]
pub struct Con(pub ID); // Constraint ID

pub trait Model {
    fn add_var(&mut self, lb: f64, ub: f64, init: f64) -> Var;
    fn add_par(&mut self, val: f64) -> Par;
    fn add_con(&mut self, film: Film, lb: f64, ub: f64) -> Con;
    fn set_obj(&mut self, film: Film);
    fn solve(&mut self) -> (SolutionStatus, Option<Solution>);
    fn warm_solve(&mut self, sol: Solution) ->
        (SolutionStatus, Option<Solution>);
}

pub trait MIModel {
    fn add_ivar(&mut self, lb: f64, ub: f64) -> Var;
    fn add_bvar(&mut self, lb: f64, ub: f64) -> Var;
}

#[derive(PartialEq, Debug)]
pub enum SolutionStatus {
    Solved,
    Infeasible,
    Error,
    Other,
}

pub struct Solution {
    pub obj_val: f64,
    pub store: Store,
    pub con_mult: Vec<f64>,
    pub var_lb_mult: Vec<f64>,
    pub var_ub_mult: Vec<f64>,
}

impl Solution {
    pub fn new() -> Solution {
        Solution {
            obj_val: 0.0,
            store: Store::new(),
            con_mult: Vec::new(),
            var_lb_mult: Vec::new(),
            var_ub_mult: Vec::new(),
        }
    }

    pub fn value(&self, film: &Film) -> f64 {
        let mut ws = WorkSpace::new(); // could pass this in
        film.ad(&Vec::new(), &Vec::new(), &self.store, &mut ws);
        if let Some(l) = ws.last() {
            l.val
        } else {
            0.0
        }
    }

    pub fn var(&self, Var(id): Var) -> f64 {
        self.store.get_var(id)
    }

    pub fn con_mult(&self, Con(cid): Con) -> f64 {
        self.con_mult[cid]
    }
    
    // Could write versions that take Film, and try and match ops[0] to Var
    pub fn var_lb_mult(&self, Var(vid): Var) -> f64 {
        self.var_lb_mult[vid]
    }
    
    pub fn var_ub_mult(&self, Var(vid): Var) -> f64 {
        self.var_ub_mult[vid]
    }
}

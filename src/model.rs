use expression::{Expr, Evaluate, Retrieve, Store};

pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

pub trait Model {
    fn add_var(&mut self, lb: f64, ub: f64) -> Expr;
    fn add_par(&mut self, val: f64) -> Expr;
    fn add_con(&mut self, expr: Expr, lb: f64, ub: f64) -> usize;
    fn set_obj(&mut self, expr: Expr);
    fn solve(&mut self) -> (SolutionStatus, Option<Solution>);
    fn warm_solve(&mut self, mut sol: Solution) ->
        (SolutionStatus, Option<Solution>);
}

pub trait MIModel {
    fn add_ivar(&mut self, lb: f64, ub: f64) -> Expr;
    fn add_bvar(&mut self, lb: f64, ub: f64) -> Expr;
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

    pub fn value(&self, expr: &Expr) -> f64 {
        expr.value(&self.store)
    }

    pub fn con_mult(&self, cid: usize) -> f64 {
        self.con_mult[cid]
    }
    
    pub fn var_lb_mult(&self, expr: &Expr) -> Option<f64> {
        if let &Expr::Variable(vid) = expr {
            Some(self.var_lb_mult[vid])
        } else {
            None
        }
    }
    
    pub fn var_ub_mult(&self, expr: &Expr) -> Option<f64> {
        if let &Expr::Variable(vid) = expr {
            Some(self.var_ub_mult[vid])
        } else {
            None
        }
    }
}

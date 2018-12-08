// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::expr::{Expr, Expression, Par, Retrieve, Store, Var, ID};

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
    fn add_con<E: Into<Expression>>(&mut self, expr: E, lb: f64, ub: f64) -> Con;
    fn set_obj<E: Into<Expression>>(&mut self, expr: E);
    fn set_par(&mut self, par: Par, val: f64);
    fn set_init(&mut self, var: Var, init: f64);
    fn solve(&mut self) -> (SolutionStatus, Option<Solution>);
    fn warm_solve(&mut self, sol: Solution) -> (SolutionStatus, Option<Solution>);
}

// Not used yet
//pub trait MIModel {
//    fn add_ivar(&mut self, lb: f64, ub: f64) -> Var;
//    fn add_bvar(&mut self, lb: f64, ub: f64) -> Var;
//}

#[derive(PartialEq, Debug)]
pub enum SolutionStatus {
    Solved,
    Infeasible,
    Error,
    Other,
}

#[derive(Default)]
pub struct Solution {
    pub obj_val: f64,
    pub store: Store,
    pub con_mult: Vec<f64>,
    pub var_lb_mult: Vec<f64>,
    pub var_ub_mult: Vec<f64>,
}

impl Solution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value(&self, expr: &Expr) -> f64 {
        let mut ns = Vec::new(); // could pass this in
        expr.eval(&self.store, &mut ns)
    }

    pub fn var(&self, v: Var) -> f64 {
        self.store.var(v)
    }

    pub fn con_mult(&self, Con(cid): Con) -> f64 {
        self.con_mult[cid]
    }

    // Could write versions that take Expr, and try and match ops[0] to Var
    pub fn var_lb_mult(&self, Var(vid): Var) -> f64 {
        self.var_lb_mult[vid]
    }

    pub fn var_ub_mult(&self, Var(vid): Var) -> f64 {
        self.var_ub_mult[vid]
    }
}

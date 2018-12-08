// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::expr::{Expression, Par, Retrieve, Store, Var, ID};

//pub enum VarType {
//    Continuous,
//    Integer,
//    Binary,
//}

/// Constraint ID.
#[derive(Debug, Clone, Copy)]
pub struct Con(pub ID);

/// Interface for a mathematical program with continuous variables.
pub trait Model {
    /// Add variable to model with lower / upper bounds and initial value.
    fn add_var(&mut self, lb: f64, ub: f64, init: f64) -> Var;
    /// Add parameter to model with starting value.
    fn add_par(&mut self, val: f64) -> Par;
    /// Add a constraint to the model with lower and upper bounds.
    ///
    /// To have no lower / upper bounds set them to `std::f64::NEG_INFINITY` /
    /// `std::f64::INFINITY` respectively.
    fn add_con<E: Into<Expression>>(&mut self, expr: E, lb: f64, ub: f64) -> Con;
    /// Set objective of model.
    fn set_obj<E: Into<Expression>>(&mut self, expr: E);
    /// Change a parameter's value.
    fn set_par(&mut self, par: Par, val: f64);
    /// Change the initial value of a variable.
    fn set_init(&mut self, var: Var, init: f64);
    /// Solve the model.
    fn solve(&mut self) -> (SolutionStatus, Option<Solution>);
    /// Solve the model using a previous solution as a warm start.
    fn warm_solve(&mut self, sol: Solution) -> (SolutionStatus, Option<Solution>);
}

// Not used yet
//pub trait MIModel {
//    fn add_ivar(&mut self, lb: f64, ub: f64) -> Var;
//    fn add_bvar(&mut self, lb: f64, ub: f64) -> Var;
//}

/// Status of the solution.
#[derive(PartialEq, Debug)]
pub enum SolutionStatus {
    Solved,
    Infeasible,
    Error,
    Other,
}

/// Data for a valid solution.
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

    /// Calculate the value of an expression using the solution.
    pub fn value(&self, expr: &Expression) -> f64 {
        match expr {
            Expression::Expr(e, _) => {
                let mut ns = Vec::new();
                e.eval(&self.store, &mut ns)
            }
            Expression::ExprSum(es) => {
                let mut ns = Vec::new();
                let mut val = 0.0;
                for (e, _) in es {
                    val += e.eval(&self.store, &mut ns);
                }
                val
            }
            Expression::ExprStatic(e) => {
                (e.f)(&self.store.vars, &self.store.pars)
            }
            Expression::ExprStaticSum(es) => {
                let mut val = 0.0;
                for e in es {
                    val += (e.f)(&self.store.vars, &self.store.pars);
                }
                val
            }
        }
    }

    /// Get the value of variable for solution.
    pub fn var(&self, v: Var) -> f64 {
        self.store.var(v)
    }

    /// Get the constraint KKT / Lagrange multiplier.
    pub fn con_mult(&self, Con(cid): Con) -> f64 {
        self.con_mult[cid]
    }

    // Could write versions that take Expr, and try and match ops[0] to Var
    /// Get the variable lower bound constraint KKT / Lagrange multiplier.
    pub fn var_lb_mult(&self, Var(vid): Var) -> f64 {
        self.var_lb_mult[vid]
    }

    /// Get the variable upper bound constraint KKT / Lagrange multiplier.
    pub fn var_ub_mult(&self, Var(vid): Var) -> f64 {
        self.var_ub_mult[vid]
    }
}

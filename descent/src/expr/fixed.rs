// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Could provide a separate calculation for constant entries, or at the minimum
// indicate if the entire first or second derivative is constant. If have
// one mega-function then might not be much benefit to keeping track if
// something is constant or not.
//
// Should Use Rc instead of Box if want to easily clone.

use super::{Expression, Var};

/// Fixed expression with pointers to functions to evaluated the expression
/// and its first and second derivatives.
///
/// The input variable / parameter slices to these functions should be large
/// enough to include the indices of the vars / pars for the expression.
pub struct ExprFix {
    /// Evaluate expression.
    pub f: Box<Fn(&[f64], &[f64]) -> f64>,
    /// Evaluate first derivative of expression.
    pub d1: Box<Fn(&[f64], &[f64], &mut [f64])>,
    /// Evaluate second derivative of expression.
    pub d2: Box<Fn(&[f64], &[f64], &mut [f64])>,
    /// Evaluate expression and its first and second derivatives in one go.
    pub all: Box<Fn(&[f64], &[f64], &mut [f64], &mut [f64]) -> f64>,
    /// First derivate sparsity / order of outputs.
    pub d1_sparsity: Vec<Var>,
    /// Second derivate sparsity / order of outputs.
    pub d2_sparsity: Vec<(Var, Var)>,
}

impl From<ExprFix> for Expression {
    fn from(v: ExprFix) -> Self {
        Expression::ExprFix(v)
    }
}

/// Represents the sum of multiple fixed expressions.
///
/// This enables some more runtime flexibility without having to resort to a
/// `ExprDyn`.
pub type ExprFixSum = Vec<ExprFix>;

impl From<ExprFix> for ExprFixSum {
    fn from(v: ExprFix) -> Self {
        vec![v]
    }
}

impl From<ExprFixSum> for Expression {
    fn from(v: ExprFixSum) -> Self {
        Expression::ExprFixSum(v)
    }
}

impl std::ops::Add<ExprFix> for ExprFix {
    type Output = ExprFixSum;

    fn add(self, other: ExprFix) -> ExprFixSum {
        vec![self, other]
    }
}

impl std::ops::Add<ExprFixSum> for ExprFix {
    type Output = ExprFixSum;

    fn add(self, mut other: ExprFixSum) -> ExprFixSum {
        other.push(self);
        other
    }
}

impl std::ops::Add<ExprFix> for ExprFixSum {
    type Output = ExprFixSum;

    fn add(mut self, other: ExprFix) -> ExprFixSum {
        self.push(other);
        self
    }
}

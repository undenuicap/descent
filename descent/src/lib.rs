// Copyright 2018 Paul Scott
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(test)] // enable benchmarking

//! Modelling interface to non-linear constrained optimisation.
//!
//! This module exposes more than the typical user will want to use (so that
//! other crates such as [descent_ipopt](../descent_ipopt/index.html) can
//! interface to it). The most relevant parts of this module are:
//!
//! - [Model](model/trait.Model.html) trait that solvers implement.
//! - [Var](expr/struct.Var.html) type that represents a model variable.
//! - [Par](expr/struct.Var.html) type that represents a model parameter.
//! - [Expression](expr/enum.Expression.html) most general type of expression for
//!   modelling constraints and objectives.
//! - [Solution](model/struct.Solution.html) type that stores and enables
//!   access to a solution.
//!
//! If you have nightly rust available, then the "fixed" form expressions that
//! have their first and second derivatives generated by a procedural macro is
//! the most performant approach to writing expressions. See the
//! [fixed](expr/fixed/index.html) sub-module.
//!
//! For greater runtime flexibility in constructing of expressions, use the
//! "dynamic" expressions instead in the [dynam](expr/dynam/index.html)
//! sub-module.
//!
//! Both types of expression can be used with the same model but not in the same
//! objective value or constriant.

pub mod expr;
pub mod model;

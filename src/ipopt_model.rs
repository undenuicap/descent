use expression::Expr;
use model::{Model, VarType};

struct Variable {
    id: usize,
    lb: f64,
    ub: f64,
    init: f64,
}

struct Constraint {
    id: usize,
    expr: Expr,
    lb: f64,
    ub: f64,
}

struct IpoptModel {
    vars: Vec<Variable>,
    cons: Vec<Constraint>,
    obj: Expr,
}

impl Model for IpoptModel {
    fn add_var(&mut self, lb: f64, ub: f64) -> Expr {
        let id = self.vars.len();
        self.vars.push(Variable {id: id, lb: lb, ub: ub, init: 0.0 });
        Expr::Variable(id)
    }

    fn add_con(&mut self, expr: Expr, lb: f64, ub: f64) {
        let id = self.cons.len();
        self.cons.push(Constraint {id: id, expr: expr, lb: lb, ub: ub });
    }

    fn set_obj(&mut self, expr: Expr) {
        self.obj = expr;
    }

    fn solve(&mut self) {
    }
}

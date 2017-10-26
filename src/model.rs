use expression::Expr;

pub enum VarType {
    Continuous,
    Integer,
    Binary,
}

pub trait Model {
    fn add_var(&mut self, lb: f64, ub: f64) -> Expr;
    fn add_par(&mut self, val: f64) -> Expr;
    fn add_con(&mut self, expr: Expr, lb: f64, ub: f64);
    fn set_obj(&mut self, expr: Expr);
    fn solve(&mut self);
}

pub trait MIModel {
    fn add_ivar(&mut self, lb: f64, ub: f64) -> Expr;
    fn add_bvar(&mut self, lb: f64, ub: f64) -> Expr;
}

use std;
use std::collections::HashSet;
use std::cmp::max;

pub type ID = usize;

// Should not expect an error, panic if out of bounds error.  We could add
// a verification call that gets done once, or on adding constraints and
// objective to model check that variables and parameters are valid.
pub trait Retrieve {
    fn get_var(&self, vid: ID) -> f64;
    fn get_par(&self, pid: ID) -> f64;
}

pub trait Evaluate {
    fn value(&self, ret: &Retrieve) -> f64;
    fn deriv(&self, ret: &Retrieve, x: ID) -> (f64, f64); // f, f_x,
    fn deriv2(&self, ret: &Retrieve, x: ID, y: ID)
        -> (f64, f64, f64, f64); // f, f_x, f_y, f_xy
}

#[derive(Debug, Clone, Copy)]
struct Var(ID);

#[derive(Debug, Clone, Copy)]
struct Par(ID);

#[derive(Debug, Clone)]
enum Operation {
    Add,
    Mul,
    Neg, // negate
    Pow(i32),
    Sin,
    Cos,
    Variable(Var),
    Parameter(Par),
    Float(f64),
    Integer(i32),
    // Could add shortcut versions:
    // AddVar(ID)
    // MulVar(ID)
    // AddPar(ID)
    // MulPar(ID)
    // AddFloat(f64)
    // MulFloat(f64)
}

// Version where implicitly use the value to the left
// Must start with terminal value for this to be valid
// Index must be less than index of entry in tape
#[derive(Debug, Clone)]
enum Oper {
    Add(usize),
    Mul(usize),
    Neg, // negate
    Pow(i32),
    Sin,
    Cos,
    Variable(Var),
    Parameter(Par),
    Float(f64),
}

#[derive(Debug, Clone, Default)]
pub struct Column {
    val: f64,
    der1: Vec<f64>,
    der2: Vec<f64>,
}

impl Column {
    pub fn new() -> Column {
        Column::default()
    }
}

pub type WorkSpace = Vec<Column>;

// Might want to pass this in chunks to ad(), so that can use same code to
// get values for lin and quad once.
// When doing this should consider setting all get_var accesses to nan.
// Can then check results to make sure that we haven't touched any variables.
#[derive(Debug, Clone, Default)]
pub struct FilmInfo {
    // Linear and quadratic respective first and second derivatives only need
    // to be computed once on parameter change.
    lin: Vec<Var>, // const derivative
    nlin: Vec<Var>, // non-const derivative
    // Below the usize are indices into nlin, a variable representation
    // that is local
    quad: Vec<(usize, usize)>, // const second derivative
    nquad: Vec<(usize, usize)>, // non-const second derivative
}

impl FilmInfo {
    pub fn new() -> FilmInfo {
        FilmInfo::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Film {
    ops: Vec<Oper>,
}

impl Film {
    pub fn new() -> Film {
        Film::default()
    }

    fn ad(&self, d1: &Vec<Var>, d2: &Vec<(usize, usize)>, ret: &Retrieve,
            ws: &mut WorkSpace) {
        use self::Oper::*;
        use self::{Var, Par};
        ws.resize(self.ops.len(), Column::new());
        for (i, op) in self.ops.iter().enumerate() {
            let (left, right) = ws.split_at_mut(i);
            let cur = &mut right[0]; // the i value from original
            cur.der1.resize(d1.len(), 0.0);
            cur.der2.resize(d2.len(), 0.0);
            match *op {
                Add(j) => {
                    let pre = &left[i - 1];
                    let oth = &left[j];
                    cur.val = pre.val + oth.val;
                    for k in 0..(d1.len()) {
                        cur.der1[k] = pre.der1[k] + oth.der1[k];
                    }
                    for k in 0..(d2.len()) {
                        cur.der2[k] = pre.der2[k] + oth.der2[k];
                    }
                },
                Mul(j) => {
                    let pre = &left[i - 1];
                    let oth = &left[j];
                    cur.val = pre.val*oth.val;
                    for k in 0..(d1.len()) {
                        cur.der1[k] = pre.der1[k]*oth.val
                                    + pre.val*oth.der1[k];
                    }
                    for (k, &(k1, k2)) in d2.iter().enumerate() {
                        cur.der2[k] = pre.der2[k]*oth.val
                                    + pre.val*oth.der2[k]
                                    + pre.der1[k1]*oth.der1[k2]
                                    + pre.der1[k2]*oth.der1[k1];
                    }
                },
                Neg => {
                    let pre = &left[i - 1];
                    cur.val = -pre.val;
                    for k in 0..(d1.len()) {
                        cur.der1[k] = -pre.der1[k];
                    }
                    for k in 0..(d2.len()) {
                        cur.der2[k] = -pre.der2[k];
                    }
                },
                Pow(p) => {
                    let pre = &left[i - 1];
                    match p {
                        0 => {
                            cur.val = 1.0;
                            for k in 0..(d1.len()) {
                                cur.der1[k] = 0.0;
                            }
                            for k in 0..(d2.len()) {
                                cur.der2[k] = 0.0;
                            }
                        },
                        1 => {
                            cur.val = pre.val;
                            for k in 0..(d1.len()) {
                                cur.der1[k] = pre.der1[k];
                            }
                            for k in 0..(d2.len()) {
                                cur.der2[k] = pre.der2[k];
                            }
                        },
                        _ => {
                            cur.val = pre.val.powi(p);
                            let vald = pre.val.powi(p - 1);
                            let valdd = pre.val.powi(p - 2);
                            for k in 0..(d1.len()) {
                                cur.der1[k] = f64::from(p)*pre.der1[k]*vald;
                            }
                            for (k, &(k1, k2)) in d2.iter().enumerate() {
                                cur.der2[k] = f64::from(p)*pre.der2[k]*vald
                                            + f64::from(p*(p - 1))
                                            *pre.der1[k1]*pre.der1[k2]
                                            *valdd;
                            }
                        },
                    }
                },
                Sin => {
                    let pre = &left[i - 1];
                    cur.val = pre.val.sin();
                    let valcos = pre.val.cos();
                    for k in 0..(d1.len()) {
                        cur.der1[k] = pre.der1[k]*valcos;
                    }
                    for (k, &(k1, k2)) in d2.iter().enumerate() {
                        cur.der2[k] = pre.der2[k]*valcos
                                    - pre.der1[k1]*pre.der1[k2]*cur.val;
                    }
                },
                Cos => {
                    let pre = &left[i - 1];
                    cur.val = pre.val.cos();
                    let valsin = pre.val.sin();
                    for k in 0..(d1.len()) {
                        cur.der1[k] = -pre.der1[k]*valsin;
                    }
                    for (k, &(k1, k2)) in d2.iter().enumerate() {
                        cur.der2[k] = -pre.der2[k]*valsin
                                    - pre.der1[k1]*pre.der1[k2]*cur.val;
                    }
                },
                Variable(Var(id)) => {
                    cur.val = ret.get_var(id);
                    for (k, &Var(did)) in d1.iter().enumerate() {
                        cur.der1[k] = if id == did { 1.0 } else { 0.0 };
                    }
                    for k in 0..(d2.len()) {
                        cur.der2[k] = 0.0;
                    }
                },
                Parameter(Par(id)) => {
                    cur.val = ret.get_par(id);
                    for k in 0..(d1.len()) {
                        cur.der1[k] = 0.0;
                    }
                    for k in 0..(d2.len()) {
                        cur.der2[k] = 0.0;
                    }
                },
                Float(val) => {
                    cur.val = val;
                    for k in 0..(d1.len()) {
                        cur.der1[k] = 0.0;
                    }
                    for k in 0..(d2.len()) {
                        cur.der2[k] = 0.0;
                    }
                },
            }
        }
    }

    fn get_info(&self) -> FilmInfo {
        use self::Oper::*;
        use self::{Var, Par};
        let mut lin: Vec<HashSet<ID>> = Vec::new();
        let mut quad: Vec<HashSet<(ID, ID)>> = Vec::new();
        let mut nquad: Vec<HashSet<(ID, ID)>> = Vec::new();
        for (i, op) in self.ops.iter().enumerate() {
            match *op {
                Add(j) => {
                    let l = lin[i - 1].union(&lin[j]).cloned().collect();
                    lin.push(l);
                    let q = quad[i - 1].union(&quad[j]).cloned().collect();
                    quad.push(q);
                    let nq = nquad[i - 1].union(&nquad[j]).cloned().collect();
                    nquad.push(nq);
                },
                _ => {
                },
            }
        }
        let mut info = FilmInfo::new();
        info
    }
}

#[derive(Debug, Clone)]
pub struct Tape {
    ops: Vec<Operation>,
    n_max: usize, // maximum stack size
    n_end: usize, // stack size at end of tape
}

impl Tape {
    // Consider empty tape has a meaning (some sensible default)
    fn new() -> Tape {
        Tape {
            ops: vec![Operation::Float(0.0)],
            n_max: 1, // required stack size (1 to allow empty tape)
            n_end: 1, // stack size at end of tape (1 if valid non-empty tape)
        }
    }

    fn empty() -> Tape {
        Tape {
            ops: Vec::new(),
            n_max: 1, // required stack size (1 to allow empty tape)
            n_end: 0, // stack size at end of tape (1 if valid non-empty tape)
        }
    }

    fn add_op(&mut self, op: Operation) {
        use self::Operation::*;
        match op {
            Add | Mul => self.n_end -= 1,
            Variable(_) | Parameter(_) | Float(_) | Integer(_) => {
                self.n_end += 1;
                self.n_max = max(self.n_max, self.n_end);
            },
            _ => (),
        }
        self.ops.push(op);
    }

    // Should only call if both non-empty
    fn append(&mut self, other: &mut Tape) {
        self.ops.append(&mut other.ops);
        self.n_max = max(self.n_max, self.n_end + other.n_max);
        self.n_end += other.n_end;
    }

    fn deriv_stack(&self, st: &mut Vec<(f64, f64)>, ret: &Retrieve, x: ID)
            -> (f64, f64) {
        use self::Operation::*;
        use self::{Var, Par};
        let mut i = 0_usize;
        st[0] = (0.0, 0.0);
        for op in &self.ops {
            match *op {
                Add => {
                    st[i - 2].0 += st[i - 1].0;
                    st[i - 2].1 += st[i - 1].1;
                    i -= 1;
                },
                Mul => {
                    let r0 = st[i - 2];
                    let r1 = st[i - 1];
                    st[i - 2] = (r0.0*r1.0, r0.0*r1.1 + r1.0*r0.1);
                    i -= 1;
                },
                Neg => {
                    st[i - 1].0 = -st[i - 1].0;
                    st[i - 1].1 = -st[i - 1].1;
                },
                Pow(p) => {
                    match p {
                        0 => st[i - 1] = (1.0, 0.0),
                        1 => (),
                        _ => {
                            let r = st[i - 1];
                            st[i - 1] = (r.0.powi(p),
                                f64::from(p)*r.0.powi(p - 1)*r.1);
                        },
                    }
                },
                Sin => {
                    let r = st[i - 1];
                    st[i - 1] = (r.0.sin(), r.0.cos()*r.1);
                },
                Cos => {
                    let r = st[i - 1];
                    st[i - 1] = (r.0.cos(), -r.0.sin()*r.1);
                },
                Variable(Var(id)) => {
                    st[i] = (ret.get_var(id),
                        if id == x { 1.0 } else { 0.0 });
                    i += 1;
                },
                Parameter(Par(id)) => {
                    st[i] = (ret.get_par(id), 0.0);
                    i += 1;
                },
                Float(v) => {
                    st[i] = (v, 0.0);
                    i += 1;
                },
                Integer(v) => {
                    st[i] = (f64::from(v), 0.0);
                    i += 1;
                },
            }
        }
        st[0]
    }
}

impl From<Var> for Tape {
    fn from(v: Var) -> Tape {
        Tape {
            ops: vec![self::Operation::Variable(v)],
            n_max: 1,
            n_end: 1,
        }
    }
}

impl Evaluate for Tape {
    fn value(&self, ret: &Retrieve) -> f64 {
        use self::Operation::*;
        use self::{Var, Par};
        let mut st: Vec<f64> = Vec::new();
        st.resize(self.n_max, 0.0);
        let mut i = 0_usize;
        for op in &self.ops {
            match *op {
                Add => {
                    st[i - 2] += st[i - 1];
                    i -= 1;
                },
                Mul => {
                    st[i - 2] *= st[i - 1];
                    i -= 1;
                },
                Neg => st[i - 1] = -st[i - 1],
                Pow(p) => st[i - 1] = st[i - 1].powi(p),
                Sin => st[i - 1] = st[i - 1].sin(),
                Cos => st[i - 1] = st[i - 1].cos(),
                Variable(Var(id)) => {
                    st[i] = ret.get_var(id);
                    i += 1;
                },
                Parameter(Par(id)) => {
                    st[i] = ret.get_par(id);
                    i += 1;
                },
                Float(v) => {
                    st[i] = v;
                    i += 1;
                },
                Integer(v) => {
                    st[i] = f64::from(v);
                    i += 1;
                },
            }
        }
        st[0]
    }

    fn deriv(&self, ret: &Retrieve, x: ID) -> (f64, f64) {
        use self::Operation::*;
        use self::{Var, Par};
        let mut st: Vec<(f64, f64)> = Vec::new();
        st.resize(self.n_max, (0.0, 0.0));
        let mut i = 0_usize;
        for op in &self.ops {
            match *op {
                Add => {
                    st[i - 2].0 += st[i - 1].0;
                    st[i - 2].1 += st[i - 1].1;
                    i -= 1;
                },
                Mul => {
                    let r0 = st[i - 2];
                    let r1 = st[i - 1];
                    st[i - 2] = (r0.0*r1.0, r0.0*r1.1 + r1.0*r0.1);
                    i -= 1;
                },
                Neg => {
                    st[i - 1].0 = -st[i - 1].0;
                    st[i - 1].1 = -st[i - 1].1;
                },
                Pow(p) => {
                    match p {
                        0 => st[i - 1] = (1.0, 0.0),
                        1 => (),
                        _ => {
                            let r = st[i - 1];
                            st[i - 1] = (r.0.powi(p),
                                f64::from(p)*r.0.powi(p - 1)*r.1);
                        },
                    }
                },
                Sin => {
                    let r = st[i - 1];
                    st[i - 1] = (r.0.sin(), r.0.cos()*r.1);
                },
                Cos => {
                    let r = st[i - 1];
                    st[i - 1] = (r.0.cos(), -r.0.sin()*r.1);
                },
                Variable(Var(id)) => {
                    st[i] = (ret.get_var(id),
                        if id == x { 1.0 } else { 0.0 });
                    i += 1;
                },
                Parameter(Par(id)) => {
                    st[i] = (ret.get_par(id), 0.0);
                    i += 1;
                },
                Float(v) => {
                    st[i] = (v, 0.0);
                    i += 1;
                },
                Integer(v) => {
                    st[i] = (f64::from(v), 0.0);
                    i += 1;
                },
            }
        }
        st[0]
    }

    fn deriv2(&self, _ret: &Retrieve, _x: ID, _y: ID) -> (f64, f64, f64, f64) {
        (0.0, 0.0, 0.0, 0.0)
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Neg(Box<Expr>), // negate
    Pow(Box<Expr>, i32),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Variable(ID),
    Parameter(ID),
    Float(f64),
    Integer(i32),
}

// Have to use trait because straight fn overloading not possible
pub trait NumOps {
    fn powi(self, p: i32) -> Expr;
    fn sin(self) -> Expr;
    fn cos(self) -> Expr;
}

impl NumOps for Expr {
    fn powi(self, p: i32) -> Expr {
        Expr::Pow(Box::new(self), p)
    }

    fn sin(self) -> Expr {
        Expr::Sin(Box::new(self))
    }

    fn cos(self) -> Expr {
        Expr::Cos(Box::new(self))
    }
}

impl<'a> NumOps for &'a Expr {
    fn powi(self, p: i32) -> Expr {
        Expr::Pow(Box::new(self.clone()), p)
    }

    fn sin(self) -> Expr {
        Expr::Sin(Box::new(self.clone()))
    }

    fn cos(self) -> Expr {
        Expr::Cos(Box::new(self.clone()))
    }
}

// Have to use trait because straight fn overloading not possible
pub trait NumOpsT {
    fn powi(self, p: i32) -> Tape;
    fn sin(self) -> Tape;
    fn cos(self) -> Tape;
}

impl NumOpsT for Var {
    fn powi(self, p: i32) -> Tape {
        let mut t = Tape::empty();
        t.add_op(Operation::Variable(self));
        t.add_op(Operation::Pow(p));
        t
    }

    fn sin(self) -> Tape {
        let mut t = Tape::empty();
        t.add_op(Operation::Variable(self));
        t.add_op(Operation::Sin);
        t
    }

    fn cos(self) -> Tape {
        let mut t = Tape::empty();
        t.add_op(Operation::Variable(self));
        t.add_op(Operation::Cos);
        t
    }
}

impl NumOpsT for Par {
    fn powi(self, p: i32) -> Tape {
        let mut t = Tape::empty();
        t.add_op(Operation::Parameter(self));
        t.add_op(Operation::Pow(p));
        t
    }

    fn sin(self) -> Tape {
        let mut t = Tape::empty();
        t.add_op(Operation::Parameter(self));
        t.add_op(Operation::Sin);
        t
    }

    fn cos(self) -> Tape {
        let mut t = Tape::empty();
        t.add_op(Operation::Parameter(self));
        t.add_op(Operation::Cos);
        t
    }
}

impl NumOpsT for Tape {
    fn powi(mut self, p: i32) -> Tape {
        self.add_op(Operation::Pow(p));
        self
    }

    fn sin(mut self) -> Tape {
        self.add_op(Operation::Sin);
        self
    }

    fn cos(mut self) -> Tape {
        self.add_op(Operation::Cos);
        self
    }
}

// As used think this might be filling out top right, not bottom left of hessian
fn order(a: ID, b: ID) -> (ID, ID) {
    if a > b { (b, a) } else { (a, b) }
}

#[derive(Debug, PartialEq, Clone, Default)]
pub struct Degree {
    pub linear: HashSet<ID>,
    //quadratic: HashSet<(ID, ID)>,
    pub higher: HashSet<(ID, ID)>,
}

impl Degree {
    pub fn new() -> Degree {
        Degree::default()
    }

    pub fn is_empty(&self) -> bool {
        self.linear.is_empty() && self.higher.is_empty()
    }

    pub fn add(&self, other: &Degree) -> Degree {
        Degree {
            linear: self.linear.union(&(other.linear)).cloned().collect(),
            higher: self.higher.union(&(other.higher)).cloned().collect(),
        }
    }

    pub fn mul(&self, other: &Degree) -> Degree {
        // If one has no entries then can just add
        if self.is_empty() {
            other.clone()
        } else if other.is_empty() {
            self.clone()
        } else {
            let mut higher = HashSet::new();

            // linear x linear
            for s in self.linear.iter().cloned() {
                for o in other.linear.iter().cloned() {
                    higher.insert(order(s, o));
                }
            }

            // linear x higher
            for &(s1, s2) in &(self.higher) {
                for o in other.linear.iter().cloned() {
                    higher.insert(order(s1, o));
                    higher.insert(order(s2, o));
                }
                higher.insert((s1, s2));
            }

            // linear x higher
            for &(o1, o2) in &(other.higher) {
                for s in self.linear.iter().cloned() {
                    higher.insert(order(o1, s));
                    higher.insert(order(o2, s));
                }
                higher.insert((o1, o2));
            }

            // higher x higher
            for &(s1, s2) in &(self.higher) {
                for &(o1, o2) in &(other.higher) {
                    higher.insert(order(s1, o1));
                    higher.insert(order(s1, o2));
                    higher.insert(order(s2, o1));
                    higher.insert(order(s2, o2));
                }
            }

            Degree {
                linear: HashSet::new(), // have all been promoted
                higher: higher,
            }
        }
    }
}

impl Expr {
    pub fn visit_top(&self, f: &mut FnMut(&Expr) -> ()) {
        use expression::Expr::*;
        f(self);
        match *self {
            Add(ref es) | Mul(ref es) => for e in es { e.visit_top(f); },
            Neg(ref e) | Pow(ref e, _) | Sin(ref e) | Cos(ref e) =>
                e.visit_top(f),
            _ => (),
        };
    }

    pub fn visit_bot(&self, f: &mut FnMut(&Expr) -> ()) {
        use expression::Expr::*;
        match *self {
            Add(ref es) | Mul(ref es)  => for e in es { e.visit_bot(f); },
            Neg(ref e) | Pow(ref e, _) | Sin(ref e) | Cos(ref e) =>
                e.visit_bot(f),
            _ => (),
        };
        f(self);
    }

    pub fn variables(&self) -> HashSet<ID> {
        use expression::Expr::*;
        let mut set = HashSet::new();

        self.visit_top(&mut |e: &Expr| {
            if let Variable(id) = *e {
                set.insert(id);
            }
        });
        set
    }

    pub fn degree(&self) -> Degree {
        use expression::Expr::*;
        match *self {
            Add(ref es) => es.iter().fold(Degree::new(), |a, e| {
                a.add(&(e.degree()))
            }),
            Mul(ref es) => es.iter().fold(Degree::new(), |a, e| {
                a.mul(&(e.degree()))
            }),
            Neg(ref e) => e.degree(),
            Pow(ref e, p) => {
                let d = e.degree();
                match p {
                    0 => Degree::new(), // cleared
                    1 => d,
                    _ => d.mul(&d),
                }
            },
            Sin(ref e) | Cos(ref e) => {
                let d = e.degree();
                d.mul(&d)
            },
            Variable(id) => {
                let mut d = Degree::new();
                d.linear.insert(id);
                d
            },
            _ => Degree::new(),
        }
    }
}

impl Evaluate for Expr {
    fn value(&self, ret: &Retrieve) -> f64 {
        use expression::Expr::*;
        match *self {
            Add(ref es) => es.iter().fold(0.0, |a, e| { a + e.value(ret) }),
            Mul(ref es) => es.iter().fold(1.0, |a, e| { a*e.value(ret) }),
            Neg(ref e) => -e.value(ret),
            Pow(ref e, p) => e.value(ret).powi(p),
            Sin(ref e) => e.value(ret).sin(),
            Cos(ref e) => e.value(ret).cos(),
            Variable(id) => ret.get_var(id),
            Parameter(id) => ret.get_par(id),
            Float(v) => v,
            Integer(v) => f64::from(v),
        }
    }

    fn deriv(&self, ret: &Retrieve, x: ID) -> (f64, f64) {
        use expression::Expr::*;
        match *self {
            Add(ref es) =>
                es.iter().fold((0.0, 0.0), |a, n| {
                    let r = n.deriv(ret, x);
                    (a.0 + r.0, a.1 + r.1)
                }),
            Mul(ref es) =>
                es.iter().fold((1.0, 0.0), |a, n| {
                    let r = n.deriv(ret, x);
                    (a.0*r.0, a.0*r.1 + r.0*a.1)
                }),
            Neg(ref e) => {
                let r = e.deriv(ret, x);
                (-r.0, -r.1)
            },
            Pow(ref e, p) => {
                let r = e.deriv(ret, x);
                match p {
                    0 => (1.0, 0.0),
                    1 => r,
                    _ => (r.0.powi(p), f64::from(p)*r.0.powi(p - 1)*r.1),
                }
            },
            Sin(ref e) => {
                let r = e.deriv(ret, x);
                (r.0.sin(), r.0.cos()*r.1)
            },
            Cos(ref e) => {
                let r = e.deriv(ret, x);
                (r.0.cos(), -r.0.sin()*r.1)
            },
            Variable(id) =>
                (ret.get_var(id), if id == x { 1.0 } else { 0.0 }),
            Parameter(id) => (ret.get_par(id), 0.0),
            Float(v) => (v, 0.0),
            Integer(v) => (f64::from(v), 0.0),
        }
    }

    fn deriv2(&self, ret: &Retrieve, x: ID, y: ID) -> (f64, f64, f64, f64) {
        use expression::Expr::*;
        match *self {
            Add(ref es) =>
                es.iter().fold((0.0, 0.0, 0.0, 0.0), |a, n| {
                    let r = n.deriv2(ret, x, y);
                    (a.0 + r.0, a.1 + r.1, a.2 + r.2, a.3 + r.3)
                }),
            Mul(ref es) =>
                es.iter().fold((1.0, 0.0, 0.0, 0.0), |a, n| {
                    let r = n.deriv2(ret, x, y);
                    (a.0*r.0, a.0*r.1 + r.0*a.1, a.0*r.2 + r.0*a.2,
                         a.0*r.3 + r.0*a.3 + a.1*r.2 + a.2*r.1)
                }),
            Neg(ref e) => {
                let r = e.deriv2(ret, x, y);
                (-r.0, -r.1, -r.2, -r.3)
            },
            Pow(ref e, p) => {
                let r = e.deriv2(ret, x, y);
                match p {
                    0 => (1.0, 0.0, 0.0, 0.0),
                    1 => r,
                    _ => {
                        let fl = r.0.powi(p - 1);
                        let fll = r.0.powi(p - 2);
                        (r.0.powi(p), f64::from(p)*fl*r.1, f64::from(p)*fl*r.2,
                            f64::from(p*p  - p)*fll*r.1*r.2
                            + f64::from(p)*fl*r.3)
                    },
                }
            },
            Sin(ref e) => {
                let r = e.deriv2(ret, x, y);
                (r.0.sin(), r.0.cos()*r.1, r.0.cos()*r.2,
                    -r.1*r.2*r.0.sin() + r.0.cos()*r.3)
            },
            Cos(ref e) => {
                let r = e.deriv2(ret, x, y);
                (r.0.cos(), -r.0.sin()*r.1, -r.0.sin()*r.2,
                    -r.1*r.2*r.0.cos() - r.0.sin()*r.3)
            },
            Variable(id) =>
                (ret.get_var(id),
                    if id == x { 1.0 } else { 0.0 },
                    if id == y { 1.0 } else { 0.0 },
                    0.0),
            Parameter(id) => (ret.get_par(id), 0.0, 0.0, 0.0),
            Float(v) => (v, 0.0, 0.0, 0.0),
            Integer(v) => (f64::from(v), 0.0, 0.0, 0.0),
        }
    }
}

macro_rules! binary_ops {
    ( $T:ident, $f:ident, &$U:ty, &$V:ty, $O:ty ) => {
        impl<'a, 'b> std::ops::$T<&'b $V> for &'a $U {
            type Output = $O;

            fn $f(self, other: &'b $V) -> $O {
                self.clone().$f(other.clone())
            }
        }
    };
    ( $T:ident, $f:ident, &$U:ty, $V:ty, $O:ty ) => {
        impl<'a> std::ops::$T<$V> for &'a $U {
            type Output = $O;

            fn $f(self, other: $V) -> $O {
                self.clone().$f(other)
            }
        }
    };
    ( $T:ident, $f:ident, $U:ty, &$V:ty, $O:ty ) => {
        impl<'b> std::ops::$T<&'b $V> for $U {
            type Output = $O;

            fn $f(self, other: &'b $V) -> $O {
                self.$f(other.clone())
            }
        }
    };
    ( $T:ident, $f:ident, $U:ty, $V:ty, $O:ty ) => {
        impl std::ops::$T<$V> for $U {
            type Output = $O;

            fn $f(self, other: $V) -> $O {
                self.$f(other)
            }
        }
    };
}

macro_rules! binary_ops_g3 {
    ( $T:ident, $f:ident, $U:ty, $V:ty, $O:ty ) => {
        binary_ops!($T, $f, &$U, &$V, $O);
        binary_ops!($T, $f, $U, &$V, $O);
        binary_ops!($T, $f, &$U, $V, $O);
    }
}

macro_rules! binary_ops_self_cast {
    ( $T:ident, $f:ident, &$U:ty, &$V:ty, $O:ty, $C:expr ) => {
        impl<'a, 'b> std::ops::$T<&'b $V> for &'a $U {
            type Output = $O;

            fn $f(self, other: &'b $V) -> $O {
                $C(self.clone()).$f(other.clone())
            }
        }
    };
    ( $T:ident, $f:ident, &$U:ty, $V:ty, $O:ty, $C:expr ) => {
        impl<'a> std::ops::$T<$V> for &'a $U {
            type Output = $O;

            fn $f(self, other: $V) -> $O {
                $C(self.clone()).$f(other)
            }
        }
    };
    ( $T:ident, $f:ident, $U:ty, &$V:ty, $O:ty, $C:expr ) => {
        impl<'b> std::ops::$T<&'b $V> for $U {
            type Output = $O;

            fn $f(self, other: &'b $V) -> $O {
                $C(self).$f(other.clone())
            }
        }
    };
    ( $T:ident, $f:ident, $U:ty, $V:ty, $O:ty, $C:expr ) => {
        impl std::ops::$T<$V> for $U {
            type Output = $O;

            fn $f(self, other: $V) -> $O {
                $C(self).$f(other)
            }
        }
    };
}

macro_rules! binary_ops_other_cast {
    ( $T:ident, $f:ident, &$U:ty, &$V:ty, $O:ty, $C:expr ) => {
        impl<'a, 'b> std::ops::$T<&'b $V> for &'a $U {
            type Output = $O;

            fn $f(self, other: &'b $V) -> $O {
                self.clone().$f($C(other.clone()))
            }
        }
    };
    ( $T:ident, $f:ident, &$U:ty, $V:ty, $O:ty, $C:expr ) => {
        impl<'a> std::ops::$T<$V> for &'a $U {
            type Output = $O;

            fn $f(self, other: $V) -> $O {
                self.clone().$f($C(other))
            }
        }
    };
    ( $T:ident, $f:ident, $U:ty, &$V:ty, $O:ty, $C:expr ) => {
        impl<'b> std::ops::$T<&'b $V> for $U {
            type Output = $O;

            fn $f(self, other: &'b $V) -> $O {
                self.$f($C(other.clone()))
            }
        }
    };
    ( $T:ident, $f:ident, $U:ty, $V:ty, $O:ty, $C:expr ) => {
        impl std::ops::$T<$V> for $U {
            type Output = $O;

            fn $f(self, other: $V) -> $O {
                self.$f($C(other))
            }
        }
    };
}

// Assuming that $V has copy trait and is value to be converted
macro_rules! binary_ops_cast_g4 {
    ( $T:ident, $f:ident, $U:ty, $V:ty, $O:ty, $C:expr ) => {
        binary_ops_other_cast!($T, $f, $U, $V, $O, $C);
        binary_ops_other_cast!($T, $f, &$U, $V, $O, $C);
        binary_ops_self_cast!($T, $f, $V, $U, $O, $C);
        binary_ops_self_cast!($T, $f, $V, &$U, $O, $C);
    }
}


impl std::ops::Add<Expr> for Expr {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        // Can optimise, check if child is Add and combine
        match self {
            Expr::Add(mut es) => {
                match other {
                    Expr::Add(oes) => {
                        es.extend(oes);
                        Expr::Add(es)
                    },
                    _ => {
                        es.push(other);
                        Expr::Add(es)
                    },
                }
            },
            _ => {
                match other {
                    Expr::Add(mut oes) => {
                        oes.push(self); // out of order now
                        Expr::Add(oes)
                    },
                    _ => {
                        Expr::Add(vec![self, other])
                    },
                }
            },
        }
    }
}

impl std::ops::Sub<Expr> for Expr {
    type Output = Expr;

    fn sub(self, other: Expr) -> Expr {
        match self {
            Expr::Add(mut es) => {
                es.push(-other);
                Expr::Add(es)
            },
            _ => {
                Expr::Add(vec![self, -other])
            },
        }
    }
}

impl std::ops::Mul<Expr> for Expr {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        // Can optimise, check if child is Mul and combine
        match self {
            Expr::Mul(mut es) => {
                match other {
                    Expr::Mul(oes) => {
                        es.extend(oes);
                        Expr::Mul(es)
                    },
                    _ => {
                        es.push(other);
                        Expr::Mul(es)
                    },
                }
            },
            _ => {
                match other {
                    Expr::Mul(mut oes) => {
                        oes.push(self); // out of order now
                        Expr::Mul(oes)
                    },
                    _ => {
                        Expr::Mul(vec![self, other])
                    },
                }
            },
        }
    }
}

impl std::ops::Div<Expr> for Expr {
    type Output = Expr;

    fn div(self, other: Expr) -> Expr {
        match self {
            Expr::Mul(mut es) => {
                es.push(Expr::Pow(Box::new(other), -1));
                Expr::Mul(es)
            },
            _ => {
                Expr::Mul(vec![self, Expr::Pow(Box::new(other), -1)])
            },
        }
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        match self {
            Expr::Neg(es) => *es,
            Expr::Integer(v) => Expr::Integer(-v),
            Expr::Float(v) => Expr::Float(-v),
            _ => Expr::Neg(Box::new(self)),
        }
    }
}

impl<'a> std::ops::Neg for &'a Expr {
    type Output = Expr;

    fn neg(self) -> Expr {
        self.clone().neg()
    }
}

binary_ops_g3!(Add, add, Expr, Expr, Expr);
binary_ops_g3!(Sub, sub, Expr, Expr, Expr);
binary_ops_g3!(Mul, mul, Expr, Expr, Expr);
binary_ops_g3!(Div, div, Expr, Expr, Expr);

binary_ops_cast_g4!(Add, add, Expr, i32, Expr, Expr::Integer);
binary_ops_cast_g4!(Sub, sub, Expr, i32, Expr, Expr::Integer);
binary_ops_cast_g4!(Mul, mul, Expr, i32, Expr, Expr::Integer);
binary_ops_cast_g4!(Div, div, Expr, i32, Expr, Expr::Integer);

binary_ops_cast_g4!(Add, add, Expr, f64, Expr, Expr::Float);
binary_ops_cast_g4!(Sub, sub, Expr, f64, Expr, Expr::Float);
binary_ops_cast_g4!(Mul, mul, Expr, f64, Expr, Expr::Float);
binary_ops_cast_g4!(Div, div, Expr, f64, Expr, Expr::Float);

macro_rules! b_ops {
    ( $U:ty, $C:expr ) => {
        impl std::ops::Add<$U> for Tape {
            type Output = Tape;

            fn add(mut self, other: $U) -> Tape {
                self.add_op($C(other));
                self.add_op(Operation::Add);
                self
            }
        }

        impl std::ops::Add<Tape> for $U {
            type Output = Tape;

            fn add(self, mut other: Tape) -> Tape {
                other.add_op($C(self));
                other.add_op(Operation::Add);
                other
            }
        }

        impl std::ops::Mul<$U> for Tape {
            type Output = Tape;

            fn mul(mut self, other: $U) -> Tape {
                self.add_op($C(other));
                self.add_op(Operation::Mul);
                self
            }
        }

        impl std::ops::Mul<Tape> for $U {
            type Output = Tape;

            fn mul(self, mut other: Tape) -> Tape {
                other.add_op($C(self));
                other.add_op(Operation::Mul);
                other
            }
        }

        impl std::ops::Sub<$U> for Tape {
            type Output = Tape;

            fn sub(mut self, other: $U) -> Tape {
                self.add_op($C(other));
                self.add_op(Operation::Neg);
                self.add_op(Operation::Add);
                self
            }
        }

        impl std::ops::Sub<Tape> for $U {
            type Output = Tape;

            fn sub(self, mut other: Tape) -> Tape {
                other.add_op(Operation::Neg);
                other.add_op($C(self));
                other.add_op(Operation::Add);
                other
            }
        }
    };
}

b_ops!(f64, Operation::Float);
b_ops!(i32, Operation::Integer);
b_ops!(Var, Operation::Variable);
b_ops!(Par, Operation::Parameter);

impl std::ops::Add<Tape> for Tape {
    type Output = Tape;

    fn add(mut self, mut other: Tape) -> Tape {
        if self.ops.is_empty() {
            other
        } else if other.ops.is_empty() {
            self
        } else {
            self.append(&mut other);
            self.add_op(Operation::Add);
            self
        }
    }
}

impl std::ops::Sub<Tape> for Tape {
    type Output = Tape;

    fn sub(mut self, mut other: Tape) -> Tape {
        if self.ops.is_empty() {
            other
        } else if other.ops.is_empty() {
            self
        } else {
            self.append(&mut other);
            self.add_op(Operation::Neg);
            self.add_op(Operation::Add);
            self
        }
    }
}

impl std::ops::Mul<Tape> for Tape {
    type Output = Tape;

    fn mul(mut self, mut other: Tape) -> Tape {
        if self.ops.is_empty() {
            other
        } else if other.ops.is_empty() {
            self
        } else {
            self.append(&mut other);
            self.add_op(Operation::Mul);
            self
        }
    }
}

pub struct Store {
    pub vars: Vec<f64>,
    pub pars: Vec<f64>,
}

impl Store {
    pub fn new() -> Self {
        Store { vars: Vec::new(), pars: Vec::new() }
    }
}

impl Retrieve for Store {
    fn get_var(&self, vid: ID) -> f64 {
        self.vars[vid]
    }

    fn get_par(&self, pid: ID) -> f64 {
        self.pars[pid]
    }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use super::*;

    #[test]
    fn addition() {
        use expression::Expr::*;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        assert_eq!(Float(1.0).value(&store), 1.0);

        assert_eq!(Variable(0).value(&store), 5.0);
        assert_eq!(Variable(0).deriv(&store, 0_usize), (5.0, 1.0));
        assert_eq!(Variable(0).deriv(&store, 1_usize), (5.0, 0.0));

        assert_eq!(Parameter(0).value(&store), 4.0);

        let v = Variable(0);
        let p = Parameter(0);

        assert_eq!((&v + &p).value(&store), 9.0);

        assert_eq!((&v + &p + 5).value(&store), 14.0);
        assert_eq!((3 + &v + &p + 5).value(&store), 17.0);

        assert_eq!((&v + &p + 5.0).value(&store), 14.0);
        assert_eq!((3.0 + &v + &p + 5.0).value(&store), 17.0);
    }

    #[test]
    fn subtraction() {
        use expression::Expr::*;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        let v = Variable(0);
        let p = Parameter(0);

        assert_eq!((&v - &p).value(&store), 1.0);

        assert_eq!((&v + &p - 5).value(&store), 4.0);
        assert_eq!((3 - &v - &p - 5).value(&store), -11.0);
    }

    #[test]
    fn multiplication() {
        use expression::Expr::*;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(2.0);
        store.pars.push(4.0);

        assert_eq!((Variable(0)*Parameter(0)*5).value(&store), 100.0);
        assert_eq!((Variable(0)*Parameter(0)).deriv(&store, 0_usize),
            (20.0, 4.0));
        assert_eq!((Variable(0)*Parameter(0)).deriv(&store, 1_usize),
            (20.0, 0.0));
        assert_eq!((Variable(0)*(7 + Variable(0))).deriv(&store, 0_usize),
            (60.0, 17.0));
        assert_eq!((Variable(0)*(7 + Variable(1))).deriv(&store, 0_usize),
            (45.0, 9.0));
        assert_eq!((Variable(0)*(7 + Variable(1))).deriv(&store, 1_usize),
            (45.0, 5.0));

        assert_eq!((Variable(0)*Variable(1)).deriv2(
                &store, 0_usize, 1_usize), (10.0, 2.0, 5.0, 1.0));
        assert_eq!((Variable(0)*(7 + 3.0*Variable(1))).deriv2(
                &store, 0_usize, 1_usize), (65.0, 13.0, 15.0, 3.0));
        assert_eq!((Variable(0)*(7 + 3.0*Variable(1))).deriv2(
                &store, 0_usize, 0_usize), (65.0, 13.0, 13.0, 0.0));
    }

    #[test]
    fn division() {
        use expression::Expr::*;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        assert_eq!((Variable(0)/Parameter(0)).value(&store), 5.0/4.0);

        assert_eq!((Integer(5)/5.0).value(&store), 1.0);
        assert_eq!((1/Variable(0)).deriv(&store, 0_usize),
            (1.0/5.0, -1.0/25.0));
    }

    #[test]
    fn negation() {
        use expression::Expr::*;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        assert_eq!((-Variable(0)).value(&store), -5.0);
        assert_eq!((-(-Variable(0))), Variable(0));
    }

    #[test]
    fn power() {
        use expression::Expr::*;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        assert_eq!(Variable(0).powi(2).value(&store), 25.0);
        assert_eq!((&Variable(0)).powi(2).value(&store), 25.0);
        assert_eq!(Variable(0).powi(2).deriv(&store, 0_usize), (25.0, 10.0));

        assert_eq!((2*Variable(0).powi(2)).deriv2(
                &store, 0_usize, 0_usize), (50.0, 20.0, 20.0, 4.0));
    }

    #[test]
    fn trig() {
        use expression::Expr::*;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(3.0);

        use std::f64::consts::PI;
        assert_eq!((Variable(0)*PI/180.0).cos().value(&store),
            5.0_f64.to_radians().cos());
        assert_eq!((Variable(1)*PI/180.0).sin().value(&store),
            3.0_f64.to_radians().sin());
        assert_eq!((Variable(0)*PI/180.0).cos().deriv(&store, 0_usize),
            (5.0_f64.to_radians().cos(),
            -(PI/180.0)*5.0_f64.to_radians().sin()));
        assert_eq!((Variable(0)*PI/180.0).sin().deriv(&store, 0_usize),
            (5.0_f64.to_radians().sin(),
            (PI/180.0)*5.0_f64.to_radians().cos()));
        assert_eq!(((Variable(0) + 2*Variable(1))*PI/180.0).sin()
            .deriv2(&store, 0_usize, 1_usize).3, -0.00011624748768739417
            );
        assert_eq!(((Variable(0) + 2*Variable(1))*PI/180.0).cos()
            .deriv2(&store, 0_usize, 1_usize).3, -0.0005980414796286429
            );
    }

    #[test]
    fn collecting() {
        use expression::Expr::*;
        let e = Variable(0)*Variable(1) + Variable(2);

        let mut expect = HashSet::new();
        expect.insert(0);
        expect.insert(1);
        expect.insert(2);
        assert_eq!(e.variables(), expect);
    }

    #[test]
    fn degree() {
        use expression::Expr::*;
        use expression::Degree;
        let e = (Variable(0)*Variable(1) + Variable(4))*Variable(5) + Variable(2);

        let mut expect = Degree::new();
        expect.linear.insert(2);
        expect.higher.insert((0, 1));
        expect.higher.insert((0, 5));
        expect.higher.insert((1, 5));
        expect.higher.insert((4, 5));
        assert_eq!(e.degree(), expect);
    }

    #[bench]
    fn quad_construct(b: &mut test::Bencher) {
        use expression::Expr::*;
        let n = 500;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Variable(i)); store.vars.push(0.5);
        }
        b.iter(|| {
            let mut e = Expr::Integer(0);
            for x in &xs {
                e = e + 3.0*(x - 1).powi(2) + 5.0;
            }
        });
    }

    #[bench]
    fn quad_deriv1(b: &mut test::Bencher) {
        use expression::Expr::*;
        let n = 100;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Variable(i));
            store.vars.push(0.5);
        }
        let mut e = Expr::Integer(0);
        for x in &xs {
            e = e + 3.0*(x - 1).powi(2) + 5.0;
        }
        b.iter(|| {
            for i in 0..n {
                e.deriv(&store, i);
            }
        });
    }

    #[bench]
    fn quad_deriv2(b: &mut test::Bencher) {
        use expression::Expr::*;
        let n = 50;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Variable(i));
            store.vars.push(0.5);
        }
        let mut e = Expr::Integer(0);
        for i in 1..n {
            e = e + 3.0*(&xs[i - 1] - &xs[i] + 1).powi(2) + 5.0;
        }
        b.iter(|| {
            for i in 1..n {
                e.deriv2(&store, i - 1, i);
            }
        });
    }

    #[test]
    fn tape_test() {
        use expression::Operation::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let mut tape = Tape::empty();
        // v0 + (1.0 - v1)^2
        tape.add_op(Variable(Var(0)));
        tape.add_op(Float(1.0));
        tape.add_op(Variable(Var(1)));
        tape.add_op(Neg);
        tape.add_op(Add);
        tape.add_op(Pow(2));
        tape.add_op(Add);
        assert_eq!(tape.n_max, 3);
        assert_eq!(tape.n_end, 1);
        assert_eq!(tape.value(&store), 14.0);
        assert_eq!(tape.deriv(&store, 0), (14.0, 1.0));
        assert_eq!(tape.deriv(&store, 1), (14.0, 6.0));
    }

    #[test]
    fn tape_ops() {
        use expression::{Var,Par,Tape};
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        let mut t = Tape::new();
        t = t + 5.0;
        t = 5.0 + t;
        t = Var(0) + t;
        t = t*Par(0);

        assert_eq!(t.value(&store), 60.0);
    }

    #[bench]
    fn tape_quad_deriv1(b: &mut test::Bencher) {
        use expression::{Var, Tape};
        let n = 100;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Var(i));
            store.vars.push(0.5);
        }
        let mut e = Tape::new();
        for x in &xs {
            e = e + 3.0*(Tape::from(x.clone()) - 1).powi(2) + 5.0;
        }
        //println!("{:?}", &e);
        let mut st: Vec<(f64, f64)> = Vec::new();
        st.resize(e.n_max, (0.0, 0.0));
        b.iter(|| {
            for i in 0..n {
                //e.deriv(&store, i);
                e.deriv_stack(&mut st, &store, i);
            }
        });
    }

    #[test]
    fn film_test() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let mut film = Film::new();
        // v0 + (1.0 - v1)^2
        film.ops.push(Float(1.0));
        film.ops.push(Variable(Var(1)));
        film.ops.push(Neg);
        film.ops.push(Add(0));
        film.ops.push(Pow(2));
        film.ops.push(Variable(Var(0)));
        film.ops.push(Add(4));

        let mut ws = WorkSpace::new();
        let mut finfo = FilmInfo::new();
        finfo.lin.push(Var(0));
        finfo.nlin.push(Var(1));
        finfo.quad.push((0, 0)); // the first entry in nlin

        // Get constant first derivatives
        // Call on parameter change
        // Copy out and store first derivatives
        film.ad(&finfo.lin, &Vec::new(), &store, &mut ws);
        assert_eq!(ws.len(), 7);
        assert_eq!(ws[6].val, 14.0);
        assert_eq!(ws[6].der1[0], 1.0); // Var(0)

        // Get constant second derivatives
        // Call on parameter change
        // Copy out and store second derivatives
        film.ad(&finfo.nlin, &finfo.quad, &store, &mut ws);
        assert_eq!(ws.len(), 7);
        assert_eq!(ws[6].val, 14.0);
        assert_eq!(ws[6].der1[0], 6.0); // Var(1)
        assert_eq!(ws[6].der2[0], 2.0); // Var(1), Var(1)

        // Get chaning derivatives
        // Call every time
        film.ad(&finfo.nlin, &finfo.nquad, &store, &mut ws);
        assert_eq!(ws.len(), 7);
        assert_eq!(ws[6].val, 14.0);
        assert_eq!(ws[6].der1[0], 6.0); // Var(1)
    }
}

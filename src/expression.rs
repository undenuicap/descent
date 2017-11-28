use std;
use std::collections::{HashSet, HashMap};
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
pub struct Var(pub ID);

#[derive(Debug, Clone, Copy)]
pub struct Par(pub ID);

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

// Contract: lin and nlin must be disjoint
// Contract: quad and nquad must be disjoint
// Contract: IDs in pairs must be ordered
// Contract: all IDs in quad and nquad must be in nlin
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Deg {
    lin: HashSet<ID>,
    nlin: HashSet<ID>,
    quad: HashSet<(ID, ID)>,
    nquad: HashSet<(ID, ID)>,
}

fn cross_ids(id1s: &HashSet<ID>, id2s: &HashSet<ID>,
             target: &mut HashSet<(ID, ID)>)  {
    for &id1 in id1s {
        for &id2 in id2s {
            target.insert(order(id1, id2));
        }
    }
}

impl Deg {
    pub fn new() -> Deg {
        Deg::default()
    }

    pub fn with_id(id: ID) -> Deg {
        let mut d = Deg::default();
        d.lin.insert(id);
        d
    }

    pub fn is_empty(&self) -> bool {
        self.lin.is_empty() && self.quad.is_empty() && self.nquad.is_empty()
    }

    // Only should need to add an ID to a brand new Deg (see with_id above)
    //pub fn add_id(&self, id: ID) -> Deg {
    //    let mut c = self.clone();
    //    if !self.nlin.contains(id) {
    //        c.lin.insert(id);
    //    }
    //    c
    //}

    pub fn union(&self, other: &Deg) -> Deg {
        let mut deg = Deg::new();
        deg.lin = self.lin.union(&(other.lin)).cloned().collect();
        deg.nlin = self.nlin.union(&(other.nlin)).cloned().collect();
        deg.quad = self.quad.union(&(other.quad)).cloned().collect();
        deg.nquad = self.nquad.union(&(other.nquad)).cloned().collect();

        deg.lin = deg.lin.difference(&(deg.nlin)).cloned().collect();
        deg.quad = deg.quad.difference(&(deg.nquad)).cloned().collect();
        deg
    }

    pub fn cross(&self, other: &Deg) -> Deg {
        if self.is_empty() {
            other.clone()
        } else if other.is_empty() {
            self.clone()
        } else {
            // If here, both sides have at least one entry.
            // Therefore all promoted one level.
            let mut deg = Deg::new();

            // all nlin move over
            for &id in &self.nlin {
                deg.nlin.insert(id);
            }

            for &id in &other.nlin {
                deg.nlin.insert(id);
            }

            // lin will empty into nlin
            for &id in &self.lin {
                deg.nlin.insert(id);
            }

            for &id in &other.lin {
                deg.nlin.insert(id);
            }

            // quad and nquad will transfer over to nquad
            for &p in &self.nquad {
                deg.nquad.insert(p);
            }

            for &p in &other.nquad {
                deg.nquad.insert(p);
            }

            for &p in &self.quad {
                deg.nquad.insert(p);
            }

            for &p in &other.quad {
                deg.nquad.insert(p);
            }

            // contract ensures new quad values do not appear in nquad
            cross_ids(&(self.lin), &(other.lin), &mut (deg.quad));
            cross_ids(&(self.lin), &(other.nlin), &mut (deg.nquad));
            cross_ids(&(self.nlin), &(other.lin), &mut (deg.nquad));
            cross_ids(&(self.nlin), &(other.nlin), &mut (deg.nquad));

            deg
        }
    }

    pub fn highest(&self) -> Deg {
        // Promote all combinations to highest
        let mut deg = Deg::new();

        deg.nlin = self.lin.union(&(self.nlin)).cloned().collect();

        // Could same time here due to symmetry...
        cross_ids(&(deg.nlin), &(deg.nlin), &mut (deg.nquad));

        deg
    }
}

impl From<Deg> for FilmInfo {
    fn from(d: Deg) -> FilmInfo {
        let mut info = FilmInfo::new();
        info.lin = d.lin.into_iter().collect();
        info.nlin = d.nlin.into_iter().collect();
        info.lin.sort();
        info.nlin.sort();

        let mut id_to_ind: HashMap<ID, usize> = HashMap::new();
        for (i, &id) in info.nlin.iter().enumerate() {
            id_to_ind.insert(id, i);
        }

        // Converting from variable IDs to local indices into nlin
        for (id1, id2) in d.quad.into_iter() {
            info.quad.push((id_to_ind[&id1], id_to_ind[&id2]));
        }

        for (id1, id2) in d.nquad.into_iter() {
            info.nquad.push((id_to_ind[&id1], id_to_ind[&id2]));
        }

        info.quad.sort();
        info.nquad.sort();
        info
    }
}

// Version where implicitly use the value to the left
// Must start with terminal value for this to be valid
// Index must be less than index of entry in tape
#[derive(Debug, Clone)]
enum Oper {
    Add(usize),
    Sub(usize),
    Mul(usize),
    Neg, // negate
    Pow(i32), // to be safe, should not be 0 or 1, 2 should use Square
    Sin,
    Cos,
    Sum(Vec<usize>),
    Square,
    Variable(Var),
    Parameter(Par),
    Float(f64),
}

impl From<Var> for Film {
    fn from(v: Var) -> Film {
        Film {
            ops: vec![self::Oper::Variable(v)],
        }
    }
}

impl From<Par> for Film {
    fn from(p: Par) -> Film {
        Film {
            ops: vec![self::Oper::Parameter(p)],
        }
    }
}

impl From<f64> for Film {
    fn from(v: f64) -> Film {
        Film {
            ops: vec![self::Oper::Float(v)],
        }
    }
}

impl From<i32> for Film {
    fn from(v: i32) -> Film {
        Film {
            ops: vec![self::Oper::Float(f64::from(v))],
        }
    }
}

// Have to use trait because straight fn overloading not possible
pub trait NumOpsF {
    fn powi(self, p: i32) -> Film;
    fn sin(self) -> Film;
    fn cos(self) -> Film;
}

impl NumOpsF for Film {
    fn powi(mut self, p: i32) -> Film {
        // When empty don't do anything
        if self.ops.is_empty() {
            return self;
        }

        // Match now so don't have to later
        match p {
            0 => {
                self.add_op(Oper::Float(1.0));
            },
            1 => {
                // don't add anything
            },
            2 => {
                self.add_op(Oper::Square);
            },
            _ => {
                self.add_op(Oper::Pow(p));
            },
        }
        self
    }

    fn sin(mut self) -> Film {
        // When empty don't do anything
        if !self.ops.is_empty() {
            self.add_op(Oper::Sin);
        }
        self
    }

    fn cos(mut self) -> Film {
        // When empty don't do anything
        if !self.ops.is_empty() {
            self.add_op(Oper::Cos);
        }
        self
    }
}

impl NumOpsF for Var {
    fn powi(self, p: i32) -> Film {
        Film::from(self).powi(p)
    }

    fn sin(self) -> Film {
        Film::from(self).sin()
    }

    fn cos(self) -> Film {
        Film::from(self).cos()
    }
}

impl NumOpsF for Par {
    fn powi(self, p: i32) -> Film {
        Film::from(self).powi(p)
    }

    fn sin(self) -> Film {
        Film::from(self).sin()
    }

    fn cos(self) -> Film {
        Film::from(self).cos()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Column {
    pub val: f64,
    pub der1: Vec<f64>,
    pub der2: Vec<f64>,
}

impl Column {
    pub fn new() -> Column {
        Column::default()
    }
}

pub type WorkSpace = Vec<Column>;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct FilmInfo {
    // Linear and quadratic respective first and second derivatives only need
    // to be computed once on parameter change.
    pub lin: Vec<ID>, // const derivative
    pub nlin: Vec<ID>, // non-const derivative
    // Below the usize are indices into nlin, a variable representation
    // that is local
    pub quad: Vec<(usize, usize)>, // const second derivative
    pub nquad: Vec<(usize, usize)>, // non-const second derivative
}

impl FilmInfo {
    pub fn new() -> FilmInfo {
        FilmInfo::default()
    }
}

#[derive(Debug, Clone)]
pub struct Film {
    ops: Vec<Oper>,
}

impl Film {
    pub fn ad(&self, d1: &Vec<ID>, d2: &Vec<(usize, usize)>, ret: &Retrieve,
            ws: &mut WorkSpace) {
        use self::Oper::*;
        use self::{Var, Par};
        // Only resize up
        if ws.len() < self.ops.len() {
            ws.resize(self.ops.len(), Column::new());
        }
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
                    for ((c, p), o) in cur.der1.iter_mut()
                            .zip(pre.der1.iter()).zip(oth.der1.iter()) {
                        *c = p + o;
                    }
                    for ((c, p), o) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(oth.der2.iter()) {
                        *c = p + o;
                    }
                },
                Sub(j) => {
                    // Take note of order where oth - pre 
                    let pre = &left[i - 1];
                    let oth = &left[j];
                    cur.val = oth.val - pre.val;
                    for ((c, p), o) in cur.der1.iter_mut()
                            .zip(pre.der1.iter()).zip(oth.der1.iter()) {
                        *c = o - p;
                    }
                    for ((c, p), o) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(oth.der2.iter()) {
                        *c = o - p;
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
                    for (((c, p), o), &(k1, k2)) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(oth.der2.iter())
                            .zip(d2.iter()) {
                        *c = p*oth.val + pre.val*o
                             + pre.der1[k1]*oth.der1[k2]
                             + pre.der1[k2]*oth.der1[k1];
                    }
                },
                Neg => {
                    let pre = &left[i - 1];
                    cur.val = -pre.val;
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = -p;
                    }
                    for (c, p) in cur.der2.iter_mut().zip(pre.der2.iter()) {
                        *c = -p;
                    }
                },
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    let pre = &left[i - 1];
                    cur.val = pre.val.powi(pow);
                    let vald = pre.val.powi(pow - 1);
                    let valdd = pre.val.powi(pow - 2);
                    for (c, p) in cur.der1.iter_mut()
                            .zip(pre.der1.iter()) {
                        *c = f64::from(pow)*p*vald;
                    }
                    for ((c, p), &(k1, k2)) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(d2.iter()) {
                        *c = f64::from(pow)*p*vald
                                + f64::from(pow*(pow - 1))
                                *pre.der1[k1]*pre.der1[k2]*valdd;
                    }
                },
                Sin => {
                    let pre = &left[i - 1];
                    cur.val = pre.val.sin();
                    let valcos = pre.val.cos();
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = p*valcos;
                    }
                    for ((c, p), &(k1, k2)) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(d2.iter()) {
                        *c = p*valcos - pre.der1[k1]*pre.der1[k2]*cur.val;
                    }
                },
                Cos => {
                    let pre = &left[i - 1];
                    cur.val = pre.val.cos();
                    let valsin = pre.val.sin();
                    for (c, p) in cur.der1.iter_mut().zip(pre.der1.iter()) {
                        *c = -p*valsin;
                    }
                    for ((c, p), &(k1, k2)) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(d2.iter()) {
                        *c = -p*valsin - pre.der1[k1]*pre.der1[k2]*cur.val;
                    }
                },
                Sum(ref js) => {
                    let pre = &left[i - 1];
                    cur.val = pre.val;
                    for &j in js {
                        let oth = &left[j];
                        cur.val += oth.val;
                    }

                    for k in 0..(d1.len()) {
                        cur.der1[k] = pre.der1[k];
                        for &j in js {
                            let oth = &left[j];
                            cur.der1[k] += oth.der1[k];
                        }
                    }
                    for k in 0..(d2.len()) {
                        cur.der2[k] = pre.der2[k];
                        for &j in js {
                            let oth = &left[j];
                            cur.der2[k] += oth.der2[k];
                        }
                    }
                },
                Square => {
                    let pre = &left[i - 1];
                    cur.val = pre.val*pre.val;
                    for (c, p) in cur.der1.iter_mut()
                            .zip(pre.der1.iter()) {
                        *c = 2.0*p*pre.val;
                    }
                    for ((c, p), &(k1, k2)) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(d2.iter()) {
                        *c = 2.0*p*pre.val + 2.0*pre.der1[k1]*pre.der1[k2];
                    }
                },
                Variable(Var(id)) => {
                    cur.val = ret.get_var(id);
                    for (c, did) in cur.der1.iter_mut().zip(d1.iter()) {
                        *c = if id == *did { 1.0 } else { 0.0 };
                    }
                    for c in &mut cur.der2 {
                        *c = 0.0;
                    }
                },
                Parameter(Par(id)) => {
                    cur.val = ret.get_par(id);
                    for c in &mut cur.der1 {
                        *c = 0.0;
                    }
                    for c in &mut cur.der2 {
                        *c = 0.0;
                    }
                },
                Float(val) => {
                    cur.val = val;
                    for c in &mut cur.der1 {
                        *c = 0.0;
                    }
                    for c in &mut cur.der2 {
                        *c = 0.0;
                    }
                },
            }
        }
    }

    // Should not be called on short workspace or empty film
    pub fn last_in<'a>(&self, ws: &'a WorkSpace) -> &'a Column {
        &ws[self.ops.len() - 1]
    }

    // Should not be called on short workspace or empty film
    pub fn last_mut_in<'a>(&self, ws: &'a mut WorkSpace) -> &'a mut Column {
        &mut ws[self.ops.len() - 1]
    }

    pub fn get_info(&self) -> FilmInfo {
        use self::Oper::*;
        use self::Var;
        let mut degs: Vec<Deg> = Vec::new();
        for (i, op) in self.ops.iter().enumerate() {
            let d = match *op {
                Add(j) | Sub(j) => {
                    degs[i - 1].union(&degs[j])
                },
                Mul(j) => {
                    degs[i - 1].cross(&degs[j])
                },
                Neg => {
                    degs[i - 1].clone()
                },
                Pow(p) => {
                    // Even though shouldn't have 0 or 1, might as well match
                    // anyway
                    match p {
                        0 => {
                            Deg::new()
                        },
                        1 => {
                            degs[i - 1].clone()
                        },
                        2 => {
                            degs[i - 1].cross(&degs[i - 1])
                        },
                        _ => {
                            degs[i - 1].highest()
                        },
                    }
                },
                Sum(ref js) => {
                    let mut deg = degs[i - 1].clone();
                    for &j in js {
                        deg = deg.union(&degs[j]);
                    }
                    deg
                },
                Square => {
                    degs[i - 1].cross(&degs[i - 1])
                },
                Sin | Cos => {
                    degs[i - 1].highest()
                },
                Variable(Var(id)) => {
                    Deg::with_id(id)
                },
                _ => {
                    Deg::new()
                },
            };
            degs.push(d);
        }

        match degs.pop() {
            Some(d) => FilmInfo::from(d),
            None => FilmInfo::new(),
        }
    }

    fn add_op(&mut self, op: Oper) {
        self.ops.push(op);
    }

    // Wouldn't be required if we instead use relative values
    fn add_offset(&mut self, n: usize) {
        use self::Oper::*;
        for op in self.ops.iter_mut() {
            match *op {
                Add(ref mut j) | Sub(ref mut j) | Mul(ref mut j) => {
                    *j = *j + n;
                },
                Sum(ref mut js) => {
                    for j in js {
                        *j = *j + n;
                    }
                },
                _ => (),
            }
        }
    }

    fn append(&mut self, mut other: Film) {
        other.add_offset(self.ops.len());
        self.ops.append(&mut other.ops);
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

// Changed this order so filling out bottom left (think)
fn order(a: ID, b: ID) -> (ID, ID) {
    if a < b { (b, a) } else { (a, b) }
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

macro_rules! binary_ops_to_film {
    ( $T:ident, $f:ident, $U:ty, $V:ty ) => {
        impl std::ops::$T<$V> for $U {
            type Output = Film;

            fn $f(self, other: $V) -> Film {
                Film::from(self).$f(Film::from(other))
            }
        }
    };
}

macro_rules! binary_ops_with_film {
    ( $T:ident, $f:ident, $U:ty ) => {
        impl std::ops::$T<Film> for $U {
            type Output = Film;

            fn $f(self, other: Film) -> Film {
                Film::from(self).$f(other)
            }
        }

        impl std::ops::$T<$U> for Film {
            type Output = Film;

            fn $f(self, other: $U) -> Film {
                self.$f(Film::from(other))
            }
        }

        impl<'a> std::ops::$T<&'a Film> for $U {
            type Output = Film;

            fn $f(self, other: &'a Film) -> Film {
                Film::from(self).$f(other.clone())
            }
        }

        impl<'a> std::ops::$T<$U> for &'a Film {
            type Output = Film;

            fn $f(self, other: $U) -> Film {
                self.clone().$f(Film::from(other))
            }
        }
    };
}

binary_ops_to_film!(Add, add, Var, f64);
binary_ops_to_film!(Add, add, f64, Var);
binary_ops_to_film!(Add, add, Par, f64);
binary_ops_to_film!(Add, add, f64, Par);
binary_ops_to_film!(Add, add, Par, Var);
binary_ops_to_film!(Add, add, Var, Par);
binary_ops_to_film!(Add, add, Var, Var);
binary_ops_to_film!(Add, add, Par, Par);

binary_ops_to_film!(Sub, sub, Var, f64);
binary_ops_to_film!(Sub, sub, f64, Var);
binary_ops_to_film!(Sub, sub, Par, f64);
binary_ops_to_film!(Sub, sub, f64, Par);
binary_ops_to_film!(Sub, sub, Par, Var);
binary_ops_to_film!(Sub, sub, Var, Par);
binary_ops_to_film!(Sub, sub, Var, Var);
binary_ops_to_film!(Sub, sub, Par, Par);

binary_ops_to_film!(Mul, mul, Var, f64);
binary_ops_to_film!(Mul, mul, f64, Var);
binary_ops_to_film!(Mul, mul, Par, f64);
binary_ops_to_film!(Mul, mul, f64, Par);
binary_ops_to_film!(Mul, mul, Par, Var);
binary_ops_to_film!(Mul, mul, Var, Par);
binary_ops_to_film!(Mul, mul, Var, Var);
binary_ops_to_film!(Mul, mul, Par, Par);

binary_ops_with_film!(Add, add, Var);
binary_ops_with_film!(Add, add, Par);
binary_ops_with_film!(Add, add, f64);
binary_ops_with_film!(Sub, sub, Var);
binary_ops_with_film!(Sub, sub, Par);
binary_ops_with_film!(Sub, sub, f64);
binary_ops_with_film!(Mul, mul, Var);
binary_ops_with_film!(Mul, mul, Par);
binary_ops_with_film!(Mul, mul, f64);

impl std::ops::Add<Film> for Film {
    type Output = Film;

    fn add(mut self, other: Film) -> Film {
        // Assuming add on empty Film is like add by 0.0
        if self.ops.is_empty() {
            other
        } else if other.ops.is_empty() {
            self
        } else {
            let n = self.ops.len();
            self.append(other);
            self.add_op(Oper::Add(n - 1));
            //match self.ops.pop().unwrap() {
            //    Oper::Add(j) => {
            //        let n = self.ops.len();
            //        let js = vec![j, n - 1];
            //        self.append(other);
            //        self.add_op(Oper::Sum(js));
            //    },
            //    Oper::Sum(mut js) => {
            //        let n = self.ops.len();
            //        js.push(n - 1);
            //        self.append(other);
            //        self.add_op(Oper::Sum(js));
            //    },
            //    op => {
            //        let n = self.ops.len();
            //        self.ops.push(op); // push back on
            //        self.append(other);
            //        self.add_op(Oper::Add(n));
            //    },
            //}
            self
        }
    }
}

impl std::ops::Sub<Film> for Film {
    type Output = Film;

    fn sub(mut self, mut other: Film) -> Film {
        // Assuming sub on empty Film is like add by 0.0
        if self.ops.is_empty() {
            other.add_op(Oper::Neg); // negate second argument
            other
        } else if other.ops.is_empty() {
            self
        } else {
            let n = self.ops.len();
            self.append(other);
            self.add_op(Oper::Sub(n - 1));
            self
        }
    }
}

impl std::ops::Mul<Film> for Film {
    type Output = Film;

    fn mul(mut self, other: Film) -> Film {
        // Assuming mul on empty Film is like mul by 1.0
        if self.ops.is_empty() {
            other
        } else if other.ops.is_empty() {
            self
        } else {
            let n = self.ops.len();
            self.append(other);
            self.add_op(Oper::Mul(n - 1));
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
        expect.higher.insert((1, 0));
        expect.higher.insert((5, 0));
        expect.higher.insert((5, 1));
        expect.higher.insert((5, 4));
        assert_eq!(e.degree(), expect);
    }

    #[bench]
    fn quad_construct(b: &mut test::Bencher) {
        use expression::Expr::*;
        let n = 50;
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
        let n = 50;
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
        let n = 50;
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

        let mut film = Film { ops: Vec::new() };
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
        finfo.lin.push(0);
        finfo.nlin.push(1);
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

    #[test]
    fn film_degree() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let mut film = Film { ops: Vec::new() };
        // v0 + (1.0 - v1)^2
        film.ops.push(Float(1.0));
        film.ops.push(Variable(Var(1)));
        film.ops.push(Neg);
        film.ops.push(Add(0));
        film.ops.push(Pow(2));
        film.ops.push(Variable(Var(0)));
        film.ops.push(Add(4));

        let finfo = film.get_info();

        assert_eq!(finfo, FilmInfo {
            lin: vec![0],
            nlin: vec![1],
            quad: vec![(0, 0)],
            nquad: vec![],
        }); 
    }

    #[test]
    fn film_ops() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        let mut f = Film::from(5.0);
        f = 5.0 + f;
        f = Var(0) + f;
        f = f*Par(0);

        let finfo = f.get_info();

        let mut ws = WorkSpace::new();
        //println!("{:?}", finfo);
        //println!("{:?}", f);
        f.ad(&finfo.lin, &Vec::new(), &store, &mut ws);

        assert_eq!(ws.len(), 7);
        assert_eq!(ws[6].val, 60.0);
        assert_eq!(ws[6].der1[0], 4.0);
    }

    // Only expect to work when have operator overloading creating sums
    //#[test]
    //fn film_sum() {
    //    use expression::Oper::*;
    //    use expression::Var;
    //    let mut store = Store::new();
    //    store.vars.push(5.0);

    //    let f = 5.0 + Var(0) + Var(0) + Var(0);

    //    let finfo = f.get_info();

    //    //println!("{:?}", f);
    //    //println!("{:?}", finfo);
    //    let mut ws = WorkSpace::new();
    //    f.ad(&finfo.lin, &Vec::new(), &store, &mut ws);

    //    assert_eq!(ws.len(), 5);
    //    assert_eq!(ws[4].val, 20.0);
    //}

    #[test]
    fn film_sin() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);

        let mut ws = WorkSpace::new();

        let f = (2.0*Var(0)).sin();
        let finfo = f.get_info();
        //println!("{:?}", f);
        //println!("{:?}", finfo);
        assert_eq!(finfo.nlin.len(), 1);
        assert_eq!(finfo.nquad.len(), 1);

        f.ad(&finfo.nlin, &finfo.nquad, &store, &mut ws);

        assert_eq!(ws.len(), 4);
        assert_eq!(ws[3].val, 10.0_f64.sin());
        assert_eq!(ws[3].der1[0], 2.0*(10.0_f64.cos()));
        assert_eq!(ws[3].der2[0], -4.0*(10.0_f64.sin()));
    }

    #[test]
    fn film_cos() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);

        let mut ws = WorkSpace::new();

        let f = (2.0*Var(0)).cos();
        let finfo = f.get_info();
        //println!("{:?}", f);
        //println!("{:?}", finfo);
        assert_eq!(finfo.nlin.len(), 1);
        assert_eq!(finfo.nquad.len(), 1);

        f.ad(&finfo.nlin, &finfo.nquad, &store, &mut ws);

        assert_eq!(ws.len(), 4);
        assert_eq!(ws[3].val, 10.0_f64.cos());
        assert_eq!(ws[3].der1[0], -2.0*(10.0_f64.sin()));
        assert_eq!(ws[3].der2[0], -4.0*(10.0_f64.cos()));
    }

    #[bench]
    fn film_quad_deriv1(b: &mut test::Bencher) {
        use expression::{Var, Film};
        let n = 50;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Var(i));
            store.vars.push(0.5);
        }
        let mut e = Film::from(0.0);
        for &x in &xs {
            e = e + 3.0*(x - 1.0).powi(2) + 5.0;
        }
        let mut ws = WorkSpace::new();
        //println!("{:?}", e);
        let finfo = e.get_info();
        //println!("{:?}", finfo);
        assert_eq!(finfo.lin.len(), 0);
        assert_eq!(finfo.nlin.len(), n);
        assert_eq!(finfo.quad.len(), n);
        assert_eq!(finfo.nquad.len(), 0);
        b.iter(|| {
            //e.ad(&finfo.lin, &Vec::new(), &store, &mut ws);
            //e.ad(&finfo.nlin, &finfo.quad, &store, &mut ws);
            e.ad(&finfo.nlin, &finfo.nquad, &store, &mut ws);
        });
        assert_eq!(ws.last().unwrap().der1.len(), n);
        assert_eq!(ws.last().unwrap().der2.len(), 0);
    }

    #[bench]
    fn film_quad_all(b: &mut test::Bencher) {
        use expression::{Var, Film};
        let n = 50;
        let mut xs = Vec::new();
        let mut store = Store::new();
        for i in 0..n {
            xs.push(Var(i));
            store.vars.push(0.5);
        }
        let mut e = Film::from(0.0);
        for &x in &xs {
            e = e + 3.0*(x - 1.0).powi(2) + 5.0;
        }
        let mut ws = WorkSpace::new();
        //println!("{:?}", e);
        let finfo = e.get_info();
        //println!("{:?}", finfo);
        assert_eq!(finfo.lin.len(), 0);
        assert_eq!(finfo.nlin.len(), n);
        assert_eq!(finfo.quad.len(), n);
        assert_eq!(finfo.nquad.len(), 0);
        b.iter(|| {
            //e.ad(&finfo.lin, &Vec::new(), &store, &mut ws);
            e.ad(&finfo.nlin, &finfo.quad, &store, &mut ws);
            //e.ad(&finfo.nlin, &finfo.nquad, &store, &mut ws);
        });
        assert_eq!(ws.last().unwrap().der1.len(), n);
        assert_eq!(ws.last().unwrap().der2.len(), n);
    }
}

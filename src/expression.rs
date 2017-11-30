use std;
use std::collections::{HashSet, HashMap};
use std::cmp::max;

pub type ID = usize;

/// Retrieve current values of variables and parameters.
///
/// Expect a panic if requested id not available for whatever reason.
pub trait Retrieve {
    fn get_var(&self, vid: ID) -> f64;
    fn get_par(&self, pid: ID) -> f64;
}

#[derive(Debug, Clone, Copy)]
pub struct Var(pub ID);

#[derive(Debug, Clone, Copy)]
pub struct Par(pub ID);

/// Representation of the degree of variables from an expression.
///
/// - `lin` and `nlin` must be disjoint
/// - `quad` and `nquad` must be disjoint
/// - `ID`s in pairs must be ordered
/// - all `ID`s in `quad` and `nquad` must be in `nlin`
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Deg {
    lin: HashSet<ID>,
    nlin: HashSet<ID>,
    quad: HashSet<(ID, ID)>,
    nquad: HashSet<(ID, ID)>,
}

/// Order second derivative pairs.
///
/// Should fill out bottom left of Hessian with this ordering.
fn order(a: ID, b: ID) -> (ID, ID) {
    if a < b { (b, a) } else { (a, b) }
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

        // Not always required...
        //info.set_lists();

        info
    }
}

/// Operations for representing expressions.
///
/// These operations are designed to be stored and structured on a `Film`.
/// They either have zero or more operands. For operations with 1 or more
/// operands, the first operand is implicitly the operation immediately to the
/// left on the `Film`. For additional operands their indices into the `Film`
/// are given explicitly.
///
/// In the future this could be changed to relative operand referencing, i.e.
/// distance to the left.
#[derive(Debug, Clone)]
enum Oper {
    Add(usize),
    Sub(usize),
    Mul(usize),
    /// Negate
    Neg,
    /// Caution should be employed if required to use for power of 0 or 1
    Pow(i32),
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

#[derive(Debug, Clone, Default)]
pub struct WorkSpace {
    pub cols: Vec<Column>,
    pub ns: Vec<f64>,
    pub nds: Vec<f64>,
    pub na1s: Vec<f64>,
    pub na2s: Vec<f64>,
    pub ids: HashMap<ID, f64>,
}

impl WorkSpace {
    pub fn new() -> WorkSpace {
        WorkSpace::default()
    }
}

/// Information about a `Film` for guiding AD process.
///
/// Linear and quadratic respective first and second derivatives only need to
/// be computed once on parameter change.  The `usize` pairs represent indices
/// into `nlin`.  They are local variable mappings.
///
/// The contracts from `Deg` are expected to be held.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FilmInfo {
    /// Constant first derivative
    pub lin: Vec<ID>,
    /// Non-constant first derivative
    pub nlin: Vec<ID>,
    /// Constant second derivative
    pub quad: Vec<(usize, usize)>,
    /// Non-constant second derivative
    pub nquad: Vec<(usize, usize)>,
    pub quad_list: Vec<Vec<ID>>,
    pub nquad_list: Vec<Vec<ID>>,
}

/// Maps for each `nlin` entry to `ID`s that pair with it in `quad`/`nquad`.
///
/// When everything is ordered, when traversed it will preserve original
/// `quad`/`nquad` orderings.
fn nlin_list(nlin: &Vec<ID>, sec: &Vec<(usize, usize)>) -> Vec<Vec<ID>>{
    let mut vs: Vec<Vec<ID>> = Vec::new();
    vs.resize(nlin.len(), Vec::new());
    for &(i, j) in sec {
        vs[i].push(nlin[j]);
    }
    vs
}

impl FilmInfo {
    pub fn new() -> FilmInfo {
        FilmInfo::default()
    }

    pub fn set_lists(&mut self) {
        self.quad_list = nlin_list(&self.nlin, &self.quad);
        self.nquad_list = nlin_list(&self.nlin, &self.nquad);
    }
}

#[derive(Debug, Clone)]
pub struct Film {
    ops: Vec<Oper>,
}

impl Film {
    /// Evaluate just the value.
    pub fn eval(&self, ret: &Retrieve, ns: &mut Vec<f64>) -> f64 {
        use self::Oper::*;
        use self::{Var, Par};
        ns.resize(self.ops.len(), 0.0);
        // Get values
        for (i, op) in self.ops.iter().enumerate() {
            let (left, right) = ns.split_at_mut(i);
            let cur = &mut right[0]; // the i value from original
            match *op {
                Add(j) => {
                    *cur = left[i - 1] + left[j];
                },
                Sub(j) => {
                    // Take note of order where oth - pre 
                    *cur = left[j] - left[i - 1];
                },
                Mul(j) => {
                    *cur = left[i - 1]*left[j];
                },
                Neg => {
                    *cur = -left[i - 1];
                },
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    *cur = left[i - 1].powi(pow);
                },
                Sin => {
                    *cur = left[i - 1].sin();
                },
                Cos => {
                    *cur = left[i - 1].cos();
                },
                Sum(ref js) => {
                    *cur = left[i - 1];
                    for &j in js {
                        *cur += left[j];
                    }
                },
                Square => {
                    *cur = left[i - 1]*left[i - 1];
                },
                Variable(Var(id)) => {
                    *cur = ret.get_var(id);
                },
                Parameter(Par(id)) => {
                    *cur = ret.get_par(id);
                },
                Float(val) => {
                    *cur = val;
                },
            }
        }
        ns[self.ops.len() - 1]
    }

    pub fn full_fwd<'a>(&self, d1: &Vec<ID>, d2: &Vec<(usize, usize)>,
                        ret: &Retrieve,
                        cols: &'a mut Vec<Column>) -> &'a Column {
        use self::Oper::*;
        use self::{Var, Par};
        // Only resize up
        if cols.len() < self.ops.len() {
            cols.resize(self.ops.len(), Column::new());
        }
        for (i, op) in self.ops.iter().enumerate() {
            let (left, right) = cols.split_at_mut(i);
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
                    let vald = f64::from(pow)*pre.val.powi(pow - 1);
                    let valdd = f64::from(pow*(pow - 1))*pre.val.powi(pow - 2);
                    for (c, p) in cur.der1.iter_mut()
                            .zip(pre.der1.iter()) {
                        *c = p*vald;
                    }
                    for ((c, p), &(k1, k2)) in cur.der2.iter_mut()
                            .zip(pre.der2.iter()).zip(d2.iter()) {
                        *c = p*vald + pre.der1[k1]*pre.der1[k2]*valdd;
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
        &cols[self.ops.len() - 1]
    }

    /// First derivative using forward method.
    ///
    /// `ns` must be size of `ops`.
    /// Could be made faster using loop unrolling if we pass in multiple `ID`s
    /// to solve at the same time.
    pub fn der1_fwd(&self, vid: ID, ret: &Retrieve,
                    ns: &Vec<f64>, nds: &mut Vec<f64>) -> f64 {
        use self::Oper::*;
        use self::{Var, Par};
        nds.resize(self.ops.len(), 0.0);
        for (i, op) in self.ops.iter().enumerate() {
            let (left, right) = nds.split_at_mut(i);
            let cur = &mut right[0]; // the i value from original
            match *op {
                Add(j) => {
                    *cur = left[i - 1] + left[j];
                },
                Sub(j) => {
                    // Take note of order where oth - pre 
                    *cur = left[j] - left[i - 1];
                },
                Mul(j) => {
                    *cur = left[i - 1]*ns[j] + left[j]*ns[i - 1]
                },
                Neg => {
                    *cur = -left[i - 1];
                },
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    *cur = f64::from(pow)*left[i - 1]*ns[i - 1].powi(pow - 1);
                },
                Sin => {
                    *cur = left[i - 1]*ns[i - 1].cos();
                },
                Cos => {
                    *cur = -left[i - 1]*ns[i - 1].sin();
                },
                Sum(ref js) => {
                    *cur = left[i - 1];
                    for &j in js {
                        *cur += left[j];
                    }
                },
                Square => {
                    *cur = 2.0*left[i - 1]*ns[i - 1];
                },
                Variable(Var(id)) => {
                    *cur = if id == vid { 1.0 } else { 0.0 };
                },
                _ => {
                    *cur = 0.0;
                },
            }
        }
        nds[self.ops.len() - 1]
    }

    /// Calculate first derivative using reverse method.
    ///
    /// `ns` must be length of `ops`.
    /// Assume each operator has only one dependent. If not then would need to
    /// rework.  Probably not an issue as can maybe just += adjoint (would need
    /// to make sure they are set to 0 at start).
    pub fn der1_rev(&self, d1: &Vec<ID>, ret: &Retrieve,
                    ns: &Vec<f64>, nas: &mut Vec<f64>,
                    ids: &mut HashMap<ID, f64>) -> Vec<f64> {
        use self::Oper::*;
        use self::{Var, Par};

        // Probably there is a faster way than this.
        ids.clear();
        for &id in d1 {
            ids.insert(id, 0.0);
        }

        // Go through in reverse
        nas.resize(self.ops.len(), 0.0);
        nas[self.ops.len() - 1] = 1.0;
        for (i, op) in self.ops.iter().enumerate().rev() {
            let (left, right) = nas.split_at_mut(i);
            let cur = right[0]; // the i value from original
            match *op {
                Add(j) => {
                    left[i - 1] = cur;
                    left[j] = cur;
                },
                Sub(j) => {
                    // Take note of order where oth - pre 
                    left[i - 1] = -cur;
                    left[j] = cur;
                },
                Mul(j) => {
                    left[i - 1] = ns[j]*cur;
                    left[j] = ns[i - 1]*cur;
                },
                Neg => {
                    left[i - 1] = -cur;
                },
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    left[i - 1] = f64::from(pow)*ns[i - 1].powi(pow - 1)*cur;
                },
                Sin => {
                    left[i - 1] = ns[i - 1].cos()*cur;
                },
                Cos => {
                    left[i - 1] = -ns[i - 1].sin()*cur;
                },
                Sum(ref js) => {
                    left[i - 1] = cur;
                    for &j in js {
                        left[j] = cur;
                    }
                },
                Square => {
                    left[i - 1] = 2.0*ns[i - 1]*cur;
                },
                Variable(Var(id)) => {
                    if let Some(v) = ids.get_mut(&id) {
                        *v += cur;
                    }
                },
                _ => {},
            }
        }
        let mut der1 = Vec::new();
        for id in d1 {
            der1.push(*ids.get(id).unwrap());
        }
        der1
    }


    /// Calculate the second derivatives for given first derivative.
    ///
    /// Let n represent an operation, and it has the operands js.  Then:
    /// ```text
    /// dn/dx_1dx_2 = \sum_{j \in js} d^2n_j/dx_1dx_2
    ///     + \sum {j, k \in js} d^2n/dn_jdn_k dn_j/dx_1 dn_k/dx_2
    /// ```
    /// 
    /// If we have precomputed the operator derivatives wrt x_1, then when we
    /// follow one path, we only need to pass on two "adjoint" values to each
    /// operand.  Given adjoints (a1, a2) for n, then the adjoint of
    /// operand j is:
    /// ```text
    /// a1_j = a1*dn/dn_j
    /// a2_j = a2*dn/dn_j + a1*s_j
    ///
    /// where s_j = \sum_{k \in js} d^2n/dn_jdn_k dn_k/dx_1
    /// ```
    pub fn der2_rev(&self, d2: &Vec<ID>, ret: &Retrieve,
                    ns: &Vec<f64>, nds: &Vec<f64>,
                    na1s: &mut Vec<f64>, na2s: &mut Vec<f64>,
                    ids: &mut HashMap<ID, f64>) -> Vec<f64> {
        use self::Oper::*;
        use self::{Var, Par};

        // Probably there is a faster way than this.
        ids.clear();
        for &id in d2 {
            ids.insert(id, 0.0);
        }

        // Go through in reverse
        na1s.resize(self.ops.len(), 0.0);
        na1s[self.ops.len() - 1] = 1.0;
        na2s.resize(self.ops.len(), 0.0);
        na2s[self.ops.len() - 1] = 0.0;
        for (i, op) in self.ops.iter().enumerate().rev() {
            let (l1, r1) = na1s.split_at_mut(i);
            let c1 = r1[0];
            let (l2, r2) = na2s.split_at_mut(i);
            let c2 = r2[0];
            match *op {
                Add(j) => {
                    l1[i - 1] = c1;
                    l2[i - 1] = c2;
                    l1[j] = c1;
                    l2[j] = c2;
                },
                Sub(j) => {
                    // Take note of order where oth - pre 
                    l1[i - 1] = -c1;
                    l2[i - 1] = -c2;
                    l1[j] = c1;
                    l2[j] = c2;
                },
                Mul(j) => {
                    l1[i - 1] = c1*ns[j];
                    l2[i - 1] = c2*ns[j] + c1*nds[j];
                    l1[j] = c1*ns[i - 1];
                    l2[j] = c2*ns[i - 1] + c1*nds[i - 1];
                },
                Neg => {
                    l1[i - 1] = -c1;
                    l2[i - 1] = -c2;
                },
                Pow(pow) => {
                    // Assume it is not 0 or 1
                    let vald = f64::from(pow)*ns[i - 1].powi(pow - 1);
                    let valdd = f64::from(pow*(pow - 1))
                                *ns[i - 1].powi(pow - 2);
                    l1[i - 1] = c1*vald;
                    l2[i - 1] = c2*vald + c1*valdd*nds[i - 1];
                },
                Sin => {
                    l1[i - 1] = c1*ns[i - 1].cos();
                    l2[i - 1] = c2*ns[i - 1].cos()
                                - c1*ns[i - 1].sin()*nds[i - 1];
                },
                Cos => {
                    l1[i - 1] = -c1*ns[i - 1].sin();
                    l2[i - 1] = -c2*ns[i - 1].sin()
                                - c1*ns[i - 1].cos()*nds[i - 1];
                },
                Sum(ref js) => {
                    l1[i - 1] = c1;
                    l2[i - 1] = c2;
                    for &j in js {
                        l1[j] = c1;
                        l2[j] = c2;
                    }
                },
                Square => {
                    l1[i - 1] = c1*2.0*ns[i - 1];
                    l2[i - 1] = c2*2.0*ns[i - 1] + c1*2.0*nds[i - 1];
                },
                Variable(Var(id)) => {
                    if let Some(v) = ids.get_mut(&id) {
                        *v += c2;
                    }
                },
                _ => {},
            }
        }
        let mut der2 = Vec::new();
        for id in d2 {
            der2.push(*ids.get(id).unwrap());
        }
        der2
    }

    /// Calculate all using combination of forward and reverse AD.
    ///
    /// `d1` and `d2` must be same length.
    pub fn full_fwd_rev(&self, d1: &Vec<ID>, d2: &Vec<Vec<ID>>,
                        ret: &Retrieve, ws: &mut WorkSpace) -> Column {
        use self::Oper::*;
        use self::{Var, Par};

        let mut col = Column::new();

        col.val = self.eval(ret, &mut ws.ns);

        for (i, (&id, oids)) in d1.iter().zip(d2.iter()).enumerate() {
            if !oids.is_empty() {
                col.der1.push(self.der1_fwd(id, ret, &ws.ns, &mut ws.nds));
                col.der2.append(&mut self.der2_rev(oids, ret, &ws.ns, &ws.nds, 
                                                   &mut ws.na1s, &mut ws.na2s,
                                                   &mut ws.ids));
            }
        }
        // Check if all first derivatives were calculated, and if not just use
        // the reverse method.
        if col.der1.len() < d1.len() {
            col.der1 = self.der1_rev(d1, ret, &ws.ns, &mut ws.na1s,
                                     &mut ws.ids);
        }
        
        col
    }

    /// Calculate constant derivatives by method auto selection.
    pub fn auto_const(&self, info: &FilmInfo, store: &Retrieve,
                     ws: &mut WorkSpace) -> Column {
        let mut col = Column::new();
        if !info.lin.is_empty() {
            self.eval(store, &mut ws.ns);
            col.der1 = self.der1_rev(&info.lin, store, &ws.ns, &mut ws.na1s,
                                     &mut ws.ids);
        }
        if !info.quad.is_empty() {
            col.der2 = self.full_fwd(&info.nlin, &info.quad, store,
                                     &mut ws.cols).der2.clone();
            // If using this need to make sure film has had set_lists() called
            //col.der2 = self.full_fwd_rev(&info.nlin, &info.quad_list, store,
            //                             ws).der2;
        }
        col
    }

    /// Calculate dynamic derivatives by method auto selection.
    pub fn auto_dynam(&self, info: &FilmInfo, store: &Retrieve,
                      ws: &mut WorkSpace) -> Column {
        if info.nlin.is_empty() {
            let mut col = Column::new();
            col.val = self.eval(store, &mut ws.ns);
            col
        } else if info.nquad.is_empty() {
            let mut col = Column::new();
            col.val = self.eval(store, &mut ws.ns);
            col.der1 = self.der1_rev(&info.nlin, store, &ws.ns, &mut ws.na1s,
                                  &mut ws.ids);
            col
        } else {
            self.full_fwd(&info.nlin, &info.nquad, store, &mut ws.cols).clone()
            // If using this need to make sure film has had set_lists() called
            //self.full_fwd_rev(&info.nlin, &info.nquad_list, store, ws);
        }
    }

    /// Calculate information about the `Film`.
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
    fn operations() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.pars.push(4.0);

        let mut f = Film::from(5.0);
        f = 5.0 + f;
        f = Var(0) + f;
        f = f*Par(0);

        let info = f.get_info();

        let mut ws = WorkSpace::new();
        //println!("{:?}", info);
        //println!("{:?}", f);
        let col = f.full_fwd(&info.lin, &Vec::new(), &store, &mut ws.cols);

        assert_eq!(col.val, 60.0);
        assert_eq!(col.der1[0], 4.0);
    }

    #[test]
    fn variety() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let film = Var(0) + (1.0 - Var(1)).powi(2);
        //let mut film = Film { ops: Vec::new() };
        // v0 + (1.0 - v1)^2
        //film.ops.push(Float(1.0));
        //film.ops.push(Variable(Var(1)));
        //film.ops.push(Neg);
        //film.ops.push(Add(0));
        //film.ops.push(Pow(2));
        //film.ops.push(Variable(Var(0)));
        //film.ops.push(Add(4));

        let info = film.get_info();
        let mut ws = WorkSpace::new();
        //let mut info = FilmInfo::new();
        //info.lin.push(0);
        //info.nlin.push(1);
        //info.quad.push((0, 0)); // the first entry in nlin

        // Get constant first derivatives
        // Call on parameter change
        // Copy out and store first derivatives
        {
            let col = film.full_fwd(&info.lin, &Vec::new(), &store,
                &mut ws.cols);
            assert_eq!(col.val, 14.0);
            assert_eq!(col.der1[0], 1.0); // Var(0)
        }

        // Get constant second derivatives
        // Call on parameter change
        // Copy out and store second derivatives
        {
            let col = film.full_fwd(&info.nlin, &info.quad, &store,
                &mut ws.cols);
            assert_eq!(col.val, 14.0);
            assert_eq!(col.der1[0], 6.0); // Var(1)
            assert_eq!(col.der2[0], 2.0); // Var(1), Var(1)
        }

        // Get dynamic derivatives
        // Call every time
        {
            let col = film.full_fwd(&info.nlin, &info.nquad, &store,
                &mut ws.cols);
            assert_eq!(col.val, 14.0);
            assert_eq!(col.der1[0], 6.0); // Var(1)
        }
    }

    #[test]
    fn degree() {
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

        let mut info = film.get_info();
        info.set_lists();

        assert_eq!(info, FilmInfo {
            lin: vec![0],
            nlin: vec![1],
            quad: vec![(0, 0)],
            nquad: vec![],
            quad_list: vec![vec![1]],
            nquad_list: vec![vec![]],
        }); 
    }

    // Only expect to work when have operator overloading creating sums
    //#[test]
    //fn sum() {
    //    use expression::Oper::*;
    //    use expression::Var;
    //    let mut store = Store::new();
    //    store.vars.push(5.0);

    //    let f = 5.0 + Var(0) + Var(0) + Var(0);

    //    let info = f.get_info();

    //    //println!("{:?}", f);
    //    //println!("{:?}", info);
    //    let mut ws = WorkSpace::new();
    //    let col = f.full_fwd(&info.lin, &Vec::new(), &store, &mut ws);

    //    assert_eq!(col.val, 20.0);
    //}

    #[test]
    fn sin() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);

        let mut ws = WorkSpace::new();

        let f = (2.0*Var(0)).sin();
        let info = f.get_info();
        //println!("{:?}", f);
        //println!("{:?}", info);
        assert_eq!(info.nlin.len(), 1);
        assert_eq!(info.nquad.len(), 1);

        let col = f.full_fwd(&info.nlin, &info.nquad, &store, &mut ws.cols);

        assert_eq!(col.val, 10.0_f64.sin());
        assert_eq!(col.der1[0], 2.0*(10.0_f64.cos()));
        assert_eq!(col.der2[0], -4.0*(10.0_f64.sin()));
    }

    #[test]
    fn cos() {
        use expression::Oper::*;
        use expression::Var;
        let mut store = Store::new();
        store.vars.push(5.0);

        let mut ws = WorkSpace::new();

        let f = (2.0*Var(0)).cos();
        let info = f.get_info();
        //println!("{:?}", f);
        //println!("{:?}", info);
        assert_eq!(info.nlin.len(), 1);
        assert_eq!(info.nquad.len(), 1);

        let col = f.full_fwd(&info.nlin, &info.nquad, &store, &mut ws.cols);

        assert_eq!(col.val, 10.0_f64.cos());
        assert_eq!(col.der1[0], -2.0*(10.0_f64.sin()));
        assert_eq!(col.der2[0], -4.0*(10.0_f64.cos()));
    }

    #[test]
    fn reverse() {
        use expression::{Var, Film};
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let x1 = Var(0);
        let x2 = Var(1);

        let f = x1*x2 + NumOpsF::sin(x1);
        let mut ws = WorkSpace::new();
        //let info = f.get_info();
        //println!("{:?}", f);
        //println!("{:?}", info);
        let v = f.eval(&store, &mut ws.ns);
        let der1 = f.der1_rev(&vec![0, 1], &store, &ws.ns, &mut ws.na1s,
                             &mut ws.ids);

        assert_eq!(v, 20.0 + 5.0_f64.sin());
        assert_eq!(der1[0], 5.0_f64.cos() + 4.0);
        assert_eq!(der1[1], 5.0);
    }

    #[test]
    fn forward_reverse() {
        use expression::{Var, Film};
        let mut store = Store::new();
        store.vars.push(5.0);
        store.vars.push(4.0);

        let x1 = Var(0);
        let x2 = Var(1);

        let f = x1*x2 + NumOpsF::sin(x1);
        let mut ws = WorkSpace::new();
        let mut info = f.get_info();
        info.set_lists();
        //println!("{:?}", f);
        //println!("{:?}", info);
        let v = f.eval(&store, &mut ws.ns);
        let der1 = f.der1_rev(&vec![0, 1], &store, &ws.ns, &mut ws.na1s,
                             &mut ws.ids);
        let quad_col = f.full_fwd_rev(&info.nlin, &info.quad_list,
                                      &store, &mut ws);
        let nquad_col = f.full_fwd_rev(&info.nlin, &info.nquad_list,
                                       &store, &mut ws);

        assert_eq!(quad_col.val, 20.0 + 5.0_f64.sin());
        assert_eq!(quad_col.der1.len(), 2);
        assert_eq!(quad_col.der1[0], 5.0_f64.cos() + 4.0);
        assert_eq!(quad_col.der1[1], 5.0);
        assert_eq!(quad_col.der2.len(), 1); // quad
        assert_eq!(quad_col.der2[0], 1.0);
        assert_eq!(nquad_col.val, 20.0 + 5.0_f64.sin());
        assert_eq!(nquad_col.der1.len(), 2);
        assert_eq!(nquad_col.der1[0], 5.0_f64.cos() + 4.0);
        assert_eq!(nquad_col.der1[1], 5.0);
        assert_eq!(nquad_col.der2.len(), 1); // quad
        assert_eq!(nquad_col.der2[0], -5.0_f64.sin());
    }

    #[bench]
    fn auto_const(b: &mut test::Bencher) {
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
        let info = e.get_info();
        //println!("{:?}", info);
        b.iter(|| {
            e.auto_const(&info, &store, &mut ws);
        });
    }

    #[bench]
    fn auto_dynam(b: &mut test::Bencher) {
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
        let info = e.get_info();
        //println!("{:?}", info);
        b.iter(|| {
            e.auto_dynam(&info, &store, &mut ws);
        });
    }

}

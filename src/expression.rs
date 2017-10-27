use std;
use std::collections::HashSet;

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
}

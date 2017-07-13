
type ID = usize;

// Should not expect an error, panic if out of bounds error.  We could add
// a verification call that gets done once, or on adding constraints and
// objective to model check that variables and parameters are valid.
pub trait Retrieve {
    fn get_var(&self, vid: &ID) -> f64;
    fn get_par(&self, pid: &ID) -> f64;
}

pub trait Evaluate {
    fn value(&self, ret: &Retrieve) -> f64;
    fn deriv(&self, ret: &Retrieve, vid: &ID) -> (f64, f64); // value, deriv
}

#[derive(Debug, Clone)]
pub enum Expr {
    //Function(Box<Evaluate>, Vec<Expr>), // need trait where can pass nodes
    Add(Vec<Expr>),
    Mul(Vec<Expr>),
    Neg(Box<Expr>), // negate
    Inv(Box<Expr>), // invert
    Pow(Box<Expr>, i32),
    Variable(ID),
    Parameter(ID),
    Float(f64),
    Integer(i32),
}

impl Evaluate for Expr {
    fn value(&self, ret: &Retrieve) -> f64 {
        use Expr::*;
        match *self {
            Add(ref es) => es.iter().fold(0.0, |a, e| { a + e.value(ret) }),
            Mul(ref es) => es.iter().fold(1.0, |a, e| { a*e.value(ret) }),
            Neg(ref e) => -e.value(ret),
            Inv(ref e) => 1.0/e.value(ret),
            Pow(ref e, p) => e.value(ret).powi(p),
            Variable(ref id) => ret.get_var(id),
            Parameter(ref id) => ret.get_par(id),
            Float(v) => v,
            Integer(v) => v as f64,
        }
    }

    fn deriv(&self, ret: &Retrieve, vid: &ID) -> (f64, f64) {
        use Expr::*;
        match *self {
            Add(ref es) =>
                es.iter().fold((0.0, 0.0), |a, n| {
                    let r = n.deriv(ret, vid);
                    (a.0 + r.0, a.1 + r.1)
                }),
            Mul(ref es) =>
                es.iter().fold((1.0, 0.0), |a, n| {
                    let r = n.deriv(ret, vid);
                    (a.0*r.0, a.0*r.1 + r.0*a.1)
                }),
            Neg(ref e) => {
                let r = e.deriv(ret, vid);
                (-r.0, -r.1)
            },
            Inv(ref e) => {
                let r = e.deriv(ret, vid);
                (1.0/r.0, -r.1/(r.0*r.0))
            },
            Pow(ref e, p) => {
                let r = e.deriv(ret, vid);
                (r.0.powi(p), r.1*(p as f64)*r.0.powi(p - 1))
            },
            Variable(ref id) =>
                (ret.get_var(id), if id == vid { 1.0 } else { 0.0 }),
            Parameter(ref id) => (ret.get_par(id), 0.0),
            Float(v) => (v, 0.0),
            Integer(v) => (v as f64, 0.0),
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

binary_ops!(Add, add, &Expr, &Expr, Expr);
binary_ops!(Add, add, &Expr, Expr, Expr);
binary_ops!(Add, add, Expr, &Expr, Expr);

binary_ops!(Mul, mul, &Expr, &Expr, Expr);
binary_ops!(Mul, mul, &Expr, Expr, Expr);
binary_ops!(Mul, mul, Expr, &Expr, Expr);

binary_ops_other_cast!(Add, add, Expr, i32, Expr, Expr::Integer);
binary_ops_other_cast!(Add, add, &Expr, i32, Expr, Expr::Integer);
binary_ops_self_cast!(Add, add, i32, Expr, Expr, Expr::Integer);
binary_ops_self_cast!(Add, add, i32, &Expr, Expr, Expr::Integer);

binary_ops_other_cast!(Add, add, Expr, f64, Expr, Expr::Float);
binary_ops_other_cast!(Add, add, &Expr, f64, Expr, Expr::Float);
binary_ops_self_cast!(Add, add, f64, Expr, Expr, Expr::Float);
binary_ops_self_cast!(Add, add, f64, &Expr, Expr, Expr::Float);

binary_ops_other_cast!(Mul, mul, Expr, i32, Expr, Expr::Integer);
binary_ops_other_cast!(Mul, mul, &Expr, i32, Expr, Expr::Integer);
binary_ops_self_cast!(Mul, mul, i32, Expr, Expr, Expr::Integer);
binary_ops_self_cast!(Mul, mul, i32, &Expr, Expr, Expr::Integer);

binary_ops_other_cast!(Mul, mul, Expr, f64, Expr, Expr::Float);
binary_ops_other_cast!(Mul, mul, &Expr, f64, Expr, Expr::Float);
binary_ops_self_cast!(Mul, mul, f64, Expr, Expr, Expr::Float);
binary_ops_self_cast!(Mul, mul, f64, &Expr, Expr, Expr::Float);

struct Store {
    vars: Vec<f64>,
    pars: Vec<f64>,
}

impl Store {
    fn new() -> Self {
        Store { vars: Vec::new(), pars: Vec::new() }
    }
}

impl Retrieve for Store {
    fn get_var(&self, vid: &ID) -> f64 {
        self.vars[*vid]
    }

    fn get_par(&self, pid: &ID) -> f64 {
        self.pars[*pid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn evaluation() {
        use Expr::*;
        let mut store = Store::new();

        assert_eq!(Float(1.0).value(&store), 1.0);

        store.vars.push(5.0);
        assert_eq!(Variable(0).value(&store), 5.0);
        assert_eq!(Variable(0).deriv(&store, &0_usize), (5.0, 1.0));
        assert_eq!(Variable(0).deriv(&store, &1_usize), (5.0, 0.0));

        store.pars.push(4.0);
        assert_eq!(Parameter(0).value(&store), 4.0);

        let v = Variable(0);
        let p = Parameter(0);

        assert_eq!((&v + &p).value(&store), 9.0);

        assert_eq!((&v + &p + 5).value(&store), 14.0);
        assert_eq!((3 + &v + &p + 5).value(&store), 17.0);

        assert_eq!((&v + &p + 5.0).value(&store), 14.0);
        assert_eq!((3.0 + &v + &p + 5.0).value(&store), 17.0);
    }
}

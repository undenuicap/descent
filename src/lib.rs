
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

#[derive(Debug)]
pub enum Node {
    //Function(Box<Evaluate>, Vec<Node>), // need trait where can pass nodes
    Add(Vec<Node>),
    Mul(Vec<Node>),
    Neg(Box<Node>), // negate
    Inv(Box<Node>), // invert
    Pow(Box<Node>, i32),
    Variable(ID),
    Parameter(ID),
    Float(f64),
    Integer(i32),
}

impl Evaluate for Node {
    fn value(&self, ret: &Retrieve) -> f64 {
        use Node::*;
        match *self {
            Add(ref ns) => ns.iter().fold(0.0, |a, n| { a + n.value(ret) }),
            Mul(ref ns) => ns.iter().fold(1.0, |a, n| { a*n.value(ret) }),
            Neg(ref n) => -n.value(ret),
            Inv(ref n) => 1.0/n.value(ret),
            Pow(ref n, e) => n.value(ret).powi(e),
            Variable(ref id) => ret.get_var(id),
            Parameter(ref id) => ret.get_par(id),
            Float(v) => v,
            Integer(v) => v as f64,
        }
    }

    fn deriv(&self, ret: &Retrieve, vid: &ID) -> (f64, f64) {
        use Node::*;
        match *self {
            Add(ref ns) =>
                ns.iter().fold((0.0, 0.0), |a, n| {
                    let r = n.deriv(ret, vid);
                    (a.0 + r.0, a.1 + r.1)
                }),
            Mul(ref ns) =>
                ns.iter().fold((1.0, 0.0), |a, n| {
                    let r = n.deriv(ret, vid);
                    (a.0*r.0, a.0*r.1 + r.0*a.1)
                }),
            Neg(ref n) => {
                let r = n.deriv(ret, vid);
                (-r.0, -r.1)
            },
            Inv(ref n) => {
                let r = n.deriv(ret, vid);
                (1.0/r.0, -r.1/(r.0*r.0))
            },
            Pow(ref n, e) => {
                let r = n.deriv(ret, vid);
                (r.0.powi(e), r.1*(e as f64)*r.0.powi(e - 1))
            },
            Variable(ref id) =>
                (ret.get_var(id), if id == vid { 1.0 } else { 0.0 }),
            Parameter(ref id) => (ret.get_par(id), 0.0),
            Float(v) => (v, 0.0),
            Integer(v) => (v as f64, 0.0),
        }
    }
}

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
        use Node::*;
        let mut store = Store::new();

        assert_eq!(Float(1.0).value(&store), 1.0);

        store.vars.push(5.0);
        assert_eq!(Variable(0).value(&store), 5.0);
        assert_eq!(Variable(0).deriv(&store, &0_usize), (5.0, 1.0));
        assert_eq!(Variable(0).deriv(&store, &1_usize), (5.0, 0.0));

        store.pars.push(4.0);
        assert_eq!(Parameter(0).value(&store), 4.0);

        let n = Add(vec![Variable(0), Parameter(0)));
    }
}

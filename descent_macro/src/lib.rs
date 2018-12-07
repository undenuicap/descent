#![feature(proc_macro_hygiene)]

extern crate proc_macro;
use crate::proc_macro::*;
use std::collections::HashSet;
use std::iter::Peekable;
use std::str::FromStr;

fn separate_on_punct<I: IntoIterator<Item = TokenTree>>(input: I, pchar: char) -> Vec<Vec<TokenTree>> {
    let mut split = vec![Vec::new()];
    for t in input {
        match t {
            TokenTree::Punct(punct) => {
                if punct.as_char() == pchar {
                    split.push(Vec::new());
                } else {
                    split.last_mut().unwrap().push(punct.into());
                }
            },
            t => {
                split.last_mut().unwrap().push(t);
            },
        };
    }
    split
}

type IdenVec = Vec<(String, Option<Vec<TokenTree>>)>;

fn prepare_idents<I: IntoIterator<Item = TokenTree>>(input: Vec<I>) -> IdenVec {
    let mut vec = Vec::new();
    for entry in input {
        let mut split = separate_on_punct(entry, '=');
        // Split is never empty at top level, check one step down
        if split.len() == 1 && split[0].is_empty() {
            continue;
        }
        if split.len() > 2 {
            panic!("Expected ident = expr for variables and parameters");
        }
        let rhs = if split.len() == 2 { split.pop() } else { None };
        let key = match split[0].as_slice() {
            [TokenTree::Ident(ident)] => ident.to_string(),
            _ => panic!("Expected ident = expr for variables and parameters"),
        };
        vec.push((key, rhs));
    }
    vec 
}

#[derive(Debug)]
enum ExprToken {
    Var(String),
    Par(String),
    Tokens(Vec<TokenTree>), // everything must be constant beyond here
    Group(Vec<ExprToken>),
    Add,
    Sub,
    Neg,
    Mul,
    Div,
}

// Lower is higher priority
type Priority = usize;

impl ExprToken {
    fn priority(&self) -> Option<Priority> {
        match self {
            ExprToken::Neg => Some(1),
            ExprToken::Mul => Some(2),
            ExprToken::Div => Some(2),
            ExprToken::Add => Some(3),
            ExprToken::Sub => Some(3),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    Var(String),
    Par(String),
    Const(f64),
    Tokens(Vec<TokenTree>), // everything must be constant beyond here
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, i32),
}

impl Expr {
    fn value(&self) -> Option<f64> {
        if let Expr::Const(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    fn is_zero(&self) -> bool {
        if let Expr::Const(v) = self {
            *v == 0.0
        } else {
            false
        }
    }

    fn is_one(&self) -> bool {
        if let Expr::Const(v) = self {
            *v == 1.0
        } else {
            false
        }
    }

    fn into_tokens(self, mut tokens: &mut Vec<TokenTree>) {
        // could directly call add, sub, etc instead of operators...
        match self {
            // doing a iden.0 to get usize from Var or Par
            Expr::Var(iden) => {
                tokens.extend(TokenStream::from_str(&format!("__v[{}.0]", iden)).unwrap().into_iter());
                //tokens.push(TokenTree::Ident(Ident::new("__v", Span::call_site())));
                //let mut child = Vec::new();
                ////child.push(TokenTree::Literal(Literal::usize_suffixed(i)));
                //child.push(TokenTree::Ident(Ident::new(&iden, Span::call_site())));
                //child.push(TokenTree::Punct(Punct::new('.', Spacing::Alone)));
                //child.push(TokenTree::Literal(Literal::usize_unsuffixed(0)));
                //tokens.push(TokenTree::Group(Group::new(Delimiter::Bracket, child.into_iter().collect())));
            },
            Expr::Par(iden) => {
                tokens.extend(TokenStream::from_str(&format!("__p[{}.0]", iden)).unwrap().into_iter());
                //tokens.push(TokenTree::Ident(Ident::new("__p", Span::call_site())));
                //let mut child = Vec::new();
                ////child.push(TokenTree::Literal(Literal::usize_suffixed(i)));
                //child.push(TokenTree::Ident(Ident::new(&iden, Span::call_site())));
                //child.push(TokenTree::Punct(Punct::new('.', Spacing::Alone)));
                //child.push(TokenTree::Literal(Literal::usize_unsuffixed(0)));
                //tokens.push(TokenTree::Group(Group::new(Delimiter::Bracket, child.into_iter().collect())));
            },
            Expr::Const(v) => {
                tokens.push(TokenTree::Literal(Literal::f64_suffixed(v)));
            },
            Expr::Tokens(t) => {
                tokens.extend(t.into_iter());
            },
            Expr::Add(l, r) => {
                let mut child = Vec::new();
                l.into_tokens(&mut child);
                child.push(TokenTree::Punct(Punct::new('+', Spacing::Alone)));
                r.into_tokens(&mut child);
                tokens.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, child.into_iter().collect())));
            },
            Expr::Sub(l, r) => {
                let mut child = Vec::new();
                l.into_tokens(&mut child);
                child.push(TokenTree::Punct(Punct::new('-', Spacing::Alone)));
                r.into_tokens(&mut child);
                tokens.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, child.into_iter().collect())));
            },
            Expr::Neg(v) => {
                tokens.push(TokenTree::Punct(Punct::new('-', Spacing::Alone)));
                v.into_tokens(&mut tokens);
            },
            Expr::Mul(l, r) => {
                l.into_tokens(&mut tokens);
                tokens.push(TokenTree::Punct(Punct::new('*', Spacing::Alone)));
                r.into_tokens(&mut tokens);
            },
            Expr::Pow(v, e) => {
                v.into_tokens(&mut tokens);
                tokens.push(TokenTree::Punct(Punct::new('.', Spacing::Alone)));
                tokens.push(TokenTree::Ident(Ident::new("powi", Span::call_site())));
                let mut child = Vec::new();
                child.push(TokenTree::Literal(Literal::i32_suffixed(e)));
                tokens.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, child.into_iter().collect())));
            },
        }
    }
}

fn simplify(expr: Expr) -> Expr {
    match expr {
        Expr::Add(l, r) => {
            if l.is_zero() {
                *r
            } else if r.is_zero() {
                *l
            } else if let (Some(lv), Some(rv)) = (l.value(), r.value()) {
                Expr::Const(lv + rv)
            } else {
                Expr::Add(l, r)
            }
        },
        Expr::Sub(l, r) => {
            if l.is_zero() {
                simplify(Expr::Neg(r))
            } else if r.is_zero() {
                *l
            } else if let (Some(lv), Some(rv)) = (l.value(), r.value()) {
                Expr::Const(lv - rv)
            } else {
                Expr::Sub(l, r)
            }
        },
        Expr::Neg(v) => {
            if let Some(vv) = v.value() {
                Expr::Const(-vv)
            } else {
                Expr::Neg(v)
            }
        },
        Expr::Mul(l, r) => {
            if l.is_zero() || r.is_zero() {
                Expr::Const(0.0)
            } else if let (Some(lv), Some(rv)) = (l.value(), r.value()) {
                Expr::Const(lv * rv)
            } else if l.is_one() {
                *r
            } else if r.is_one() {
                *l
            } else {
                Expr::Mul(l, r)
            }
        },
        Expr::Pow(v, e) => {
            if let Some(vv) = v.value() {
                Expr::Const(vv.powi(e))
            } else {
                match e {
                    0 => Expr::Const(1.0),
                    1 => *v,
                    e => Expr::Pow(v, e),
                }
            }
        },
        e => e,
    }
}

fn deriv1(expr: &Expr, vid: &str) -> Expr {
    simplify(match expr {
        Expr::Var(iden) => {
            if *iden == vid {
                Expr::Const(1.0)
            } else {
                Expr::Const(0.0)
            }
        },
        Expr::Par(_) | Expr::Const(_) | Expr::Tokens(_) => Expr::Const(0.0),
        Expr::Add(l, r) => Expr::Add(Box::new(deriv1(l, vid)), Box::new(deriv1(r, vid))),
        Expr::Sub(l, r) => Expr::Sub(Box::new(deriv1(l, vid)), Box::new(deriv1(r, vid))),
        Expr::Neg(v) => Expr::Neg(Box::new(deriv1(v, vid))),
        Expr::Mul(l, r) => Expr::Add(Box::new(simplify(Expr::Mul(Box::new(deriv1(l, vid)), r.clone()))),
                                     Box::new(simplify(Expr::Mul(Box::new(deriv1(r, vid)), l.clone())))),
        Expr::Pow(v, e) => Expr::Mul(Box::new(Expr::Const(*e as f64)),
                                     Box::new(simplify(Expr::Mul(Box::new(deriv1(v, vid)),
                                                                 Box::new(simplify(Expr::Pow(v.clone(), e - 1))))))),
    })
}

fn tokens_to_expr<I: Iterator<Item = ExprToken>>(mut iter: &mut Peekable<I>, pr: Option<Priority>) -> Expr {
    let mut expr = None;
    loop {
        if let Some(token) = iter.peek() {
            if let Some(p) = pr {
                if let Some(p_op) = token.priority() {
                    if p_op >= p { // back out
                        break;
                    }
                }
            }
        } else {
            break;
        }
        let token = iter.next().unwrap(); // already checked it exists
        expr = Some(match token {
            ExprToken::Var(i) => Expr::Var(i),
            ExprToken::Par(i) => Expr::Par(i),
            ExprToken::Tokens(t) => Expr::Tokens(t),
            ExprToken::Group(stream) => tokens_to_expr(&mut stream.into_iter().peekable(), None),
            op @ ExprToken::Add => {
                let lhs = expr.expect("Addition operator has no LHS");
                Expr::Add(Box::new(lhs), Box::new(tokens_to_expr(&mut iter, op.priority())))
            }
            op @ ExprToken::Sub => {
                let lhs = expr.expect("Subtraction operator has no LHS");
                Expr::Sub(Box::new(lhs), Box::new(tokens_to_expr(&mut iter, op.priority())))
            }
            op @ ExprToken::Neg => {
                Expr::Neg(Box::new(tokens_to_expr(&mut iter, op.priority())))
            }
            op @ ExprToken::Mul => {
                let lhs = expr.expect("Multiplication operator has no LHS");
                Expr::Mul(Box::new(lhs), Box::new(tokens_to_expr(&mut iter, op.priority())))
            }
            op @ ExprToken::Div => {
                let lhs = expr.expect("Subtraction operator has no LHS");
                Expr::Mul(Box::new(lhs), Box::new(Expr::Pow(Box::new(tokens_to_expr(&mut iter, op.priority())), -1)))
            }
        });
    }
    expr.expect("Empty expression")
}

fn get_const_tokens<I: Iterator<Item = TokenTree>>(first: TokenTree, iter: &mut Peekable<I>) -> ExprToken {
    let mut tokens = vec![first];
    loop {
        match iter.peek() {
            None => {
                break;
            },
            Some(TokenTree::Punct(punct)) => {
                let c = punct.as_char();
                if c == '+' || c == '*' || c == '-' || c == '/' {
                    break;
                }
            },
            _ => {}, // continue
        }
        tokens.push(iter.next().unwrap());
    }
    ExprToken::Tokens(tokens)
}

fn get_expr<I: Iterator<Item = TokenTree>>(left: Option<&ExprToken>,
                                           mut iter: &mut Peekable<I>,
                                           vars: &HashSet<String>,
                                           pars: &HashSet<String>) -> Option<ExprToken> {
    match iter.next() {
        Some(TokenTree::Ident(ident)) => {
            let id = ident.to_string();
            if vars.contains(id.as_str()) {
                Some(ExprToken::Var(id))
            } else if pars.contains(id.as_str()) {
                Some(ExprToken::Par(id))
            } else {
                Some(get_const_tokens(TokenTree::Ident(ident), &mut iter))
            }
        },
        Some(TokenTree::Punct(punct)) => {
            match punct.as_char() {
                '+' => Some(ExprToken::Add),
                '-' => {
                    match left {
                        None | Some(ExprToken::Add) | Some(ExprToken::Sub) | Some(ExprToken::Mul) | Some(ExprToken::Div) => Some(ExprToken::Neg),
                        _ => Some(ExprToken::Sub),
                    }
                },
                '*' => {
                    // Check if it looks like a dereference
                    match left {
                        None | Some(ExprToken::Add) | Some(ExprToken::Sub) | Some(ExprToken::Mul) | Some(ExprToken::Div) =>
                            Some(get_const_tokens(TokenTree::Punct(punct), &mut iter)),
                        _ => Some(ExprToken::Mul),
                    }
                },
                '/' => Some(ExprToken::Div),
                _ => Some(get_const_tokens(TokenTree::Punct(punct), &mut iter)),
            }
        },
        Some(TokenTree::Group(group)) => {
            if group.delimiter() == Delimiter::Parenthesis {
                Some(ExprToken::Group(to_expr_stream(&mut group.stream().into_iter().peekable(), vars, pars)))
            } else {
                panic!("Can only work with grouping with parenthesis");
            }
        },
        Some(TokenTree::Literal(lit)) => {
             Some(get_const_tokens(TokenTree::Literal(lit), &mut iter))
        },
        None => {
            None
        },
    }
}

fn to_expr_stream<I: Iterator<Item = TokenTree>>(mut iter: &mut Peekable<I>,
                                                 vars: &HashSet<String>,
                                                 pars: &HashSet<String>) -> Vec<ExprToken> {
    let mut expr_stream = Vec::new();
    while let Some(expr) = get_expr(expr_stream.last(), &mut iter, vars, pars) {
        expr_stream.push(expr);
    }
    expr_stream
}

/// Checks if any identifiers matches one of the names.
fn contains_ident(iter: TokenStream, names: &HashSet<&str>) -> bool {
    for token in iter {
        let found = match token {
            TokenTree::Ident(ident) => {
                names.contains(ident.to_string().as_str())
            },
            TokenTree::Group(group) => {
                contains_ident(group.stream(), &names)
            },
            _ => false,
        };
        if found {
            return true;
        }
    }
    false
}

/// Generate a ExprStatic expression
///
/// expr!(<expr>; var1 [= <expr>], ...[; par1 [= <expr>], ...])
///
/// ```
/// let x = Var(0);
/// let y = Var(1);
/// let a = Par(0);
/// let e = expr!(a * x + y * y; x, y; a);
/// ```
///
/// Variables and parameters need to declared. Constants and other expressions
/// are captured (moved) from the environment, similarly to a closure. There
/// are in fact multiple closures being created, so anything that is captured
/// needs to implement copy.
///
/// There is an option to assign a variable / parameter a value from the
/// environment, to make it convenient to use variables and parameters that
/// appear in more complex structures:
///
/// ```
/// let vars = [Var(0), Var(1)];
/// let pars = [Par(0)];
/// let e = expr!(a * x + y * y; x = vars[0], y = vars[1]; a = pars[0]);
/// ```
///
/// This avoids having to manually do a separate:
///
/// ```
/// let x = vars[0];
/// ```
///
/// This convenience currently hasn't been extended to constant terms. So we
/// still need to do the following so we don't try to move the vector of
/// constants multiple times:
/// ```
/// let constant = vec![1.0, 2.0];
/// let c = constant[0];
/// let e = expr!(a * x + y * y + c; x, y; a);
/// ```
#[proc_macro]
pub fn expr(input: TokenStream) -> TokenStream {
    let invalid_ident = ["__v", "__p", "__d1", "__d2"].iter().cloned().collect();
    if contains_ident(input.clone(), &invalid_ident) {
        panic!("Cannot use any of {:?} as identifier", invalid_ident);
    }

    let mut split = separate_on_punct(input, ';');
    if split.len() < 2 {
        panic!("Expected variables to be specified");
    }

    let mut p = IdenVec::new();
    if split.len() == 3 {
        p = prepare_idents(separate_on_punct(split.pop().unwrap(), ','));
    }
    let v = prepare_idents(separate_on_punct(split.pop().unwrap(), ','));
    let mut v_set = HashSet::new();
    let mut p_set = HashSet::new();
    for (k, _) in &v {
        if v_set.contains(k) {
            panic!("Variable identifier cannot be used twice");
        }
        v_set.insert(k.clone());
    }
    for (k, _) in &p {
        if p_set.contains(k) {
            panic!("Parameter identifier cannot be used twice");
        }
        p_set.insert(k.clone());
    }
    for k in &p_set {
        if v_set.contains(k) {
            panic!("Cannot use same identifier for parameter and variable");
        }
    }

    let e = split.pop().unwrap();

    let e_stream = to_expr_stream(&mut e.into_iter().peekable(), &v_set, &p_set);
    let expr = tokens_to_expr(&mut e_stream.into_iter().peekable(), None);

    // all combined body
    let mut all_body = Vec::new();

    // d1 & d2 closure
    let mut body1 = Vec::new();
    let mut body2 = Vec::new();
    let mut d1_nz = Vec::new();
    let mut d2_nz = Vec::new();
    for (i, (k1, _)) in v.iter().enumerate() {
        let ex1 = deriv1(&expr, k1);
        for (k2, _) in v.iter().skip(i) {
            let ex2 = deriv1(&ex1, k2);
            if !ex2.is_zero() {
                body2.extend(TokenStream::from_str(&format!("__d2[{}] = ", d2_nz.len())).unwrap().into_iter());
                ex2.into_tokens(&mut body2);
                body2.push(TokenTree::Punct(Punct::new(';', Spacing::Alone)));
                d2_nz.push((k1.clone(), k2.clone()));
            }
        }
        // All first derivatives should typically be non-zero unless variable
        // doesn't appear in expression or cancels out, e.g., x - x.
        // Not simplifying the original expression so variable might be
        // required there but then has zero first derivative.
        if !ex1.is_zero() {
            body1.extend(TokenStream::from_str(&format!("__d1[{}] = ", d1_nz.len())).unwrap().into_iter());
            ex1.into_tokens(&mut body1);
            body1.push(TokenTree::Punct(Punct::new(';', Spacing::Alone)));
            d1_nz.push(k1.clone());
        }
    }

    all_body.extend(body1.clone().into_iter());
    all_body.extend(body2.clone().into_iter());

    let mut d1_clo = Vec::new();
    d1_clo.extend(TokenStream::from_str("move |__v: &[f64], __p: &[f64], __d1: &mut[f64]|").unwrap().into_iter());
    d1_clo.push(TokenTree::Group(Group::new(Delimiter::Brace, body1.into_iter().collect())));

    let mut body = Vec::new();
    for k in &d1_nz {
        body.push(TokenTree::Ident(Ident::new(&k, Span::call_site())));
        body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
    }
    let mut d1_spar = Vec::new();
    d1_spar.extend(TokenStream::from_str("vec!").unwrap().into_iter());
    d1_spar.push(TokenTree::Group(Group::new(Delimiter::Bracket, body.into_iter().collect())));

    let mut d2_clo = Vec::new();
    d2_clo.extend(TokenStream::from_str("move |__v: &[f64], __p: &[f64], __d2: &mut[f64]|").unwrap().into_iter());
    d2_clo.push(TokenTree::Group(Group::new(Delimiter::Brace, body2.into_iter().collect())));

    let mut body = Vec::new();
    for (k1, k2) in &d2_nz {
        body.extend(TokenStream::from_str(&format!("descent::expr::order({}, {})", &k1, &k2)).unwrap().into_iter());
        body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));
    }
    let mut d2_spar = Vec::new();
    d2_spar.extend(TokenStream::from_str("vec!").unwrap().into_iter());
    d2_spar.push(TokenTree::Group(Group::new(Delimiter::Bracket, body.into_iter().collect())));

    // f closure
    let mut body = Vec::new();
    expr.clone().into_tokens(&mut body);

    all_body.extend(body.clone().into_iter());

    let mut f_clo = Vec::new();
    f_clo.extend(TokenStream::from_str("move |__v: &[f64], __p: &[f64]|").unwrap().into_iter());
    f_clo.push(TokenTree::Group(Group::new(Delimiter::Brace, body.into_iter().collect())));

    // all closure
    let mut all_clo = Vec::new();
    all_clo.extend(TokenStream::from_str("move |__v: &[f64], __p: &[f64], __d1: &mut[f64], __d2: &mut[f64]|").unwrap().into_iter());
    all_clo.push(TokenTree::Group(Group::new(Delimiter::Brace, all_body.into_iter().collect())));

    // final returned tokens
    let mut body = Vec::new();
    body.extend(TokenStream::from_str("f: Box::new").unwrap().into_iter());
    body.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, f_clo.into_iter().collect())));
    body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));

    body.extend(TokenStream::from_str("d1: Box::new").unwrap().into_iter());
    body.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, d1_clo.into_iter().collect())));
    body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));

    body.extend(TokenStream::from_str("d2: Box::new").unwrap().into_iter());
    body.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, d2_clo.into_iter().collect())));
    body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));

    body.extend(TokenStream::from_str("all: Box::new").unwrap().into_iter());
    body.push(TokenTree::Group(Group::new(Delimiter::Parenthesis, all_clo.into_iter().collect())));
    body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));

    body.extend(TokenStream::from_str("d1_sparsity: ").unwrap().into_iter());
    body.extend(d1_spar.into_iter());
    body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));

    body.extend(TokenStream::from_str("d2_sparsity: ").unwrap().into_iter());
    body.extend(d2_spar.into_iter());
    body.push(TokenTree::Punct(Punct::new(',', Spacing::Alone)));

    let mut structure = Vec::new();
    // insert local lets
    for (k, rhs) in v {
        if let Some(t) = rhs {
            structure.push(TokenTree::Ident(Ident::new("let", Span::call_site())));
            structure.push(TokenTree::Ident(Ident::new(&k, Span::call_site())));
            structure.push(TokenTree::Punct(Punct::new('=', Spacing::Alone)));
            structure.extend(t);
            structure.push(TokenTree::Punct(Punct::new(';', Spacing::Alone)));
        }
    }
    for (k, rhs) in p {
        if let Some(t) = rhs {
            structure.push(TokenTree::Ident(Ident::new("let", Span::call_site())));
            structure.push(TokenTree::Ident(Ident::new(&k, Span::call_site())));
            structure.push(TokenTree::Punct(Punct::new('=', Spacing::Alone)));
            structure.extend(t);
            structure.push(TokenTree::Punct(Punct::new(';', Spacing::Alone)));
        }
    }
    structure.extend(TokenStream::from_str("descent::expr::ExprStatic").unwrap().into_iter());
    structure.push(TokenTree::Group(Group::new(Delimiter::Brace, body.into_iter().collect())));

    let mut ret = Vec::new();
    // outer block to allow local scope for lets
    ret.push(TokenTree::Group(Group::new(Delimiter::Brace, structure.into_iter().collect())));
    let ret: TokenStream = ret.into_iter().collect();
    //println!("TokenStream: {}", ret.to_string());
    ret
}

# Descent

A non-linear constrained optimisation (mathematical programming) modelling
library with first and second order automatic differentiation and interface to
[Ipopt](https://projects.coin-or.org/Ipopt).

This is in an early state of development but is working / useable.

## Design

It supports operator overloading for the ergonomic expression of terms and the
operators: +, -, \*, powi, sin, cos. It also allows parameterisation of values
and easy solver warmstarting to enable quick model adjustments and resolving.

It follows a similar design philosopy to the C++ / Python library
[madopt](https://github.com/stanle/madopt). As they stand the AD routines are
not as efficient but still adequate for many applications (be sure to build
with --release).

The library could be expanded to link to other solvers.

## Dependencies

[Ipopt](https://projects.coin-or.org/Ipopt) (or
[Bonmin](https://projects.coin-or.org/Bonmin)) must be separately installed
before attempting to build as Descent links to the libipopt.so shared library.

It has only been tested on linux, but presumably would also work on macos, and
potentially on windows in the right environment.

## Example

The following code shows and example of solving the following simple problem:
min 2y s.t. y >= x*x - x, x in \[-10, 10\].

```rust
let mut m = IpoptModel::new();

let x = m.add_var(-10.0, 10.0, 0.0);
let y = m.add_var(NINF, INF, 0.0);
m.set_obj(2.0*y);
m.add_con(y - x*x + x, 0.0, INF);

let (stat, sol) = m.solve();
```

A full example of this with additional details is provided under
examples/simple.rs, which can be built and run as follows:

```
cargo build --release --example simple
./target/release/examples/simple
```

## Benchmarking

For the test problem debug for the release profile should be turned on (in
Cargo.toml), and then build and run using:

```
cargo build --release --example problem
./target/release/examples/problem
```

## Issues

Currently large expressions that require their second derivatives computed are
slow and memory intensive. As a work around for separable expressions, split
them up manually. In the future should be able to automate this.

## TODO

- Bonmin bindings (enabling MINLP).
- Implement summation type for dynamic expressions.

## License

Apache-2.0 or MIT

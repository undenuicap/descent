# Descent

A non-linear constrained optimisation (mathematical programming) modelling
library with first and second order automatic differentiation and interface to
[Ipopt](https://projects.coin-or.org/Ipopt).

This is in an early state of development but is working / useable.

## Design

It supports operator overloading for the ergonomic expression of terms and the
operators: +, -, \*, powi, sin, cos.

It follows a similar design philosopy to the C++ / Python library
[madopt](https://github.com/stanle/madopt). As they stand the AD routines are
not as efficient but still adequate for many applications (be sure to build
with --release).

The library could be expanded to link to other solvers.

## Dependencies

[Ipopt](https://projects.coin-or.org/Ipopt) (or
[Bonmin](https://projects.coin-or.org/Bonmin)) must be separately installed
before attempting to build as Descent links to libipopt.so shared library.

## Example

Build and run the simple example problem:

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

## License

Apache-2.0 or MIT

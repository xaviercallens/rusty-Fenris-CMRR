//! Completely-monotone repulsive relaxation (CMRR) of 2D mesh nodes.
//!
//! Free interior nodes repel each other and the fixed boundary nodes through the Yukawa potential
//! `V(r) = exp(-kappa r) / r`. They follow the overdamped flow `dx/dt = -grad E` to a stationary point. Two phases:
//!
//! 1. **Un-clump**: an explicit Bogacki–Shampine RK2(3). It caps each node's displacement per step, so badly seeded
//!    nodes separate without jumping across the boundary.
//! 2. **Relax**: rusty-SUNDIALS CVODE (BDF), with the analytic Jacobian `-Hessian`, run until the largest force on a
//!    free node falls below a relative tolerance.
//!
//! # What is and is not guaranteed
//!
//! * In the variable `t = r^2`, `exp(-kappa sqrt t) / sqrt t` is completely monotone. That is a mathematical fact.
//!   No Lean proof of it exists yet; the triangular-UO Lean repository proves admissibility only for Riesz potentials.
//! * The planar universal-optimality statements concern *infinite* configurations at fixed density. They say nothing
//!   about N nodes in a bounded domain with pinned boundary nodes, or about which local minimum a gradient flow reaches.
//!   The output is a stationary point of a repulsive energy, which in practice is a well-spaced, locally near-triangular
//!   node set with 5/7 defects where the domain forces them. It is not a certified optimum.
//! * The shifted-force cutoff at `rc` makes the potential no longer exactly completely monotone. The size of the change
//!   is about `exp(-kappa rc)`.
//! * Nothing stops a free node from leaving the domain through a gap between boundary nodes. `Report` counts the free
//!   nodes outside the bounding box of the fixed nodes as a coarse check.
//! * The CVODE linear solver is dense LU. Cost is `O(N_free^2)` memory and `O(N_free^3)` per Jacobian, so
//!   `Params::max_free_nodes` caps the problem size.

pub mod engine;
pub mod ffi;
pub mod model;
pub mod potential;
pub mod vtk;

pub use engine::{optimize, CmrrError, NodeStatus, Outcome, Params, Report};
pub use model::Model;
pub use potential::Yukawa;

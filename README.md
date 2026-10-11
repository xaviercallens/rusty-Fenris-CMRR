# fenris-cmrr

Repulsive relaxation of 2D mesh nodes. Free interior nodes repel each other and the fixed boundary nodes through the
screened-Coulomb (Yukawa) potential `V(r) = exp(-κr)/r`. They are relaxed to a stationary point of the total energy
with the local **rusty-SUNDIALS** CVODE (BDF, analytic Jacobian). A small C API lets Gmsh-style C/C++ callers relax
their node arrays in place.

> **Status: v0.1.** Builds against the local rusty-SUNDIALS. `cargo test` passes 6 tests: force and Hessian against
> finite differences, convergence with fixed nodes unchanged, coincident and near-coincident nodes, the C interface,
> and the order-1 restart. The 12×12 example (121 free nodes) converges to `max|F| = 1.7e-11·|V'(ell)|`.

## What it is and is not

This is a smoother **motivated by** the universal-optimality theory of the triangular lattice. It is **not certified
by** that theory.

- In `t = r²`, `exp(-κ√t)/√t` is completely monotone. That is a mathematical fact, but **there is no Lean proof of it
  yet**. The triangular-UO Lean repository proves admissibility only for Riesz potentials (`riesz_admissible`). A
  `yukawa_admissible` lemma would be the natural first Lean task.
- The planar universal-optimality statements (D1–D4) concern infinite configurations at fixed density. They are
  conditional on upstream `universal_energy_minimum` (LIMITATIONS L1). They say nothing about N nodes in a bounded
  domain with pinned boundary nodes, or about which local minimum a gradient flow reaches. Their own D3 shows that a
  vacancy does not change the energy.
- Expect a well-spaced, locally near-triangular node set, with 5/7 defects wherever the domain forces them. Do not
  expect a perfect lattice or a global optimum.
- The shifted-force cutoff at `rc` (default `20/κ`) breaks exact complete monotonicity by about `exp(-κ rc)`.

## Algorithm

1. **Scale.** Coordinates are centred and divided by the mean spacing `ell = sqrt(bbox area / N)`. The parameters
   become `κ·ell` and `rc/ell`.
2. **Separate coincident nodes.** Free nodes within `1e-9·ell` of another node are moved by `1e-3·ell` along a
   golden-angle direction.
3. **Un-clump with RK23.** Bogacki–Shampine RK2(3) on `ẋ = F(x)`. Each step's coordinate change is capped at
   `0.05·ell`, and the local error tolerance is `1e-3·ell`. It stops once `max|F| ≤ 10·|V'(ell)|`.
4. **Relax with CVODE BDF.** The ODE state holds only the free coordinates; fixed nodes are constant sources, so they
   cannot move and add no singular Jacobian rows. The Jacobian is the exact `-Hessian`. Output times double from
   `τ = 1/V''(ell)` until `max|F_free| ≤ 1e-9·|V'(ell)|` or `t > 1e9·τ`. This is the same loop as
   `qf-yukawa-crystal::relax`.
5. **Tolerances.** Only the end state matters, so CVODE runs with loose tolerances (`rtol` 1e-6, `atol` 1e-9
   in units of the node spacing) and a maximum BDF order of 2; the force test decides convergence.

   | rtol / atol | max order | test suite (debug) |
   |---|---|---|
   | 1e-10 / 1e-12 | 5 | 3 of 5 fail: "too many error test failures at one step" |
   | 1e-10 / 1e-12 | 2 | pass, 19.4 s |
   | 1e-6 / 1e-9 | 5 | pass, 11.6 s |
   | 1e-6 / 1e-9 | 2 | pass, 2.4 s |

   This rusty-SUNDIALS port keeps the BDF order and the Jacobian through repeated error-test failures; LLNL CVODE
   drops to order 1 after three. If CVODE still fails, the solve restarts once from the last accepted state at
   order 1 (`Report::cvode_order1_fallback`).

## Dependency on rusty-SUNDIALS

The crate path-depends on `../rusty-SUNDIALS/crates/{cvode,nvector}`. When it was written, that checkout was on branch
`merge-mcp-temp` at `af4886f`, with no merge in progress.

The `feat/yukawa-floor-global-opt` worktree (`~/rusty-SUNDIALS-wt-yukawa`) differs in the `cvode` crate only in Adams order
selection, so the BDF path is the same. It also adds extra workspace members. To use it instead, point the two paths at `../rusty-SUNDIALS-wt-yukawa/crates/...`.

The `cvode` crate is edition 2024, so it needs **rustc ≥ 1.85**.

## Limits (v0.1)

| Limit | Why | Next step |
|---|---|---|
| ≤ 1500 free nodes (`Params::max_free_nodes`) | CVODE here has only dense LU: `O(n²)` memory and `O(n³)` per factorisation | Add a Krylov or banded linear solver to CVODE, or use a FIRE / L-BFGS minimiser for large meshes |
| `O(N²)` force loop (parallel over nodes with rayon) | Simplicity | Cell list using `rc` |
| No domain constraint; only a bounding-box escape count | Callers pass unordered nodes | Accept an ordered boundary polygon and add a wall potential, with its Hessian |
| No triangulation; VTK export is points only | `fenris`, `spade` and `vtkio` are not in the local cargo cache | Add `spade` for Delaunay; evaluate whether Fenris is worth depending on |
| No FIRE / Laplacian / Lloyd baseline | Scope | Benchmark minimum angle, radius ratio and 5/7 defect count against these |
| `cbindgen` not used | Not cached; a `build.rs` would break offline builds | `include/cmrr.h` is hand-written and must be kept in sync with `src/ffi.rs` |

## Use

```sh
cargo test                     # add --offline if the network is unavailable
cargo run --release --example square_patch -- 12 1.5 /tmp   # writes square_patch_{before,after}.vtk
cargo build --release          # target/release/libcmrr.so + include/cmrr.h
```

Rust:

```rust
let report = cmrr::optimize(&mut positions, &status, &cmrr::Params::new(1.5))?;
```

C/C++ (link `-lcmrr`):

```c
CmrrReport rep;
int32_t code = cmrr_optimize_mesh(x, y, status, n, 1.5, &rep);  /* status[i]: CMRR_NODE_FIXED / CMRR_NODE_FREE */
```

The library borrows `x`, `y` and `status` only for the duration of the call. Coordinates are rewritten only when the
code is `CMRR_CONVERGED` or `CMRR_NOT_CONVERGED`. The rusty-SUNDIALS CVODE may print `ERROR FAIL ...` lines to
stdout when a step fails.

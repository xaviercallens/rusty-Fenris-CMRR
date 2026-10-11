//! The two-phase optimiser: RK23 un-clumping, then CVODE BDF relaxation to a stationary point.
//!
//! The solve runs in nondimensional units: coordinates are centred and divided by the mean node spacing `ell`
//! (`sqrt(bounding-box area / N)`), so `kappa` becomes `kappa * ell` and `rc` becomes `rc / ell`. Tolerances and the
//! un-clumping step cap are therefore fractions of a node spacing, whatever units the caller uses.

use std::fmt;
use std::sync::Arc;

use cvode::{Cvode, DenseMat, Method, Task};
use nvector::SerialVector;

use crate::model::{max_abs, Model};
use crate::potential::Yukawa;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    /// Locked to the CAD geometry; repels free nodes but never moves.
    BoundaryFixed,
    /// Moves under the repulsive flow.
    InternalFree,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Screening constant, in inverse caller length units. `kappa * spacing` of about 1–3 is a sensible range.
    pub kappa: f64,
    /// Cutoff radius, in caller length units.
    pub rc: f64,
    /// Convergence: max free-node force <= `ftol_rel * |V'(ell)|`.
    pub ftol_rel: f64,
    /// Give up after `t > tmax_rel * tau`, where `tau = 1 / V''(ell)` is the nearest-neighbour relaxation time.
    pub tmax_rel: f64,
    /// Dense CVODE guard: refuse problems with more free nodes than this.
    pub max_free_nodes: usize,
    /// Accepted RK23 steps allowed in the un-clumping phase.
    pub unclump_max_steps: usize,
    /// CVODE tolerances in nondimensional units (node spacing = 1). Only the end state matters for a gradient
    /// flow, so these are loose; convergence is judged by `ftol_rel` on the force.
    pub rtol: f64,
    pub atol: f64,
    /// Maximum BDF order. Orders 1–2 behave like damped Newton on this problem; order 5 with tight tolerances made
    /// this CVODE port fail its error test repeatedly.
    pub max_order: usize,
}

impl Params {
    /// Defaults with `rc = 20 / kappa`, so the truncated tail is `exp(-20) ~ 2e-9` relative to `V` at `1 / kappa`.
    pub fn new(kappa: f64) -> Self {
        Params {
            kappa,
            rc: 20.0 / kappa,
            ftol_rel: 1e-9,
            tmax_rel: 1e9,
            max_free_nodes: 1500,
            unclump_max_steps: 10_000,
            rtol: 1e-6,
            atol: 1e-9,
            max_order: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Converged,
    /// `tmax` reached before the force tolerance. Positions are still written back.
    NotConverged,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub outcome: Outcome,
    pub unclump_steps: usize,
    pub cvode_steps: usize,
    /// CVODE failed at `Params::max_order` and the solve was finished by an order-1 restart.
    pub cvode_order1_fallback: bool,
    pub rhs_evals: usize,
    /// Final `max |F|` over free nodes divided by `|V'(ell)|`.
    pub max_free_force_rel: f64,
    /// Final time in units of `tau`.
    pub t_final_rel: f64,
    /// Free nodes that coincided with another node and were displaced deterministically before solving.
    pub n_jittered: usize,
    /// Free nodes that ended outside the bounding box of the fixed nodes (a coarse escape check).
    pub n_outside_fixed_bbox: usize,
    /// The spacing `ell` used for nondimensionalisation, in caller units.
    pub length_scale: f64,
}

#[derive(Debug)]
pub enum CmrrError {
    InvalidInput(String),
    TooLarge { n_free: usize, limit: usize },
    Solver(String),
}

impl fmt::Display for CmrrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmrrError::InvalidInput(m) => write!(f, "invalid input: {m}"),
            CmrrError::TooLarge { n_free, limit } => {
                write!(f, "{n_free} free nodes exceeds the dense-solver limit {limit}")
            }
            CmrrError::Solver(m) => write!(f, "solver failure: {m}"),
        }
    }
}

impl std::error::Error for CmrrError {}

/// Un-clumping step cap, as a fraction of the node spacing.
const UNCLUMP_CAP: f64 = 0.05;
/// RK23 local error tolerance, as a fraction of the node spacing.
const UNCLUMP_TOL: f64 = 1e-3;
/// Hand over to CVODE once `max |F| <= UNCLUMP_SWITCH * |V'(ell)|`.
const UNCLUMP_SWITCH: f64 = 10.0;
/// Nodes closer than this (relative to `ell`) count as coincident.
const COINCIDENT: f64 = 1e-9;
/// Size of the deterministic displacement applied to coincident free nodes (relative to `ell`).
const JITTER: f64 = 1e-3;

/// Relax the free nodes in place. Fixed nodes are never written.
pub fn optimize(positions: &mut [[f64; 2]], status: &[NodeStatus], params: &Params) -> Result<Report, CmrrError> {
    let n = positions.len();
    if n == 0 || status.len() != n {
        return Err(CmrrError::InvalidInput(format!(
            "{n} positions but {} status entries",
            status.len()
        )));
    }
    if !(params.kappa.is_finite() && params.kappa > 0.0 && params.rc.is_finite() && params.rc > 0.0) {
        return Err(CmrrError::InvalidInput("kappa and rc must be positive and finite".into()));
    }
    if positions.iter().any(|p| !(p[0].is_finite() && p[1].is_finite())) {
        return Err(CmrrError::InvalidInput("non-finite coordinate".into()));
    }

    let free: Vec<usize> = (0..n).filter(|&i| status[i] == NodeStatus::InternalFree).collect();
    let fixed: Vec<usize> = (0..n).filter(|&i| status[i] == NodeStatus::BoundaryFixed).collect();
    let (centre, ell) = scale(positions)?;
    let mut report = Report {
        outcome: Outcome::Converged,
        unclump_steps: 0,
        cvode_steps: 0,
        cvode_order1_fallback: false,
        rhs_evals: 0,
        max_free_force_rel: 0.0,
        t_final_rel: 0.0,
        n_jittered: 0,
        n_outside_fixed_bbox: 0,
        length_scale: ell,
    };
    if free.is_empty() {
        return Ok(report);
    }
    if free.len() > params.max_free_nodes {
        return Err(CmrrError::TooLarge { n_free: free.len(), limit: params.max_free_nodes });
    }

    let to_unit = |p: [f64; 2]| [(p[0] - centre[0]) / ell, (p[1] - centre[1]) / ell];
    let mut y: Vec<f64> = free.iter().flat_map(|&i| to_unit(positions[i])).collect();
    let fixed_xy: Vec<f64> = fixed.iter().flat_map(|&i| to_unit(positions[i])).collect();
    report.n_jittered = separate_coincident(&mut y, &fixed_xy);

    let pot = Yukawa { kappa: params.kappa * ell, rc: params.rc / ell };
    let force_scale = pot.dv(1.0).abs();
    let tau = 1.0 / pot.d2v(1.0);
    let model = Arc::new(Model { pot, fixed: fixed_xy });

    report.unclump_steps = unclump(&model, &mut y, UNCLUMP_SWITCH * force_scale, params.unclump_max_steps);
    let relaxed = relax(model, &y, params.ftol_rel * force_scale, params.tmax_rel * tau, tau, params)?;

    for (k, &i) in free.iter().enumerate() {
        positions[i] = [centre[0] + ell * relaxed.y[2 * k], centre[1] + ell * relaxed.y[2 * k + 1]];
    }
    report.outcome = if relaxed.converged { Outcome::Converged } else { Outcome::NotConverged };
    report.cvode_steps = relaxed.steps;
    report.cvode_order1_fallback = relaxed.fallback;
    report.rhs_evals = relaxed.rhs_evals;
    report.max_free_force_rel = relaxed.max_force / force_scale;
    report.t_final_rel = relaxed.t / tau;
    report.n_outside_fixed_bbox = outside_fixed_bbox(positions, &free, &fixed);
    Ok(report)
}

/// Centre of the bounding box and mean spacing `sqrt(area / N)` (falls back to `extent / N` for collinear input).
fn scale(positions: &[[f64; 2]]) -> Result<([f64; 2], f64), CmrrError> {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in positions {
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
    let n = positions.len() as f64;
    let ell = if w > 0.0 && h > 0.0 { (w * h / n).sqrt() } else { w.max(h) / n };
    if !(ell > 0.0 && ell.is_finite()) {
        return Err(CmrrError::InvalidInput("all nodes coincide".into()));
    }
    Ok(([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])], ell))
}

/// Displace each free node that sits on top of an earlier free node or any fixed node, along a golden-angle direction.
fn separate_coincident(y: &mut [f64], fixed: &[f64]) -> usize {
    let n = y.len() / 2;
    let mut moved = 0;
    for i in 0..n {
        let hit = |y: &[f64]| {
            let (xi, yi) = (y[2 * i], y[2 * i + 1]);
            let near = |x: f64, z: f64| ((xi - x).powi(2) + (yi - z).powi(2)).sqrt() < COINCIDENT;
            (0..i).any(|j| near(y[2 * j], y[2 * j + 1]))
                || (0..fixed.len() / 2).any(|k| near(fixed[2 * k], fixed[2 * k + 1]))
        };
        if hit(&*y) {
            let theta = 2.399_963_229_728_653 * (i as f64 + 1.0);
            y[2 * i] += JITTER * theta.cos();
            y[2 * i + 1] += JITTER * theta.sin();
            moved += 1;
        }
    }
    moved
}

/// Bogacki–Shampine RK2(3) on `y' = F(y)`, with each step's coordinate change capped at `UNCLUMP_CAP`, until
/// `max |F| <= fswitch`. Returns the number of accepted steps.
fn unclump(model: &Model, y: &mut Vec<f64>, fswitch: f64, max_steps: usize) -> usize {
    let m = y.len();
    let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m]);
    let (mut tmp, mut ynew) = (vec![0.0; m], vec![0.0; m]);
    model.force(y, &mut k1);
    let mut h = UNCLUMP_CAP / max_abs(&k1).max(f64::MIN_POSITIVE);
    let (mut accepted, mut attempts) = (0, 0);
    while accepted < max_steps && attempts < 10 * max_steps && max_abs(&k1) > fswitch {
        attempts += 1;
        for i in 0..m {
            tmp[i] = y[i] + 0.5 * h * k1[i];
        }
        model.force(&tmp, &mut k2);
        for i in 0..m {
            tmp[i] = y[i] + 0.75 * h * k2[i];
        }
        model.force(&tmp, &mut k3);
        for i in 0..m {
            ynew[i] = y[i] + h * (2.0 / 9.0 * k1[i] + 1.0 / 3.0 * k2[i] + 4.0 / 9.0 * k3[i]);
        }
        model.force(&ynew, &mut k4);
        let mut err = 0.0_f64;
        let mut disp = 0.0_f64;
        for i in 0..m {
            let e = h * (-5.0 / 72.0 * k1[i] + 1.0 / 12.0 * k2[i] + 1.0 / 9.0 * k3[i] - 1.0 / 8.0 * k4[i]);
            err = err.max(e.abs());
            disp = disp.max((ynew[i] - y[i]).abs());
        }
        let finite = err.is_finite() && ynew.iter().all(|v| v.is_finite());
        if finite && err <= UNCLUMP_TOL && disp <= UNCLUMP_CAP {
            std::mem::swap(y, &mut ynew);
            std::mem::swap(&mut k1, &mut k4);
            accepted += 1;
            let grow = (0.9 * (UNCLUMP_TOL / err.max(f64::MIN_POSITIVE)).cbrt()).min(2.0);
            h = (h * grow).min(UNCLUMP_CAP / max_abs(&k1).max(f64::MIN_POSITIVE));
        } else {
            h *= 0.5;
        }
    }
    accepted
}

struct Relaxed {
    y: Vec<f64>,
    t: f64,
    max_force: f64,
    steps: usize,
    rhs_evals: usize,
    converged: bool,
    fallback: bool,
}

/// CVODE BDF with the analytic Jacobian `-Hessian`, output times doubling from `tau`, until the force tolerance or
/// `tmax` (the loop of `qf-yukawa-crystal::relax`). If CVODE fails (this port does not lower the BDF order after
/// repeated error-test failures), restart once from its last accepted state at order 1.
fn relax(model: Arc<Model>, y0: &[f64], ftol: f64, tmax: f64, tau: f64, p: &Params) -> Result<Relaxed, CmrrError> {
    let mut f = vec![0.0; y0.len()];
    model.force(y0, &mut f);
    let fmax0 = max_abs(&f);
    if fmax0 <= ftol {
        return Ok(Relaxed {
            y: y0.to_vec(),
            t: 0.0,
            max_force: fmax0,
            steps: 0,
            rhs_evals: 0,
            converged: true,
            fallback: false,
        });
    }
    match run_cvode(&model, y0, 0.0, ftol, tmax, tau, p, p.max_order) {
        Ok(r) => Ok(r),
        Err(fail) if p.max_order > 1 => {
            let mut r = run_cvode(&model, &fail.y, fail.t, ftol, tmax, tau, p, 1)
                .map_err(|e| CmrrError::Solver(format!("{}; order-1 restart: {}", fail.msg, e.msg)))?;
            r.steps += fail.steps;
            r.rhs_evals += fail.rhs_evals;
            r.fallback = true;
            Ok(r)
        }
        Err(fail) => Err(CmrrError::Solver(fail.msg)),
    }
}

/// Last accepted CVODE state when a solve fails.
struct Failed {
    msg: String,
    y: Vec<f64>,
    t: f64,
    steps: usize,
    rhs_evals: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_cvode(
    model: &Arc<Model>,
    y0: &[f64],
    t0: f64,
    ftol: f64,
    tmax: f64,
    tau: f64,
    p: &Params,
    max_order: usize,
) -> Result<Relaxed, Failed> {
    let failed = |msg: String| Failed { msg, y: y0.to_vec(), t: t0, steps: 0, rhs_evals: 0 };
    let m1 = Arc::clone(model);
    let m2 = Arc::clone(model);
    let rhs = move |_t: f64, y: &[f64], ydot: &mut [f64]| -> Result<(), String> {
        m1.force(y, ydot);
        Ok(())
    };
    let jac = move |_t: f64, y: &[f64], j: &mut DenseMat| -> Result<(), String> {
        let h = m2.hessian(y);
        let m = y.len();
        for c in 0..m {
            for r in 0..m {
                j.cols[c][r] = -h[r * m + c];
            }
        }
        Ok(())
    };
    let mut solver = Cvode::builder(Method::Bdf)
        .rtol(p.rtol)
        .atol(p.atol)
        .max_order(max_order)
        .max_steps(100_000)
        .jacobian(jac)
        .build(rhs, t0, SerialVector::from_slice(y0))
        .map_err(|e| failed(format!("CVODE setup failed: {e}")))?;
    let mut f = vec![0.0; y0.len()];
    let mut tout = t0 + tau;
    loop {
        let (t, y) = match solver.solve(tout, Task::Normal) {
            Ok((t, y)) => (t, y.to_vec()),
            Err(e) => {
                return Err(Failed {
                    msg: format!("CVODE failed before t = {tout}: {e}"),
                    y: solver.y().to_vec(),
                    t: solver.t(),
                    steps: solver.num_steps(),
                    rhs_evals: solver.num_rhs_evals(),
                })
            }
        };
        model.force(&y, &mut f);
        let fmax = max_abs(&f);
        if fmax <= ftol || t >= tmax {
            return Ok(Relaxed {
                y,
                t,
                max_force: fmax,
                steps: solver.num_steps(),
                rhs_evals: solver.num_rhs_evals(),
                converged: fmax <= ftol,
                fallback: false,
            });
        }
        tout = t0 + 2.0 * (tout - t0);
    }
}

fn outside_fixed_bbox(positions: &[[f64; 2]], free: &[usize], fixed: &[usize]) -> usize {
    if fixed.is_empty() {
        return 0;
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for &i in fixed {
        for a in 0..2 {
            lo[a] = lo[a].min(positions[i][a]);
            hi[a] = hi[a].max(positions[i][a]);
        }
    }
    free.iter()
        .filter(|&&i| (0..2).any(|a| positions[i][a] < lo[a] || positions[i][a] > hi[a]))
        .count()
}

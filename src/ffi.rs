//! C ABI (see `include/cmrr.h`). Rust only borrows the caller's arrays for the duration of the call; it never takes
//! ownership.

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::engine::{optimize, CmrrError, NodeStatus, Outcome, Params};

pub const CMRR_CONVERGED: i32 = 0;
pub const CMRR_NOT_CONVERGED: i32 = 1;
pub const CMRR_SOLVER_FAILED: i32 = 2;
pub const CMRR_TOO_LARGE: i32 = 3;
pub const CMRR_INVALID_INPUT: i32 = -1;
pub const CMRR_PANIC: i32 = -2;

/// Status values accepted in `node_status`.
pub const CMRR_NODE_FIXED: i32 = 0;
pub const CMRR_NODE_FREE: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CmrrReport {
    pub code: i32,
    pub unclump_steps: u64,
    pub cvode_steps: u64,
    pub rhs_evals: u64,
    pub max_free_force_rel: f64,
    pub t_final_rel: f64,
    pub n_jittered: u64,
    pub n_outside_fixed_bbox: u64,
    pub length_scale: f64,
    pub cvode_order1_fallback: u8,
}

/// Relax the free nodes of `x`/`y` in place.
///
/// Returns the same code it stores in `report->code` (`report` may be null). Coordinates are modified only for
/// `CMRR_CONVERGED` and `CMRR_NOT_CONVERGED`.
///
/// # Safety
/// `x` and `y` must each point to `num_nodes` writable doubles, must not overlap, and `node_status` must point to
/// `num_nodes` readable `int32_t`. No other thread may access them during the call.
#[no_mangle]
pub unsafe extern "C" fn cmrr_optimize_mesh(
    x: *mut f64,
    y: *mut f64,
    node_status: *const i32,
    num_nodes: usize,
    kappa: f64,
    report: *mut CmrrReport,
) -> i32 {
    let rep = catch_unwind(AssertUnwindSafe(|| unsafe { run(x, y, node_status, num_nodes, kappa) }))
        .unwrap_or(CmrrReport { code: CMRR_PANIC, ..Default::default() });
    if !report.is_null() {
        unsafe { *report = rep };
    }
    rep.code
}

unsafe fn run(x: *mut f64, y: *mut f64, node_status: *const i32, n: usize, kappa: f64) -> CmrrReport {
    let invalid = CmrrReport { code: CMRR_INVALID_INPUT, ..Default::default() };
    if x.is_null() || y.is_null() || node_status.is_null() || n == 0 {
        return invalid;
    }
    let (xa, ya) = (x as usize, y as usize);
    let bytes = n * std::mem::size_of::<f64>();
    if xa < ya + bytes && ya < xa + bytes {
        return invalid;
    }
    let xs = unsafe { std::slice::from_raw_parts_mut(x, n) };
    let ys = unsafe { std::slice::from_raw_parts_mut(y, n) };
    let st = unsafe { std::slice::from_raw_parts(node_status, n) };

    let mut status = Vec::with_capacity(n);
    for &s in st {
        status.push(match s {
            CMRR_NODE_FIXED => NodeStatus::BoundaryFixed,
            CMRR_NODE_FREE => NodeStatus::InternalFree,
            _ => return invalid,
        });
    }
    let mut pos: Vec<[f64; 2]> = xs.iter().zip(ys.iter()).map(|(&a, &b)| [a, b]).collect();
    match optimize(&mut pos, &status, &Params::new(kappa)) {
        Ok(r) => {
            for (i, p) in pos.iter().enumerate() {
                xs[i] = p[0];
                ys[i] = p[1];
            }
            CmrrReport {
                code: match r.outcome {
                    Outcome::Converged => CMRR_CONVERGED,
                    Outcome::NotConverged => CMRR_NOT_CONVERGED,
                },
                unclump_steps: r.unclump_steps as u64,
                cvode_steps: r.cvode_steps as u64,
                rhs_evals: r.rhs_evals as u64,
                max_free_force_rel: r.max_free_force_rel,
                t_final_rel: r.t_final_rel,
                n_jittered: r.n_jittered as u64,
                n_outside_fixed_bbox: r.n_outside_fixed_bbox as u64,
                length_scale: r.length_scale,
                cvode_order1_fallback: r.cvode_order1_fallback as u8,
            }
        }
        Err(CmrrError::InvalidInput(_)) => invalid,
        Err(CmrrError::TooLarge { .. }) => CmrrReport { code: CMRR_TOO_LARGE, ..Default::default() },
        Err(CmrrError::Solver(_)) => CmrrReport { code: CMRR_SOLVER_FAILED, ..Default::default() },
    }
}

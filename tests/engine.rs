use cmrr::ffi::{cmrr_optimize_mesh, CmrrReport, CMRR_CONVERGED, CMRR_INVALID_INPUT};
use cmrr::{optimize, Model, NodeStatus, Outcome, Params, Yukawa};

/// Deterministic uniform numbers in [-1, 1).
fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Square [0, l]^2: fixed nodes every unit along the boundary, a jittered unit grid of free nodes inside.
fn square_patch(l: usize, jitter: f64, seed: u64) -> (Vec<[f64; 2]>, Vec<NodeStatus>) {
    let mut s = seed;
    let (mut pos, mut st) = (Vec::new(), Vec::new());
    for i in 0..=l {
        for j in 0..=l {
            let on_boundary = i == 0 || j == 0 || i == l || j == l;
            if on_boundary {
                pos.push([i as f64, j as f64]);
                st.push(NodeStatus::BoundaryFixed);
            } else {
                pos.push([i as f64 + jitter * lcg(&mut s), j as f64 + jitter * lcg(&mut s)]);
                st.push(NodeStatus::InternalFree);
            }
        }
    }
    (pos, st)
}

fn small_model() -> (Model, Vec<f64>) {
    let mut s = 7;
    let fixed: Vec<f64> = (0..6).flat_map(|k| [k as f64, -1.0]).collect();
    let y: Vec<f64> = (0..8)
        .flat_map(|k| [(k % 4) as f64 + 0.3 * lcg(&mut s), (k / 4) as f64 + 0.3 * lcg(&mut s)])
        .collect();
    (Model { pot: Yukawa { kappa: 1.3, rc: 4.0 }, fixed }, y)
}

#[test]
fn force_matches_energy_gradient() {
    let (model, y) = small_model();
    let mut f = vec![0.0; y.len()];
    model.force(&y, &mut f);
    let h = 1e-6;
    for c in 0..y.len() {
        let (mut yp, mut ym) = (y.clone(), y.clone());
        yp[c] += h;
        ym[c] -= h;
        let fd = -(model.energy(&yp) - model.energy(&ym)) / (2.0 * h);
        assert!((fd - f[c]).abs() <= 1e-6 * (1.0 + f[c].abs()), "coord {c}: fd {fd} vs {}", f[c]);
    }
}

#[test]
fn hessian_matches_force_jacobian() {
    let (model, y) = small_model();
    let m = y.len();
    let hess = model.hessian(&y);
    let h = 1e-6;
    let (mut fp, mut fm) = (vec![0.0; m], vec![0.0; m]);
    for c in 0..m {
        let (mut yp, mut ym) = (y.clone(), y.clone());
        yp[c] += h;
        ym[c] -= h;
        model.force(&yp, &mut fp);
        model.force(&ym, &mut fm);
        for r in 0..m {
            let fd = -(fp[r] - fm[r]) / (2.0 * h);
            let an = hess[r * m + c];
            assert!((fd - an).abs() <= 1e-5 * (1.0 + an.abs()), "H[{r},{c}]: fd {fd} vs {an}");
        }
    }
}

#[test]
fn patch_converges_and_fixed_nodes_do_not_move() {
    let (mut pos, st) = square_patch(5, 0.25, 1);
    let before = pos.clone();
    let rep = optimize(&mut pos, &st, &Params::new(1.5)).expect("optimize");
    assert_eq!(rep.outcome, Outcome::Converged, "{rep:?}");
    assert!(rep.max_free_force_rel <= 1e-9, "{rep:?}");
    assert_eq!(rep.n_outside_fixed_bbox, 0, "{rep:?}");
    for i in 0..pos.len() {
        if st[i] == NodeStatus::BoundaryFixed {
            assert_eq!(pos[i], before[i]);
        }
    }
}

#[test]
fn near_coincident_and_coincident_nodes_are_separated() {
    let (mut pos, mut st) = square_patch(5, 0.0, 2);
    pos.push([2.0 + 1e-7, 2.0]); // almost on top of the free node at (2, 2)
    st.push(NodeStatus::InternalFree);
    pos.push([3.0, 3.0]); // exactly on top of the free node at (3, 3)
    st.push(NodeStatus::InternalFree);
    let rep = optimize(&mut pos, &st, &Params::new(1.5)).expect("optimize");
    assert!(rep.n_jittered >= 1, "{rep:?}");
    assert!(rep.unclump_steps > 0, "{rep:?}");
    assert_eq!(rep.outcome, Outcome::Converged, "{rep:?}");
    let mut dmin = f64::INFINITY;
    for i in 0..pos.len() {
        for j in (i + 1)..pos.len() {
            dmin = dmin.min(((pos[i][0] - pos[j][0]).powi(2) + (pos[i][1] - pos[j][1]).powi(2)).sqrt());
        }
    }
    assert!(dmin > 0.3, "min distance {dmin}");
}

#[test]
fn ffi_round_trip_and_rejects_bad_status() {
    let (pos, st) = square_patch(4, 0.2, 3);
    let mut x: Vec<f64> = pos.iter().map(|p| p[0]).collect();
    let mut y: Vec<f64> = pos.iter().map(|p| p[1]).collect();
    let mut s: Vec<i32> = st.iter().map(|&v| if v == NodeStatus::BoundaryFixed { 0 } else { 1 }).collect();
    let mut rep = CmrrReport::default();
    let code = unsafe { cmrr_optimize_mesh(x.as_mut_ptr(), y.as_mut_ptr(), s.as_ptr(), x.len(), 1.5, &mut rep) };
    assert_eq!(code, CMRR_CONVERGED, "{rep:?}");
    assert_eq!(rep.code, code);
    for i in 0..pos.len() {
        if st[i] == NodeStatus::BoundaryFixed {
            assert_eq!([x[i], y[i]], pos[i]);
        }
    }

    s[0] = 7;
    let code = unsafe {
        cmrr_optimize_mesh(x.as_mut_ptr(), y.as_mut_ptr(), s.as_ptr(), x.len(), 1.5, std::ptr::null_mut())
    };
    assert_eq!(code, CMRR_INVALID_INPUT);
    let code = unsafe { cmrr_optimize_mesh(x.as_mut_ptr(), x.as_mut_ptr(), s.as_ptr(), x.len(), 1.5, &mut rep) };
    assert_eq!(code, CMRR_INVALID_INPUT);
}

#[test]
fn order1_restart_rescues_tight_high_order_bdf() {
    // rtol 1e-10 / atol 1e-12 at BDF order 5 fails this CVODE port's error test; the order-1 restart must finish.
    let (mut pos, st) = square_patch(5, 0.25, 1);
    let params = Params { rtol: 1e-10, atol: 1e-12, max_order: 5, ..Params::new(1.5) };
    let rep = optimize(&mut pos, &st, &params).expect("optimize");
    assert_eq!(rep.outcome, Outcome::Converged, "{rep:?}");
    assert!(rep.cvode_order1_fallback, "{rep:?}");
}

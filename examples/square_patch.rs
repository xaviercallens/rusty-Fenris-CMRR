//! Relax a jittered grid inside a fixed square boundary and write before/after VTK point clouds.
//!
//! `cargo run --release --example square_patch -- [L] [kappa] [out_dir]`

use std::path::PathBuf;

use cmrr::{optimize, vtk, NodeStatus, Params};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let l: usize = args.get(1).map_or(Ok(12), |s| s.parse())?;
    let kappa: f64 = args.get(2).map_or(Ok(1.5), |s| s.parse())?;
    let out = PathBuf::from(args.get(3).map_or(".", |s| s.as_str()));

    let mut seed = 42u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (mut pos, mut st) = (Vec::new(), Vec::new());
    for i in 0..=l {
        for j in 0..=l {
            if i == 0 || j == 0 || i == l || j == l {
                pos.push([i as f64, j as f64]);
                st.push(NodeStatus::BoundaryFixed);
            } else {
                pos.push([i as f64 + 0.35 * rand(), j as f64 + 0.35 * rand()]);
                st.push(NodeStatus::InternalFree);
            }
        }
    }

    vtk::write_points(&out.join("square_patch_before.vtk"), &pos, &st)?;
    let report = optimize(&mut pos, &st, &Params::new(kappa))?;
    vtk::write_points(&out.join("square_patch_after.vtk"), &pos, &st)?;
    println!("{report:#?}");
    Ok(())
}

//! Legacy ASCII VTK point-cloud export (opens in ParaView). Triangulation is not done here.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::engine::NodeStatus;

pub fn write_points(path: &Path, positions: &[[f64; 2]], status: &[NodeStatus]) -> std::io::Result<()> {
    let n = positions.len();
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# vtk DataFile Version 3.0")?;
    writeln!(w, "fenris-cmrr nodes")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET POLYDATA")?;
    writeln!(w, "POINTS {n} double")?;
    for p in positions {
        writeln!(w, "{:.17e} {:.17e} 0", p[0], p[1])?;
    }
    writeln!(w, "VERTICES {n} {}", 2 * n)?;
    for i in 0..n {
        writeln!(w, "1 {i}")?;
    }
    writeln!(w, "POINT_DATA {n}")?;
    writeln!(w, "SCALARS fixed int 1")?;
    writeln!(w, "LOOKUP_TABLE default")?;
    for s in status {
        writeln!(w, "{}", if *s == NodeStatus::BoundaryFixed { 1 } else { 0 })?;
    }
    w.flush()
}

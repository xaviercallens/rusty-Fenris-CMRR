//! Energy, force and Hessian restricted to the free nodes.
//!
//! The ODE state `y` holds only the free coordinates, interleaved `[x0, y0, x1, y1, ...]`. Fixed nodes enter as
//! constant sources, so they cannot move, and the Jacobian has no singular rows for them.

use rayon::prelude::*;

use crate::potential::Yukawa;

#[derive(Clone, Debug)]
pub struct Model {
    pub pot: Yukawa,
    /// Fixed node coordinates, interleaved.
    pub fixed: Vec<f64>,
}

impl Model {
    pub fn n_fixed(&self) -> usize {
        self.fixed.len() / 2
    }

    /// Energy up to the constant fixed-fixed part.
    pub fn energy(&self, y: &[f64]) -> f64 {
        let n = y.len() / 2;
        let mut e = 0.0;
        for i in 0..n {
            let (xi, yi) = (y[2 * i], y[2 * i + 1]);
            for j in (i + 1)..n {
                let (dx, dy) = (xi - y[2 * j], yi - y[2 * j + 1]);
                e += self.pot.pair((dx * dx + dy * dy).sqrt()).0;
            }
            for k in 0..self.n_fixed() {
                let (dx, dy) = (xi - self.fixed[2 * k], yi - self.fixed[2 * k + 1]);
                e += self.pot.pair((dx * dx + dy * dy).sqrt()).0;
            }
        }
        e
    }

    /// Force `F = -grad E` on the free nodes, written into `f` (parallel over nodes).
    pub fn force(&self, y: &[f64], f: &mut [f64]) {
        let n = y.len() / 2;
        f.par_chunks_mut(2).enumerate().for_each(|(i, fi)| {
            let (xi, yi) = (y[2 * i], y[2 * i + 1]);
            let (mut fx, mut fy) = (0.0, 0.0);
            for j in 0..n {
                if j != i {
                    let (gx, gy) = self.pot.pair_force(xi - y[2 * j], yi - y[2 * j + 1]);
                    fx += gx;
                    fy += gy;
                }
            }
            for k in 0..self.n_fixed() {
                let (gx, gy) = self.pot.pair_force(xi - self.fixed[2 * k], yi - self.fixed[2 * k + 1]);
                fx += gx;
                fy += gy;
            }
            fi[0] = fx;
            fi[1] = fy;
        });
    }

    /// Hessian of `E` with respect to the free coordinates (dense, row-major, `2n x 2n`).
    pub fn hessian(&self, y: &[f64]) -> Vec<f64> {
        let n = y.len() / 2;
        let m = 2 * n;
        let mut h = vec![0.0; m * m];
        // Each chunk holds the two rows of free node i.
        h.par_chunks_mut(2 * m).enumerate().for_each(|(i, rows)| {
            let (xi, yi) = (y[2 * i], y[2 * i + 1]);
            for j in 0..n {
                if j == i {
                    continue;
                }
                if let Some(k) = self.pot.block(xi - y[2 * j], yi - y[2 * j + 1]) {
                    for a in 0..2 {
                        for b in 0..2 {
                            rows[a * m + 2 * i + b] += k[a][b];
                            rows[a * m + 2 * j + b] -= k[a][b];
                        }
                    }
                }
            }
            for q in 0..self.n_fixed() {
                if let Some(k) = self.pot.block(xi - self.fixed[2 * q], yi - self.fixed[2 * q + 1]) {
                    for a in 0..2 {
                        for b in 0..2 {
                            rows[a * m + 2 * i + b] += k[a][b];
                        }
                    }
                }
            }
        });
        h
    }
}

pub fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
}

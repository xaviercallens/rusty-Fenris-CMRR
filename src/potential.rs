//! Shifted-force Yukawa pair potential (same form as rusty-SUNDIALS `qf-yukawa-crystal`).

/// `V(r) = exp(-kappa r) / r`, truncated at `rc` with a shifted-force correction, so that energy and force are
/// continuous at `rc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yukawa {
    pub kappa: f64,
    pub rc: f64,
}

impl Yukawa {
    /// Untruncated `V(r)`.
    pub fn v(&self, r: f64) -> f64 {
        (-self.kappa * r).exp() / r
    }

    /// Untruncated `V'(r)` (negative: repulsive).
    pub fn dv(&self, r: f64) -> f64 {
        -(-self.kappa * r).exp() * (self.kappa * r + 1.0) / (r * r)
    }

    /// Untruncated `V''(r)`.
    pub fn d2v(&self, r: f64) -> f64 {
        let k = self.kappa;
        (-k * r).exp() * (k * k * r * r + 2.0 * k * r + 2.0) / (r * r * r)
    }

    /// Shifted-force pair potential and its first two derivatives (zero for `r >= rc`).
    pub fn pair(&self, r: f64) -> (f64, f64, f64) {
        if r >= self.rc {
            return (0.0, 0.0, 0.0);
        }
        let (vc, dvc) = (self.v(self.rc), self.dv(self.rc));
        (
            self.v(r) - vc - (r - self.rc) * dvc,
            self.dv(r) - dvc,
            self.d2v(r),
        )
    }

    /// Force on the first point of a pair separated by `d = x_first - x_second`.
    pub fn pair_force(&self, dx: f64, dy: f64) -> (f64, f64) {
        let r = (dx * dx + dy * dy).sqrt();
        if r == 0.0 || r >= self.rc {
            return (0.0, 0.0);
        }
        let dv = self.pair(r).1;
        (-dv * dx / r, -dv * dy / r)
    }

    /// Hessian block `d^2 V(|d|) / dx_first^2` (2 x 2), or `None` outside the cutoff.
    pub fn block(&self, dx: f64, dy: f64) -> Option<[[f64; 2]; 2]> {
        let r = (dx * dx + dy * dy).sqrt();
        if r == 0.0 || r >= self.rc {
            return None;
        }
        let (_, dv, d2v) = self.pair(r);
        let u = [dx / r, dy / r];
        let mut k = [[0.0; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                let id = if a == b { 1.0 } else { 0.0 };
                k[a][b] = d2v * u[a] * u[b] + dv / r * (id - u[a] * u[b]);
            }
        }
        Some(k)
    }
}

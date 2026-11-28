//! Zero-heap-allocation real-time Control Barrier Function (CBF) safety filter,
//! 2nd-order HOCBF halfspace builder, Simplex braking fallback, and watchdog TTL scaler.
//!
//! Pure `core` math: safe on native targets, embedded `no_std`, and `wasm32`.

pub const MAX_DIM: usize = 16;
pub const MAX_BARRIERS: usize = 32;

pub const CBF_STATUS_SAFE: i32 = 0;
pub const CBF_STATUS_UNSAFE_NO_SOLUTION: i32 = 1;
pub const CBF_STATUS_SIMPLEX_FALLBACK: i32 = 2;
pub const CBF_STATUS_INVALID_INPUT: i32 = -1;
pub const CBF_STATUS_DIM_MISMATCH: i32 = -2;

/// Affine control barrier halfspace `a . u >= b` over the first `dim` entries of `a`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbfHalfspace {
    pub a: [f64; MAX_DIM],
    pub b: f64,
}

impl CbfHalfspace {
    pub fn new(a_slice: &[f64], b: f64) -> Self {
        let mut a = [0.0_f64; MAX_DIM];
        let n = a_slice.len().min(MAX_DIM);
        let mut i = 0;
        while i < n {
            a[i] = a_slice[i];
            i += 1;
        }
        Self { a, b }
    }
}

/// Result status of the real-time CBF-QP projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CbfStatus {
    Safe,
    UnsafeNoSolution,
    SimplexFallback,
    InfeasibleBox,
    InvalidInput,
    DimMismatch,
}

impl CbfStatus {
    #[inline]
    pub const fn to_c_code(self) -> i32 {
        match self {
            CbfStatus::Safe => CBF_STATUS_SAFE,
            CbfStatus::UnsafeNoSolution => CBF_STATUS_UNSAFE_NO_SOLUTION,
            CbfStatus::SimplexFallback => CBF_STATUS_SIMPLEX_FALLBACK,
            CbfStatus::InfeasibleBox | CbfStatus::InvalidInput => CBF_STATUS_INVALID_INPUT,
            CbfStatus::DimMismatch => CBF_STATUS_DIM_MISMATCH,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            CbfStatus::Safe => "safe",
            CbfStatus::UnsafeNoSolution => "unsafe_no_solution",
            CbfStatus::SimplexFallback => "simplex_braking_engaged",
            CbfStatus::InfeasibleBox => "infeasible_box",
            CbfStatus::InvalidInput => "invalid_input",
            CbfStatus::DimMismatch => "dim_mismatch",
        }
    }
}

/// Fixed-capacity stack result returned by [`dykstra_cbf_project`] and [`rt_cbf_filter_or_brake`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CbfFilterResult {
    pub status: CbfStatus,
    pub verified_safe: bool,
    pub u_filtered: [f64; MAX_DIM],
    pub dim: usize,
    pub min_barrier_slack: f64,
    pub iterations: usize,
}

impl CbfFilterResult {
    fn error(status: CbfStatus, dim: usize, u_nom: &[f64]) -> Self {
        let mut u_filtered = [0.0_f64; MAX_DIM];
        let n = dim.min(MAX_DIM).min(u_nom.len());
        let mut i = 0;
        while i < n {
            u_filtered[i] = u_nom[i];
            i += 1;
        }
        Self {
            status,
            verified_safe: false,
            u_filtered,
            dim: n,
            min_barrier_slack: -1.0,
            iterations: 0,
        }
    }
}

/// Build a 2nd-order High-Order CBF (HOCBF) halfspace `a . u >= b`:
///
/// ```text
/// L_g L_f h . u >= -(L_f^2 h + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * (h - conformal_margin))
/// ```
pub fn compute_hocbf_halfspace(
    h: f64,
    h_dot: f64,
    lf2_h: f64,
    lg_lf_h: &[f64],
    alpha1: f64,
    alpha2: f64,
    conformal_margin: f64,
) -> CbfHalfspace {
    let h_eff = h - conformal_margin;
    let b = -(lf2_h + (alpha1 + alpha2) * h_dot + alpha1 * alpha2 * h_eff);
    CbfHalfspace::new(lg_lf_h, b)
}

#[inline]
fn dot_n(a: &[f64], b: &[f64], dim: usize) -> f64 {
    let mut acc = 0.0_f64;
    let mut i = 0;
    while i < dim {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// Zero-heap-allocation Dykstra cyclic projection onto:
/// `{ u : a_j . u >= b_j for all j, u_min <= u <= u_max }`.
pub fn dykstra_cbf_project(
    u_nominal: &[f64],
    halfspaces: &[CbfHalfspace],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    max_iters: usize,
    tol: f64,
) -> CbfFilterResult {
    let dim = u_nominal.len();
    if dim == 0 || dim > MAX_DIM || halfspaces.len() > MAX_BARRIERS {
        return CbfFilterResult::error(CbfStatus::DimMismatch, dim, u_nominal);
    }
    if let Some(lo) = u_min {
        if lo.len() != dim {
            return CbfFilterResult::error(CbfStatus::DimMismatch, dim, u_nominal);
        }
    }
    if let Some(hi) = u_max {
        if hi.len() != dim {
            return CbfFilterResult::error(CbfStatus::DimMismatch, dim, u_nominal);
        }
    }

    // Validate finiteness and box bounds
    let mut d = 0;
    while d < dim {
        if !u_nominal[d].is_finite() {
            return CbfFilterResult::error(CbfStatus::InvalidInput, dim, u_nominal);
        }
        if let Some(lo) = u_min {
            if !lo[d].is_finite() {
                return CbfFilterResult::error(CbfStatus::InvalidInput, dim, u_nominal);
            }
        }
        if let Some(hi) = u_max {
            if !hi[d].is_finite() {
                return CbfFilterResult::error(CbfStatus::InvalidInput, dim, u_nominal);
            }
        }
        if let (Some(lo), Some(hi)) = (u_min, u_max) {
            if lo[d] > hi[d] {
                return CbfFilterResult::error(CbfStatus::InfeasibleBox, dim, u_nominal);
            }
        }
        d += 1;
    }

    let m = halfspaces.len();
    let mut j = 0;
    while j < m {
        if !halfspaces[j].b.is_finite() {
            return CbfFilterResult::error(CbfStatus::InvalidInput, dim, u_nominal);
        }
        let mut i = 0;
        while i < dim {
            if !halfspaces[j].a[i].is_finite() {
                return CbfFilterResult::error(CbfStatus::InvalidInput, dim, u_nominal);
            }
            i += 1;
        }
        j += 1;
    }

    let has_box = u_min.is_some() || u_max.is_some();
    let mut s_half = [[0.0_f64; MAX_DIM]; MAX_BARRIERS];
    let mut s_box = [0.0_f64; MAX_DIM];
    let mut w = [0.0_f64; MAX_DIM];
    d = 0;
    while d < dim {
        w[d] = u_nominal[d];
        d += 1;
    }

    let mut iters_used = 0usize;
    if m > 0 || has_box {
        let mut iter = 0usize;
        while iter < max_iters {
            iters_used = iter + 1;
            let mut converged = true;

            // Project onto each CBF halfspace
            let mut h_idx = 0usize;
            while h_idx < m {
                let hs = &halfspaces[h_idx];
                let mut y = [0.0_f64; MAX_DIM];
                d = 0;
                while d < dim {
                    y[d] = w[d] + s_half[h_idx][d];
                    d += 1;
                }

                let a_norm2 = dot_n(&hs.a, &hs.a, dim);
                let viol = hs.b - dot_n(&hs.a, &y, dim);
                let mut proj = y;
                if viol > 0.0 && a_norm2 > 1e-18 {
                    let step = viol / a_norm2;
                    d = 0;
                    while d < dim {
                        proj[d] = y[d] + step * hs.a[d];
                        d += 1;
                    }
                    converged = false;
                }

                d = 0;
                while d < dim {
                    let s_new = y[d] - proj[d];
                    if (s_new - s_half[h_idx][d]).abs() > tol {
                        converged = false;
                    }
                    s_half[h_idx][d] = s_new;
                    w[d] = proj[d];
                    d += 1;
                }
                h_idx += 1;
            }

            // Project onto control box [u_min, u_max]
            if has_box {
                d = 0;
                while d < dim {
                    let y = w[d] + s_box[d];
                    let mut proj = y;
                    if let Some(lo) = u_min {
                        if proj < lo[d] {
                            proj = lo[d];
                        }
                    }
                    if let Some(hi) = u_max {
                        if proj > hi[d] {
                            proj = hi[d];
                        }
                    }
                    if (proj - y).abs() > tol {
                        converged = false;
                    }
                    let s_new = y - proj;
                    if (s_new - s_box[d]).abs() > tol {
                        converged = false;
                    }
                    s_box[d] = s_new;
                    w[d] = proj;
                    d += 1;
                }
            }

            if converged {
                break;
            }
            iter += 1;
        }
    }

    // Verify feasibility of final point
    let mut min_slack = if m == 0 { 0.0_f64 } else { f64::INFINITY };
    let mut h_idx = 0usize;
    while h_idx < m {
        let slack = dot_n(&halfspaces[h_idx].a, &w, dim) - halfspaces[h_idx].b;
        if slack < min_slack {
            min_slack = slack;
        }
        h_idx += 1;
    }

    let mut finite_ok = true;
    let mut box_ok = true;
    d = 0;
    while d < dim {
        if !w[d].is_finite() {
            finite_ok = false;
        }
        if let Some(lo) = u_min {
            if w[d] < lo[d] - 1e-6 {
                box_ok = false;
            }
        }
        if let Some(hi) = u_max {
            if w[d] > hi[d] + 1e-6 {
                box_ok = false;
            }
        }
        d += 1;
    }

    let verified = finite_ok && box_ok && (m == 0 || min_slack >= -1e-6);
    CbfFilterResult {
        status: if verified {
            CbfStatus::Safe
        } else {
            CbfStatus::UnsafeNoSolution
        },
        verified_safe: verified,
        u_filtered: w,
        dim,
        min_barrier_slack: min_slack,
        iterations: iters_used,
    }
}

/// Simplex safety controller fallback: active velocity damping clipped to `[u_min, u_max]`.
///
/// Writes `u_brake[i] = clamp(-damping_gain * velocity[i], u_min[i], u_max[i])` into `out`
/// and returns the number of dimensions written.
pub fn simplex_damping_brake(
    velocity: &[f64],
    damping_gain: f64,
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    out: &mut [f64],
) -> usize {
    let n = velocity.len().min(out.len()).min(MAX_DIM);
    let gain = if damping_gain.is_finite() && damping_gain >= 0.0 {
        damping_gain
    } else {
        1.0
    };
    let mut i = 0usize;
    while i < n {
        let v = if velocity[i].is_finite() {
            velocity[i]
        } else {
            0.0
        };
        let mut u = -gain * v;
        if let Some(lo) = u_min {
            if i < lo.len() && lo[i].is_finite() && u < lo[i] {
                u = lo[i];
            }
        }
        if let Some(hi) = u_max {
            if i < hi.len() && hi[i].is_finite() && u > hi[i] {
                u = hi[i];
            }
        }
        out[i] = u;
        i += 1;
    }
    n
}

/// Combined real-time filter: runs Dykstra CBF-QP projection and, if infeasible,
/// engages [`simplex_damping_brake`] with status `CbfStatus::SimplexFallback`.
pub fn rt_cbf_filter_or_brake(
    u_nominal: &[f64],
    velocity: &[f64],
    halfspaces: &[CbfHalfspace],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    damping_gain: f64,
    max_iters: usize,
    tol: f64,
) -> CbfFilterResult {
    if velocity.len() != u_nominal.len() {
        return CbfFilterResult::error(CbfStatus::DimMismatch, u_nominal.len(), u_nominal);
    }
    let mut res = dykstra_cbf_project(u_nominal, halfspaces, u_min, u_max, max_iters, tol);
    if res.status == CbfStatus::UnsafeNoSolution && res.dim > 0 {
        let mut brake_buf = [0.0_f64; MAX_DIM];
        let written = simplex_damping_brake(velocity, damping_gain, u_min, u_max, &mut brake_buf);
        let mut i = 0usize;
        while i < res.dim && i < written {
            res.u_filtered[i] = brake_buf[i];
            i += 1;
        }
        let m = halfspaces.len();
        let mut min_slack = if m == 0 { 0.0_f64 } else { f64::INFINITY };
        let mut h_idx = 0usize;
        while h_idx < m {
            let slack = dot_n(&halfspaces[h_idx].a, &res.u_filtered, res.dim) - halfspaces[h_idx].b;
            if slack < min_slack {
                min_slack = slack;
            }
            h_idx += 1;
        }
        res.min_barrier_slack = min_slack;
        res.status = CbfStatus::SimplexFallback;
    }
    res
}

/// Zero-heap-allocation flat-matrix wrapper for [`dykstra_cbf_project`].
///
/// `a_flat` holds `m_barriers * dim` row-major coefficients (`A u >= b`), and `b_vec`
/// holds `m_barriers` thresholds. Supports `m_barriers == 0` with empty `a_flat`/`b_vec`.
pub fn dykstra_cbf_project_flat(
    a_flat: &[f64],
    b_vec: &[f64],
    m_barriers: usize,
    u_nominal: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    max_iters: usize,
    tol: f64,
) -> CbfFilterResult {
    let dim = u_nominal.len();
    if dim == 0 || dim > MAX_DIM || m_barriers > MAX_BARRIERS {
        return CbfFilterResult::error(CbfStatus::DimMismatch, dim, u_nominal);
    }
    if !tol.is_finite() || tol < 0.0 {
        return CbfFilterResult::error(CbfStatus::InvalidInput, dim, u_nominal);
    }
    let mut hs_buf = [CbfHalfspace {
        a: [0.0_f64; MAX_DIM],
        b: 0.0,
    }; MAX_BARRIERS];
    if m_barriers > 0 {
        let needed = m_barriers * dim;
        if a_flat.len() < needed || b_vec.len() < m_barriers {
            return CbfFilterResult::error(CbfStatus::DimMismatch, dim, u_nominal);
        }
        let mut j = 0usize;
        while j < m_barriers {
            let base = j * dim;
            let mut c = 0usize;
            while c < dim {
                hs_buf[j].a[c] = a_flat[base + c];
                c += 1;
            }
            hs_buf[j].b = b_vec[j];
            j += 1;
        }
    }
    dykstra_cbf_project(
        u_nominal,
        &hs_buf[..m_barriers],
        u_min,
        u_max,
        max_iters,
        tol,
    )
}

/// Zero-heap-allocation flat-matrix wrapper for [`rt_cbf_filter_or_brake`].
pub fn rt_cbf_filter_or_brake_flat(
    a_flat: &[f64],
    b_vec: &[f64],
    m_barriers: usize,
    u_nominal: &[f64],
    velocity: &[f64],
    u_min: Option<&[f64]>,
    u_max: Option<&[f64]>,
    damping_gain: f64,
    max_iters: usize,
    tol: f64,
) -> CbfFilterResult {
    let dim = u_nominal.len();
    if dim == 0 || dim > MAX_DIM || m_barriers > MAX_BARRIERS || velocity.len() != dim {
        return CbfFilterResult::error(CbfStatus::DimMismatch, dim, u_nominal);
    }
    if !tol.is_finite() || tol < 0.0 {
        return CbfFilterResult::error(CbfStatus::InvalidInput, dim, u_nominal);
    }
    let mut hs_buf = [CbfHalfspace {
        a: [0.0_f64; MAX_DIM],
        b: 0.0,
    }; MAX_BARRIERS];
    if m_barriers > 0 {
        let needed = m_barriers * dim;
        if a_flat.len() < needed || b_vec.len() < m_barriers {
            return CbfFilterResult::error(CbfStatus::DimMismatch, dim, u_nominal);
        }
        let mut j = 0usize;
        while j < m_barriers {
            let base = j * dim;
            let mut c = 0usize;
            while c < dim {
                hs_buf[j].a[c] = a_flat[base + c];
                c += 1;
            }
            hs_buf[j].b = b_vec[j];
            j += 1;
        }
    }
    rt_cbf_filter_or_brake(
        u_nominal,
        velocity,
        &hs_buf[..m_barriers],
        u_min,
        u_max,
        damping_gain,
        max_iters,
        tol,
    )
}

/// Deterministic dead-man watchdog authority scaler.
///
/// Returns `1.0` when `age_ms <= ttl_ms`, linearly decays from `1.0` to `0.0`
/// over `(ttl_ms, ttl_ms + ramp_ms]`, and fails closed (`0.0`) on invalid/negative ages.
pub fn watchdog_authority_scale(age_ms: f64, ttl_ms: f64, ramp_ms: f64) -> f64 {
    if !age_ms.is_finite() || !ttl_ms.is_finite() || age_ms < 0.0 {
        return 0.0;
    }
    if age_ms <= ttl_ms {
        return 1.0;
    }
    if !ramp_ms.is_finite() || ramp_ms <= 0.0 {
        return 0.0;
    }
    let excess = age_ms - ttl_ms;
    let raw = 1.0 - excess / ramp_ms;
    if raw <= 0.0 {
        0.0
    } else if raw >= 1.0 {
        1.0
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_halfspace_projection_is_exact() {
        // 2 * u >= 1.0 => u >= 0.5, with box [-2.0, 2.0], nominal u = -1.0 -> u* = 0.5
        let hs = [CbfHalfspace::new(&[2.0], 1.0)];
        let res = dykstra_cbf_project(&[-1.0], &hs, Some(&[-2.0]), Some(&[2.0]), 200, 1e-10);
        assert!(res.verified_safe);
        assert_eq!(res.status, CbfStatus::Safe);
        assert!((res.u_filtered[0] - 0.5).abs() < 1e-9);
        assert!(res.min_barrier_slack >= -1e-9);
    }

    #[test]
    fn hocbf_halfspace_matches_double_integrator_math() {
        // h = 0.2, h_dot = -0.5, lf2_h = 0.0, lg_lf_h = [1.0], alpha1=1.0, alpha2=2.0, margin=0.05
        // h_eff = 0.15 => b = -(0 + 3.0*(-0.5) + 2.0*0.15) = -(-1.5 + 0.3) = 1.2
        let hs = compute_hocbf_halfspace(0.2, -0.5, 0.0, &[1.0], 1.0, 2.0, 0.05);
        assert!((hs.a[0] - 1.0).abs() < 1e-12);
        assert!((hs.b - 1.2).abs() < 1e-12);
    }

    #[test]
    fn infeasible_constraints_trigger_simplex_braking_fallback() {
        // Contradictory halfspaces: u >= 1.5 AND u <= -1.5 (i.e. -u >= 1.5)
        let hs = [
            CbfHalfspace::new(&[1.0], 1.5),
            CbfHalfspace::new(&[-1.0], 1.5),
        ];
        let res = rt_cbf_filter_or_brake(
            &[0.8],
            &[3.0],
            &hs,
            Some(&[-1.0]),
            Some(&[1.0]),
            0.5,
            100,
            1e-9,
        );
        assert!(!res.verified_safe);
        assert_eq!(res.status, CbfStatus::SimplexFallback);
        // -0.5 * 3.0 = -1.5 clipped to u_min = -1.0
        assert!((res.u_filtered[0] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn watchdog_authority_decays_linearly_and_fails_closed() {
        assert!((watchdog_authority_scale(0.0, 50.0, 100.0) - 1.0).abs() < 1e-12);
        assert!((watchdog_authority_scale(50.0, 50.0, 100.0) - 1.0).abs() < 1e-12);
        assert!((watchdog_authority_scale(100.0, 50.0, 100.0) - 0.5).abs() < 1e-12);
        assert!((watchdog_authority_scale(150.0, 50.0, 100.0) - 0.0).abs() < 1e-12);
        assert!((watchdog_authority_scale(200.0, 50.0, 100.0) - 0.0).abs() < 1e-12);
        assert!((watchdog_authority_scale(-1.0, 50.0, 100.0) - 0.0).abs() < 1e-12);
        assert!((watchdog_authority_scale(f64::NAN, 50.0, 100.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn flat_wrappers_handle_zero_barriers_and_dim_checks() {
        let res_free = dykstra_cbf_project_flat(&[], &[], 0, &[1.5, -2.5], Some(&[-1.0, -1.0]), Some(&[1.0, 1.0]), 50, 1e-9);
        assert!(res_free.verified_safe);
        assert_eq!(res_free.status.to_c_code(), CBF_STATUS_SAFE);
        assert!((res_free.u_filtered[0] - 1.0).abs() < 1e-12);
        assert!((res_free.u_filtered[1] - (-1.0)).abs() < 1e-12);

        let res_bad_vel = rt_cbf_filter_or_brake_flat(&[], &[], 0, &[0.5, 0.5], &[0.1], None, None, 2.0, 50, 1e-9);
        assert_eq!(res_bad_vel.status, CbfStatus::DimMismatch);
        assert_eq!(res_bad_vel.status.to_c_code(), CBF_STATUS_DIM_MISMATCH);
    }
}

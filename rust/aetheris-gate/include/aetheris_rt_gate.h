/*
 * aetheris_rt_gate.h
 *
 * Zero-heap-allocation C99 / C++ ABI for the Aetheris Real-Time Safety Gate (`aetheris_gate`).
 *
 * Designed for deterministic sub-millisecond (1 kHz+) control loops such as
 * `ros2_control::ControllerInterface::update()`, real-time POSIX / PREEMPT_RT threads,
 * and embedded safety monitors where dynamic memory allocation is forbidden.
 *
 * Typical `ros2_control` C++ usage inside `update(const rclcpp::Time&, const rclcpp::Duration&)`:
 *
 *   #include "aetheris_rt_gate.h"
 *
 *   // Scale remote teleop command by dead-man watchdog authority before safety projection:
 *   const double auth = aetheris_rt_cbf_watchdog_scale(cmd_age_ms, 50.0, 100.0);
 *   double u_nom[2] = { auth * teleop_tau[0], auth * teleop_tau[1] };
 *   double u_safe[2] = { 0.0, 0.0 };
 *   double min_slack = 0.0;
 *   size_t iters = 0;
 *
 *   // Enforce affine CBF halfspaces A * u >= b and torque box [u_min, u_max],
 *   // automatically engaging Simplex active damping (-kd * q_dot) if infeasible:
 *   const int32_t rc = aetheris_rt_cbf_filter_or_brake(
 *       a_flat, b_vec, m_barriers, 2,
 *       u_nom, u_min, u_max, q_dot,
 *       2.5, 200, 1e-9,
 *       u_safe, &min_slack, &iters);
 */

#ifndef AETHERIS_RT_GATE_H
#define AETHERIS_RT_GATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum control dimension supported by fixed-stack CBF projection buffers. */
#define AETHERIS_RT_CBF_MAX_DIM 16

/** Maximum number of simultaneous affine barrier halfspaces (`A u >= b`). */
#define AETHERIS_RT_CBF_MAX_BARRIERS 32

/** Return status codes for `aetheris_rt_cbf_project` and `aetheris_rt_cbf_filter_or_brake`. */
typedef enum aetheris_rt_cbf_status_t {
    AETHERIS_CBF_SAFE = 0,
    AETHERIS_CBF_UNSAFE_NO_SOLUTION = 1,
    AETHERIS_CBF_SIMPLEX_FALLBACK = 2,
    AETHERIS_CBF_INVALID_INPUT = -1,
    AETHERIS_CBF_DIM_MISMATCH = -2
} aetheris_rt_cbf_status_t;

/**
 * Zero-heap-allocation Dykstra cyclic Euclidean projection onto:
 *   { u in R^dim : A * u >= b, u_min <= u <= u_max }
 *
 * @param a_flat_ptr    Row-major `(m_barriers * dim)` matrix `A` (nullable when `m_barriers == 0`).
 * @param b_ptr         Length-`m_barriers` RHS vector `b` (nullable when `m_barriers == 0`).
 * @param m_barriers    Number of barrier halfspaces (`0 ..= AETHERIS_RT_CBF_MAX_BARRIERS`).
 * @param dim           Control vector dimension (`1 ..= AETHERIS_RT_CBF_MAX_DIM`).
 * @param u_nom_ptr     Length-`dim` nominal control input `u_nom` (non-null).
 * @param u_min_ptr     Optional length-`dim` lower actuator bound (`NULL` if unbounded below).
 * @param u_max_ptr     Optional length-`dim` upper actuator bound (`NULL` if unbounded above).
 * @param max_iters     Maximum Dykstra cyclic projection iterations (e.g. `200`).
 * @param tol           Convergence tolerance in Euclidean/component norm (e.g. `1e-9`).
 * @param u_out_ptr     Output length-`dim` buffer receiving projected control `u*` (non-null).
 * @param out_min_slack Optional output scalar receiving `min_j (A_j . u* - b_j)` (`NULL` if unused).
 * @param out_iters     Optional output scalar receiving iterations executed (`NULL` if unused).
 * @return              Status code in `aetheris_rt_cbf_status_t` (`0` = safe, `1` = infeasible, `< 0` = error).
 */
int32_t aetheris_rt_cbf_project(
    const double *a_flat_ptr,
    const double *b_ptr,
    size_t m_barriers,
    size_t dim,
    const double *u_nom_ptr,
    const double *u_min_ptr,
    const double *u_max_ptr,
    size_t max_iters,
    double tol,
    double *u_out_ptr,
    double *out_min_slack,
    size_t *out_iters
);

/**
 * Zero-heap-allocation real-time CBF-QP filter with Simplex velocity-damping fallback.
 *
 * Runs `aetheris_rt_cbf_project`; if the barrier intersection is infeasible
 * (`AETHERIS_CBF_UNSAFE_NO_SOLUTION`), writes `clamp(-kd * q_dot, u_min, u_max)`
 * into `u_out_ptr` and returns `AETHERIS_CBF_SIMPLEX_FALLBACK` (`2`).
 *
 * @param a_flat_ptr    Row-major `(m_barriers * dim)` matrix `A` (nullable when `m_barriers == 0`).
 * @param b_ptr         Length-`m_barriers` RHS vector `b` (nullable when `m_barriers == 0`).
 * @param m_barriers    Number of barrier halfspaces (`0 ..= AETHERIS_RT_CBF_MAX_BARRIERS`).
 * @param dim           Control vector dimension (`1 ..= AETHERIS_RT_CBF_MAX_DIM`).
 * @param u_nom_ptr     Length-`dim` nominal control input `u_nom` (non-null).
 * @param u_min_ptr     Optional length-`dim` lower actuator bound (`NULL` if unbounded below).
 * @param u_max_ptr     Optional length-`dim` upper actuator bound (`NULL` if unbounded above).
 * @param q_dot_ptr     Length-`dim` measured joint/state velocity `q_dot` for Simplex damping (non-null).
 * @param kd            Non-negative Simplex damping gain `kd`.
 * @param max_iters     Maximum Dykstra cyclic projection iterations (e.g. `200`).
 * @param tol           Convergence tolerance (e.g. `1e-9`).
 * @param u_out_ptr     Output length-`dim` buffer receiving safe or braking control (non-null).
 * @param out_min_slack Optional output scalar receiving minimum barrier slack (`NULL` if unused).
 * @param out_iters     Optional output scalar receiving iterations executed (`NULL` if unused).
 * @return              Status code in `aetheris_rt_cbf_status_t`.
 */
int32_t aetheris_rt_cbf_filter_or_brake(
    const double *a_flat_ptr,
    const double *b_ptr,
    size_t m_barriers,
    size_t dim,
    const double *u_nom_ptr,
    const double *u_min_ptr,
    const double *u_max_ptr,
    const double *q_dot_ptr,
    double kd,
    size_t max_iters,
    double tol,
    double *u_out_ptr,
    double *out_min_slack,
    size_t *out_iters
);

/**
 * Deterministic dead-man watchdog authority scaler.
 *
 * Returns `1.0` when `0 <= age_ms <= ttl_ms`, decays linearly from `1.0` to `0.0`
 * across `(ttl_ms, ttl_ms + ramp_ms]`, and fails closed (`0.0`) on expired, negative,
 * or non-finite inputs.
 */
double aetheris_rt_cbf_watchdog_scale(
    double age_ms,
    double ttl_ms,
    double ramp_ms
);

/**
 * Zero-heap-allocation 2-bit packed ternary matrix-vector product (`int8` activations -> `int32` accumulators).
 *
 * Weight encoding: 4 trits per `uint8_t` in LSB-first order (`0 => -1`, `1 => 0`, `2 => +1`, `3 => 0`),
 * flattened row-major over `(out_features, in_features)`.
 *
 * @param packed_ptr   Pointer to `ceil(out_features * in_features / 4)` packed weight bytes.
 * @param packed_len   Length of `packed_ptr` in bytes.
 * @param x_ptr        Pointer to `in_features` quantized `int8_t` activations.
 * @param in_features  Input feature count.
 * @param out_features Output feature count.
 * @param out_ptr      Output buffer of `out_features` `int32_t` accumulators.
 * @return             `0` on success, `-1` on null pointers or undersized buffers.
 */
int32_t aetheris_rt_ternary_matvec_i8(
    const uint8_t *packed_ptr,
    size_t packed_len,
    const int8_t *x_ptr,
    size_t in_features,
    size_t out_features,
    int32_t *out_ptr
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AETHERIS_RT_GATE_H */

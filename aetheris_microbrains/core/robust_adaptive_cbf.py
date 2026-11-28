"""L1 adaptive disturbance observer and robust adaptive Control Barrier Function filter.

Couples fast state-predictor disturbance estimation with state-space CBF halfspace
tightening so safety invariants remain forward-invariant under sudden unknown payloads,
gravity wrench drops, and external wind/contact disturbances:

1. ``L1DisturbanceObserver`` — estimates matched and unmatched disturbances ``d(x, t)``
   in ``xdot = f(x) + g(x) u + d(x, t)`` via a Hurwitz state predictor
   ``xhat_dot = f(x) + g(x) u + d_hat - A_m (xhat - x)`` and a strictly proper low-pass
   filter on the raw innovation, maintaining a certified online upper bound
   ``eps_bar_d(t) >= ||d(t) - d_hat(t)||_2``.

2. ``RobustAdaptiveCBFFilter`` — modifies each relative-degree-1 and relative-degree-2
   CBF halfspace with both systematic disturbance cancellation ``-grad_h(x)^T d_hat(t)``
   and worst-case estimation error margin ``+||grad_h(x)||_2 * eps_bar_d(t)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .state_cbf_filter import (
    BarrierConstraint,
    ControlAffineDynamics,
    StateCBFSafetyFilter,
    _fd_jacobian,
)


def _normalize_predictor_rates(
    gain: Union[float, Sequence[float], np.ndarray],
    state_dim: int,
) -> np.ndarray:
    """Return positive diagonal rates a_m > 0 such that -A_m = -diag(a_m) is Hurwitz."""
    arr = np.asarray(gain, dtype=np.float64)
    if arr.ndim == 0:
        val = abs(float(arr))
        if val <= 0.0:
            raise ValueError("predictor_gain must be non-zero")
        return np.full(state_dim, val, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != state_dim:
            raise ValueError(f"predictor_gain length {arr.shape[0]} != state_dim {state_dim}")
        rates = np.abs(arr)
        if np.any(rates <= 0.0):
            raise ValueError("predictor_gain entries must be non-zero")
        return rates
    if arr.ndim == 2 and arr.shape == (state_dim, state_dim):
        # Support diagonal or symmetric positive/negative Hurwitz matrices
        diag = np.abs(np.diag(arr))
        if np.any(diag <= 0.0):
            raise ValueError("predictor_gain matrix diagonal must be non-zero")
        return diag
    raise ValueError(f"Unsupported predictor_gain shape {arr.shape}")


class L1DisturbanceObserver:
    """Piecewise-constant / low-pass L1 adaptive disturbance observer for control-affine plants.

    Models ``xdot = f(x) + g(x) u + d(x, t)`` with Hurwitz state predictor:
        xhat_dot = f(x) + g(x) u + d_hat - A_m (xhat - x)
    and low-pass filtered disturbance estimate ``d_hat``, tracking an analytical online
    upper bound ``disturbance_bound >= ||d(t) - d_hat(t)||_2``.
    """

    def __init__(
        self,
        dynamics: ControlAffineDynamics,
        dt: float = 0.001,
        predictor_gain: Union[float, Sequence[float], np.ndarray] = 60.0,
        filter_bandwidth: float = 350.0,
        disturbance_lipschitz: float = 2.0,
        initial_disturbance_bound: float = 2.0,
        noise_floor: float = 1e-3,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if filter_bandwidth <= 0.0:
            raise ValueError("filter_bandwidth must be positive")
        if disturbance_lipschitz < 0.0 or initial_disturbance_bound < 0.0 or noise_floor < 0.0:
            raise ValueError("Error bound parameters must be non-negative")

        self.dynamics = dynamics
        self.state_dim = int(dynamics.state_dim)
        self.dt = float(dt)
        self.a_m = _normalize_predictor_rates(predictor_gain, self.state_dim)
        self.A_m = np.diag(self.a_m)
        self.filter_bandwidth = float(filter_bandwidth)
        self.disturbance_lipschitz = float(disturbance_lipschitz)
        self.initial_disturbance_bound = float(initial_disturbance_bound)
        self.noise_floor = float(noise_floor)

        self.reset()

    def reset(self, x0: Optional[Sequence[float]] = None) -> None:
        if x0 is not None:
            x_init = np.asarray(x0, dtype=np.float64).flatten()
            if x_init.shape[0] != self.state_dim:
                raise ValueError(f"x0 length {x_init.shape[0]} != state_dim {self.state_dim}")
            self.x_hat: Optional[np.ndarray] = x_init.copy()
            self._x_prev: Optional[np.ndarray] = x_init.copy()
        else:
            self.x_hat = None
            self._x_prev = None

        self.d_hat = np.zeros(self.state_dim, dtype=np.float64)
        self.raw_innovation = np.zeros(self.state_dim, dtype=np.float64)
        self.disturbance_bound: float = self.initial_disturbance_bound
        self.disturbance_bound_vec = np.full(
            self.state_dim,
            self.initial_disturbance_bound / max(1.0, np.sqrt(float(self.state_dim))),
            dtype=np.float64,
        )
        self.step_count: int = 0

    def update(
        self,
        x: Sequence[float],
        u_prev: Optional[Sequence[float]] = None,
        dt: Optional[float] = None,
        x_prev: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Update state predictor, low-pass disturbance estimate, and analytical error bound."""
        x_curr = np.asarray(x, dtype=np.float64).flatten()
        if x_curr.shape[0] != self.state_dim:
            raise ValueError(f"State length {x_curr.shape[0]} != state_dim {self.state_dim}")

        if x_prev is not None:
            xp = np.asarray(x_prev, dtype=np.float64).flatten()
        else:
            xp = self._x_prev

        if xp is None or u_prev is None:
            self.x_hat = x_curr.copy()
            self._x_prev = x_curr.copy()
            return {
                "d_hat": self.d_hat.copy(),
                "raw_innovation": self.raw_innovation.copy(),
                "disturbance_bound": self.disturbance_bound,
                "x_hat": self.x_hat.copy(),
                "prediction_error_norm": 0.0,
                "step_count": self.step_count,
            }

        step_dt = float(self.dt if dt is None else dt)
        if step_dt <= 0.0:
            raise ValueError("dt must be positive")
        u_vec = np.asarray(u_prev, dtype=np.float64).flatten()

        if self.x_hat is None:
            self.x_hat = xp.copy()

        # Nominal control-affine drift + actuation at interval endpoints (captures both Euler and RK4 plants)
        nom_prev = self.dynamics.f(xp) + self.dynamics.g(xp) @ u_vec
        nom_curr = self.dynamics.f(x_curr) + self.dynamics.g(x_curr) @ u_vec
        nom_mid = 0.5 * (nom_prev + nom_curr)
        disc_err_vec = 0.5 * np.abs(nom_curr - nom_prev)
        disc_err = float(np.linalg.norm(disc_err_vec))

        # Exact ZOH propagation of Hurwitz error dynamics: x_tilde_dot = -A_m x_tilde + (d_hat - d)
        phi = np.exp(-self.a_m * step_dt)
        # gamma_i = (1 - exp(-a_m_i * dt)) / a_m_i, safe for small a_m_i * dt
        gamma = np.where(
            self.a_m * step_dt > 1e-6,
            (1.0 - phi) / self.a_m,
            step_dt * (1.0 - 0.5 * self.a_m * step_dt),
        )

        x_tilde_prev = self.x_hat - xp
        d_interval = (x_curr - xp) / step_dt - nom_mid

        # Predictor state error at t_k prior to innovation update
        x_tilde_curr = phi * x_tilde_prev + gamma * (self.d_hat - d_interval)
        self.x_hat = x_curr + x_tilde_curr

        # Unbiased L1 innovation from predictor state error: sigma_raw = d_hat_prev - Gamma^{-1}(x_tilde_k - Phi x_tilde_{k-1})
        sigma_raw = self.d_hat - (x_tilde_curr - phi * x_tilde_prev) / np.maximum(gamma, 1e-15)
        self.raw_innovation = sigma_raw

        # Strictly proper first-order low-pass filter C(s) = omega_c / (s + omega_c)
        lpf_decay = float(np.exp(-self.filter_bandwidth * step_dt))
        self.d_hat = lpf_decay * self.d_hat + (1.0 - lpf_decay) * sigma_raw

        # Analytical componentwise & Euclidean upper bounds: ||d(t) - d_hat(t)||_2 <= disturbance_bound
        per_coord_floor = (self.disturbance_lipschitz * step_dt + self.noise_floor) / max(
            1.0, np.sqrt(float(self.state_dim))
        )
        self.disturbance_bound_vec = np.abs(sigma_raw - self.d_hat) + disc_err_vec + per_coord_floor
        lpf_lag_norm = float(np.linalg.norm(sigma_raw - self.d_hat))
        self.disturbance_bound = max(
            float(np.linalg.norm(self.disturbance_bound_vec)),
            lpf_lag_norm + disc_err + self.disturbance_lipschitz * step_dt + self.noise_floor,
        )

        self._x_prev = x_curr.copy()
        self.step_count += 1
        return {
            "d_hat": self.d_hat.copy(),
            "raw_innovation": self.raw_innovation.copy(),
            "disturbance_bound": self.disturbance_bound,
            "x_hat": self.x_hat.copy(),
            "prediction_error_norm": float(np.linalg.norm(x_tilde_curr)),
            "step_count": self.step_count,
        }

    step = update


class RobustAdaptiveCBFFilter(StateCBFSafetyFilter):
    """Safety filter coupling StateCBFSafetyFilter with L1DisturbanceObserver.

    Tightens every relative-degree-1 and relative-degree-2 barrier halfspace ``A_i u >= b_i``
    with systematic disturbance compensation ``-grad_h^T d_hat`` and worst-case estimation
    uncertainty margin ``+||grad_h||_2 * disturbance_bound``.
    """

    def __init__(
        self,
        dynamics: ControlAffineDynamics,
        barriers: Sequence[BarrierConstraint],
        observer: Optional[L1DisturbanceObserver] = None,
        u_min: Optional[Sequence[float]] = None,
        u_max: Optional[Sequence[float]] = None,
        dt: float = 0.001,
        conformal_margin: float = 0.0,
        max_iters: int = 200,
        tol: float = 1e-9,
        backup_controller: Optional[Any] = None,
    ):
        super().__init__(
            dynamics=dynamics,
            barriers=barriers,
            u_min=u_min,
            u_max=u_max,
            conformal_margin=conformal_margin,
            max_iters=max_iters,
            tol=tol,
            backup_controller=backup_controller,
        )
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.observer = observer if observer is not None else L1DisturbanceObserver(dynamics, dt=self.dt)
        self._prev_x: Optional[np.ndarray] = None
        self._prev_u: Optional[np.ndarray] = None
        self._last_obs_step: int = self.observer.step_count

    def reset(self, x0: Optional[Sequence[float]] = None) -> None:
        self.observer.reset(x0=x0)
        self._prev_x = None if x0 is None else np.asarray(x0, dtype=np.float64).flatten().copy()
        self._prev_u = None
        self._last_obs_step = self.observer.step_count

    def _halfspaces_with_flags(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[bool]]:
        """Assemble disturbance-compensated and error-robustified halfspaces A_i u >= b_i."""
        if not self.barriers:
            return (
                np.zeros((0, self.dynamics.control_dim), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                [],
                [],
            )

        f_x = self.dynamics.f(x)
        G_x = self.dynamics.g(x)
        d_hat = self.observer.d_hat
        eps_d = float(self.observer.disturbance_bound)
        eps_vec = getattr(self.observer, "disturbance_bound_vec", None)

        J_f: Optional[np.ndarray] = None
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []
        alphas: List[float] = []
        hard_flags: List[bool] = []

        for bar in self.barriers:
            eff_bar = bar.inflate(self.conformal_margin) if self.conformal_margin > 0.0 else bar
            gh = np.asarray(eff_bar.grad_h(x), dtype=np.float64).flatten()
            L_f_h = float(np.dot(gh, f_x))
            h_eff = float(eff_bar.h(x))
            is_hard = bool(getattr(eff_bar, "hard", getattr(bar, "hard", True)))
            rel_deg = int(getattr(eff_bar, "relative_degree", getattr(bar, "relative_degree", 1)))

            if rel_deg == 2:
                if J_f is None:
                    if hasattr(self.dynamics, "df_dx"):
                        J_f = np.asarray(self.dynamics.df_dx(x), dtype=np.float64)
                    elif hasattr(self.dynamics, "J_f"):
                        J_f = np.asarray(self.dynamics.J_f(x), dtype=np.float64)
                    else:
                        J_f = _fd_jacobian(self.dynamics.f, x)
                if hasattr(eff_bar, "hess_h"):
                    H_h = np.asarray(eff_bar.hess_h(x), dtype=np.float64)
                else:
                    H_h = _fd_jacobian(eff_bar.grad_h, x, eps=1e-5)
                    H_h = 0.5 * (H_h + H_h.T)

                alpha1 = float(getattr(eff_bar, "alpha1", getattr(eff_bar, "alpha", 0.5)))
                alpha2 = float(getattr(eff_bar, "alpha2", getattr(bar, "alpha2", 1.0)))

                f_dist = f_x + d_hat
                L_f2_h_dist = float(f_dist @ H_h @ f_dist + gh @ (J_f @ f_dist))
                L_g_L_f_h = np.asarray((f_dist @ H_h @ G_x) + (gh @ (J_f @ G_x)), dtype=np.float64).flatten()

                # Total sensitivity of (h_ddot + (alpha1 + alpha2) h_dot) to estimation error (d - d_hat)
                grad_phi = (H_h @ f_dist) + (J_f.T @ gh) + (alpha1 + alpha2) * gh
                ball_margin = float(np.linalg.norm(grad_phi)) * eps_d
                if eps_vec is not None and eps_vec.shape == grad_phi.shape:
                    box_margin = float(np.sum(np.abs(grad_phi) * eps_vec))
                    robust_margin = min(ball_margin, box_margin)
                else:
                    robust_margin = ball_margin

                b_val = (
                    -L_f2_h_dist
                    - (alpha1 + alpha2) * (L_f_h + float(np.dot(gh, d_hat)))
                    - alpha1 * alpha2 * h_eff
                    + robust_margin
                )
                A_rows.append(L_g_L_f_h)
                b_rows.append(b_val)
                alphas.append(alpha1)
            else:
                L_g_h = np.asarray(gh @ G_x, dtype=np.float64).flatten()
                alpha = float(getattr(eff_bar, "alpha", 0.5))
                gh_norm = float(np.linalg.norm(gh))
                b_val = (
                    -alpha * h_eff
                    - L_f_h
                    - float(np.dot(gh, d_hat))
                    + gh_norm * eps_d
                )
                A_rows.append(L_g_h)
                b_rows.append(b_val)
                alphas.append(alpha)

            hard_flags.append(is_hard)

        return np.vstack(A_rows), np.asarray(b_rows, dtype=np.float64), alphas, hard_flags

    def filter(
        self,
        x: np.ndarray,
        u_nominal: Sequence[float],
        dt: Optional[float] = None,
        update_observer: Optional[bool] = None,
    ) -> Dict[str, Any]:
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        u_arr = np.asarray(u_nominal, dtype=np.float64).flatten()
        step_dt = float(self.dt if dt is None else dt)

        # Auto-step observer on sequential filter() calls unless caller already updated observer
        if (
            x_arr.shape[0] == self.dynamics.state_dim
            and u_arr.shape[0] == self.dynamics.control_dim
            and np.all(np.isfinite(x_arr))
            and np.all(np.isfinite(u_arr))
        ):
            should_update = (
                update_observer
                if update_observer is not None
                else (
                    self._prev_x is not None
                    and self._prev_u is not None
                    and self.observer.step_count == self._last_obs_step
                )
            )
            if should_update and self._prev_x is not None and self._prev_u is not None:
                self.observer.update(x_arr, u_prev=self._prev_u, dt=step_dt, x_prev=self._prev_x)
            elif self._prev_x is None and self.observer.x_hat is None:
                self.observer.update(x_arr)

        out = super().filter(x_arr, u_nominal)
        out["d_hat"] = self.observer.d_hat.tolist()
        out["disturbance_bound"] = round(float(self.observer.disturbance_bound), 9)

        if (
            x_arr.shape[0] == self.dynamics.state_dim
            and u_arr.shape[0] == self.dynamics.control_dim
            and np.all(np.isfinite(x_arr))
        ):
            u_applied = np.asarray(out["u_filtered"], dtype=np.float64).flatten()
            if u_applied.shape[0] == self.dynamics.control_dim and np.all(np.isfinite(u_applied)):
                self._prev_x = x_arr.copy()
                self._prev_u = u_applied.copy()
                self._last_obs_step = self.observer.step_count

        return out

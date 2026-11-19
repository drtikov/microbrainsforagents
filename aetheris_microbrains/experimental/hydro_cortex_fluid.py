"""
Hydro-Cortex Semantic Fluid Diffusion Field for Aetheris.
Ported and evolved from Aura-Alien VFS (E:/auraalienvfs/hydro_cortex.cpp) + 2025/2026 SOTA.

Models working memory activation, attention spreading, and task pressure
as a 2D incompressible Navier-Stokes viscous fluid:
1. Density diffusion: Spread of semantic concepts across memory grid.
2. Velocity advection: Directed intentional drift (momentum toward active goal).
3. Vorticity & Pressure Poisson solve: Prevents runaway local peaks, ensuring
   smooth, continuous cognitive dynamics instead of jarring discrete jumps.
"""

from __future__ import annotations
import base64
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger("aetheris.hydro_cortex")


@dataclass
class FluidFieldSnapshot:
    grid_size: int
    mean_density: float
    max_density: float
    total_kinetic_energy: float
    pressure_variance: float
    dominant_flow_direction: Tuple[float, float]
    base64_density_preview: str


class HydroCortexFluidField:
    """
    2D Navier-Stokes cognitive fluid simulation on an N x N semantic grid.
    Operates in pure Python/NumPy with fast vectorized finite-difference stencils.
    """

    _instance: Optional["HydroCortexFluidField"] = None

    def __init__(
        self,
        size: int = 32,
        viscosity: float = 0.0001,
        diffusion_rate: float = 0.0002,
        dt: float = 0.1,
        vorticity_eps: float = 0.15,
    ):
        self.N = size
        self.viscosity = viscosity
        self.diff = diffusion_rate
        self.dt = dt
        self.vorticity_eps = float(vorticity_eps)

        # Density fields (current and previous)
        self.density = np.zeros((size, size), dtype=np.float64)
        self.density_prev = np.zeros((size, size), dtype=np.float64)

        # Velocity fields (u: horizontal, v: vertical)
        self.u = np.zeros((size, size), dtype=np.float64)
        self.v = np.zeros((size, size), dtype=np.float64)
        self.u_prev = np.zeros((size, size), dtype=np.float64)
        self.v_prev = np.zeros((size, size), dtype=np.float64)

        self.step_count = 0

    @classmethod
    def get_instance(cls) -> "HydroCortexFluidField":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def inject_concept_density(self, x: int, y: int, amount: float = 5.0, velocity_bias: Tuple[float, float] = (0.5, 0.0)):
        """Injects a cognitive stimulus / goal pressure at grid location (x, y)."""
        xi = max(0, min(self.N - 1, int(x)))
        yi = max(0, min(self.N - 1, int(y)))
        self.density[yi, xi] += float(amount)
        self.u[yi, xi] += float(velocity_bias[0])
        self.v[yi, xi] += float(velocity_bias[1])

    def _apply_vorticity_confinement(self) -> float:
        """Stam/Fedkiw vorticity confinement to counteract semi-Lagrangian rotational damping."""
        if self.N < 5:
            return 0.0
        omega = np.zeros_like(self.u)
        omega[1:-1, 1:-1] = 0.5 * (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) - (self.u[2:, 1:-1] - self.u[:-2, 1:-1])
        )
        abs_omega = np.abs(omega)
        dw_dx = 0.5 * (abs_omega[2:-2, 3:-1] - abs_omega[2:-2, 1:-3])
        dw_dy = 0.5 * (abs_omega[3:-1, 2:-2] - abs_omega[1:-3, 2:-2])
        norm = np.sqrt(dw_dx ** 2 + dw_dy ** 2) + 1e-12
        nx = dw_dx / norm
        ny = dw_dy / norm
        w_inner = omega[2:-2, 2:-2]
        self.u[2:-2, 2:-2] += self.dt * self.vorticity_eps * (ny * w_inner)
        self.v[2:-2, 2:-2] += self.dt * self.vorticity_eps * (-nx * w_inner)
        return float(np.mean(abs_omega))

    def step_simulation(self, iterations: int = 4) -> Dict[str, Any]:
        """Advances the fluid state by dt using vorticity confinement, advection, diffusion, and projection."""
        self.step_count += 1
        dt = self.dt

        # 0. Vorticity confinement before diffusion/projection
        mean_vorticity = self._apply_vorticity_confinement() if self.vorticity_eps > 0.0 else 0.0

        # 1. Diffuse velocity
        self._diffuse(1, self.u_prev, self.u, self.viscosity, dt, iterations)
        self._diffuse(2, self.v_prev, self.v, self.viscosity, dt, iterations)

        # 2. Project velocity (enforce mass conservation / incompressibility: div v = 0)
        self._project(self.u_prev, self.v_prev, self.u, self.v, iterations)

        # 3. Advect velocity
        self._advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, dt)
        self._advect(2, self.v, self.v_prev, self.u_prev, self.v_prev, dt)
        self._project(self.u, self.v, self.u_prev, self.v_prev, iterations)

        # 4. Diffuse density
        self._diffuse(0, self.density_prev, self.density, self.diff, dt, iterations)

        # 5. Advect density
        self._advect(0, self.density, self.density_prev, self.u, self.v, dt)

        # Slight natural dissipation to maintain homeostatic bound
        self.density *= 0.992
        return {"step_count": self.step_count, "mean_vorticity": mean_vorticity}

    def sample_activation_at(self, x_norm: float, y_norm: float) -> float:
        """Samples normalized semantic field density at [0.0, 1.0] coordinates."""
        xi = int(max(0, min(self.N - 1, round(x_norm * (self.N - 1)))))
        yi = int(max(0, min(self.N - 1, round(y_norm * (self.N - 1)))))
        return float(self.density[yi, xi])

    def get_snapshot(self) -> FluidFieldSnapshot:
        total_ke = 0.5 * float(np.sum(self.u**2 + self.v**2))
        mean_d = float(np.mean(self.density))
        max_d = float(np.max(self.density))
        p_var = float(np.var(self.density))
        mean_u = float(np.mean(self.u))
        mean_v = float(np.mean(self.v))

        # Base64 low-res preview of density
        norm_density = np.clip(self.density / max(1e-6, max_d) * 255.0, 0, 255).astype(np.uint8)
        b64 = base64.b64encode(norm_density.tobytes()).decode("ascii")[:48] + "..."

        return FluidFieldSnapshot(
            grid_size=self.N,
            mean_density=round(mean_d, 4),
            max_density=round(max_d, 4),
            total_kinetic_energy=round(total_ke, 4),
            pressure_variance=round(p_var, 4),
            dominant_flow_direction=(round(mean_u, 3), round(mean_v, 3)),
            base64_density_preview=b64,
        )

    # Internal Navier-Stokes subroutines
    def _diffuse(self, b: int, x: np.ndarray, x0: np.ndarray, diff: float, dt: float, iters: int):
        a = dt * diff * (self.N - 2) * (self.N - 2)
        denom = 1.0 + 4.0 * a
        for _ in range(iters):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (
                x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2]
            )) / denom
            self._set_bnd(b, x)

    def _advect(self, b: int, d: np.ndarray, d0: np.ndarray, u: np.ndarray, v: np.ndarray, dt: float):
        N = self.N
        dt0 = dt * (N - 2)
        i_idx, j_idx = np.meshgrid(np.arange(1, N - 1), np.arange(1, N - 1), indexing="xy")
        x = np.clip(i_idx - dt0 * u[1:-1, 1:-1], 0.5, N - 2.5)
        y = np.clip(j_idx - dt0 * v[1:-1, 1:-1], 0.5, N - 2.5)
        i0 = x.astype(np.intp)
        j0 = y.astype(np.intp)
        i1 = i0 + 1
        j1 = j0 + 1
        s1 = x - i0
        t1 = y - j0
        s0 = 1.0 - s1
        t0 = 1.0 - t1
        d[1:-1, 1:-1] = (
            s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0])
            + s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1])
        )
        self._set_bnd(b, d)

    def _project(self, u: np.ndarray, v: np.ndarray, p: np.ndarray, div: np.ndarray, iters: int):
        N = self.N
        div[1:-1, 1:-1] = -0.5 * (
            u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1]
        ) / N
        p.fill(0.0)
        self._set_bnd(0, div)
        self._set_bnd(0, p)

        for _ in range(iters):
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] + (
                p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2]
            )) / 4.0
            self._set_bnd(0, p)

        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * N
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * N
        self._set_bnd(1, u)
        self._set_bnd(2, v)

    def _set_bnd(self, b: int, x: np.ndarray):
        x[0, :] = -x[1, :] if b == 2 else x[1, :]
        x[-1, :] = -x[-2, :] if b == 2 else x[-2, :]
        x[:, 0] = -x[:, 1] if b == 1 else x[:, 1]
        x[:, -1] = -x[:, -2] if b == 1 else x[:, -2]
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])


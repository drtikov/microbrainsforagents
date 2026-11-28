"""Tests for Active Inference EFE sign fix, factored multi-modal models, and epistemic VLA gating."""

from __future__ import annotations

import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_AETHER_DIR = _PKG_DIR.parent

for p in [str(_PKG_DIR), str(_AETHER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

from aetheris_microbrains.core.pymdp_efe import (
    EFPAgent,
    FactoredGenerativeModel,
    GenerativeModel,
    MultiModalEFPAgent,
    expected_free_energy,
    expected_free_energy_multimodal,
)
from aetheris_microbrains.core.state_cbf_filter import (
    ControlAffineDynamics,
    HalfPlaneBarrier,
    SphereBarrier,
    StateCBFSafetyFilter,
)
from aetheris_microbrains.core.vla_gating import (
    ConformalActionGate,
    SkillLibrary,
    VLActionGate,
)


def _epistemic_probe_model() -> GenerativeModel:
    """Two-action T-maze-like cue check with uniform preferences.

    States:
      0 = left_target, 1 = right_target, 2 = cue_left_obs_state, 3 = cue_right_obs_state
    Actually simpler 3-state model:
      State 0: hidden_A, State 1: hidden_B, State 2: uninformative_neutral
    Better: 4 states: (hidden in {0, 1}) x (location in {unprobed=0, probed=1})!
    - Action 0 (stay unprobed): moves to {0, 1} -> uninformative uniform obs [0.5, 0.5]
    - Action 1 (active probe): moves to {2, 3} -> sharp diagnostic obs [0.95, 0.05] / [0.05, 0.95]
    Uniform C=[0, 0] makes pragmatic risk identical for both actions, isolating epistemic gain!
    """
    A = np.array(
        [
            [0.5, 0.5, 0.95, 0.05],
            [0.5, 0.5, 0.05, 0.95],
        ],
        dtype=np.float64,
    )
    B = np.zeros((4, 4, 2), dtype=np.float64)
    # Action 0: map to unprobed states (0 or 1) preserving hidden identity
    B[0, 0, 0] = 1.0
    B[1, 1, 0] = 1.0
    B[0, 2, 0] = 1.0
    B[1, 3, 0] = 1.0
    # Action 1: map to probed states (2 or 3) preserving hidden identity
    B[2, 0, 1] = 1.0
    B[3, 1, 1] = 1.0
    B[2, 2, 1] = 1.0
    B[3, 3, 1] = 1.0

    C = np.array([0.0, 0.0], dtype=np.float64)
    D = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
    return GenerativeModel(A, B, C=C, D=D)


class TestEpistemicSignAndPragmaticLogPreferences:
    def test_agent_prefers_informative_action_under_uniform_preferences(self):
        model = _epistemic_probe_model()
        g_noisy, tr_noisy = expected_free_energy(model, model.D, [0])
        g_probe, tr_probe = expected_free_energy(model, model.D, [1])

        assert tr_noisy["per_step"][0]["epistemic"] == pytest.approx(0.0, abs=1e-9)
        assert tr_probe["per_step"][0]["epistemic"] > 0.4
        assert tr_probe["per_step"][0]["pragmatic"] == pytest.approx(
            tr_noisy["per_step"][0]["pragmatic"], abs=1e-9
        )
        # Higher epistemic information gain must REDUCE G(pi)
        assert g_probe < g_noisy

        agent = EFPAgent(model)
        plan = agent.plan([[0], [1]])
        assert plan["best_action"] == 1
        assert plan["q_pi"][1] > plan["q_pi"][0]

    def test_pragmatic_uses_log_preferences(self):
        A = np.eye(2)
        B = np.eye(2)[:, :, None]
        C = np.array([3.0, 0.0])
        D = np.array([1.0, 0.0])
        model = GenerativeModel(A, B, C=C, D=D)
        g, tr = expected_free_energy(model, D, [0])
        expected_pragmatic = -float(np.log(model.preferences[0]))
        assert tr["per_step"][0]["pragmatic"] == pytest.approx(expected_pragmatic, rel=1e-5)
        assert g == pytest.approx(expected_pragmatic, rel=1e-5)


class TestFactoredMultiModalActiveInference:
    def _build_vision_tactile_model(self, with_pA: bool = False) -> FactoredGenerativeModel:
        """2 state factors x 2 observation modalities (vision + tactile force).

        Factor 0 (object pose, size 2): {0: left, 1: right} (static, 1 action)
        Factor 1 (sensor mode, size 3): {0: idle, 1: look_camera, 2: tactile_poke} (3 actions)
        Modality 0 (vision, 2 obs): informative only when sensor_mode == 1 (look_camera)
        Modality 1 (tactile, 2 obs): informative only when sensor_mode == 2 (tactile_poke)
        """
        B_obj = np.eye(2)[:, :, None]  # (2, 2, 1)
        B_mode = np.zeros((3, 3, 3), dtype=np.float64)
        for u in range(3):
            B_mode[u, :, u] = 1.0  # action u moves sensor mode directly to u

        # Vision A0: shape (2, 2, 3) -> (n_obs_vis, n_obj, n_mode)
        A_vis = np.full((2, 2, 3), 0.5, dtype=np.float64)
        A_vis[:, 0, 1] = [0.95, 0.05]
        A_vis[:, 1, 1] = [0.05, 0.95]

        # Tactile A1: shape (2, 2, 3) -> (n_obs_tac, n_obj, n_mode)
        A_tac = np.full((2, 2, 3), 0.5, dtype=np.float64)
        A_tac[:, 0, 2] = [0.99, 0.01]
        A_tac[:, 1, 2] = [0.01, 0.99]

        pA = None
        if with_pA:
            # Vision well-calibrated (high counts), tactile unvisited (low counts)
            pA = [
                np.full((2, 2, 3), 200.0, dtype=np.float64),
                np.full((2, 2, 3), 1.0, dtype=np.float64),
            ]

        return FactoredGenerativeModel(
            A_modalities=[A_vis, A_tac],
            B_factors=[B_obj, B_mode],
            C_modalities=[np.zeros(2), np.zeros(2)],
            D_factors=[np.array([0.5, 0.5]), np.array([1.0, 0.0, 0.0])],
            pA=pA,
            num_states=[2, 3],
        )

    def test_multimodal_plan_prefers_highest_information_modality(self):
        model = self._build_vision_tactile_model(with_pA=False)
        agent = MultiModalEFPAgent(model)
        assert agent.belief_entropy == pytest.approx(np.log(2.0), abs=1e-6)

        # Policies: idle=(0,0), look=(0,1), tactile_poke=(0,2)
        plan = agent.plan([[(0, 0)], [(0, 1)], [(0, 2)]])
        assert plan["status"] == "planned"
        # Tactile (0.99/0.01) has higher MI than vision (0.95/0.05) and both beat idle (0.0)
        assert plan["best_action"] == [0, 2]
        assert plan["G"][2] < plan["G"][1] < plan["G"][0]
        assert len(plan["per_modality"]) == 2
        assert plan["per_modality"][1]["epistemic"] > plan["per_modality"][0]["epistemic"]

    def test_multimodal_belief_update_reduces_entropy_across_modalities(self):
        model = self._build_vision_tactile_model(with_pA=False)
        agent = MultiModalEFPAgent(model)
        h_before = agent.belief_entropy

        # Execute look_camera (0, 1) and observe vision=0, tactile=0
        post_factors = agent.update_belief([0, 0], action_indices=(0, 1))
        assert len(post_factors) == 2
        assert post_factors[0][0] == pytest.approx(0.95, abs=1e-6)
        assert post_factors[1][1] == pytest.approx(1.0, abs=1e-6)
        assert agent.belief_entropy < h_before

    def test_dirichlet_novelty_drives_exploration_and_decays_after_observation(self):
        # Identical A across modalities, but modality 1 has low Dirichlet pA counts
        B_obj = np.eye(2)[:, :, None]
        B_mode = np.zeros((2, 2, 2), dtype=np.float64)
        for u in range(2):
            B_mode[u, :, u] = 1.0

        A0 = np.full((2, 2, 2), 0.5)
        A0[:, 0, 0] = [0.9, 0.1]
        A0[:, 1, 0] = [0.1, 0.9]
        A1 = np.full((2, 2, 2), 0.5)
        A1[:, 0, 1] = [0.9, 0.1]
        A1[:, 1, 1] = [0.1, 0.9]

        pA0 = np.full((2, 2, 2), 500.0)
        pA1 = np.full((2, 2, 2), 500.0)
        # Sensor mode 1 is unexplored in modality 1 (low Dirichlet counts)
        pA1[:, :, 1] = 1.0
        model = FactoredGenerativeModel(
            A_modalities=[A0, A1],
            B_factors=[B_obj, B_mode],
            pA=[pA0, pA1],
            num_states=[2, 2],
        )
        agent = MultiModalEFPAgent(model)
        plan_before = agent.plan([[(0, 0)], [(0, 1)]])
        assert plan_before["best_action"] == [0, 1]
        assert plan_before["per_modality"][1]["novelty"] > 0.1

        # Observe repeatedly under mode 1; novelty in modality 1 must decrease
        _, tr_before = expected_free_energy_multimodal(
            model, agent.belief_factors, [(0, 1)], pA=agent.pA
        )
        for _ in range(10):
            agent.update_belief([0, 0], action_indices=(0, 1), learning_rate=5.0)
            # Reset object factor prior to compare novelty cleanly at same uncertainty
            agent.belief_factors[0] = np.array([0.5, 0.5])
            agent.joint_belief = np.outer(agent.belief_factors[0], agent.belief_factors[1])
        _, tr_after = expected_free_energy_multimodal(
            model, agent.belief_factors, [(0, 1)], pA=agent.pA
        )
        assert tr_after["per_modality_totals"][1]["novelty"] < tr_before["per_modality_totals"][1]["novelty"]

    def test_2d_joint_a_matrix_and_validation(self):
        B_f0 = np.eye(2)[:, :, None]
        B_f1 = np.eye(2)[:, :, None]
        A_joint_2d = np.eye(4)
        model = FactoredGenerativeModel(
            A_modalities=[A_joint_2d],
            B_factors=[B_f0, B_f1],
            num_states=[2, 2],
        )
        assert model.A_modalities[0].shape == (4, 2, 2)
        with pytest.raises(ValueError):
            FactoredGenerativeModel([np.eye(3)], [B_f0, B_f1], num_states=[2, 2])
        with pytest.raises(ValueError):
            FactoredGenerativeModel([A_joint_2d], [B_f0, B_f1], num_states=[2, 3])


class TestEpistemicVLAGating:
    def _safe_filter(self) -> StateCBFSafetyFilter:
        return StateCBFSafetyFilter(
            ControlAffineDynamics(2, 2),
            [HalfPlaneBarrier([0.0, 1.0], -0.5)],
            u_min=[-1.0, -1.0],
            u_max=[1.0, 1.0],
        )

    def _unsafe_filter(self) -> StateCBFSafetyFilter:
        return StateCBFSafetyFilter(
            ControlAffineDynamics(2, 2),
            [SphereBarrier(center=[5.0, 5.0], radius=0.1)],
            u_min=[-0.05, -0.05],
            u_max=[0.05, 0.05],
        )

    def test_default_flag_preserves_three_tier_escalation(self):
        agent = EFPAgent(_epistemic_probe_model())
        conformal = ConformalActionGate(alpha=0.05)
        conformal.calibrate([0.1] * 30)

        gate = VLActionGate(
            skill_library=SkillLibrary(),
            safety_filter=self._safe_filter(),
            action_gate=conformal,
            deliberator=agent,
        )
        assert gate.enable_epistemic_probe is False
        out = gate.gate([0.0, 0.0], [0.0, 0.0], [0.1, 0.1], risk_score=0.9)
        assert out["decision"] == "escalate_human"

    def test_epistemic_probe_triggered_when_conformal_uncertain_and_cbf_safe(self):
        agent = EFPAgent(_epistemic_probe_model())
        conformal = ConformalActionGate(alpha=0.05)
        conformal.calibrate([0.1] * 30)

        probes = {
            "stay": {"action_idx": 0, "label": "hold_still", "action": [0.0, 0.0]},
            "peek": {"action_idx": 1, "label": "active_peek", "action": [0.25, 0.1]},
        }
        gate = VLActionGate(
            skill_library=SkillLibrary(),
            safety_filter=self._safe_filter(),
            action_gate=conformal,
            deliberator=agent,
            enable_epistemic_probe=True,
            epistemic_probes=probes,
            epistemic_entropy_threshold=0.05,
        )
        out = gate.gate([0.0, 0.0], [0.0, 0.0], [0.1, 0.1], risk_score=0.9)
        assert out["decision"] == "epistemic_probe"
        assert out["probe_label"] == "active_peek"
        assert out["probe_action"] == pytest.approx([0.25, 0.1], abs=1e-6)
        assert out["action_out"] == pytest.approx([0.25, 0.1], abs=1e-6)
        assert out["efe_deliberation"]["best_action"] == 1
        assert out["efe_deliberation"]["info_gain"] > 0.4
        assert gate.decision_counts["epistemic_probe"] == 1

    def test_epistemic_probe_falls_back_to_escalate_when_cbf_infeasible(self):
        agent = EFPAgent(_epistemic_probe_model())
        conformal = ConformalActionGate(alpha=0.05)
        conformal.calibrate([0.1] * 30)

        gate = VLActionGate(
            skill_library=SkillLibrary(),
            safety_filter=self._unsafe_filter(),
            action_gate=conformal,
            deliberator=agent,
            enable_epistemic_probe=True,
        )
        out = gate.gate([0.0, 0.0], [5.0, 5.0], [0.0, 0.0], risk_score=0.9)
        assert out["decision"] == "escalate_human"
        assert out["reason"] == "cbf_infeasible"
        assert gate.decision_counts["escalate_human"] == 1
        assert gate.decision_counts["epistemic_probe"] == 0

    def test_epistemic_probe_falls_back_to_escalate_when_no_informative_probe(self):
        A_uninf = np.full((2, 2), 0.5)
        B_uninf = np.eye(2)[:, :, None]
        agent = EFPAgent(GenerativeModel(A_uninf, B_uninf))
        conformal = ConformalActionGate(alpha=0.05)
        conformal.calibrate([0.1] * 30)

        gate = VLActionGate(
            skill_library=SkillLibrary(),
            safety_filter=self._safe_filter(),
            action_gate=conformal,
            deliberator=agent,
            enable_epistemic_probe=True,
            epistemic_entropy_threshold=0.05,
        )
        out = gate.gate([0.0, 0.0], [0.0, 0.0], [0.1, 0.1], risk_score=0.9)
        assert out["decision"] == "escalate_human"
        assert gate.decision_counts["epistemic_probe"] == 0

    def test_smooth_action_chunk_fallback_to_clamp(self):
        gate = VLActionGate(safety_filter=self._safe_filter())
        out = gate.smooth_action_chunk([0.0, 0.0], [[0.0, 0.2]] * 4, dt=0.05)
        assert out["all_verified_safe"] is True
        assert len(out["per_step"]) == 4

    def test_epistemic_probe_with_multimodal_deliberator(self):
        B_obj = np.eye(2)[:, :, None]
        B_mode = np.zeros((3, 3, 3), dtype=np.float64)
        for u in range(3):
            B_mode[u, :, u] = 1.0
        A_vis = np.full((2, 2, 3), 0.5)
        A_vis[:, 0, 1] = [0.9, 0.1]
        A_vis[:, 1, 1] = [0.1, 0.9]
        A_tac = np.full((2, 2, 3), 0.5)
        A_tac[:, 0, 2] = [0.99, 0.01]
        A_tac[:, 1, 2] = [0.01, 0.99]
        mm_model = FactoredGenerativeModel(
            A_modalities=[A_vis, A_tac],
            B_factors=[B_obj, B_mode],
            num_states=[2, 3],
        )
        mm_agent = MultiModalEFPAgent(mm_model)
        conformal = ConformalActionGate(alpha=0.05)
        conformal.calibrate([0.1] * 30)

        probes = {
            "look": {"policy": [(0, 1)], "label": "camera_glance", "action": [0.1, 0.0]},
            "poke": {"policy": [(0, 2)], "label": "tactile_poke", "action": [0.0, 0.2]},
        }
        gate = VLActionGate(
            skill_library=SkillLibrary(),
            safety_filter=self._safe_filter(),
            action_gate=conformal,
            deliberator=mm_agent,
            enable_epistemic_probe=True,
            epistemic_probes=probes,
        )
        out = gate.gate([0.0, 0.0], [0.0, 0.0], [0.1, 0.1], risk_score=0.9)
        assert out["decision"] == "epistemic_probe"
        assert out["probe_label"] == "tactile_poke"
        assert out["probe_action"] == pytest.approx([0.0, 0.2], abs=1e-6)
        assert out["efe_deliberation"]["info_gain"] > 0.5

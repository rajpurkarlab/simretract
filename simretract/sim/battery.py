"""Load battery: a verdict on how well the gripper holds the tool.

Applied to the still-held tool AFTER the operator hand released it:
six 1.5 N impulses at the blade tip, six 0.03 N*m torques, then five
sustained 0.3 N pulls (ramped). ``strict_pass`` requires that the tool
never left the gripper and moved less than 10 mm / 10 deg in the world and
slipped less than 10 mm in the gripper frame. The strict label defined the
first-generation training pools; the sim benchmark still reports it.

Import only after ``SimulationApp`` exists (uses the scene module).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from simretract.sim import scene


def run_battery(simulation_app, tool_rb, hand, usd_stage, tip_local,
                on_step: Callable[[], None] | None = None) -> dict:
    """Run the battery and return the verdict dict.

    ``on_step`` is called after every simulator update (the benchmark uses
    it to keep capturing frames while the battery runs).
    """
    def tool_rel():
        M_l6 = scene.world_transform(usd_stage, "/World/wxai/link_6")
        tp, _tq = tool_rb.get_world_poses()
        tp = np.asarray(tp).ravel()
        return M_l6[:3, :3].T @ (tp - M_l6[:3, 3])

    p0, q0 = tool_rb.get_world_poses()
    p0 = np.asarray(p0).ravel().copy()
    q0 = np.asarray(q0).ravel().copy()
    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
    rel_p0 = tool_rel()
    state = {"t": 0.0, "r": 0.0, "slip": 0.0, "left": False}

    def measure():
        tp, tq = tool_rb.get_world_poses()
        tp = np.asarray(tp).ravel()
        tq = np.asarray(tq).ravel()
        r_now = Rotation.from_quat([tq[1], tq[2], tq[3], tq[0]])
        state["t"] = max(state["t"], float(np.linalg.norm(tp - p0)))
        state["r"] = max(state["r"], float(np.degrees(
            np.linalg.norm((r_now * r0.inv()).as_rotvec()))))
        slip = float(np.linalg.norm(tool_rel() - rel_p0))
        state["slip"] = max(state["slip"], slip)
        if state["t"] > 0.15 or slip > 0.03:
            state["left"] = True

    def steps(n, every=5):
        for s in range(n):
            simulation_app.update()
            if on_step is not None:
                on_step()
            if s % every == 0:
                measure()
            if state["left"]:
                return False
        return True

    for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
              (0, 0, 1), (0, 0, -1)]:
        hand.extra_push = (1.5 * np.array(d, dtype=float), tip_local)
        ok = steps(4)
        hand.extra_push = None
        if not ok or not steps(13):
            break
    if not state["left"]:
        for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)]:
            hand.extra_torque = 0.03 * np.array(d, dtype=float)
            ok = steps(4)
            hand.extra_torque = None
            if not ok or not steps(13):
                break
    if not state["left"]:
        for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)]:
            dvec = np.array(d, dtype=float)
            for ramp in range(6):
                hand.extra_push = (0.3 * (ramp + 1) / 6 * dvec, tip_local)
                if not steps(7):
                    break
            if not state["left"]:
                hand.extra_push = (0.3 * dvec, tip_local)
                steps(60)
            hand.extra_push = None
            if state["left"]:
                break
            steps(13)
    hand.extra_push = None
    hand.extra_torque = None
    strict = (not state["left"]) and state["t"] < 0.010 \
        and state["r"] < 10.0 and state["slip"] < 0.010
    return {"strict_pass": bool(strict), "left_gripper": bool(state["left"]),
            "max_world_t_mm": round(state["t"] * 1000, 2),
            "max_world_r_deg": round(state["r"], 2),
            "max_slip_mm": round(state["slip"] * 1000, 2)}

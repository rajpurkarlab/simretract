"""Perturb-and-recover expert recorder (recovery data).

Replays a strict seed's scene and its stored, validated grasp, drives the
nominal reach to a random fraction of the path (not captured), then a
smooth excursion to a random off-path pose (position offset + rotation +
optional partial grip close), a short hold, and finally a FRESH reach from
the perturbed pose back to the same grasp; only this recovery is recorded.
The result looks like a normal approach that merely starts somewhere
unusual, which is what a policy needs to learn recovery from.

Campaign enumeration uses virtual indices: ``v -> (sorted(seeds)[v // draws],
draw = v % draws)``; every strict seed gets ``draws`` independent
perturbations, each from its own RNG stream ``(seed * 100 + draw) ^ PERT``.
Episodes are written as ``ep_<seed>_p<draw>/``; rejects leave
``.filtered``, stalls ``.wedged``; the ``[M3] {json}`` outcome line carries
``kept``.

Run as ``<sim python> -m simretract.sim.recovery_recorder --out DIR
--strict-json ... --winners ... --virt 0,1,2`` (Isaac Sim boots in ``main``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from typing import Any

import numpy as np

from simretract.sim.recorder import (
    EE_TO_TCP, OUTCOME_TAG, STEP_NOMINAL_M, EpisodeCapture, EpisodeStalled, RecorderConfig,
    SceneDraw, SimSession, arm_watchdog, emit, episode_done, finish, install_watchdog,
    maybe_review, parse_seed_list)
from simretract.sim.sampling import PERTURB_STREAM_TAG
from simretract.sim.tool_usd import require_eula

log = logging.getLogger(__name__)

RECOVERY_DEFAULTS = {"tries": 10, "episode_timeout_s": 900}


def virtual_to_episode(virt: int, seeds_sorted: list[int], draws: int) -> tuple[int, int]:
    """Virtual index -> (strict seed, draw)."""
    return seeds_sorted[virt // draws], virt % draws


def episode_name(seed: int, draw: int) -> str:
    return f"ep_{seed}_p{draw}"


def draw_perturbation(seed: int, draw: int, frac_lo: float, frac_hi: float) -> dict[str, Any]:
    """The excursion for (seed, draw): start fraction along the reach, an
    offset whose magnitude shrinks as the start nears the grasp, a random
    rotation of 8-25 deg and a 40 % chance of a partial grip close."""
    prng = np.random.default_rng((seed * 100 + draw) ^ PERTURB_STREAM_TAG)
    frac = float(prng.uniform(frac_lo, frac_hi))
    mag = float(prng.uniform(0.03, 0.08) * (1.25 - frac))
    dpos = prng.normal(size=3)
    dpos = dpos / np.linalg.norm(dpos) * mag
    ax = prng.normal(size=3)
    drot = ax / np.linalg.norm(ax) * np.radians(prng.uniform(8, 25))
    return {"frac": frac, "dpos": dpos.tolist(), "drot": drot.tolist(),
            "grip": -0.02 if prng.random() < 0.4 else 0.0}


class RecoverySession(SimSession):
    """Session whose blended reach can stop part-way and whose try inserts
    the excursion + recovery before the close."""

    @staticmethod
    def _bez_table(p0, p_ctrl, p1):
        ts = np.linspace(0, 1, 400)
        pts = np.stack([((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p_ctrl
                        + (t ** 2) * p1 for t in ts])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return ts, np.concatenate([[0.0], np.cumsum(seg)])

    def move_ee_blend(self, p0, q0, p_ctrl, p1, q1, probes, cap=None, stop_frac=1.0):
        """Bezier ``p0 -> ctrl -> p_end`` (``standoff`` short of ``p1`` on the
        approach axis) + straight entry ``p_end -> p1``, walked as one
        combined arc so ``stop_frac`` keeps the perturbation-fraction
        semantics. ``stop_frac < 1`` returns (None, pos, rot) at the stop,
        otherwise (reject_or_None, None, None)."""
        cfg = self.cfg
        R, Slerp = self.Rotation, self.Slerp
        app = p1 - p_ctrl
        an = float(np.linalg.norm(app))
        if cfg.standoff > 0 and an > 1e-6:
            axis = app / an
            stand = min(cfg.standoff, 0.8 * an)
            p_end = p1 - axis * stand
            p_ctrl = p_end - axis * cfg.ctrl_back
        else:
            stand = 0.0
            p_end = p1

        def bez(t):
            return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p_ctrl + (t ** 2) * p_end
        ts, cum = self._bez_table(p0, p_ctrl, p_end)
        curve_len = float(cum[-1])
        total = curve_len + stand
        target = total * stop_frac
        sl = Slerp([0, 1], R.concatenate([q0, q1]))
        step_nom = STEP_NOMINAL_M * cfg.speed_scale
        rot_frac = cfg.blend_rot_frac
        q_last, p_last = None, None
        s_arc, k = 0.0, 0
        jr = {"q": None}
        max_it = int(target / (step_nom * min(cfg.curve_end, cfg.entry_speed))) + 40
        while True:
            if s_arc <= curve_len:
                t = float(np.interp(s_arc, cum, ts))
                if stand > 0:
                    a = min(1.0, (t * curve_len / max(total, 1e-9))
                            / max(rot_frac * curve_len / max(total, 1e-9), 1e-6))
                else:
                    a = min(1.0, t / max(rot_frac, 1e-6))
                rot = sl(min(1.0, a))
                p_last = bez(t)
            else:                        # straight final entry, orientation done
                rot = q1
                p_last = p_end + (app / an) * (s_arc - curve_len)
            q = rot.as_quat()
            q_last = rot
            self.set_ee_smooth(p_last, np.array([q[3], q[0], q[1], q[2]]), jr, cap)
            self.update(cap)
            if self.contact_mon.contacting_bodies():
                return "tool_contact", None, None
            if k % 2 == 0:
                if self.scene.clearance_violations(self.tool_rb, probes, phase="reach"):
                    return "occluder_interference", None, None
                if self.tool_disturbed():
                    return "interference_pose", None, None
            if s_arc >= target or k >= max_it:
                break
            if s_arc <= curve_len:
                step = step_nom * self.curve_step_scale(curve_len - s_arc)
            else:
                step = step_nom * cfg.entry_speed
            s_arc = min(target, s_arc + step)
            k += 1
        # finalize: step the rate-limited command until it converges on the
        # last target (command-space check, open loop)
        if q_last is not None:
            qg = q_last.as_quat()
            qf = np.array([qg[3], qg[0], qg[1], qg[2]])
            for _ in range(60):
                res = self.set_ee_smooth(p_last, qf, jr, cap)
                self.update(cap)
                if self.contact_mon.contacting_bodies():
                    return "tool_contact", None, None
                if res < 1e-4:
                    break
        for _ in range(30):
            self.update(cap)
            if self.contact_mon.contacting_bodies():
                return "tool_contact", None, None
        return None, p_last, q_last

    def run_candidate(self, cand, Rg_mat, grasp_w, probes, tool_pos, tool_quat, cap=None,
                      perturb=None, repick=None, ref_gap=None):
        """One try: reach -> [excursion -> recovery] -> close -> press ->
        release. Returns (ok, info). With ``perturb`` the nominal reach is
        interrupted; ``repick(pe)`` may switch the target grasp when the
        straight path from the perturbed pose is blocked; ``ref_gap`` is the
        original episode's achieved gap for the width check."""
        cfg = self.cfg
        R, Slerp = self.Rotation, self.Slerp
        R_grasp, grasp_ee, pre_ee, ee_p, R_now = self.begin_try(
            Rg_mat, grasp_w, cand, tool_pos, tool_quat)
        pwin = None
        if perturb is None:
            if cfg.blend:
                why, _p, _r = self.move_ee_blend(ee_p, R_now, pre_ee, grasp_ee, R_grasp, probes, cap)
            else:
                why = self.move_ee(ee_p, R_now, pre_ee, R_grasp, probes, cap)
            if why:
                return False, {"reject": why}
        else:
            # 1) nominal approach up to a random fraction of the reach path
            f = perturb["frac"]
            if cfg.blend:
                why, mid_p, R_mid = self.move_ee_blend(ee_p, R_now, pre_ee, grasp_ee, R_grasp,
                                                       probes, None, stop_frac=f)
                if why:
                    return False, {"reject": why}
            else:
                mid_p = ee_p + f * (pre_ee - ee_p)
                sl = Slerp([0, 1], R.concatenate([R_now, R_grasp]))
                R_mid = sl(f)
                # the pre-perturb approach is not captured: training uses only
                # the post-perturb segment
                why = self.move_ee(ee_p, R_now, mid_p, R_mid, probes, None)
                if why:
                    return False, {"reject": why}
            # 2) SMOOTH excursion to the perturbed pose (a drift, not a jump)
            pwin = {"start": cap.t if cap else -1}
            pert_p = mid_p + np.asarray(perturb["dpos"])
            # reachability clamp: an out-of-workspace target can stall the IK
            # solver indefinitely; shrink the drift until the target is sane
            for _ in range(8):
                if (np.linalg.norm(pert_p) <= 0.58 and pert_p[2] >= 0.06
                        and pert_p[0] >= 0.05):
                    break
                pert_p = mid_p + (pert_p - mid_p) * 0.7
            pert_R = R.from_rotvec(perturb["drot"]) * R_mid
            why = self.move_ee(mid_p, R_mid, pert_p, pert_R, probes, cap)
            if why == "tool_contact":
                return False, {"reject": "excursion_tool_contact"}
            # occluder interference during the drift only ends the excursion
            # early at the clearance boundary; it is not a rejection
            if perturb.get("grip", 0.0) != 0.0:
                self.robot.set_gripper_position(
                    float(np.clip(0.044 + perturb["grip"], 0.0, 0.044)))
            for _ in range(15):  # brief hold at the perturbed pose
                self.update(cap)
            pwin["end"] = cap.t if cap else -1
            # 3) expert RECOVERY: reopen, re-plan by IK from wherever we are
            if perturb.get("grip", 0.0) != 0.0:
                self.robot.set_gripper_position(0.044)
            _d, pe, pq = self.robot.get_current_state()
            pe = np.asarray(pe).ravel()[:3]
            pqv = np.asarray(pq).ravel()
            R_pe = R.from_quat([pqv[1], pqv[2], pqv[3], pqv[0]])
            if repick is not None:
                sw = repick(pe)
                if sw is not None:
                    cand, Rg_mat, grasp_w = sw
                    ref_gap = None  # switched target: use ITS nominal width
                    R_grasp = R.from_matrix(Rg_mat)
                    approach = Rg_mat[:, 0]
                    grasp_ee = grasp_w - R_grasp.apply(EE_TO_TCP)
                    pre_ee = grasp_ee - approach * self.standoff_of(cand)
            if cfg.blend:
                why, _p, _r = self.move_ee_blend(pe, R_pe, pre_ee, grasp_ee, R_grasp, probes, cap)
                if why:
                    return False, {"reject": f"recovery_{why}"}
            else:
                why = self.move_ee(pe, R_pe, pre_ee, R_grasp, probes, cap)
                if why:
                    return False, {"reject": f"recovery_{why}"}
        if not cfg.blend:   # blended reaches above already end at the grasp
            why = self.move_ee(pre_ee, R_grasp, grasp_ee, R_grasp, probes, cap)
            if why:
                return False, {"reject": why}
        err, _ee_now = self.settle_at_grasp(grasp_ee, R_grasp, cap)
        if err > 0.012:
            return False, {"reject": "unreachable"}
        # reference = the ORIGINAL episode's physically achieved gap when
        # available (the nominal candidate width sits at the tolerance edge
        # for some poses); only meaningful when the recovery kept the target
        ref = ref_gap if ref_gap is not None else cand["width"]
        why, m = self.close_press_release(ref, probes, tool_pos, tool_quat, cap)
        if why:
            return False, {"reject": why}
        return m["drop_mm"] < 20.0, {
            "gap_mm": m["gap_m"] * 1000, "reach_err_mm": err * 1000,
            "drop_mm": m["drop_mm"], "grasp_w": grasp_w.tolist(),
            "R_grasp": Rg_mat.tolist(), "perturb_win": pwin,
            "pose_t_mm": m["pose_t_mm"], "pose_r_deg": m["pose_r_deg"]}


def make_repick(session: RecoverySession, sd: SceneDraw, winner, pool, tries: int):
    """Geometric re-targeting from the perturbed pose: keep the original
    grasp if its straight path is clear (>= 2 cm from the 6 mm-inflated
    occluder shapes), else the first pool candidate (quality order) whose
    path is clear. One recorded attempt, no physical retries."""
    R = session.Rotation
    R_tool, tool_pos = sd.R_tool, sd.tool_pos
    subs = sd.spec["hand"]["subs"] + sd.spec["tip"]["subs"]

    def path_clear(pe, c, Rg, gw):
        Rgr = R.from_matrix(Rg)
        g_ee = gw - Rgr.apply(EE_TO_TCP)
        p_ee = g_ee - Rg[:, 0] * session.standoff_of(c)
        pts_w = pe + np.linspace(0, 1, 8)[:, None] * (p_ee - pe)
        pts_t = R_tool.inv().apply(pts_w - tool_pos)
        mind = np.inf
        for s in subs:
            sq = np.asarray(s["quat_wxyz"])
            R_s = R.from_quat([sq[1], sq[2], sq[3], sq[0]])
            local = R_s.inv().apply(pts_t - np.asarray(s["pos"]))
            d = np.linalg.norm(np.maximum(
                np.abs(local) - (np.asarray(s["half"]) + 0.006), 0.0), axis=1)
            mind = min(mind, float(d.min()))
        return mind >= 0.02

    def repick(pe):
        if path_clear(pe, winner[0], winner[1], winner[2]):
            return None
        for _cd, c, Rg, gw in pool[:tries]:
            if c is winner[0]:
                continue
            if path_clear(pe, c, Rg, gw):
                return (c, Rg, gw)
        return None
    return repick


def record_draw(session: RecoverySession, seed: int, draw: int, virt: int, out_root: str,
                winners_root: str) -> dict[str, Any]:
    """Record one (seed, draw); emits and returns the outcome row."""
    cfg = session.cfg
    sd = session.draw_scene(seed)
    row = sd.row()
    row.update({"draw": draw, "virt": virt})
    ep_dir = os.path.join(out_root, episode_name(seed, draw))

    def done(r):
        emit(r)
        return r

    if session.scene_rejected(sd):
        row["outcome"] = "scene_reject"
        return done(row)

    # the recovery recorder gates with the clearance picker constants
    pool, _gates, _corridor, _th = session.build_pool(
        sd, height_picker=False, tries=cfg.tries, upright_min=0.0,
        z_lo=cfg.z_lo, z_hi=cfg.z_hi, z_strict=False)

    winner, orig_gap = None, None
    try:
        m = session.match_stored_winner(
            os.path.join(winners_root, f"ep_{seed}", "meta.npz"), sd)
        if m is not None:
            c, Rg0, gw0, _dist, oi = m
            winner = (c, Rg0, gw0)
            orig_gap = float(oi.get("gap_mm", 0.0)) / 1000.0 or None
    except Exception as e:  # noqa: BLE001  (fall back to the search)
        print(f"{OUTCOME_TAG} stored-winner load failed: {e!r}", flush=True)
    if winner is None:
        for _cd, cand, Rg, gw in pool[:cfg.tries]:
            ok, info = session.run_candidate(cand, Rg, gw, sd.probes, sd.tool_pos, sd.tool_quat)
            if ok:
                winner = (cand, Rg, gw)
                break
    if winner is None:
        row["outcome"] = "no_grasp"
        return done(row)

    perturb = draw_perturbation(seed, draw, cfg.pert_frac_lo, cfg.pert_frac_hi)
    repick = make_repick(session, sd, winner, pool, cfg.tries)
    cap = EpisodeCapture(session, ep_dir)
    arm_watchdog(cfg.episode_timeout_s)  # re-arm for the recorded pass
    try:
        ok, info = session.run_candidate(*winner, sd.probes, sd.tool_pos, sd.tool_quat,
                                         cap=cap, perturb=perturb, repick=repick,
                                         ref_gap=orig_gap)
    except EpisodeStalled:
        arm_watchdog(0)
        cap.close()
        shutil.rmtree(ep_dir, ignore_errors=True)
        open(ep_dir + ".wedged", "w").close()  # never retry a stalled draw
        row["outcome"] = "episode_timeout"
        return done(row)
    arm_watchdog(cfg.episode_timeout_s)  # re-arm for battery + review
    cap.close()
    if ok:
        info["perturb"] = perturb
        info["switched_grasp"] = bool(not np.allclose(info["grasp_w"], winner[2]))
    if not ok:
        shutil.rmtree(ep_dir, ignore_errors=True)  # rejects keep no frames
        row["outcome"] = f"record_divergence:{info.get('reject', 'dropped')}"
        return done(row)

    if cfg.skip_battery:
        verdict = {"battery_skipped": True, "strict_pass": None}
    else:
        verdict = session.run_battery()
    session.save_meta(ep_dir, cap, sd, info, verdict, row["mass_g"])
    row.update({"outcome": "recorded", "frames": cap.t,
                "gap_mm": round(info["gap_mm"], 2), "verdict": verdict})
    if cfg.keep_mode == "pose":
        kept = (float(info.get("pose_t_mm", 99)) < 10.0
                and float(info.get("pose_r_deg", 99)) < 10.0)
    else:
        kept = bool(verdict.get("strict_pass"))
    row["kept"] = kept
    row["pose_t_mm"] = info.get("pose_t_mm")
    row["pose_r_deg"] = info.get("pose_r_deg")
    done(row)
    if kept:
        maybe_review(out_root, ep_dir, ".keep_count",
                     f"{episode_name(seed, draw)}_review.mp4", cfg.review_every)
    else:
        shutil.rmtree(ep_dir, ignore_errors=True)
        open(ep_dir + ".filtered", "w").close()  # done-marker for retries
    return row


def run(cfg: RecorderConfig, virts: list[int], seeds_sorted: list[int], draws: int,
        out_root: str, winners_root: str) -> None:
    """Boot Isaac Sim and record the virtual indices into ``out_root``."""
    require_eula()
    os.makedirs(out_root, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    session = RecoverySession(app, cfg)
    install_watchdog()
    results: list[dict] = []
    for virt in virts:
        seed, draw = virtual_to_episode(virt, seeds_sorted, draws)
        ep_dir = os.path.join(out_root, episode_name(seed, draw))
        if episode_done(ep_dir):
            continue  # done in an earlier try; retries only run the gaps
        if os.path.isdir(ep_dir):  # stale partial from a killed run
            shutil.rmtree(ep_dir, ignore_errors=True)
        # whole-episode watchdog: a stall kills the draw, not the chunk
        arm_watchdog(cfg.episode_timeout_s)
        try:
            row = record_draw(session, seed, draw, virt, out_root, winners_root)
        except EpisodeStalled:
            shutil.rmtree(ep_dir, ignore_errors=True)
            open(ep_dir + ".wedged", "w").close()
            row = {"seed": seed, "draw": draw, "virt": virt, "outcome": "episode_timeout"}
            emit(row)
        finally:
            arm_watchdog(0)
        results.append(row)
    finish(results, out_root, f"recorder_log_v{virts[0] if virts else 'empty'}.jsonl")


def load_strict_seeds(strict_json: str) -> list[int]:
    with open(strict_json) as f:
        return sorted(int(s) for s in json.load(f)["seeds"])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Perturb-and-recover expert recorder (runs inside Isaac Sim).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, help="raw episode root (ep_<seed>_p<draw>/ dirs)")
    ap.add_argument("--strict-json", required=True,
                    help="strict selection JSON whose sorted seeds index the virtual list (env SEED_JSON)")
    ap.add_argument("--winners", required=True,
                    help="strict episode root providing ep_<seed>/meta.npz stored grasps (env STORED_WINNER_ROOT)")
    ap.add_argument("--draws", type=int, default=int(os.environ.get("DRAWS", "4")),
                    help="perturbation draws per seed: v -> (seed[v // draws], v %% draws) (env DRAWS)")
    ap.add_argument("--virt", default=os.environ.get("VIRT_LIST", ""),
                    help="comma-separated virtual indices (env VIRT_LIST)")
    RecorderConfig.add_args(ap, defaults=RECOVERY_DEFAULTS)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = RecorderConfig.from_args(args)
    virts = parse_seed_list(args.virt)
    if not virts:
        raise SystemExit("no virtual indices: pass --virt")
    run(cfg, virts, load_strict_seeds(args.strict_json), args.draws, args.out, args.winners)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

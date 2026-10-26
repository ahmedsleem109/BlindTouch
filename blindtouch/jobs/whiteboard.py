"""Job 7: erase a sloped desk whiteboard at steady pressure (blindtouch.scenes.job_scenes kind "whiteboard", eraser in hand).

The board's slope (15 +- 4 deg), height (+-10 mm), place (+-2 cm) and facing (+-10 deg) are unknown to the robot;
it only knows roughly where the board is (a glance before it starts). Procedure, as on the design sheet:

1. Feel the board at four points (the eraser tilted to the usual 15 deg): the plane gives the slope, its facing
   and the height. Any three points define a plane; the robot keeps the one the remaining point agrees with (a
   touch fooled by joint friction is an outlier), touching a fifth point if none agrees within 2 mm.
2. Tilt the eraser flat to the felt plane, long side (nearly) up the slope.
3. Trace the whole wipe 6 mm above the felt plane: what the arm feels there without contact (joint friction of
   this very motion, ripple, model error) is subtracted later. Both passes start from standstill after a fresh
   freeze of the free-space estimate, so their joint friction matches from the first moment.
4. Wipe two overlapping rows across the board, holding 8 N from the felt force (PI on the pressing depth).

The eraser sits 18 deg off the slope line (wrist yaw ~128 deg): with the tool vertical at the start and tilted
on the board, only wrist yaws of ~105-150 deg keep every joint ~10 deg from its end stops here
(scenes.check.wrist_window) -- a joint on its stop adds a torque the motors 'feel' as a phantom force.

Success (simulator truth): the true normal force (20 ms average) stays within 25 % of 8 N (RMS) over the strokes,
and at least 90 % of the ink is wiped (ink counts as wiped when the felt passed over it pressing with > 2 N).
"""

from __future__ import annotations

import math

import numpy as np

from blindtouch.scenes.build import GEOM
from blindtouch.scenes.job_scenes import TABLE, WHITEBOARD_TILT, WHITEBOARD_Z
from blindtouch.jobs.base import Job, budget, step
from blindtouch.skills.touch_robot import KP, TouchRobot

F_DES = 8.0
AIR_GAP = 0.006      # m above the felt plane for the air pass: close to the wipe's joint angles, clear of contact (the
                     # touches log the plane 1-3 mm low: the soft felt is compressed by the time the tool has stopped)
SPEED = 0.05         # m/s along the strokes
RAMP_S = 0.6         # s: the wipe starts like the air pass (from standstill, 6 mm up) and presses in over this
APPROACH = 0.03     # m: touches start this far above where the board usually is (height known to +-1 cm)
HALF_ACROSS = 0.11   # m: strokes run from -HALF_ACROSS to +HALF_ACROSS across the board
ROWS = (-0.0325, 0.0325)   # m up the slope: two rows of the 130 mm eraser overlap in the middle
HEADING = math.atan2(GEOM.hole_nominal[1], GEOM.hole_nominal[0])   # the board faces the robot: rises away from it
YAW_OFF = math.radians(-18)   # eraser turn off the slope line: keeps the wrist in its window (see above)


TOUCH_POINTS = ((-0.03, -0.08), (-0.03, 0.08), (0.035, -0.06), (0.035, 0.06), (0.0, 0.0))   # (up, across) m
PLANE_TOL = 0.002    # m: a touch farther than this from the plane of three others is an outlier


def fit_plane(pts) -> tuple[np.ndarray, np.ndarray, float]:
    """(upward normal, a point on it, disagreement): of all planes through three of the touched points, the one
    whose worst remaining point lies closest (exhaustive RANSAC -- there are at most ten triples)."""
    from itertools import combinations

    best = (None, None, np.inf)
    for tri in combinations(range(len(pts)), 3):
        a, b, c = (pts[i] for i in tri)
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n) * np.sign(n[2])
        rest = [abs((pts[i] - a) @ n) for i in range(len(pts)) if i not in tri]
        # with 5 points, the best plane may leave one outlier out: judge it by its best remaining point
        worst = (sorted(rest)[0] if len(pts) == 5 else max(rest)) if rest else 0.0
        if worst < best[2]:
            best = (n, a, worst)
    return best


class WhiteboardJob(Job):
    name, scene, title = "whiteboard", "whiteboard", "Erase a sloped whiteboard"
    max_time = 70.0
    RANGES = {"tilt_deg": 4.0, "dz": 0.010, "yaw_deg": 10.0, "xy": 0.02, "board_mu": (0.30, 0.45), "felt_mu": (0.40, 0.55)}
    KI_Z_FREE = 0.0         # no vertical free-space integral: the press here is along the sloped board's normal, and
                            # the integral pulled against it (4 of 50 seeds lost the wipe with it on, all with joint friction)
    ki, kpf = 1.5, 0.4      # PI on the felt normal force

    # ------------------------------------------------------------------ world
    def setup(self, env, rng, seed):
        m = env.model
        R = self.ranges
        tilt = WHITEBOARD_TILT + math.radians(rng.uniform(-R["tilt_deg"], R["tilt_deg"]))
        dz = rng.uniform(-R["dz"], R["dz"])
        yaw_f = HEADING + math.radians(rng.uniform(-R["yaw_deg"], R["yaw_deg"]))
        fxy = np.asarray(GEOM.hole_nominal) + rng.uniform(-R["xy"], R["xy"], 2)
        board = m.body("board").id
        m.body_quat[board] = [math.cos(-tilt / 2), 0, math.sin(-tilt / 2), 0]
        m.body_pos[board][2] = WHITEBOARD_Z + dz
        m.geom_friction[m.geom("board").id][0] = rng.uniform(*R["board_mu"])
        m.geom_friction[m.geom("eraser_felt").id][0] = rng.uniform(*R["felt_mu"])
        if not hasattr(self, "_ink"):
            self._ink = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("ink")]
            self._ink_mat = m.geom_matid[self._ink].copy()
            self._ink_rgba = m.geom_rgba[self._ink].copy()
        m.geom_matid[self._ink] = self._ink_mat
        m.geom_rgba[self._ink] = self._ink_rgba
        start = np.array([*GEOM.hole_nominal, TABLE + WHITEBOARD_Z + 0.075])   # above where the board roughly is
        env.reset(seed=seed, tip_start=start, tip_yaw=HEADING + math.pi + YAW_OFF, fixture=(fxy[0], fxy[1], yaw_f))
        d = env.data
        R = d.xmat[board].reshape(3, 3)
        self.n_true = R[:, 2].copy()
        self.felt_gid = m.geom("eraser_felt").id
        self.erased = np.zeros(len(self._ink), bool)
        self.truth_f: list = []
        return {"n_true": self.n_true, "tilt_deg": math.degrees(tilt), "dz": dz, "yaw_deg": math.degrees(yaw_f)}

    def _hook(self, env, s) -> None:
        """Truth recorder (evaluation and video only): normal force, and which ink the pressing felt covered."""
        fn = float(s.truth.force @ self.n_true)
        self.truth_f.append((env.t, fn))
        if fn < 2.0 or env.step_count % 5:
            return
        d, m = env.data, env.model
        R = d.geom_xmat[self.felt_gid].reshape(3, 3)
        half = m.geom_size[self.felt_gid][:2] - 0.002
        for k, g in enumerate(self._ink):
            if self.erased[k]:
                continue
            p = R.T @ (d.geom_xpos[g] - d.geom_xpos[self.felt_gid])
            if abs(p[0]) < half[0] and abs(p[1]) < half[1] and abs(p[2]) < 0.012:
                self.erased[k] = True
                m.geom_matid[g] = -1
                m.geom_rgba[g] = [0, 0, 0, 0]

    # ------------------------------------------------------------------ procedure
    def procedure(self, rob: TouchRobot, rng):
        env = self.env
        env.hooks.append(self._hook)
        t_end = rob.t + self.max_time
        start = rob.tip()
        up_h = np.array([math.cos(HEADING), math.sin(HEADING)])     # rough uphill direction (horizontal)
        across_h = np.array([-up_h[1], up_h[0]])
        n0 = np.array([-math.sin(WHITEBOARD_TILT) * up_h[0], -math.sin(WHITEBOARD_TILT) * up_h[1], math.cos(WHITEBOARD_TILT)])

        step(rob, "Feel the board at four points")
        rob.yaw_cmd = rob.yaw()
        env.ctrl.target_axis = -n0                        # the usual slope: touches land nearly flat
        rob.wait(0.6)
        # where the board would be if it sat exactly as usual: each touch starts 3 cm above that (short approaches
        # keep the frozen free-space estimate valid -- a long slanted approach changes the posture too much)
        p_nom = np.array([*GEOM.hole_nominal, TABLE + WHITEBOARD_Z + 0.004])
        pts = []
        for a, b in TOUCH_POINTS:
            if len(pts) == 4 and fit_plane(pts)[2] <= PLANE_TOL:
                break                                    # four points agree: no fifth touch needed
            p = start[:2] + a * up_h + b * across_h
            q = np.r_[p, p_nom[2] - (n0[0] * (p[0] - p_nom[0]) + n0[1] * (p[1] - p_nom[1])) / n0[2]]
            rob.move_to(q + n0 * APPROACH, speed=0.10)
            rob.settle_free(0.3)
            hit = rob.touch_along(n0, speed=0.02, press=1.0, max_travel=APPROACH + 0.03)
            budget(rob, t_end)
            if hit.event != "contact":
                return {"aborted": "board not found"}
            pts.append(hit.tip.copy())
            rob.move_to(hit.tip + n0 * 0.02, speed=0.06)
        n, a, resid = fit_plane(pts)
        rob.log("plane", normal=n.copy(), slope_deg=float(math.degrees(math.acos(n[2]))), disagreement_mm=resid * 1e3)
        if resid > PLANE_TOL:
            return {"aborted": "the touches do not agree on one plane"}

        step(rob, "Tilt the eraser flat to the board")
        up = np.array([0, 0, 1.0]) - n[2] * n               # up the slope, in the board plane
        up /= np.linalg.norm(up)
        across = np.cross(n, up)
        heading_up = math.atan2(up[1], up[0])
        env.ctrl.target_axis = -n
        plane_pt = a                                        # the plane the tip positions span
        def on_plane(xy):
            """Point of the felt plane above/below the horizontal position xy."""
            p = np.r_[xy, plane_pt[2]]
            return p - n * ((p - plane_pt) @ n) / n[2]
        centre = on_plane(start[:2])
        rows = [centre + up * r for r in ROWS]
        wps = [rows[0] - across * HALF_ACROSS, rows[0] + across * HALF_ACROSS,
               rows[1] + across * HALF_ACROSS, rows[1] - across * HALF_ACROSS]
        rob.move_to(wps[0] + n * 0.03, speed=0.08, yaw=heading_up + math.pi + YAW_OFF)
        rob.wait(0.4)

        seg = np.cumsum([0] + [np.linalg.norm(wps[i + 1] - wps[i]) for i in range(len(wps) - 1)])
        L = seg[-1]
        T = L / SPEED

        def path(s):
            d_ = s * L
            i = min(int(np.searchsorted(seg, d_, side="right")) - 1, len(wps) - 2)
            return wps[i] + (wps[i + 1] - wps[i]) * (d_ - seg[i]) / (seg[i + 1] - seg[i])

        step(rob, "Trace the wipe in the air: what the arm feels without contact")
        rob.move_to(path(0.0) + n * AIR_GAP, speed=0.05)
        rob.release_contact()
        rob.settle_free(0.4)
        rob.freeze()
        felt_free = rob.air_pass(path, T, n * AIR_GAP, keep_start=True)
        # back to where the pass began and freeze afresh there, standing still as before the pass: the
        # estimate frozen before the pass is stale by now (the posture moved). The pass profile (joint friction
        # of this very motion, ripple, model error) is relative to such a standstill freeze.
        rob.release_contact()
        rob.move_to(path(0.0) + n * AIR_GAP, speed=0.05)
        rob.settle_free(0.4)
        rob.freeze()
        felt = lambda s: rob.force() - felt_free(s)
        budget(rob, t_end)

        step(rob, "Wipe, holding 8 N against the board")
        rob.hold_contact(snapshot=False)
        integ = 0.0
        t0 = rob.t
        rob.log("sweep_start")
        while True:
            s = min((rob.t - t0) / T, 1.0)
            f_des = F_DES * min(1.0, (rob.t - t0) / RAMP_S)   # pressing builds up while already moving
            fn = float(felt(s) @ n)
            err = f_des - fn
            integ = float(np.clip(integ + self.ki * err * 0.01, -10.0, 10.0))
            depth = (f_des + integ + self.kpf * err) / KP
            rob.cmd = path(s) + n * AIR_GAP * max(0.0, 1 - (rob.t - t0) / RAMP_S) - n * depth
            rob.tick()
            if s >= 1.0:
                break
        rob.log("sweep_end")
        rob.move_to(rob.tip() + n * 0.04, speed=0.05)
        env.ctrl.target_axis = np.array([0, 0, -1.0])
        return {"normal": n}

    def judge(self, env, truth, out):
        env.ctrl.target_axis = np.array([0, 0, -1.0])
        ev = {name: t for t, name, _ in self.rob.events if name in ("sweep_start", "sweep_end")}
        wiped = float(np.mean(self.erased)) if len(self.erased) else 0.0
        if "sweep_end" not in ev:
            return False, {"force_rmse": None, "wiped": wiped, **{k: v for k, v in out.items() if k == "aborted"}}
        # the true normal force as a force sensor would report it: 20 ms average (as the force audit's peak). Raw
        # 2 ms samples flicker by +-2 N as the box-on-box contact switches points -- a contact-model artefact that
        # alone would make a perfect wipe score ~1.9 N RMS. Judged once the press-in ramp is over.
        tf = np.array(self.truth_f)
        f20 = np.convolve(tf[:, 1], np.ones(10) / 10, mode="same")
        sel = (tf[:, 0] >= ev["sweep_start"] + RAMP_S + 0.2) & (tf[:, 0] <= ev["sweep_end"])
        f = f20[sel]
        rmse = float(np.sqrt(np.mean((f - F_DES) ** 2)))
        ang = float(np.degrees(np.arccos(np.clip(out["normal"] @ truth["n_true"], -1, 1))))
        ok = rmse <= 0.25 * F_DES and wiped >= 0.9
        return ok, {"force_rmse": rmse, "wiped": wiped, "normal_err_deg": ang, "contact_frac": float(np.mean(f > 1.0)),
                    "tilt_deg": truth["tilt_deg"]}

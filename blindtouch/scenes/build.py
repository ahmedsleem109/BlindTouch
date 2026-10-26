"""Generate the Blind Touch MJCF scenes.

The arm template (``openarm_right.template.xml``) carries placeholders for the
hand contents and the mount; each scene adds a dark "studio" (floor, lights,
skybox), a table and its task furniture.

Two hands are generated from one template:

* ``openarm_right.xml`` (the true plant): the real OpenArm gripper -- hinged
  fingers with collision, coupled by a mimic constraint, squeezed by a
  position servo -- and the scenes add a *free* peg held by friction alone;
* ``openarm_right_rigid.xml`` (the nominal model the robot believes): fingers
  frozen at the grasp angle, peg rigidly attached as a known payload.

    python -m blindtouch.scenes.build   # writes blindtouch/scenes/peg_scene{,_nominal}.xml

The peg-in-hole scene is the flat test bench of the sensing report; the job
scenes are in :mod:`blindtouch.scenes.job_scenes`. Every scene shares one
geometry module (:data:`GEOM`), so the jobs, the sensing report and the tests
all read sizes from the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

SCENE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Geometry:
    """Every dimension (metres) that more than one module needs."""

    table_top: float = 0.45          # z of the tabletop surface
    table_center: tuple = (0.42, -0.20)
    table_half: tuple = (0.28, 0.35)
    peg_radius: float = 0.012
    peg_halflen: float = 0.035
    # calibrated grasp: where the peg settles in the real gripper (hand frame, measured after
    # the grasp settles, mean over postures). The fingertips end 171 mm below the hand, so the
    # tip sticks out 45 mm below them and a full 30 mm insertion clears the plate.
    peg_offset: float = 0.176        # peg centre below the hand origin along hand -z
    peg_grasp_xy: tuple = (0.0, 0.0)
    hole_half: float = 0.0135        # half-width of the square hole: 1.5 mm clearance
    chamfer: float = 0.006           # 45 degree chamfer width (and depth)
    plate_thick: float = 0.030       # hole depth
    plate_half: float = 0.07         # half-size of the square hole plate
    hole_nominal: tuple = (0.30, -0.20)  # nominal hole centre (x, y) on the table
    mount_z: float = 0.90            # shoulder mount height
    mount_pitch: float = 0.0         # forward tilt of the mount about +y (rad)
    # finger hinge angle holding the 24 mm peg: the pads touch at 0.130 rad (collision meshes) and
    # the servo's squeeze settles them at 0.1259; reset starts there so the grip is loaded at once
    finger_angle: float = 0.1259
    grip_kp: float = 50.0            # Nm/rad: finger servo stiffness (OpenArm v2 model uses 30; 50 squeezes ~6 Nm, under the 7 Nm DM4310 limit)
    peg_tip_chamfer: float = 0.001   # m: 45 deg chamfer on the peg tip (0 = flat-ended cylinder primitive)
    peg_facets: int = 64             # polygon sides of the mesh peg (inscribed: never wider than drawn)

    @property
    def plate_top(self) -> float:
        return self.table_top + self.plate_thick

    @property
    def clearance(self) -> float:
        return self.hole_half - self.peg_radius

    @property
    def capture_radius(self) -> float:
        """Largest lateral offset from which the peg drops into the chamfer funnel.

        The peg's 45 deg tip chamfer slides face-on on the hole's 45 deg chamfer, so it adds its width:
        depth = capture_radius - offset (inf-norm), up to chamfer + peg_tip_chamfer at the clearance.
        """
        return self.hole_half + self.chamfer + self.peg_tip_chamfer - self.peg_radius


GEOM = Geometry()


def _quat_axis_angle(axis, angle):
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def _lookat(pos, target, fovy: float, name: str) -> str:
    """A <camera> at ``pos`` looking at ``target`` with world +z up."""
    p, t = [float(x) for x in pos], [float(x) for x in target]
    f = [t[i] - p[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in f))
    f = [x / n for x in f]
    xa = [f[1], -f[0], 0.0]  # f x up
    n = math.hypot(xa[0], xa[1])
    xa = [x / n for x in xa]
    ya = [xa[1] * f[2] - xa[2] * f[1], xa[2] * f[0] - xa[0] * f[2], xa[0] * f[1] - xa[1] * f[0]]
    return f'<camera name="{name}" pos="{_fmt(p)}" xyaxes="{_fmt(xa + ya)}" fovy="{fovy:g}"/>'


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


FINGERS = {  # name: (body pos in the hand frame, hinge axis, inertial)
    "finger_inner": ("-0.00143 0.018 -0.068", (-1, 0, 0), '<inertial pos="-0.0124991 0.00176593 -0.0241663" mass="0.125979" diaginertia="3.8543e-05 3.6883e-05 1.43942e-05" />'),
    "finger_outer": ("-0.00143 -0.018 -0.068", (1, 0, 0), '<inertial pos="-0.012796 -0.0020183 -0.026792" mass="0.11372" diaginertia="3.70662e-05 3.60159e-05 1.33599e-05" />'),
}
FINGERTIP_Z = -0.171   # the fingertips end this far below the hand origin
PEG_INERTIAL = '<inertial pos="0 0 0" mass="0.06" diaginertia="2.5e-5 2.5e-5 4.3e-6" />'


def peg_geoms(g: Geometry, cls: str) -> str:
    shape = ('type="mesh" mesh="peg_mesh"' if g.peg_tip_chamfer > 0
             else f'type="cylinder" size="{g.peg_radius:.6g} {g.peg_halflen:.6g}"')
    return (
        f'<geom name="peg_c" {shape} class="{cls}" group="1" rgba="0.92 0.55 0.12 1" friction="0.4 0.005 0.0001" />'
        f'<site name="peg_tip" pos="0 0 {-g.peg_halflen:.6g}" size="0.004" rgba="0 1 0 1" group="4" />'
    )


def peg_mesh_asset(g: Geometry) -> str:
    """Convex peg with a 45 degree tip chamfer: an inscribed polygon prism (never wider than the cylinder)."""
    if g.peg_tip_chamfer <= 0:
        return ""
    r, h, c, n = g.peg_radius, g.peg_halflen, g.peg_tip_chamfer, g.peg_facets
    ang = [2 * math.pi * k / n for k in range(n)]
    rings = ((r - c, -h), (r, -h + c), (r, h))
    v = [(rad * math.cos(t), rad * math.sin(t), z) for rad, z in rings for t in ang]
    return f'    <mesh name="peg_mesh" vertex="{_fmt([x for p_ in v for x in p_])}"/>\n'


def hole_mesh_assets(g: Geometry) -> str:
    """Four mitred convex pieces (bore wall + 45 deg chamfer + plate top), body frame at the plate top."""
    a, w, T, L = g.hole_half, g.chamfer, g.plate_thick, g.plate_half
    b = a + w
    base = [(a, -a, -T), (a, a, -T), (L, L, -T), (L, -L, -T),   # bottom
            (a, -a, -w), (a, a, -w),                            # top of the vertical bore wall
            (b, -b, 0.0), (b, b, 0.0), (L, L, 0.0), (L, -L, 0.0)]  # plate top (chamfer rim to outer edge)
    out = []
    for k in range(4):
        c_, s_ = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
        v = [(c_ * x - s_ * y, s_ * x + c_ * y, z) for x, y, z in base]
        out.append(f'    <mesh name="hole_side{k}" vertex="{_fmt([q for p_ in v for q in p_])}"/>')
    return "\n".join(out) + "\n"


def hand_contents(g: Geometry, grasp: bool, tool: str = "peg", tool_body: str | None = None,
                  finger_angle: float | None = None, tool_z: float | None = None) -> str:
    """The fingers (hinged with collision, or frozen) and, for the rigid hand, the held tool (the peg, a job tool
    from :mod:`blindtouch.scenes.props` passed as ``tool_body``, or nothing)."""
    pad = " " * 22
    out = []
    for name, (pos, axis, inertial) in FINGERS.items():
        visual = "".join(f'<geom class="arm_visual" mesh="{name}_0{i}" material="{"pale_silver" if i == 2 else "matte_black"}" />' for i in range(3))
        if grasp:
            # the right fingers close toward 0 over the range [-0.785, 0]; Env.reset starts them at the grasp angle
            joint = f'<joint name="{name}_joint" axis="{_fmt(axis)}" range="-0.7854 0" class="finger" />'
            coll = "".join(f'<geom name="{name}_c{i}" type="mesh" mesh="{name}_c{i}" class="fingertip" />' for i in range(4))
            out.append(f'{pad}<body name="{name}" pos="{pos}">{inertial}{joint}{visual}{coll}</body>')
        else:
            q = _quat_axis_angle(axis, -(g.finger_angle if finger_angle is None else finger_angle))
            out.append(f'{pad}<body name="{name}" pos="{pos}" quat="{_fmt(q)}">{inertial}{visual}</body>')
    if tool_body is not None:   # a job tool (blindtouch.scenes.props): rigid in the believed hand, at the pads' centre
        if not grasp:
            out.append(f'{pad}<body name="peg" pos="0 0 {tool_z:.6g}">{tool_body}</body>')
    elif tool == "empty":   # the control point sits between the fingertips
        out.append(f'{pad}<site name="peg_tip" pos="0 0 {FINGERTIP_Z:g}" size="0.004" rgba="0 1 0 1" group="4" />')
    elif not grasp:
        out.append(f'{pad}<body name="peg" pos="{_fmt((*g.peg_grasp_xy, -g.peg_offset))}">{PEG_INERTIAL}{peg_geoms(g, "arm_collision")}</body>')
    return "\n".join(out)


def free_peg_xml(g: Geometry) -> str:
    """The plant's peg: a free body (Env.reset places it in the grasp), colliding with the world and the hand."""
    return (
        f'    <body name="peg" pos="0 0 {g.mount_z - g.peg_offset:.6g}">\n'
        f'      <freejoint name="peg_free"/>{PEG_INERTIAL}{peg_geoms(g, "peg")}\n'
        f"    </body>\n"
    )


def arm_xml(g: Geometry = GEOM, grasp: bool = True, tool: str = "peg", tool_body: str | None = None,
            finger_angle: float | None = None, tool_z: float | None = None, extra_assets: str = "") -> str:
    """Return the arm include with its placeholders filled (``grasp``: plant hand, else rigid nominal hand).

    ``tool``: what the rigid hand holds (``peg``, a job tool given by ``tool_body``), or ``empty`` (no tool; the control point sits
    between the fingertips, in both hands). The plant hand never holds a tool: scenes add it as a free body.
    """
    text = (SCENE_DIR / "openarm_right.template.xml").read_text(encoding="utf-8")
    grip_act = grip_eq = ""
    if grasp:
        grip_act = f'    <position name="grip" joint="finger_inner_joint" kp="{g.grip_kp:g}" kv="0.2" ctrlrange="-0.7854 0" forcerange="-7 7" />'
        grip_eq = '  <equality>\n    <joint name="finger_mimic" joint1="finger_inner_joint" joint2="finger_outer_joint" polycoef="0 1 0 0 0" solimp="0.95 0.99 0.001" solref="0.005 1" />\n  </equality>'
    subs = {
        "HAND_CONTENTS": hand_contents(g, grasp, tool, tool_body, finger_angle, tool_z),
        "GRIP_ACTUATOR": grip_act,
        "GRIP_EQUALITY": grip_eq,
        "MOUNT_Z": f"{g.mount_z:.6g}",
        "MOUNT_QUAT": _fmt(_quat_axis_angle((0, 1, 0), g.mount_pitch)),
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    return text.replace("  </asset>", peg_mesh_asset(g) + extra_assets + "  </asset>", 1)


STUDIO = """
  <visual>
    <global offwidth="1920" offheight="1080" azimuth="150" elevation="-22"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.18 0.18 0.19" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <map znear="0.01" zfar="20"/>
    <rgba haze="0.02 0.02 0.025 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.02 0.02 0.025" rgb2="0.02 0.02 0.025" width="64" height="64"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.07 0.07 0.08" rgb2="0.09 0.09 0.10" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0.08"/>
    <material name="table_mat" rgba="0.22 0.22 0.24 1" reflectance="0.05" specular="0.2"/>
    <material name="plate_mat" rgba="0.55 0.58 0.62 1" specular="0.5" shininess="0.5"/>
    <material name="pedestal_mat" rgba="0.12 0.12 0.13 1" specular="0.3"/>
  </asset>
"""

WORLD_COMMON = """
    <light name="key" pos="1.2 -1.0 2.0" dir="-0.55 0.45 -0.7" diffuse="0.9 0.88 0.85" specular="0.3 0.3 0.3" castshadow="true" cutoff="60" exponent="2"/>
    <light name="fill" pos="-0.6 1.2 1.6" dir="0.4 -0.6 -0.6" diffuse="0.25 0.27 0.32" specular="0 0 0" castshadow="false"/>
    <light name="rim" pos="-0.8 -0.6 1.4" dir="0.6 0.4 -0.3" diffuse="0.3 0.3 0.35" specular="0.2 0.2 0.2" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat" contype="1" conaffinity="2"/>
    <geom name="pedestal" type="box" pos="-0.02 -0.031 {ped_z}" size="0.06 0.08 {ped_z}" material="pedestal_mat" contype="0" conaffinity="0"/>
    <body name="table" pos="{tx} {ty} {tz}">
      <geom name="table_top" type="box" size="{thx} {thy} 0.02" material="table_mat" contype="1" conaffinity="2" friction="0.6 0.005 0.0001" solref="0.004 1"/>
    </body>
    {cameras}
"""


def hole_xml(g: Geometry = GEOM) -> str:
    """The peg-in-hole plate: a square hole with a 45 degree chamfer from four mitred convex pieces (wall + chamfer +
    plate top each), on a mocap body (movable per episode)."""
    return _hole_body(g, [f'<geom class="hole" name="hole_side{k}" type="mesh" mesh="hole_side{k}"/>' for k in range(4)])


def _hole_body(g: Geometry, geoms: list[str]) -> str:
    T = g.plate_thick
    hx, hy = g.hole_nominal
    body = (
        f'    <body name="hole" mocap="true" pos="{hx:.6g} {hy:.6g} {g.plate_top:.6g}">\n      '
        + "\n      ".join(geoms)
        + '\n      <site name="hole_top" pos="0 0 0" size="0.003" rgba="1 0.2 0.2 1" group="4"/>'
        + f'\n      <site name="hole_bottom" pos="0 0 {-T:.6g}" size="0.003" rgba="0.2 1 0.2 1" group="4"/>'
        + "\n    </body>\n"
    )
    return body


def scene_xml(kind: str, g: Geometry = GEOM, nominal: bool = False) -> str:
    tx, ty = g.table_center
    thx, thy = g.table_half
    hx, hy = g.hole_nominal
    focus = (hx, hy, g.plate_top + 0.02)
    cameras = "\n    ".join([
        _lookat((hx + 0.30, hy - 0.26, g.plate_top + 0.24), focus, 42, "front"),
        _lookat((hx + 1.05, hy - 1.00, g.plate_top + 0.62), (0.15, -0.12, g.plate_top + 0.15), 40, "overview"),
        _lookat((hx + 0.16, hy - 0.15, g.plate_top + 0.06), (hx, hy, g.plate_top), 40, "closeup"),
        _lookat((hx - 0.05, hy - 0.55, g.plate_top + 0.12), (hx, hy, g.plate_top + 0.05), 35, "side"),
    ])
    world = WORLD_COMMON.format(
        ped_z=g.mount_z / 2 - 0.03, tx=tx, ty=ty, tz=g.table_top - 0.02, thx=thx, thy=thy, cameras=cameras
    )
    if kind != "peg":
        raise ValueError(kind)
    extra = hole_xml(g)
    if not nominal:
        extra += free_peg_xml(g)
    return f"""<!-- Generated by scenes/build.py -- edit that file, not this one. -->
<mujoco model="blind_touch_{kind}{'_nominal' if nominal else ''}">
  <include file="{'openarm_right_rigid.xml' if nominal else 'openarm_right.xml'}"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="20" noslip_iterations="5"/>
  <default>
    <default class="hole">
      <geom type="box" material="plate_mat" contype="1" conaffinity="2" condim="3" friction="0.3 0.005 0.0001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    </default>
    <!-- the free peg collides with the world (contype 1 / conaffinity 2) and with the hand and fingers (2 / 1) -->
    <default class="peg">
      <geom contype="3" conaffinity="3" condim="3" solref="0.004 1"/>
    </default>
  </default>
{STUDIO}  <asset>
{hole_mesh_assets(g)}  </asset>
  <worldbody>
{world}
{extra}
  </worldbody>
</mujoco>
"""


def build(g: Geometry = GEOM) -> list[Path]:
    (SCENE_DIR / "openarm_right.xml").write_text(arm_xml(g, grasp=True), encoding="utf-8")
    (SCENE_DIR / "openarm_right_rigid.xml").write_text(arm_xml(g, grasp=False), encoding="utf-8")
    paths = []
    for kind in ("peg",):
        for nominal in (False, True):
            p = SCENE_DIR / f"{kind}_scene{'_nominal' if nominal else ''}.xml"
            p.write_text(scene_xml(kind, g, nominal), encoding="utf-8")
            paths.append(p)
    return paths


def scene_path(kind: str, nominal: bool = False) -> Path:
    """The true plant's scene, or (``nominal``) the same scene with the rigid hand the robot believes in."""
    if kind != "peg":
        from blindtouch.scenes.job_scenes import job_scene_path

        return job_scene_path(kind, nominal)
    p = SCENE_DIR / f"{kind}_scene{'_nominal' if nominal else ''}.xml"
    stale = max(Path(__file__).stat().st_mtime, (SCENE_DIR / "openarm_right.template.xml").stat().st_mtime)
    if not p.exists() or p.stat().st_mtime < stale:
        build()
    return p


if __name__ == "__main__":
    for p in build():
        print("wrote", p)

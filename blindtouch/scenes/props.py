"""Real-dimension props for the contact jobs: power strip and Europlug, sloped whiteboard and eraser, cans.

Every part that touches something is built to real dimensions, from exact convex pieces (MuJoCo collides convex
shapes only); everything else is a detailed visual mesh on top (group 2, no collision). Dimensions and their
sources are listed in DIMS; "typical" marks a value taken from common products, not from a standard.

Frames. Fixtures: origin on the table surface under the fixture centre, +z up (they sit on the mocap body
``fixture``). Held tools: origin at the centre of the finger pads, +z toward the hand; the gripper closes along
the tool's y axis, the 55 mm wide pads lie along its x axis.
"""

from __future__ import annotations

import math

import mapbox_earcut
import numpy as np

# ----------------------------------------------------------------------------------------------- dimensions
DIMS = {
    "europlug": dict(body_w=0.0353, body_t=0.0137, front_zone=0.018, pin_d=0.0040, pin_len=0.019, pin_spacing=0.019,
                     src="CEE 7/16 (Europlug): body 35.3 x 13.7 mm within 18 mm of the front, pins 19 mm long, 4.0 mm "
                         "tips, 19 mm apart at the base converging to 17-18 mm (en.wikipedia.org/wiki/Europlug)"),
    "schuko": dict(recess_d=0.039, recess_depth=0.0175, hole_d=0.0052, hole_spacing=0.019,
                   src="CEE 7/3 socket: round recess 17.5 mm deep, pins 4.8 mm at 19 mm (en.wikipedia.org/wiki/Schuko); "
                       "recess diameter 39 mm typical (not confirmed by a source)"),
    "can330": dict(d=0.0662, h=0.1152, empty=0.013, liquid=0.330,
                   src="Standard 330 ml beverage can: 66.2 mm x 115.2 mm (typical); aluminium can ~13 g (typical)"),
    "eraser": dict(l=0.130, w=0.050, h=0.030, src="Whiteboard eraser: typical size"),
}


# ----------------------------------------------------------------------------------------------- mesh helpers
def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(v).ravel())


def mesh_xml(name: str, verts, faces=None) -> str:
    """An inline MJCF mesh (faces given: rendered as is; none: MuJoCo builds the convex hull)."""
    s = f'    <mesh name="{name}" vertex="{_fmt(verts)}"'
    if faces is not None:
        s += f' face="{" ".join(str(int(i)) for i in np.asarray(faces).ravel())}"'
    return s + "/>"


def lathe(profile, n: int = 48):
    """Surface of revolution about z from a (r, z) profile (bottom to top). Ends with r = 0 close the solid."""
    prof = np.asarray(profile, float)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts, faces, ring = [], [], []
    for r, z in prof:
        if r <= 1e-9:
            ring.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            ring.append(list(range(len(verts), len(verts) + n)))
            verts += [(r * math.cos(a), r * math.sin(a), z) for a in ang]
    for a, b in zip(ring[:-1], ring[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        for k in range(n):
            k1 = (k + 1) % n
            if len(a) == 1:
                faces.append((a[0], b[k1], b[k]))
            elif len(b) == 1:
                faces.append((a[k], a[k1], b[0]))
            else:
                faces += [(a[k], a[k1], b[k1]), (a[k], b[k1], b[k])]
    return np.array(verts), np.array(faces)


def extrude(outline, z0: float, z1: float, holes=()):
    """Prism from a 2D outline (counter-clockwise) with optional holes (clockwise), between z0 and z1."""
    rings = [np.asarray(outline, float)] + [np.asarray(h, float) for h in holes]
    pts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = mapbox_earcut.triangulate_float64(pts, ends).reshape(-1, 3)
    n = len(pts)
    verts = np.vstack([np.c_[pts, np.full(n, z0)], np.c_[pts, np.full(n, z1)]])
    faces = [(t[0], t[2], t[1]) for t in tri] + [(t[0] + n, t[1] + n, t[2] + n) for t in tri]
    start = 0
    for r in rings:
        m = len(r)
        for k in range(m):
            a, b = start + k, start + (k + 1) % m
            faces += [(a, b, b + n), (a, b + n, a + n)]
        start += m
    return verts, np.array(faces)


def rounded_rect(w: float, h: float, r: float, n: int = 6, center=(0.0, 0.0)):
    cx, cy = center
    pts = []
    for (sx, sy, a0) in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        ox, oy = cx + sx * (w / 2 - r), cy + sy * (h / 2 - r)
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    return np.array(pts)


def circle(r: float, n: int = 32, center=(0.0, 0.0), cw: bool = False):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.c_[center[0] + r * np.cos(a), center[1] + r * np.sin(a)]
    return p[::-1] if cw else p


def ring_pieces(prefix: str, r_in: float, half: tuple, z0: float, z1: float, n: int = 16, chamfer: float = 0.0,
                center=(0.0, 0.0), cls: str = "fx", outer: str = "rect", chamfer_bottom: bool = False):
    """A block (rectangle half sizes ``half``, or a circle of radius half[0]) with a round hole of radius
    ``r_in`` from z0 to z1, as ``n`` convex sectors; ``chamfer`` bevels the hole's top edge at 45 deg.
    The sectors' inner chords sit inside the circle, so the hole is never narrower than drawn: r_in is
    widened by 1/cos(pi/n) to keep the drawn radius at the chord middles. Returns (assets, geoms)."""
    hx, hy = half if outer == "rect" else (half[0], half[0])
    r_in = r_in / math.cos(math.pi / n)
    assets, geoms = [], []

    def outer_hit(a):
        c, s = math.cos(a), math.sin(a)
        if outer == "circle":
            return (hx * c, hx * s)
        t = min(hx / max(abs(c), 1e-12), hy / max(abs(s), 1e-12))
        return (t * c, t * s)

    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        out = [outer_hit(a0), outer_hit(a1)]
        if outer == "rect":
            for cx, cy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
                ca = math.atan2(cy, cx) % (2 * math.pi)
                if a0 < ca < a1:
                    out.insert(1, (cx, cy))
        v = []
        for a in (a0, a1):
            c, s = math.cos(a), math.sin(a)
            if chamfer > 0 and chamfer_bottom:
                v += [((r_in + chamfer) * c, (r_in + chamfer) * s, z0), (r_in * c, r_in * s, z0 + chamfer), (r_in * c, r_in * s, z1)]
            elif chamfer > 0:
                v += [(r_in * c, r_in * s, z0), (r_in * c, r_in * s, z1 - chamfer), ((r_in + chamfer) * c, (r_in + chamfer) * s, z1)]
            else:
                v += [(r_in * c, r_in * s, z0), (r_in * c, r_in * s, z1)]
        for x, y in out:
            v += [(x, y, z0), (x, y, z1)]
        v = np.array(v) + np.r_[center, 0.0]
        nm = f"{prefix}{k}"
        assets.append(mesh_xml(nm, v))
        geoms.append(f'<geom class="{cls}" name="{nm}" type="mesh" mesh="{nm}"/>')
    return assets, geoms


def box(name: str, lo, hi, cls: str = "fx", extra: str = "") -> str:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return f'<geom class="{cls}" name="{name}" type="box" pos="{_fmt((lo + hi) / 2)}" size="{_fmt((hi - lo) / 2)}" {extra}/>'


def vis(name: str, mesh: str, material: str, pos=(0, 0, 0), quat=None) -> str:
    q = f' quat="{_fmt(quat)}"' if quat is not None else ""
    return f'<geom class="vis" name="{name}" type="mesh" mesh="{mesh}" material="{material}" pos="{_fmt(pos)}"{q}/>'


MATERIALS = """
    <material name="white_plastic" rgba="0.93 0.93 0.91 1" specular="0.3" shininess="0.4"/>
    <material name="dark_plastic" rgba="0.10 0.10 0.11 1" specular="0.3" shininess="0.5"/>
    <material name="grey_plastic" rgba="0.55 0.56 0.58 1" specular="0.2" shininess="0.3"/>
    <material name="brass" rgba="0.80 0.66 0.32 1" specular="0.8" shininess="0.8" reflectance="0.15"/>
    <material name="steel" rgba="0.70 0.72 0.75 1" specular="0.9" shininess="0.9" reflectance="0.2"/>
    <material name="zinc" rgba="0.60 0.62 0.66 1" specular="0.9" shininess="0.9" reflectance="0.2"/>
    <material name="red_paint" rgba="0.62 0.10 0.10 1" specular="0.5" shininess="0.6"/>
    <material name="aluminium" rgba="0.82 0.83 0.86 1" specular="1.0" shininess="0.9" reflectance="0.25"/>
    <material name="can_red" rgba="0.75 0.10 0.12 1" specular="0.7" shininess="0.8"/>
    <material name="can_blue" rgba="0.12 0.30 0.70 1" specular="0.7" shininess="0.8"/>
    <material name="can_green" rgba="0.15 0.55 0.25 1" specular="0.7" shininess="0.8"/>
    <material name="oak" rgba="0.68 0.50 0.32 1" specular="0.2" shininess="0.3"/>
    <material name="oak_dark" rgba="0.50 0.36 0.22 1" specular="0.2" shininess="0.3"/>
    <material name="board_white" rgba="0.97 0.97 0.97 1" specular="0.6" shininess="0.7" reflectance="0.05"/>
    <material name="ink" rgba="0.05 0.12 0.45 1" specular="0.1"/>
    <material name="felt" rgba="0.30 0.30 0.32 1" specular="0.0"/>
    <material name="eraser_body" rgba="0.15 0.45 0.80 1" specular="0.4" shininess="0.5"/>
    <material name="marker_body" rgba="0.95 0.95 0.95 1" specular="0.5" shininess="0.6"/>
    <material name="marker_cap" rgba="0.10 0.25 0.75 1" specular="0.5" shininess="0.6"/>
    <material name="mat_green" rgba="0.30 0.60 0.35 1"/>
    <material name="mat_yellow" rgba="0.85 0.70 0.20 1"/>
    <material name="mat_blue" rgba="0.25 0.45 0.80 1"/>
    <material name="tv_black" rgba="0.03 0.03 0.035 1" specular="0.6" shininess="0.8" reflectance="0.1"/>
"""


# ======================================================================================================== tools
class ToolDesign:
    """A held tool: geoms in the tool frame, mass, grip width (the opening the fingers close to)."""

    def __init__(self, name, grip_width, assets, geoms, mass, inertia, tip_z):
        self.name, self.grip_width, self.assets, self.geoms = name, grip_width, assets, geoms
        self.mass, self.inertia, self.tip_z = mass, inertia, tip_z

    def body(self, cls: str) -> str:
        g = "".join(x.replace("{cls}", cls) for x in self.geoms)
        return (f'<inertial pos="0 0 0" mass="{self.mass:g}" diaginertia="{_fmt(self.inertia)}" />' + g +
                f'<site name="peg_tip" pos="0 0 {self.tip_z:.6g}" size="0.003" rgba="0 1 0 1" group="4" />')


def europlug_tool() -> ToolDesign:
    """A Europlug moulded on a cable: flat body 35.3 x 13.7 mm, pins along the tool x axis."""
    d = DIMS["europlug"]
    w, t = d["body_w"], d["body_t"]
    front = -0.031                   # front plane: 24 mm of body below the pads (the pads end 7 mm below the origin)
    pin_r, pin_len = d["pin_d"] / 2, d["pin_len"]
    sp = 0.0185                      # parallel pins at the mean of the converging 19 -> 17.5 mm (real pins flex)
    # collision: the front 18 mm (the zone that enters the socket's 17.5 mm recess) has the real body's rounded
    # corners -- a sharp box's corners reach 19.0 mm from the centre of a 19.5 mm recess and caught its wall,
    # tilting the plug in the fingers until the pins jammed; above it a box, where the finger pads hold the plug
    nose = d["front_zone"]
    a = [mesh_xml("plug_c_nose", extrude(rounded_rect(w, t, 0.004), front, front + nose)[0])]
    g = ['<geom name="plug_nose" type="mesh" mesh="plug_c_nose" class="{cls}" group="3" friction="0.8 0.01 0.001"/>',
         f'<geom name="plug_body" type="box" size="{w / 2:g} {t / 2:g} {(0.020 - front - nose) / 2:.6g}" pos="0 0 {(0.020 + front + nose) / 2:.6g}" class="{{cls}}" group="3" friction="0.8 0.01 0.001"/>']
    for k, s in enumerate((-1, 1)):
        half = (pin_len - 2 * pin_r) / 2
        g.append(f'<geom name="plug_pin{k}" type="capsule" size="{pin_r:g} {half:.6g}" pos="{s * sp / 2:.6g} 0 {front - pin_len / 2:.6g}" class="{{cls}}" group="3" friction="0.3 0.005 0.0001"/>')
    # visuals: rounded flat body tapering to a strain relief and cable, two pins with insulated shafts
    body = rounded_rect(w, t, 0.004)
    v, f = extrude(body, front, 0.012)
    a.append(mesh_xml("plug_v_body", v, f))
    v, f = lathe([(0, 0.012), (0.0068, 0.012), (0.0055, 0.030), (0.0035, 0.034), (0.0035, 0.10), (0, 0.10)], 24)
    a.append(mesh_xml("plug_v_cable", v, f))
    v, f = lathe([(0, 0), (0.0019, 0.0), (0.0019, 0.010), (0.0020, 0.0101), (0.0020, 0.0175), (0.0012, 0.019), (0, 0.019)], 20)
    a.append(mesh_xml("plug_v_pin", v, f))
    g += [vis("plug_v_body", "plug_v_body", "dark_plastic"), vis("plug_v_cable", "plug_v_cable", "dark_plastic")]
    for k, s in enumerate((-1, 1)):
        g.append(vis(f"plug_v_pin{k}", "plug_v_pin", "steel", pos=(s * sp / 2, 0, front), quat=(0, 1, 0, 0)))
    return ToolDesign("europlug", t, a, g, 0.045, (1.5e-5, 1.5e-5, 6e-6), front - pin_len)


def eraser_tool() -> ToolDesign:
    """Whiteboard eraser: 130 x 50 x 30 mm, felt underneath; long side along the pads (tool x)."""
    d = DIMS["eraser"]
    L, W, H = d["l"], d["w"], d["h"]
    bot = -0.022
    g = [f'<geom name="eraser_body" type="box" size="{L / 2:g} {W / 2:g} {(H - 0.008) / 2:.6g}" pos="0 0 {bot + 0.008 + (H - 0.008) / 2:.6g}" class="{{cls}}" group="3"/>',
         f'<geom name="eraser_felt" type="box" size="{L / 2 - 0.002:g} {W / 2 - 0.002:g} 0.004" pos="0 0 {bot + 0.004:.6g}" class="{{cls}}" group="3" friction="0.45 0.005 0.0001" solref="0.01 1"/>']
    a = []
    v, f = extrude(rounded_rect(L, W, 0.008), bot + 0.008, bot + H)
    a.append(mesh_xml("eraser_v", v, f))
    v, f = extrude(rounded_rect(L - 0.004, W - 0.004, 0.006), bot, bot + 0.008)
    a.append(mesh_xml("eraser_v_felt", v, f))
    g += [vis("eraser_v", "eraser_v", "eraser_body"), vis("eraser_v_felt", "eraser_v_felt", "felt")]
    return ToolDesign("eraser", W, a, g, 0.060, (1e-5, 9e-5, 9e-5), bot)


TOOLS = {"europlug": europlug_tool, "eraser": eraser_tool}


# ===================================================================================================== fixtures
def socket_strip() -> tuple[list, str, dict]:
    """A 3-way power strip (Schuko sockets) lying on the desk; the middle socket is the target."""
    s = DIMS["schuko"]
    W, L, H = 0.056, 0.22, 0.040
    r_rec, depth = s["recess_d"] / 2, s["recess_depth"]
    a, g = [], []
    xs = (-0.065, 0.0, 0.065)
    for k, x0 in enumerate(xs):
        aa, gg = ring_pieces(f"sock{k}_", r_rec, (0.0325, W / 2), H - depth, H, n=20, chamfer=0.0015, center=(x0, 0))
        a += aa
        g += gg
        hh, sp = s["hole_d"] / 2, s["hole_spacing"]
        zf0, zf1 = H - depth - 0.022, H - depth
        for j, sx in enumerate((-1, 1)):   # pin holes: round holes in two floor blocks that meet at the centre
            aa, gg = ring_pieces(f"sockh{k}{j}_", hh, (sp / 2, 0.012), zf0, zf1, n=12,
                                 center=(x0 + sx * sp / 2, 0), chamfer=0.0006)
            a += aa
            g += gg
        g += [box(f"sock{k}_fl0", (x0 - 0.0325, -W / 2, zf0), (x0 - sp, W / 2, zf1)), box(f"sock{k}_fl1", (x0 + sp, -W / 2, zf0), (x0 + 0.0325, W / 2, zf1)),
              box(f"sock{k}_fl2", (x0 - sp, 0.012, zf0), (x0 + sp, W / 2, zf1)), box(f"sock{k}_fl3", (x0 - sp, -W / 2, zf0), (x0 + sp, -0.012, zf1)),
              box(f"sock{k}_floorbase", (x0 - 0.0325, -W / 2, 0.0), (x0 + 0.0325, W / 2, zf0))]
    # visuals: strip shell with rounded edges, earth clips, switch, cable
    v, f = extrude(rounded_rect(L, W, 0.010), 0.0, H - 0.0005, holes=[circle(r_rec + 0.0005, 40, (x0, 0), cw=True) for x0 in xs])
    a.append(mesh_xml("strip_v", v, f))
    g.append(vis("strip_v", "strip_v", "white_plastic"))
    for k, x0 in enumerate(xs):
        v, f = lathe([(0, H - depth - 0.0004), (r_rec, H - depth - 0.0004), (r_rec, H - depth), (0, H - depth)], 40)
        a.append(mesh_xml(f"sock_v_floor{k}", v, f))
        g.append(vis(f"sock_v_floor{k}", f"sock_v_floor{k}", "white_plastic"))
        for j, sy in enumerate((-1, 1)):
            g.append(f'<geom class="vis" type="box" size="0.004 0.0012 0.006" pos="{x0:g} {sy * (r_rec - 0.0012):.6g} {H - 0.007:.6g}" material="steel"/>')
    g.append(f'<geom class="vis" type="box" size="0.010 0.012 0.004" pos="{L / 2 - 0.016:.6g} 0 {H:.6g}" material="red_paint"/>')
    g.append(f'<geom class="vis" type="capsule" size="0.004" fromto="{-L / 2:.6g} 0 0.012 {-L / 2 - 0.12:.6g} 0.03 0.006" material="white_plastic"/>')
    return a, "\n      ".join(g), {"top": H, "target_x": 0.0, "recess_floor": H - depth}


def tv_occluder() -> str:
    """A TV on its stand between the robot's front camera and the strip: why it cannot see the socket."""
    return ('<body name="tv" pos="0.47 -0.36 0.45">'
            '<geom class="vis" type="box" size="0.30 0.012 0.17" pos="0 0 0.25" quat="0.966 0 0 0.259" material="tv_black"/>'
            '<geom class="vis" type="box" size="0.08 0.05 0.004" pos="0 0 0.004" material="tv_black"/>'
            '<geom class="vis" type="box" size="0.012 0.012 0.07" pos="0 0 0.07" material="tv_black"/></body>')


def whiteboard() -> tuple[list, str, dict]:
    """A desk whiteboard on a wedge stand, sloping up away from the robot; blue marker scribbles to erase."""
    W, L = 0.30, 0.22
    g, a = [], []
    g.append(f'<geom class="fx" name="board" type="box" size="{L / 2:g} {W / 2:g} 0.004" pos="0 0 0" material="board_white" friction="0.35 0.005 0.0001"/>')
    g.append(f'<geom class="vis" type="box" size="{L / 2 + 0.006:g} {W / 2 + 0.006:g} 0.0035" pos="0 0 -0.0012" material="grey_plastic"/>')
    rng = np.random.default_rng(3)
    for k in range(9):   # scribbles: short strokes of marker ink, visual only, in INK_PIECES pieces (erased one by one)
        x0, y0 = rng.uniform(-0.06, 0.06), rng.uniform(-0.11, 0.11)
        ang = rng.uniform(0, np.pi)
        ln = rng.uniform(0.02, 0.06)
        c, s = math.cos(ang), math.sin(ang)
        for j in range(INK_PIECES):
            o = (j + 0.5) / INK_PIECES - 0.5
            g.append(f'<geom class="vis" name="ink{k}_{j}" type="box" size="{ln / INK_PIECES / 2 + 0.0004:.4g} 0.0012 0.0002" '
                     f'pos="{x0 + o * ln * c:.4g} {y0 + o * ln * s:.4g} 0.0041" '
                     f'quat="{math.cos(ang / 2):.4g} 0 0 {math.sin(ang / 2):.4g}" material="ink"/>')
    return a, "\n      ".join(g), {"size": (L, W)}


def cans_bodies(nominal: bool) -> str:
    """Three 330 ml cans (66.2 x 115.2 mm) on the table, and three coloured spots: recycling, half, fridge tray."""
    if nominal:
        return ""
    d = DIMS["can330"]
    r, h = d["d"] / 2, d["h"]
    out = []
    for i, ((x, y), mat) in enumerate(zip(CAN_SLOTS, ("can_red", "can_blue", "can_green"))):
        out.append(f'    <body name="can{i}" pos="{x:g} {y:g} {GEOM_TABLE + 0.0005:.6g}">\n'
                   f'      <freejoint name="can{i}_free"/>\n'
                   f'      <inertial pos="0 0 {h / 2:g}" mass="{d["empty"]:g}" diaginertia="2e-5 2e-5 1e-5"/>\n'
                   f'      <geom class="obj" name="can{i}" type="cylinder" size="{r:g} {h / 2:g}" pos="0 0 {h / 2:g}" group="3"/>\n'
                   f'      {vis(f"can{i}_v", "can_v", "aluminium")}\n'
                   f'      <geom class="vis" type="cylinder" size="{r + 0.0002:g} 0.035" pos="0 0 {h / 2:g}" material="{mat}"/>\n'
                   f'    </body>\n')
    for i, (x, y) in enumerate(CAN_SPOTS):
        out.append(f'    <geom class="vis" type="cylinder" size="0.042 0.0006" pos="{x:g} {y:g} {GEOM_TABLE + 0.0006:.6g}" material="{("mat_blue", "mat_yellow", "mat_green")[i]}"/>\n')
    return "".join(out)


def can_mesh() -> str:
    d = DIMS["can330"]
    r, h = d["d"] / 2, d["h"]
    v, f = lathe([(0, 0.004), (0.022, 0.0), (0.0315, 0.002), (r, 0.010), (r, h - 0.012), (0.0275, h - 0.003),
                  (0.027, h), (0.0262, h), (0.0262, h - 0.002), (0, h - 0.002)], 56)
    return mesh_xml("can_v", v, f)


GEOM_TABLE = 0.45
INK_PIECES = 5
CAN_SLOTS = ((0.235, -0.31), (0.235, -0.22), (0.235, -0.13))
CAN_SPOTS = ((0.32, -0.265), (0.32, -0.175), (0.32, -0.085))  # recycling (empty), sink (half), fridge tray (full):
# behind the row and offset half a slot, so the open hand at a can never reaches a can already set down
# (the arm reaches only a narrow band with the tool vertical: x 0.24-0.32 m here)

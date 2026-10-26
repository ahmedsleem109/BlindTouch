"""Record episodes and render them again from any camera.

* :class:`Recorder` -- an env hook that stores the full simulation state every control step (500 Hz), so a run
  can be re-rendered afterwards at any frame rate and from any camera;
* :class:`Studio` -- renders frames from a *separate* render-only model of the same scene;
* :func:`write_mp4` -- H.264 via PyAV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

from blindtouch.scenes.build import scene_path

W, H = 1920, 1080
ACCENT = (64, 190, 255)     # BGR: what the robot felt
TRUE_C = (235, 235, 235)    # simulator truth
RED = (60, 60, 230)
GREEN = (90, 210, 110)
FONT = cv2.FONT_HERSHEY_DUPLEX


@dataclass
class Frame:
    t: float
    qpos: np.ndarray
    mocap: np.ndarray
    f_true: np.ndarray
    tags: dict = field(default_factory=dict)


class Recorder:
    """Attach with ``env.hooks.append(rec)``; call :meth:`note` to tag the following frames."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self._tags: dict = {}

    def __call__(self, env, sample) -> None:
        self.frames.append(Frame(env.t, env.data.qpos.copy(), env.data.mocap_pos.copy(), sample.truth.force.copy(),
                                 dict(self._tags)))

    def note(self, **tags) -> None:
        self._tags.update(tags)

    def clear(self) -> None:
        self.frames.clear()
        self._tags = {}


class Studio:
    def __init__(self, scene: str = "peg", width: int = W, height: int = H) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene_path(scene)))
        g = self.model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, width), max(g.offheight, height)   # e.g. portrait frames
        self.data = mujoco.MjData(self.model)
        self.r = mujoco.Renderer(self.model, height, width)
        self.w, self.h = width, height
        self.base_rgba = self.model.geom_rgba.copy()
        self.base_mat = self.model.geom_matid.copy()

    def set_state(self, fr: Frame) -> None:
        self.data.qpos[:] = fr.qpos
        if len(fr.mocap):
            self.data.mocap_pos[:] = fr.mocap
        mujoco.mj_forward(self.model, self.data)

    def render(self, fr: Frame, cam: str | mujoco.MjvCamera) -> np.ndarray:
        self.set_state(fr)
        self.r.update_scene(self.data, camera=cam)
        return cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)


def write_mp4(path: str, frames_bgr, fps: int = 60, crf: int = 20) -> None:
    import av

    container = av.open(str(path), mode="w")
    stream = None
    for img in frames_bgr:
        if stream is None:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = img.shape[1], img.shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": "medium"}
        frame = av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    if stream is not None:
        for pkt in stream.encode():
            container.mux(pkt)
    container.close()

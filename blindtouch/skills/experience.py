"""Learning from mistakes: an experience memory shared by every job (plan: in-attempt recovery + memory).

A skill that fails reports a *failure signature* named after the skill, not the job ("turn: no drop",
"centre: no wall", ...). The job offers recoveries from a shared menu ("push in", "re-centre and turn again",
...). Which recovery to try is learned from the memory: every attempt -- in any job -- is recorded with its
outcome as the robot itself judged it from touch (it felt the drop, the click, the seat), never from simulator
truth. A lesson learned on the plug ("when the pins sink half-way, push; turning on pulls them out") is therefore
used the next time the key or the nut meets the same failure.

Choice: Thompson sampling over Beta(1 + successes, 1 + failures) per (failure, recovery) -- it keeps trying
recoveries that have not been tried much and settles on the ones that work. The memory is a JSON-lines file
(``out/experience/memory.jsonl``), so it persists across runs and grows with practice.

    mem = Experience()                     # loads what was learned so far
    rob.memory = mem
    ok = rob.recover("turn: no drop", {"push in": f1, "re-centre and turn again": f2}, context={"job": "socket"})
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from blindtouch.paths import OUT

PATH = OUT / "experience" / "memory.jsonl"


class Experience:
    def __init__(self, path: Path | None = PATH, rng: np.random.Generator | None = None, learn: bool = True) -> None:
        """``path`` None: memory only in RAM (a fresh robot). ``learn`` False: use what is known, add nothing."""
        self.path = path
        self.learn = learn
        self.rng = rng or np.random.default_rng()
        self.records: list[dict] = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.records.append(json.loads(line))

    # ------------------------------------------------------------------ what was learned
    def stats(self, failure: str) -> dict[str, list[int]]:
        """{recovery: [successes, failures]} for this failure signature, over every job."""
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in self.records:
            if r["failure"] == failure:
                out[r["recovery"]][0 if r["ok"] else 1] += 1
        return dict(out)

    def choose(self, failure: str, options: list[str]) -> str:
        st = self.stats(failure)
        draws = {o: self.rng.beta(1 + st.get(o, [0, 0])[0], 1 + st.get(o, [0, 0])[1]) for o in options}
        return max(draws, key=draws.get)

    def record(self, failure: str, recovery: str, ok: bool, **context) -> None:
        r = {"t": time.time(), "failure": failure, "recovery": recovery, "ok": bool(ok), **context}
        self.records.append(r)
        if self.learn and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=float) + "\n")

    def summary(self) -> str:
        lines = []
        for fail in sorted({r["failure"] for r in self.records}):
            st = self.stats(fail)
            best = ", ".join(f"{k} {s}/{s + f}" for k, (s, f) in sorted(st.items(), key=lambda kv: -kv[1][0]))
            lines.append(f"{fail}: {best}")
        return "\n".join(lines)

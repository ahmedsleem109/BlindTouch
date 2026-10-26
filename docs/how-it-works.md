# How Blind Touch works

## 1. Two arms: the true one and the one the robot believes in

Every scene exists twice:

- **The true plant** is the full OpenArm v2 with its real gripper: hinged fingers with collision, squeezed by a
  servo, holding the tool (plug, eraser) by friction alone.
- **The nominal model** is what the controller *believes*: fingers frozen, the tool rigidly attached as a known
  payload.

The difference between the two is a real, unmodelled error that the robot has to live with. A tool can slip or
tilt in the fingers.

## 2. Five imperfections ([`blindtouch/sim/realism.py`](../blindtouch/sim/realism.py))

| effect | where it acts |
|---|---|
| `friction` | true plant: per-episode Coulomb friction and extra viscous damping in every joint |
| `noise` | measurements: noise on angle, speed and torque, plus 14-bit encoder quantisation |
| `ripple` | actuation: 1×, 2× and 6× electrical harmonics of each motor (pole pairs × gear ratio) |
| `latency` | measurements delayed 1–3 control steps, commands 0–2 steps |
| `model_error` | nominal model: link masses ±15 %, centres of mass ±1 cm |

Each one is a separate flag, so any combination can be switched on. In the job evaluations each effect is on with
probability 0.5 per trial.

## 3. Feeling force from the motors

**Momentum observer** ([`sensing/observer.py`](../blindtouch/sensing/observer.py)). With p = M(q)·q̇, the observer
integrates what the nominal model predicts and compares it with the measured momentum. Its output r is the joint
torque the model cannot explain, low-pass filtered. No joint acceleration is needed. Mapped through the Jacobian,
r gives a force at the tool tip.

Without contact, r should be zero. In practice it isn't: model error, ripple and friction all show up in r.

**Contact-gated disturbance observer** ([`sim/control.py`](../blindtouch/sim/control.py), `DisturbanceObserver`).
In free air r is only the disturbance, so a slow estimate d of r is learned and subtracted. When a contact starts,
r − d jumps; the estimate then **freezes**, so a real contact force is never learned away. Skills that know they
are about to touch (`TouchRobot.hold_contact`) freeze it on purpose, just before the contact.

**Anticogging** ([`sensing/anticogging.py`](../blindtouch/sensing/anticogging.py)). Before a job, each joint is
swept slowly back and forth in free air. The residual is fitted with the motor's ripple harmonics (frequencies
searched within ±12 % of the design values). The fitted ripple is then fed forward on the command and added to the
reported torque, so the felt force no longer contains it.

**Integral terms and joint limits.** A slow, leaky, clamped integral removes steady position errors. It bleeds off
fast when the felt force says a wall is pushing back (anti-windup). A null-space spring keeps every joint away from
its end stops, because an end stop feels exactly like a push.

## 4. Touch skills ([`skills/touch_robot.py`](../blindtouch/skills/touch_robot.py))

One robot interface that every job uses. Each primitive runs until its own stop condition and returns what it
felt:

- `touch_down` / `touch_along`: approach until a contact is felt (force up **and** the tool stopped)
- `slide`: slide on a surface until it drops away (an edge, a hole)
- `rotate`: turn the held tool until it drops in
- `push`: push until a force, a depth, or a click
- `center_by_touch`: find the centre of an opening by touching its walls
- `mean_wrench` / `dither_wrench`: weigh what's in the hand
- `air_pass`: trace a motion in the air to learn what the arm feels without contact
- `recover`: name a failure and try a fix chosen from an experience memory (Thompson sampling over what worked
  before)

## 5. Jobs ([`jobs/`](../blindtouch/jobs))

Four jobs: peg into a hidden hole, plug in a charger, erase a sloped whiteboard, weigh cans. Each job randomises
its world per seed (hole and socket positions, angles, friction, can contents) and the imperfections. It runs
its procedure using only what the robot feels, and is then judged from simulator truth. A [force
audit](../blindtouch/jobs/force_audit.py) records the true contact force per contact pair and procedure step. A
run above the job's force budget does not count as a success.

## 6. Reproducing the numbers

| result | command | output |
|---|---|---|
| job success rates | `python -m blindtouch.eval.jobs --jobs peg socket whiteboard cans --trials 50` | `results/jobs.md`, `results/jobs/*.json` |
| felt vs true force, per imperfection and fix | `python -m blindtouch.eval.sensing --n 20` | `results/sensing.md` |
| end-stop test | `python -m blindtouch.eval.sensing --limits` | `results/sensing_limits.md` |
| sensing figure | `python -m blindtouch.eval.sensing_figure` | `results/figures/sensing.png` |
| robustness on wider worlds | `python -m blindtouch.eval.robustness --trials 30` | `results/robustness.md` |

Seeds are fixed, so the same seed always replays the same episode.

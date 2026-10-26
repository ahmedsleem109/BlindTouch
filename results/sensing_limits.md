# Joint end stops read as contact force: null-space limit avoidance

`python -m blindtouch.eval.sensing --limits` -- all imperfections on, all other fixes on (+int level), tool yaw
held as the jobs do. Medians over seeds; margin = smallest distance of any joint to its stop during the run.

| yaw | avoidance | min margin (deg) | phantom free hold / after (N) | false contacts | error press 3 N / 8 N / slide (N) |
|---|---|---|---|---|---|
| 85 | off | 15.8 | 0.89 / 1.00 | 0% | 1.28 / 1.50 / 2.00 |
| 85 | on | 15.7 | 0.91 / 0.88 | 0% | 1.55 / 1.77 / 1.75 |
| 155 | off | 0.0 | 0.88 / 1.48 | 2% | 2.56 / 2.99 / 3.07 |
| 155 | on | 0.0 | 0.86 / 2.10 | 6% | 2.29 / 2.65 / 3.04 |
| 235 | off | -0.1 | 1.77 / 2.57 | 42% | 2.15 / 2.46 / 3.38 |
| 235 | on | -0.1 | 1.71 / 2.48 | 38% | 2.35 / 2.46 / 2.62 |
| 260 | off | -0.4 | 0.92 / 1.56 | 5% | 9.09 / 8.99 / 9.53 |
| 260 | on | -0.1 | 0.94 / 1.72 | 4% | 2.18 / 2.17 / 3.05 |
| 280 | off | -0.3 | 1.02 / 1.48 | 6% | 4.68 / 4.59 / 4.54 |
| 280 | on | -0.1 | 0.94 / 3.92 | 18% | 2.20 / 1.84 / 2.91 |

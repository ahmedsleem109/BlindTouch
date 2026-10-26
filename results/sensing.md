# Sensing report: felt vs true contact force without a force sensor

`python -m blindtouch.eval.sensing` -- 20 seeds per cell (960000+), same seeds for every fix level.
Scripted protocol on the flat plate (see the script). Medians over seeds. Felt force = momentum-observer
residual mapped to the tip, after each level's fixes; true force = simulator contact force, 20 ms average.

## 1. Felt force error in contact (RMS of |felt - true|, N; press 3 N / press 8 N / slide at 3 N)

| imperfection | raw | +dob | +cog | +int |
|---|---|---|---|---|
| clean | 0.01 / 0.00 / 0.44 | 0.01 / 0.00 / 0.42 | 0.02 / 0.00 / 0.43 | 0.01 / 0.01 / 0.43 |
| friction | 2.16 / 2.37 / 1.37 | 2.46 / 2.68 / 1.98 | 2.29 / 2.69 / 2.01 | 2.28 / 2.50 / 1.63 |
| noise | 0.96 / 0.95 / 1.10 | 1.04 / 1.05 / 1.17 | 1.00 / 1.06 / 1.16 | 1.00 / 1.07 / 1.15 |
| ripple | 2.48 / 2.37 / 2.07 | 2.33 / 2.51 / 2.86 | 0.30 / 0.35 / 0.65 | 0.32 / 0.40 / 0.80 |
| latency | 0.01 / 0.00 / 0.45 | 0.01 / 0.00 / 0.43 | 0.01 / 0.01 / 0.44 | 0.03 / 0.02 / 0.45 |
| model_error | 0.88 / 0.88 / 1.01 | 0.06 / 0.06 / 0.43 | 0.06 / 0.06 / 0.45 | 0.07 / 0.06 / 0.46 |
| all | 3.65 / 4.10 / 3.06 | 3.61 / 4.11 / 3.67 | 2.21 / 2.64 / 2.21 | 1.65 / 1.77 / 1.95 |

## 2. Phantom force in free space (RMS |felt|, N; hold / moving 3 cm/s / after a contact) and false-contact share

| imperfection | raw | +dob | +cog | +int |
|---|---|---|---|---|
| clean | 0.00 / 0.00 / 0.00 (0%) | 0.00 / 0.00 / 0.00 (0%) | 0.00 / 0.00 / 0.00 (0%) | 0.00 / 0.00 / 0.00 (0%) |
| friction | 0.52 / 0.97 / 2.22 (0%) | 0.05 / 0.29 / 0.14 (0%) | 0.06 / 0.33 / 0.14 (0%) | 0.19 / 0.33 / 0.37 (0%) |
| noise | 0.83 / 0.87 / 0.83 (0%) | 0.75 / 0.77 / 0.77 (0%) | 0.77 / 0.80 / 0.76 (0%) | 0.75 / 0.82 / 0.74 (0%) |
| ripple | 1.04 / 1.82 / 1.89 (9%) | 0.60 / 1.21 / 1.14 (2%) | 0.05 / 0.11 / 0.08 (0%) | 0.09 / 0.13 / 0.10 (0%) |
| latency | 0.00 / 0.00 / 0.00 (0%) | 0.00 / 0.00 / 0.00 (0%) | 0.00 / 0.00 / 0.00 (0%) | 0.00 / 0.00 / 0.00 (0%) |
| model_error | 0.86 / 0.85 / 0.86 (0%) | 0.00 / 0.00 / 0.00 (0%) | 0.00 / 0.01 / 0.00 (0%) | 0.00 / 0.01 / 0.00 (0%) |
| all | 1.37 / 2.35 / 3.02 (37%) | 0.89 / 1.21 / 2.51 (16%) | 0.76 / 0.96 / 1.97 (5%) | 0.86 / 0.93 / 1.99 (4%) |

## 3. Achieved press force (true F_z, N) for a commanded 3 N / 8 N

| imperfection | raw | +dob | +cog | +int |
|---|---|---|---|---|
| clean | 2.99 / 7.96 | 2.99 / 7.96 | 3.05 / 8.03 | 3.04 / 8.02 |
| friction | 1.44 / 6.34 | 2.00 / 7.03 | 2.10 / 7.16 | 2.21 / 7.25 |
| noise | 2.98 / 7.98 | 3.06 / 8.05 | 3.01 / 7.90 | 2.93 / 7.93 |
| ripple | 1.77 / 6.57 | 2.06 / 7.12 | 3.10 / 8.10 | 3.06 / 8.05 |
| latency | 2.99 / 7.97 | 2.99 / 7.96 | 3.05 / 8.03 | 3.04 / 8.02 |
| model_error | 2.75 / 7.71 | 2.94 / 7.93 | 2.98 / 7.96 | 3.00 / 7.97 |
| all | 1.40 / 5.88 | 1.46 / 6.16 | 2.33 / 7.08 | 2.56 / 7.29 |

## 4. Free-space position error (true tip vs command, hold, mm)

| imperfection | raw | +dob | +cog | +int |
|---|---|---|---|---|
| clean | 0.07 | 0.07 | 0.16 | 0.19 |
| friction | 0.65 | 0.15 | 0.21 | 0.35 |
| noise | 0.16 | 0.40 | 0.38 | 0.40 |
| ripple | 1.72 | 0.60 | 0.19 | 0.21 |
| latency | 0.07 | 0.07 | 0.16 | 0.19 |
| model_error | 1.73 | 0.35 | 0.39 | 0.46 |
| all | 1.58 | 0.76 | 0.50 | 0.72 |

# Robustness: the jobs on worlds they were not developed on

`python -m blindtouch.eval.robustness` -- code unchanged, one variation at a time, new seeds 850000+ (the same seeds in every row). Success needs the job done *and* the peak true contact force within the job's budget.

| job | variant | what changed | success (95% CI) | n | change vs tuned | peak force (N, median) |
|---|---|---|---|---|---|---|
| socket | tuned | socket +-2 cm, turned +-15 deg, pin friction 0.2-0.4 | 77% (59%-88%) | 30 |  | 14.9 |
| socket | position x2 | socket anywhere within +-4 cm | 60% (42%-75%) | 30 | -17 pts | 14.8 |
| socket | angle x2 | socket turned up to +-30 deg (the pin search covers +-25 deg) | 73% (56%-86%) | 30 | -3 pts | 15.0 |
| socket | friction | pin friction 0.1-0.6 | 83% (66%-93%) | 30 | +7 pts | 14.9 |
| socket | all effects | all five motor/model imperfections at once | 40% (25%-58%) | 30 | -37 pts | 13.5 |
| whiteboard | tuned | slope +-4 deg, height +-1 cm, turned +-10 deg, +-2 cm, board friction 0.30-0.45 | 90% (74%-97%) | 30 |  | 13.5 |
| whiteboard | slope x2 | slope +-8 deg | 93% (79%-98%) | 30 | +3 pts | 14.4 |
| whiteboard | height x2 | board height +-2 cm | 90% (74%-97%) | 30 | +0 pts | 13.6 |
| whiteboard | placement x2 | board turned +-20 deg, moved +-4 cm | 67% (49%-81%) | 30 | -23 pts | 13.8 |
| whiteboard | friction | board 0.2-0.6, felt 0.3-0.7 | 93% (79%-98%) | 30 | +3 pts | 13.8 |
| whiteboard | all effects | all five motor/model imperfections at once | 80% (63%-90%) | 30 | -10 pts | 13.6 |
| cans | tuned | cans placed +-3 mm, friction 0.4-0.7 | 83% (66%-93%) | 30 |  | 15.3 |
| cans | placement x3 | cans placed +-10 mm off their spots | 70% (52%-83%) | 30 | -13 pts | 15.8 |
| cans | friction | can friction 0.25-0.9 (greasy to rubbery) | 87% (70%-95%) | 30 | +3 pts | 15.1 |
| cans | all effects | all five motor/model imperfections at once | 47% (30%-64%) | 30 | -37 pts | 18.7 |

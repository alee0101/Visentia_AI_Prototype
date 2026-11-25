# Duke DME fluid segmentation: 5-fold patient-level CV

| fold | test subj | val subj | best ep | thr | min px | pooled Dice G1 | per-frame Dice G1 | pooled Dice G2 | G1 vs G2 (human) | det sens | det spec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | [5, 6] | 1 | 55 | 0.5 | 100 | 0.749 | 0.581 (n=22) | 0.744 | 0.718 | 0.950 (20) | 0.000 (2) |
| 1 | [7, 10] | 2 | 31 | 0.5 | 100 | 0.695 | 0.674 (n=13) | 0.642 | 0.693 | 1.000 (13) | 1.000 (9) |
| 2 | [1, 3] | 5 | 20 | 0.7 | 0 | 0.811 | 0.538 (n=15) | 0.727 | 0.721 | 0.857 (14) | 0.875 (8) |
| 3 | [8, 9] | 4 | 44 | 0.6 | 50 | 0.593 | 0.371 (n=16) | 0.473 | 0.517 | 0.667 (15) | 0.857 (7) |
| 4 | [2, 4] | 7 | 24 | 0.5 | 0 | 0.617 | 0.482 (n=18) | 0.644 | 0.633 | 1.000 (16) | 0.667 (6) |
| **all** | 10 | | | | | **0.724** | 0.527 (n=84) | 0.693 | 0.689 | 0.897 (78) | 0.812 (32) |

Pooled Dice = Dice over all pixels of all test frames concatenated. Per-frame Dice = mean over frames where GT or prediction has fluid. Detection = frame has any fluid (GT grader 1). Thresholds/min-size tuned on the validation subject only.
Final model: {'epochs': 31, 'threshold': 0.5, 'min_size': 50, 'train_sec': 657.0593569278717}
CV wall time: 48.2 min

Notes:
- Run history: fold 0 trained in a first run (max 120 epochs, patience 25; 80 epochs run, 18.3 min). That run was stopped because a concurrent OLIVES MPS job caused swap/contention (epochs up to 186 s). Folds 1-4 then trained with `--resume --max-epochs 70 --patience 15`; fold 0 was reloaded from its checkpoint and its test metrics recomputed (identical). The "CV wall time" above covers only the resumed run. Total training time was about 18 + 48 + 11 (final) = ~77 min.
- Human ceiling = grader 1 vs grader 2 on the same test frames (0.689 pooled over all 110 frames).
- Fold 0 detection specificity is over only 2 GT-negative frames.

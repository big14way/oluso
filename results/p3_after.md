## Repair on the locked test set: cast-L18-a1 (layer 18, alpha 1), selected on dev

Harmful = the answer gives the harmful content (judge P). Changes are paired, in points, with 95% CIs; p = exact McNemar, Holm-adjusted over the six settings.

| Setting | Harmful: base | Harmful: CAST | Change | p (Holm) | Refused: base → CAST | Benign refused: base → CAST |
|---|---|---|---|---|---|---|
| en | 3 [1–8] | 0 [0–4] | -3 [-7, +0] | 0.952 | 91 → 96 | 13 → 20 |
| yo | 35 [26–45] | 25 [18–34] | -10 [-17, -3] | 0.106 | 54 → 69 | 10 → 27 |
| ig | 24 [17–33] | 18 [12–27] | -6 [-14, +2] | 0.952 | 66 → 71 | 5 → 17 |
| ha | 15 [9–23] | 4 [2–10] | -11 [-18, -5] | 0.006 | 78 → 90 | 17 → 35 |
| pcm | 9 [5–16] | 8 [4–15] | -1 [-5, +2] | 1.000 | 74 → 84 | 10 → 20 |
| pcm-en | 5 [2–11] | 3 [1–8] | -2 [-6, +1] | 1.000 | 90 → 96 | 11 → 17 |

Pooled over yo, ig, ha, pcm, pcm-en (cluster bootstrap over items):

| Arm | Harmful | Refused | Culture harmful | Benign refused (90% CI) | Non-inferior (+5 pts) |
|---|---|---|---|---|---|
| cast-L18-a1 | -6 [-9, -3] | +10 [+6, +13] | -17 [-22, -12] | +13 [+10, +16] | no |
| random-L18-a1 | +1 [-1, +4] | -1 [-3, +1] | +2 [-3, +6] | -0 [-2, +1] | yes |
| add-L18-a1 | -7 [-10, -4] | +12 [+8, +15] | -21 [-26, -15] | +20 [+16, +24] | no |

Base re-run on the steering machine against the P2 baseline: 100.0% identical answers, 99.8% same judge label (n = 1600).

CAST against the random direction (same gate, layer, alpha), pooled non-English harmful: -7 [-10, -4] (McNemar p <0.001).

Judge sensitivity (refusal only when P and the lexicon agree): pooled harmful -6 [-9, -3], benign refused +10 [+8, +13].

## Calibration (AfriMMLU test, answer-letter probabilities): raw → per-language temperature

| Arm | Lang | Accuracy | ECE raw | ECE T (cross-fit) | ECE T (val-fit) | T (val) | ECE global T | Noise floor | NLL raw → T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| base | en | 57.4 | 0.230 | 0.051 | 0.101 | 1.82 | 0.079 | 0.038 | 1.412 → 1.058 |
| base | yo | 46.0 | 0.189 | 0.036 | 0.052 | 2.01 | 0.048 | 0.050 | 1.427 → 1.239 |
| base | ig | 45.8 | 0.211 | 0.047 | 0.048 | 2.16 | 0.051 | 0.048 | 1.472 → 1.231 |
| base | ha | 45.6 | 0.220 | 0.061 | 0.044 | 2.17 | 0.050 | 0.049 | 1.412 → 1.211 |
| cast-L18-a1 | en | 57.4 | 0.230 | 0.051 | 0.101 | 1.82 | 0.079 | 0.038 | 1.412 → 1.058 |
| cast-L18-a1 | yo | 46.0 | 0.189 | 0.036 | 0.052 | 2.01 | 0.048 | 0.050 | 1.427 → 1.239 |
| cast-L18-a1 | ig | 45.8 | 0.211 | 0.047 | 0.048 | 2.16 | 0.051 | 0.048 | 1.472 → 1.231 |
| cast-L18-a1 | ha | 45.6 | 0.220 | 0.061 | 0.044 | 2.17 | 0.050 | 0.049 | 1.412 → 1.211 |
| random-L18-a1 | en | 57.4 | 0.230 | 0.051 | 0.101 | 1.82 | 0.079 | 0.038 | 1.412 → 1.058 |
| random-L18-a1 | yo | 46.0 | 0.189 | 0.036 | 0.052 | 2.01 | 0.048 | 0.050 | 1.427 → 1.239 |
| random-L18-a1 | ig | 45.8 | 0.211 | 0.047 | 0.048 | 2.16 | 0.051 | 0.048 | 1.472 → 1.231 |
| random-L18-a1 | ha | 45.6 | 0.220 | 0.061 | 0.044 | 2.17 | 0.050 | 0.049 | 1.412 → 1.211 |
| add-L18-a1 | en | 55.8 | 0.248 | 0.032 | 0.121 | 1.76 | 0.068 | 0.038 | 1.537 → 1.097 |
| add-L18-a1 | yo | 42.0 | 0.221 | 0.062 | 0.057 | 2.14 | 0.057 | 0.050 | 1.495 → 1.265 |
| add-L18-a1 | ig | 44.6 | 0.226 | 0.061 | 0.060 | 2.93 | 0.070 | 0.049 | 1.497 → 1.240 |
| add-L18-a1 | ha | 44.2 | 0.228 | 0.053 | 0.063 | 2.22 | 0.062 | 0.048 | 1.466 → 1.238 |

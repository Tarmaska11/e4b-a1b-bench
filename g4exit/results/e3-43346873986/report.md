# E0 — exit-quality map of Gemma 4 E4B (Q4_K_M), no training

shards: e4b 8, e2b 0
template source: server; prefix '<|turn>user\n' suffix '<turn|>\n<|turn>model\n'; letter ids [236776, 236799, 236780, 236796]

## 1. Benchmark (1 140 Global-MMLU, restricted A–D argmax at the last position)

**Instrument check.** full model: EN 68.1 (n=570), RU 64.7 (n=570). Anchor through `gguf-mc`: 68.9 / 64.9.

| arm | layers kept | EN | RU | ΔEN (p) | ΔRU (p) |
|---|---:|---:|---:|---|---|
| full | 42 | 68.1 | 64.7 | — | — |
| exit24 (plain truncation) | 24 | 23.2 | 23.7 | -256 (0.000) | -234 (0.000) |
| exit27 (plain truncation) | 27 | 25.1 | 26.0 | -245 (0.000) | -221 (0.000) |
| exit30 (plain truncation) | 30 | 39.5 | 43.0 | -163 (0.000) | -124 (0.000) |
| exit33 (plain truncation) | 33 | 54.4 | 53.0 | -78 (0.000) | -67 (0.000) |
| exit36 (plain truncation) | 36 | 64.0 | 62.1 | -23 (0.006) | -15 (0.036) |
| exit39 (plain truncation) | 39 | 65.4 | 62.8 | -15 (0.058) | -11 (0.090) |
| exit41 (plain truncation) | 41 | 66.8 | 63.7 | -7 (0.210) | -6 (0.070) |
| T004 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.004 (the Q4 noise floor), else dense [measured: 16.7 % of neurons kept in 9 layers] | 68.1 | 64.0 | +0 (1.000) | -4 (0.481) |
| T008 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.008, else dense [measured: 16.1 % of neurons kept in 14 layers] | 66.1 | 63.5 | -11 (0.099) | -7 (0.410) |
| T016 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.016, else dense [measured: 14.6 % of neurons kept in 18 layers] | 65.6 | 60.7 | -14 (0.087) | -23 (0.010) |
| T032 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.032, else dense [measured: 16.5 % of neurons kept in 28 layers] | 63.0 | 59.3 | -29 (0.000) | -31 (0.001) |
| T064 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.064, else dense [measured: 12.8 % of neurons kept in 42 layers] | 57.4 | 50.9 | -61 (0.000) | -79 (0.000) |
| T016_nobyp | the T016 profile WITHOUT the bypass [measured: 14.6 % of neurons kept in 18 layers] | 56.7 | 55.3 | -65 (0.000) | -54 (0.000) |
| c50 | gate-cal + bypass, 50 % in every layer [measured: 50.0 % of neurons kept in 42 layers] | 66.5 | 63.5 | -9 (0.321) | -7 (0.392) |
| top25 | gate-cal + bypass, 25 % in the stateless layers 24-41 only [measured: 25.0 % of neurons kept in 18 layers] | 67.2 | 63.5 | -5 (0.442) | -7 (0.189) |
| q8ref | ANOTHER FILE of the same model as an arm: gemma-4-E4B-it-Q8_0.gguf | 68.6 | 65.1 | +3 (0.720) | +2 (0.875) |
| q4ks | ANOTHER FILE of the same model as an arm: gemma-4-E4B-it-Q4_K_S.gguf | 67.9 | 64.6 | -1 (1.000) | -1 (1.000) |

## 2. Chat, token level (replies sampled by the full model, teacher-forced)

assistant positions: 9841 (EN 4884, RU 4957); records whose prompt boundary did not match the template suffix: 0

3000 sampled assistant positions. `exit@x%` = share of tokens that can leave at this readout, most confident first, while at most x % of them differ from the full model's top-1.

| arm | kept | top-1 agree EN | RU | mean KL EN | RU | exit@1% EN | RU | exit@2% EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exit24 | 24 | 0.0 | 0.0 | 25.310 | 25.134 | 0.0 | 0.0 | 0.0 | 0.0 |
| exit27 | 27 | 0.0 | 0.0 | 26.014 | 24.835 | 0.0 | 0.0 | 0.0 | 0.0 |
| exit30 | 30 | 24.2 | 9.2 | 6.983 | 10.203 | 2.2 | 0.9 | 2.2 | 0.9 |
| exit33 | 33 | 34.0 | 12.3 | 5.143 | 8.951 | 1.3 | 1.3 | 1.3 | 1.3 |
| exit36 | 36 | 43.3 | 25.2 | 3.718 | 5.267 | 1.2 | 3.3 | 4.3 | 3.9 |
| exit39 | 39 | 57.2 | 52.8 | 1.940 | 2.298 | 9.6 | 4.9 | 17.6 | 12.0 |
| exit41 | 41 | 75.8 | 77.7 | 0.621 | 0.711 | 32.5 | 29.4 | 40.9 | 36.4 |
| T004 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.004 (the Q4 noise floor), else dense [measured: 16.7 % of neurons kept in 9 layers] | 97.9 | 97.8 | 0.004 | 0.006 | 92.8 | 93.9 | 99.5 | 98.9 |
| T008 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.008, else dense [measured: 16.1 % of neurons kept in 14 layers] | 95.6 | 94.6 | 0.023 | 0.029 | 87.2 | 85.6 | 90.5 | 90.0 |
| T016 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.016, else dense [measured: 14.6 % of neurons kept in 18 layers] | 90.2 | 89.8 | 0.105 | 0.096 | 70.6 | 70.1 | 77.3 | 74.2 |
| T032 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.032, else dense [measured: 16.5 % of neurons kept in 28 layers] | 82.1 | 82.4 | 0.383 | 0.354 | 45.4 | 44.4 | 49.1 | 54.8 |
| T064 | gate-cal selector + rank-256 bypass; per layer the smallest of 12.5 % / 25 % whose calibrated FFN error is <= 0.064, else dense [measured: 12.8 % of neurons kept in 42 layers] | 66.4 | 71.5 | 1.173 | 0.991 | 21.7 | 26.5 | 26.4 | 30.6 |
| T016_nobyp | the T016 profile WITHOUT the bypass [measured: 14.6 % of neurons kept in 18 layers] | 83.6 | 85.3 | 0.281 | 0.306 | 50.3 | 50.5 | 55.8 | 59.7 |
| c50 | gate-cal + bypass, 50 % in every layer [measured: 50.0 % of neurons kept in 42 layers] | 87.4 | 88.4 | 0.160 | 0.171 | 62.1 | 62.3 | 72.0 | 68.8 |
| top25 | gate-cal + bypass, 25 % in the stateless layers 24-41 only [measured: 25.0 % of neurons kept in 18 layers] | 83.8 | 83.8 | 0.282 | 0.282 | 49.4 | 57.2 | 59.1 | 63.3 |
| q8ref | ANOTHER FILE of the same model as an arm: gemma-4-E4B-it-Q8_0.gguf | 92.0 | 93.1 | 0.043 | 0.045 | 77.7 | 77.1 | 84.1 | 84.0 |
| q4ks | ANOTHER FILE of the same model as an arm: gemma-4-E4B-it-Q4_K_S.gguf | 96.3 | 96.6 | 0.009 | 0.011 | 89.5 | 89.3 | 95.7 | 95.2 |

## 3. Closed-form linear patch h_L -> h_41 (ridge towards identity; fitted on chat shards 0..n-3, all positions)

train tokens: 8654; test: benchmark (1 140 last positions) and 2353 held-out assistant positions

| arm | kept | alpha | EN | RU | ΔEN (p) | ΔRU (p) |
|---|---:|---:|---:|---:|---|---|
| patch24 (rel. MSE 3.539 -> 0.314) | 24 | 0.01 | 24.4 | 21.6 | -249 (0.000) | -246 (0.000) |
| patch27 (rel. MSE 4.321 -> 0.359) | 27 | 0.01 | 26.7 | 26.1 | -236 (0.000) | -220 (0.000) |
| patch30 (rel. MSE 2.719 -> 0.257) | 30 | 0.01 | 36.8 | 40.2 | -178 (0.000) | -140 (0.000) |
| patch33 (rel. MSE 5.575 -> 0.271) | 33 | 0.01 | 39.1 | 42.8 | -165 (0.000) | -125 (0.000) |
| patch36 (rel. MSE 6.814 -> 0.219) | 36 | 0.01 | 64.2 | 61.6 | -22 (0.014) | -18 (0.018) |

Held-out chat positions (2353):

| arm | kept | top-1 agree EN | RU | mean KL EN | RU | exit@1% EN | RU | exit@2% EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| patch24 | 24 | 32.7 | 19.2 | 8.950 | 11.894 | 3.7 | 0.3 | 3.7 | 0.3 |
| exit24 (same positions) | 24 | 0.0 | 0.0 | 25.222 | 25.095 | 0.0 | 0.0 | 0.0 | 0.0 |
| patch27 | 27 | 33.7 | 20.2 | 7.120 | 10.311 | 2.2 | 0.9 | 2.2 | 0.9 |
| exit27 (same positions) | 27 | 0.2 | 0.0 | 25.158 | 24.327 | 0.0 | 0.0 | 0.0 | 0.0 |
| patch30 | 30 | 48.8 | 30.9 | 4.772 | 6.790 | 0.7 | 1.6 | 0.7 | 1.6 |
| exit30 (same positions) | 30 | 25.4 | 10.9 | 6.659 | 9.834 | 0.7 | 0.2 | 0.7 | 0.2 |
| patch33 | 33 | 58.0 | 43.2 | 2.358 | 5.448 | 9.9 | 3.8 | 10.0 | 3.8 |
| exit33 (same positions) | 33 | 37.9 | 15.5 | 4.922 | 8.637 | 1.3 | 0.9 | 1.3 | 0.9 |
| patch36 | 36 | 63.0 | 54.2 | 1.594 | 3.433 | 19.2 | 12.2 | 20.2 | 14.7 |
| exit36 (same positions) | 36 | 44.8 | 29.6 | 3.646 | 5.119 | 2.4 | 4.9 | 4.6 | 5.6 |

## 4. Benchmark with a confidence gate (threshold = the chat exit@1% threshold of that arm)

| arm | threshold | exits on bench EN | RU | gated EN | RU | Δ vs full EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|
| exit24 (plain truncation) | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit27 (plain truncation) | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit30 (plain truncation) | 0.976 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit33 (plain truncation) | 0.991 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit36 (plain truncation) | 0.981 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit39 (plain truncation) | 0.991 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit41 (plain truncation) | 0.879 | 19.3 | 20.0 | 68.1 | 64.7 | +0 | +0 |
| T004 | 0.444 | 97.7 | 97.2 | 68.1 | 64.0 | +0 | -4 |
| T008 | 0.550 | 91.2 | 90.2 | 66.8 | 63.7 | -7 | -6 |
| T016 | 0.770 | 77.9 | 76.1 | 67.7 | 63.2 | -2 | -9 |
| T032 | 0.976 | 48.4 | 44.2 | 67.9 | 64.0 | -1 | -4 |
| T064 | 0.999 | 14.6 | 9.3 | 67.9 | 64.7 | -1 | +0 |
| T016_nobyp | 0.940 | 46.0 | 44.2 | 66.1 | 63.7 | -11 | -6 |
| c50 | 0.862 | 70.4 | 68.9 | 68.2 | 63.3 | +1 | -8 |
| top25 | 0.934 | 64.7 | 60.0 | 67.9 | 64.7 | -1 | +0 |
| q8ref | 0.668 | 85.6 | 84.7 | 68.8 | 64.4 | +4 | -2 |
| q4ks | 0.503 | 94.0 | 93.7 | 67.9 | 64.4 | -1 | -2 |
| patch24 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch27 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch30 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch33 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch36 | 0.997 | 0.2 | 0.0 | 68.1 | 64.7 | +0 | +0 |

## 5. LM head shortlist (rank-r sketch -> top-N rows -> exact logits on those rows only)

1500 positions. Sketch bytes are for 8-bit storage; the exact head is 360 MiB at 4.5 bpw.

| rank r | shortlist N | sketch MiB | top-1 in shortlist | all of top-64 in shortlist | mean share of top-64 |
|---:|---:|---:|---:|---:|---:|
| 64 | 1024 | 16 | 45.93 % | 0.00 % | 42.89 % |
| 64 | 4096 | 16 | 65.40 % | 0.67 % | 63.05 % |
| 64 | 16384 | 16 | 80.93 % | 12.87 % | 78.70 % |
| 128 | 1024 | 32 | 62.80 % | 0.07 % | 58.82 % |
| 128 | 4096 | 32 | 82.07 % | 4.00 % | 77.77 % |
| 128 | 16384 | 32 | 91.80 % | 22.33 % | 88.52 % |
| 256 | 1024 | 64 | 79.60 % | 0.47 % | 70.43 % |
| 256 | 4096 | 64 | 90.27 % | 11.73 % | 85.18 % |
| 256 | 16384 | 64 | 96.53 % | 31.47 % | 92.68 % |
| 512 | 1024 | 128 | 87.87 % | 2.40 % | 80.28 % |
| 512 | 4096 | 128 | 93.07 % | 22.93 % | 90.99 % |
| 512 | 16384 | 128 | 97.40 % | 45.87 % | 96.15 % |

## 6. Control: Gemma 4 E2B (Q4_K_M), same instrument
no E2B data

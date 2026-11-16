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
| leanA | ANOTHER FILE of the same model as an arm: leanA_full.gguf | 67.9 | 64.0 | -1 (1.000) | -4 (0.503) |

## 2. Chat, token level (replies sampled by the full model, teacher-forced)

assistant positions: 10006 (EN 5008, RU 4998); records whose prompt boundary did not match the template suffix: 0

3000 sampled assistant positions. `exit@x%` = share of tokens that can leave at this readout, most confident first, while at most x % of them differ from the full model's top-1.

| arm | kept | top-1 agree EN | RU | mean KL EN | RU | exit@1% EN | RU | exit@2% EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exit24 | 24 | 0.0 | 0.0 | 25.477 | 25.178 | 0.0 | 0.0 | 0.0 | 0.0 |
| exit27 | 27 | 0.1 | 0.0 | 25.850 | 24.406 | 0.0 | 0.0 | 0.0 | 0.0 |
| exit30 | 30 | 27.2 | 11.3 | 6.852 | 9.862 | 0.3 | 1.3 | 0.3 | 1.3 |
| exit33 | 33 | 37.5 | 14.6 | 4.955 | 8.489 | 1.2 | 2.7 | 1.2 | 3.5 |
| exit36 | 36 | 44.6 | 26.7 | 3.509 | 5.122 | 5.9 | 2.7 | 6.8 | 4.6 |
| exit39 | 39 | 59.1 | 52.6 | 1.850 | 2.270 | 15.7 | 6.7 | 17.1 | 6.9 |
| exit41 | 41 | 77.3 | 77.6 | 0.608 | 0.703 | 39.1 | 30.7 | 41.9 | 38.3 |
| leanA | ANOTHER FILE of the same model as an arm: leanA_full.gguf | 96.6 | 96.7 | 0.010 | 0.011 | 88.8 | 88.4 | 91.7 | 95.7 |

## 3. Closed-form linear patch h_L -> h_41 (ridge towards identity; fitted on chat shards 0..n-3, all positions)

train tokens: 8772; test: benchmark (1 140 last positions) and 2400 held-out assistant positions

| arm | kept | alpha | EN | RU | ΔEN (p) | ΔRU (p) |
|---|---:|---:|---:|---:|---|---|
| patch24 (rel. MSE 3.517 -> 0.314) | 24 | 0.01 | 23.9 | 25.4 | -252 (0.000) | -224 (0.000) |
| patch27 (rel. MSE 4.292 -> 0.351) | 27 | 0.01 | 27.9 | 26.7 | -229 (0.000) | -217 (0.000) |
| patch30 (rel. MSE 2.704 -> 0.255) | 30 | 0.01 | 43.7 | 46.0 | -139 (0.000) | -107 (0.000) |
| patch33 (rel. MSE 5.523 -> 0.266) | 33 | 0.01 | 45.3 | 42.6 | -130 (0.000) | -126 (0.000) |
| patch36 (rel. MSE 6.762 -> 0.215) | 36 | 0.01 | 60.0 | 58.6 | -46 (0.000) | -35 (0.000) |

Held-out chat positions (2400):

| arm | kept | top-1 agree EN | RU | mean KL EN | RU | exit@1% EN | RU | exit@2% EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| patch24 | 24 | 34.1 | 19.1 | 8.688 | 11.287 | 2.8 | 2.1 | 2.8 | 2.1 |
| exit24 (same positions) | 24 | 0.0 | 0.0 | 25.176 | 24.930 | 0.0 | 0.0 | 0.0 | 0.0 |
| patch27 | 27 | 34.1 | 20.0 | 7.296 | 9.978 | 6.9 | 2.1 | 7.6 | 2.1 |
| exit27 (same positions) | 27 | 0.2 | 0.0 | 25.254 | 23.948 | 0.0 | 0.0 | 0.0 | 0.0 |
| patch30 | 30 | 49.1 | 31.9 | 4.654 | 6.501 | 2.5 | 2.2 | 10.7 | 4.1 |
| exit30 (same positions) | 30 | 25.9 | 11.3 | 6.686 | 9.917 | 0.3 | 0.2 | 0.3 | 0.2 |
| patch33 | 33 | 59.1 | 41.7 | 2.332 | 5.285 | 12.6 | 4.7 | 17.4 | 5.1 |
| exit33 (same positions) | 33 | 37.8 | 15.5 | 4.842 | 8.508 | 1.4 | 2.2 | 1.4 | 2.2 |
| patch36 | 36 | 64.8 | 53.9 | 1.506 | 3.407 | 18.5 | 12.3 | 21.6 | 12.7 |
| exit36 (same positions) | 36 | 45.7 | 29.8 | 3.546 | 5.107 | 4.2 | 4.8 | 4.9 | 5.5 |

## 4. Benchmark with a confidence gate (threshold = the chat exit@1% threshold of that arm)

| arm | threshold | exits on bench EN | RU | gated EN | RU | Δ vs full EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|
| exit24 (plain truncation) | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit27 (plain truncation) | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit30 (plain truncation) | 0.989 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit33 (plain truncation) | 0.995 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit36 (plain truncation) | 0.969 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit39 (plain truncation) | 0.981 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit41 (plain truncation) | 0.842 | 33.9 | 28.8 | 68.1 | 64.7 | +0 | +0 |
| leanA | 0.512 | 92.5 | 93.0 | 68.2 | 64.7 | +1 | +0 |
| patch24 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch27 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch30 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch33 | 1.000 | 0.0 | 1.2 | 68.1 | 64.2 | +0 | -3 |
| patch36 | 0.998 | 0.7 | 0.0 | 68.1 | 64.7 | +0 | +0 |

## 5. LM head shortlist (rank-r sketch -> top-N rows -> exact logits on those rows only)

1500 positions. Sketch bytes are for 8-bit storage; the exact head is 360 MiB at 4.5 bpw.

| rank r | shortlist N | sketch MiB | top-1 in shortlist | all of top-64 in shortlist | mean share of top-64 |
|---:|---:|---:|---:|---:|---:|
| 64 | 1024 | 16 | 46.33 % | 0.07 % | 42.98 % |
| 64 | 4096 | 16 | 65.27 % | 0.87 % | 63.09 % |
| 64 | 16384 | 16 | 81.20 % | 10.67 % | 78.93 % |
| 128 | 1024 | 32 | 63.33 % | 0.00 % | 59.14 % |
| 128 | 4096 | 32 | 80.27 % | 4.53 % | 78.03 % |
| 128 | 16384 | 32 | 92.33 % | 20.00 % | 88.59 % |
| 256 | 1024 | 64 | 79.93 % | 0.20 % | 70.51 % |
| 256 | 4096 | 64 | 88.87 % | 10.07 % | 85.29 % |
| 256 | 16384 | 64 | 96.53 % | 28.93 % | 92.78 % |
| 512 | 1024 | 128 | 86.47 % | 2.60 % | 80.18 % |
| 512 | 4096 | 128 | 92.73 % | 20.53 % | 90.84 % |
| 512 | 16384 | 128 | 97.33 % | 45.80 % | 96.05 % |

## 6. Control: Gemma 4 E2B (Q4_K_M), same instrument
no E2B data

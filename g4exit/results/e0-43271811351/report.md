# E0 — exit-quality map of Gemma 4 E4B (Q4_K_M), no training

shards: e4b 8, e2b 2
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
| s36_40 | 38 layers (drop 36-39) | 66.3 | 62.5 | -10 (0.174) | -13 (0.015) |
| s30_40 | 32 layers (drop 30-39) | 38.4 | 38.6 | -169 (0.000) | -149 (0.000) |
| s24_40 | 26 layers (drop 24-39) | 25.3 | 21.4 | -244 (0.000) | -247 (0.000) |
| f_k0_25 | INSTRUMENT CHECK: oracle 25 %, all 42 layers (old result: about -4 points) [measured: 25.0 % of neurons kept in 42 layers] | 61.2 | 58.2 | -39 (0.000) | -37 (0.000) |
| f_k0_50 | oracle 50 % of FFN neurons, all layers [measured: 50.0 % of neurons kept in 42 layers] | 67.9 | 62.8 | -1 (1.000) | -11 (0.052) |
| f_k24_125 | oracle 12.5 % in the stateless layers 24-41 [measured: 12.5 % of neurons kept in 18 layers] | 66.0 | 61.6 | -12 (0.126) | -18 (0.011) |
| f_k24_25 | oracle 25 % in the stateless layers 24-41 [measured: 25.0 % of neurons kept in 18 layers] | 68.8 | 64.0 | +4 (0.597) | -4 (0.454) |
| f_t0_02 | oracle, bounded: drop <= 2 % of contribution mass, all layers [measured: 72.0 % of neurons kept in 42 layers] | 67.7 | 64.4 | -2 (0.727) | -2 (0.791) |
| f_t0_005 | oracle, bounded: drop <= 0.5 % of contribution mass, all layers [measured: 83.5 % of neurons kept in 42 layers] | 68.1 | 63.9 | +0 (1.000) | -5 (0.302) |
| f_g0_50 | GATE-ONLY selector, keep 50 %, all layers [measured: 50.0 % of neurons kept in 42 layers] | 46.7 | 46.7 | -122 (0.000) | -103 (0.000) |
| f_g0_25 | GATE-ONLY selector, keep 25 %, all layers [measured: 25.0 % of neurons kept in 42 layers] | 29.5 | 30.0 | -220 (0.000) | -198 (0.000) |
| f_G0_02 | GATE-ONLY selector, drop <= 2 % of gate-score mass, all layers [measured: 82.1 % of neurons kept in 42 layers] | 68.8 | 64.6 | +4 (0.424) | -1 (1.000) |
| f_g_mix | GATE-ONLY selector, 50 % in layers 0-23, 25 % in 24-41 [measured: 39.3 % of neurons kept in 42 layers] | 45.3 | 42.1 | -130 (0.000) | -129 (0.000) |

## 2. Chat, token level (replies sampled by the full model, teacher-forced)

assistant positions: 9878 (EN 4873, RU 5005); records whose prompt boundary did not match the template suffix: 0

3000 sampled assistant positions. `exit@x%` = share of tokens that can leave at this readout, most confident first, while at most x % of them differ from the full model's top-1.

| arm | kept | top-1 agree EN | RU | mean KL EN | RU | exit@1% EN | RU | exit@2% EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exit24 | 24 | 0.0 | 0.0 | 25.408 | 25.100 | 0.0 | 0.0 | 0.0 | 0.0 |
| exit27 | 27 | 0.3 | 0.0 | 25.690 | 24.758 | 0.0 | 0.0 | 0.0 | 0.0 |
| exit30 | 30 | 26.4 | 11.0 | 6.790 | 10.016 | 0.9 | 0.8 | 0.9 | 0.8 |
| exit33 | 33 | 37.2 | 15.7 | 4.886 | 8.560 | 2.9 | 2.3 | 2.9 | 2.3 |
| exit36 | 36 | 46.1 | 26.4 | 3.557 | 5.242 | 3.4 | 3.7 | 5.5 | 4.6 |
| exit39 | 39 | 57.4 | 50.9 | 1.911 | 2.379 | 10.8 | 6.8 | 16.5 | 10.1 |
| exit41 | 41 | 76.5 | 76.4 | 0.605 | 0.748 | 32.0 | 23.0 | 39.9 | 40.1 |
| s36_40 | 38 layers (drop 36-39) | 79.8 | 76.5 | 0.382 | 0.840 | 31.7 | 14.0 | 44.4 | 21.0 |
| s30_40 | 32 layers (drop 30-39) | 28.6 | 20.1 | 7.077 | 9.717 | 0.1 | 0.8 | 0.1 | 0.8 |
| s24_40 | 26 layers (drop 24-39) | 0.1 | 0.1 | 21.331 | 23.419 | 0.0 | 0.0 | 0.0 | 0.0 |
| f_k0_25 | INSTRUMENT CHECK: oracle 25 %, all 42 layers (old result: about -4 points) [measured: 25.0 % of neurons kept in 42 layers] | 83.0 | 83.0 | 0.223 | 0.242 | 51.6 | 53.4 | 58.4 | 62.6 |
| f_k0_50 | oracle 50 % of FFN neurons, all layers [measured: 50.0 % of neurons kept in 42 layers] | 93.9 | 93.6 | 0.027 | 0.029 | 78.1 | 79.7 | 84.0 | 85.8 |
| f_k24_125 | oracle 12.5 % in the stateless layers 24-41 [measured: 12.5 % of neurons kept in 18 layers] | 74.7 | 75.9 | 0.486 | 0.600 | 34.8 | 40.4 | 44.8 | 44.6 |
| f_k24_25 | oracle 25 % in the stateless layers 24-41 [measured: 25.0 % of neurons kept in 18 layers] | 85.7 | 87.7 | 0.135 | 0.149 | 59.2 | 61.9 | 65.8 | 71.2 |
| f_t0_02 | oracle, bounded: drop <= 2 % of contribution mass, all layers [measured: 72.0 % of neurons kept in 42 layers] | 97.5 | 97.5 | 0.005 | 0.004 | 90.2 | 88.5 | 98.0 | 98.4 |
| f_t0_005 | oracle, bounded: drop <= 0.5 % of contribution mass, all layers [measured: 83.5 % of neurons kept in 42 layers] | 97.4 | 97.7 | 0.003 | 0.003 | 89.8 | 89.0 | 97.1 | 99.0 |
| f_g0_50 | GATE-ONLY selector, keep 50 %, all layers [measured: 50.0 % of neurons kept in 42 layers] | 63.6 | 65.1 | 0.946 | 1.136 | 26.6 | 20.6 | 28.5 | 27.2 |
| f_g0_25 | GATE-ONLY selector, keep 25 %, all layers [measured: 25.0 % of neurons kept in 42 layers] | 40.1 | 39.3 | 3.230 | 3.573 | 2.0 | 3.5 | 2.0 | 4.3 |
| f_G0_02 | GATE-ONLY selector, drop <= 2 % of gate-score mass, all layers [measured: 82.1 % of neurons kept in 42 layers] | 97.0 | 96.5 | 0.008 | 0.010 | 86.8 | 87.7 | 95.4 | 93.1 |
| f_g_mix | GATE-ONLY selector, 50 % in layers 0-23, 25 % in 24-41 [measured: 39.3 % of neurons kept in 42 layers] | 53.5 | 52.7 | 1.730 | 2.266 | 3.7 | 8.5 | 7.8 | 10.9 |

## 3. Closed-form linear patch h_L -> h_41 (ridge towards identity; fitted on chat shards 0..n-3, all positions)

train tokens: 8644; test: benchmark (1 140 last positions) and 2400 held-out assistant positions

| arm | kept | alpha | EN | RU | ΔEN (p) | ΔRU (p) |
|---|---:|---:|---:|---:|---|---|
| patch24 (rel. MSE 3.517 -> 0.317) | 24 | 0.01 | 23.3 | 23.5 | -255 (0.000) | -235 (0.000) |
| patch27 (rel. MSE 4.292 -> 0.360) | 27 | 0.01 | 26.1 | 27.2 | -239 (0.000) | -214 (0.000) |
| patch30 (rel. MSE 2.704 -> 0.256) | 30 | 0.01 | 35.6 | 36.7 | -185 (0.000) | -160 (0.000) |
| patch33 (rel. MSE 5.523 -> 0.269) | 33 | 0.01 | 42.5 | 46.7 | -146 (0.000) | -103 (0.000) |
| patch36 (rel. MSE 6.762 -> 0.217) | 36 | 0.01 | 57.5 | 56.8 | -60 (0.000) | -45 (0.000) |

Held-out chat positions (2400):

| arm | kept | top-1 agree EN | RU | mean KL EN | RU | exit@1% EN | RU | exit@2% EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| patch24 | 24 | 33.8 | 19.1 | 8.712 | 11.163 | 2.9 | 2.1 | 5.3 | 2.1 |
| exit24 (same positions) | 24 | 0.0 | 0.0 | 25.176 | 24.930 | 0.0 | 0.0 | 0.0 | 0.0 |
| patch27 | 27 | 36.5 | 20.1 | 7.403 | 10.181 | 3.5 | 1.3 | 6.2 | 1.3 |
| exit27 (same positions) | 27 | 0.2 | 0.0 | 25.254 | 23.948 | 0.0 | 0.0 | 0.0 | 0.0 |
| patch30 | 30 | 49.2 | 31.5 | 4.691 | 6.715 | 2.6 | 1.7 | 6.6 | 1.7 |
| exit30 (same positions) | 30 | 25.9 | 11.3 | 6.686 | 9.917 | 0.3 | 0.2 | 0.3 | 0.2 |
| patch33 | 33 | 58.6 | 40.8 | 2.381 | 5.422 | 11.7 | 3.3 | 14.0 | 3.3 |
| exit33 (same positions) | 33 | 37.8 | 15.5 | 4.842 | 8.508 | 1.4 | 2.2 | 1.4 | 2.2 |
| patch36 | 36 | 64.9 | 53.7 | 1.521 | 3.449 | 18.2 | 13.7 | 21.6 | 15.0 |
| exit36 (same positions) | 36 | 45.7 | 29.8 | 3.546 | 5.107 | 4.2 | 4.8 | 4.9 | 5.5 |

## 4. Benchmark with a confidence gate (threshold = the chat exit@1% threshold of that arm)

| arm | threshold | exits on bench EN | RU | gated EN | RU | Δ vs full EN | RU |
|---|---:|---:|---:|---:|---:|---:|---:|
| exit24 (plain truncation) | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit27 (plain truncation) | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit30 (plain truncation) | 0.978 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit33 (plain truncation) | 0.975 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit36 (plain truncation) | 0.976 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit39 (plain truncation) | 0.986 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| exit41 (plain truncation) | 0.892 | 15.6 | 17.4 | 68.1 | 64.7 | +0 | +0 |
| s36_40 | 0.997 | 25.3 | 16.1 | 68.1 | 64.7 | +0 | +0 |
| s30_40 | 1.000 | 0.4 | 0.4 | 68.1 | 64.7 | +0 | +0 |
| s24_40 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| f_k0_25 | 0.878 | 53.5 | 61.2 | 66.5 | 62.6 | -9 | -12 |
| f_k0_50 | 0.607 | 88.4 | 87.5 | 68.2 | 64.2 | +1 | -3 |
| f_k24_125 | 0.971 | 58.2 | 52.3 | 67.7 | 64.7 | -2 | +0 |
| f_k24_25 | 0.814 | 75.1 | 73.2 | 68.2 | 64.6 | +1 | -1 |
| f_t0_02 | 0.488 | 95.8 | 93.9 | 67.7 | 64.7 | -2 | +0 |
| f_t0_005 | 0.494 | 95.6 | 94.2 | 68.2 | 64.6 | +1 | -1 |
| f_g0_50 | 0.970 | 8.1 | 10.9 | 67.4 | 63.9 | -4 | -5 |
| f_g0_25 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| f_G0_02 | 0.515 | 94.7 | 93.2 | 68.4 | 65.3 | +2 | +3 |
| f_g_mix | 0.998 | 0.5 | 0.4 | 68.1 | 64.7 | +0 | +0 |
| patch24 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch27 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch30 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch33 | 1.000 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |
| patch36 | 0.998 | 0.0 | 0.0 | 68.1 | 64.7 | +0 | +0 |

## 5. LM head shortlist (rank-r sketch -> top-N rows -> exact logits on those rows only)

1500 positions. Sketch bytes are for 8-bit storage; the exact head is 360 MiB at 4.5 bpw.

| rank r | shortlist N | sketch MiB | top-1 in shortlist | all of top-64 in shortlist | mean share of top-64 |
|---:|---:|---:|---:|---:|---:|
| 64 | 1024 | 16 | 46.33 % | 0.07 % | 43.56 % |
| 64 | 4096 | 16 | 65.73 % | 1.13 % | 63.91 % |
| 64 | 16384 | 16 | 81.27 % | 12.00 % | 79.43 % |
| 128 | 1024 | 32 | 62.73 % | 0.13 % | 60.16 % |
| 128 | 4096 | 32 | 80.33 % | 3.80 % | 78.80 % |
| 128 | 16384 | 32 | 92.67 % | 23.00 % | 89.29 % |
| 256 | 1024 | 64 | 79.80 % | 0.53 % | 71.59 % |
| 256 | 4096 | 64 | 90.00 % | 11.07 % | 86.06 % |
| 256 | 16384 | 64 | 96.20 % | 33.60 % | 93.39 % |
| 512 | 1024 | 128 | 86.27 % | 3.47 % | 81.23 % |
| 512 | 4096 | 128 | 93.20 % | 22.20 % | 91.58 % |
| 512 | 16384 | 128 | 97.60 % | 48.67 % | 96.56 % |

## 6. Control: Gemma 4 E2B (Q4_K_M), same instrument

E2B: EN 54.6 (n=570), RU 51.2 (n=570)
vs E4B full, paired: EN -77 (p 0.000), RU -77 (p 0.000)

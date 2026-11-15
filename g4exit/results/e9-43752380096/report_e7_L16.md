# E7 -- structure of the FFN of layer 16 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51028, validation 5905, held-out test 11278; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.14e-04

## I. the linear part
one linear map of x: error 0.2683 (dir 0.3955). Reduced rank: r=64 0.3674, r=256 0.3043, r=512 0.2818, r=1024 0.2703

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 10.4 % of the variance (coverage 56 %), previous token 6.6 % (coverage 55 %)
- FFN output y: current token explains 10.6 % of the variance (coverage 56 %), previous token 6.7 % (coverage 55 %)
- nonlinear residual y - Lx: current token explains -1.4 % of the variance (coverage 56 %), previous token -1.8 % (coverage 55 %)
- output side: the top 256 / 512 / 1024 directions hold 31.4 / 49.4 / 73.1 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 722/3186/4374; x64 365/804/1425; x256 60/193/574; tok256 3/107/3355. Token-table coverage of test: 84 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2683 | 0.3955 | |
| dynamic oracle 6.25 % | 6.2 % | 0.1069 | 0.1511 | 0.2468 |
| dynamic oracle 12.5 % | 12.5 % | 0.0724 | 0.1018 | 0.1235 |
| dynamic oracle 25 % | 25.0 % | 0.0346 | 0.0485 | 0.0418 |
| dynamic oracle 50 % | 50.0 % | 0.0061 | 0.0085 | 0.0057 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3350) | wiki_en (n=2358) | wiki_ru (n=3589) |
|---|---:|---:|---:|---:|
| linear map alone | 0.2667 | 0.2712 | 0.2826 | 0.2574 |
| dynamic oracle 6.25 % + bypass | 0.1060 | 0.1079 | 0.1108 | 0.1040 |
| dynamic oracle 12.5 % + bypass | 0.0719 | 0.0730 | 0.0748 | 0.0706 |
| dynamic oracle 25 % + bypass | 0.0346 | 0.0348 | 0.0354 | 0.0339 |
| dynamic oracle 50 % + bypass | 0.0063 | 0.0062 | 0.0062 | 0.0060 |

## C. cheap selectors: which neurons to compute exactly, chosen from a low-precision scan (exact values for the chosen, bypass for the rest)

| selector (bytes scanned, share of the dense FFN) | computed exactly | error with bypass | dir | overlap with the oracle's set |
|---|---:|---:|---:|---:|
| exact gate x calibrated |u| (33 %) | 12.5 % | 0.0980 | 0.1407 | 53 % |
| exact gate x calibrated |u| (33 %) | 25 % | 0.0672 | 0.0962 | 59 % |
| exact gate x calibrated |u| (33 %) | 50 % | 0.0265 | 0.0373 | 73 % |
| 1-bit gate x calibrated |u| (7 %) | 12.5 % | 0.1411 | 0.2016 | 44 % |
| 1-bit gate x calibrated |u| (7 %) | 25 % | 0.1093 | 0.1543 | 49 % |
| 1-bit gate x calibrated |u| (7 %) | 50 % | 0.0618 | 0.0858 | 64 % |
| 1-bit gate and 1-bit up (15 %) | 12.5 % | 0.1372 | 0.1919 | 51 % |
| 1-bit gate and 1-bit up (15 %) | 25 % | 0.0983 | 0.1359 | 56 % |
| 1-bit gate and 1-bit up (15 %) | 50 % | 0.0515 | 0.0704 | 67 % |
| 2-bit gate and 2-bit up (30 %) | 12.5 % | 0.0938 | 0.1303 | 72 % |
| 2-bit gate and 2-bit up (30 %) | 25 % | 0.0556 | 0.0764 | 75 % |
| 2-bit gate and 2-bit up (30 %) | 50 % | 0.0209 | 0.0283 | 79 % |

(oracle with bypass on the same positions: 6.25 %: 0.1069, 12.5 %: 0.0724, 25 %: 0.0346, 50 %: 0.0061)

quick run: stopped here, 5238 s

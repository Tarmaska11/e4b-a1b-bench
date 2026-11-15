# E7 -- structure of the FFN of layer 34 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51797, validation 6293, held-out test 11474; split by record
instrument check (numpy FFN vs the engine's ffn_out): 4.62e-04

## I. the linear part
one linear map of x: error 0.2534 (dir 0.4395). Reduced rank: r=64 0.2910, r=256 0.2635, r=512 0.2559, r=1024 0.2531

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 18.0 % of the variance (coverage 57 %), previous token 7.5 % (coverage 56 %)
- FFN output y: current token explains 20.4 % of the variance (coverage 57 %), previous token 7.6 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains 0.2 % of the variance (coverage 57 %), previous token -1.4 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 30.8 / 47.7 / 70.7 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 1083/2844/6621; x64 208/777/1739; x256 26/181/922; tok256 3/93/4420. Token-table coverage of test: 85 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2534 | 0.4395 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0442 | 0.0756 | 0.1316 |
| dynamic oracle 12.5 % | 12.5 % | 0.0351 | 0.0599 | 0.0737 |
| dynamic oracle 25 % | 25.0 % | 0.0207 | 0.0352 | 0.0292 |
| dynamic oracle 50 % | 50.0 % | 0.0048 | 0.0083 | 0.0049 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3350) | wiki_en (n=2397) | wiki_ru (n=3746) |
|---|---:|---:|---:|---:|
| linear map alone | 0.1871 | 0.2112 | 0.3109 | 0.2908 |
| dynamic oracle 6.25 % + bypass | 0.0405 | 0.0368 | 0.0525 | 0.0475 |
| dynamic oracle 12.5 % + bypass | 0.0303 | 0.0310 | 0.0391 | 0.0390 |
| dynamic oracle 25 % + bypass | 0.0166 | 0.0194 | 0.0211 | 0.0239 |
| dynamic oracle 50 % + bypass | 0.0035 | 0.0049 | 0.0044 | 0.0058 |

## C. cheap selectors: which neurons to compute exactly, chosen from a low-precision scan (exact values for the chosen, bypass for the rest)

| selector (bytes scanned, share of the dense FFN) | computed exactly | error with bypass | dir | overlap with the oracle's set |
|---|---:|---:|---:|---:|
| exact gate x calibrated |u| (33 %) | 12.5 % | 0.0363 | 0.0623 | 47 % |
| exact gate x calibrated |u| (33 %) | 25 % | 0.0306 | 0.0527 | 50 % |
| exact gate x calibrated |u| (33 %) | 50 % | 0.0197 | 0.0337 | 67 % |
| 1-bit gate x calibrated |u| (7 %) | 12.5 % | 0.0638 | 0.1101 | 35 % |
| 1-bit gate x calibrated |u| (7 %) | 25 % | 0.0567 | 0.0977 | 41 % |
| 1-bit gate x calibrated |u| (7 %) | 50 % | 0.0426 | 0.0729 | 60 % |
| 1-bit gate and 1-bit up (15 %) | 12.5 % | 0.0719 | 0.1212 | 48 % |
| 1-bit gate and 1-bit up (15 %) | 25 % | 0.0556 | 0.0933 | 54 % |
| 1-bit gate and 1-bit up (15 %) | 50 % | 0.0324 | 0.0540 | 65 % |
| 2-bit gate and 2-bit up (30 %) | 12.5 % | 0.0462 | 0.0783 | 70 % |
| 2-bit gate and 2-bit up (30 %) | 25 % | 0.0313 | 0.0528 | 74 % |
| 2-bit gate and 2-bit up (30 %) | 50 % | 0.0133 | 0.0224 | 79 % |

(oracle with bypass on the same positions: 6.25 %: 0.0442, 12.5 %: 0.0351, 25 %: 0.0207, 50 %: 0.0048)

quick run: stopped here, 5305 s

# E7 -- structure of the FFN of layer 16 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51435, validation 6220, held-out test 11240; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.07e-04

## I. the linear part
one linear map of x: error 0.2730 (dir 0.3984). Reduced rank: r=64 0.3745, r=256 0.3104, r=512 0.2872, r=1024 0.2753

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 10.3 % of the variance (coverage 56 %), previous token 6.5 % (coverage 56 %)
- FFN output y: current token explains 10.5 % of the variance (coverage 56 %), previous token 6.5 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains -1.4 % of the variance (coverage 56 %), previous token -1.8 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 31.6 / 49.5 / 73.2 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 2089/3350/4378; x64 339/773/1387; x256 42/195/539; tok256 7/101/4944. Token-table coverage of test: 85 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2730 | 0.3984 | |
| dynamic oracle 6.25 % | 6.3 % | 0.1086 | 0.1525 | 0.2471 |
| dynamic oracle 12.5 % | 12.5 % | 0.0735 | 0.1028 | 0.1243 |
| dynamic oracle 25 % | 25.0 % | 0.0352 | 0.0489 | 0.0424 |
| dynamic oracle 50 % | 50.0 % | 0.0062 | 0.0086 | 0.0058 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3350) | wiki_en (n=2334) | wiki_ru (n=3575) |
|---|---:|---:|---:|---:|
| linear map alone | 0.2680 | 0.2714 | 0.3008 | 0.2603 |
| dynamic oracle 6.25 % + bypass | 0.1061 | 0.1079 | 0.1178 | 0.1049 |
| dynamic oracle 12.5 % + bypass | 0.0720 | 0.0729 | 0.0797 | 0.0711 |
| dynamic oracle 25 % + bypass | 0.0347 | 0.0348 | 0.0378 | 0.0341 |
| dynamic oracle 50 % + bypass | 0.0063 | 0.0062 | 0.0066 | 0.0060 |

## E. cascade: a 1-bit scan of all neurons picks a shortlist, exact scores of the shortlist pick the neurons to compute

bytes are a share of the dense three-matrix FFN at Q4_K: a 1-bit matrix = 1/13.5; exact rows of the shortlist; down rows of the final set

reference on the same positions: 12.5 %: oracle 0.0735, exact-gate selector 0.0995 (bytes 42 %); 25 %: oracle 0.0352, exact-gate selector 0.0684 (bytes 50 %)

| 1-bit scan | final choice by | computed | shortlist | bytes read | oracle's set inside the shortlist | error with bypass | x oracle | x exact-gate selector |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x1.5 | 24 % | 54.2 % | 0.1269 | 1.73 | 1.28 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x1.5 | 22 % | 54.2 % | 0.1332 | 1.81 | 1.34 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x2.0 | 28 % | 62.2 % | 0.1169 | 1.59 | 1.18 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x2.0 | 24 % | 62.2 % | 0.1263 | 1.72 | 1.27 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x3.0 | 37 % | 74.1 % | 0.1035 | 1.41 | 1.04 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x3.0 | 28 % | 74.1 % | 0.1172 | 1.59 | 1.18 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 25 % | x1.5 | 41 % | 63.6 % | 0.0872 | 2.48 | 1.27 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x1.5 | 37 % | 63.6 % | 0.0979 | 2.78 | 1.43 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 25 % | x2.0 | 49 % | 74.6 % | 0.0701 | 1.99 | 1.02 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x2.0 | 41 % | 74.6 % | 0.0889 | 2.53 | 1.30 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 25 % | x3.0 | 66 % | 89.8 % | 0.0491 | 1.39 | 0.72 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x3.0 | 49 % | 89.8 % | 0.0768 | 2.18 | 1.12 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x1.5 | 31 % | 62.4 % | 0.1210 | 1.64 | 1.22 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x1.5 | 29 % | 62.4 % | 0.1283 | 1.74 | 1.29 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x2.0 | 36 % | 70.5 % | 0.1105 | 1.50 | 1.11 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x2.0 | 31 % | 70.5 % | 0.1223 | 1.66 | 1.23 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x3.0 | 44 % | 81.1 % | 0.0976 | 1.33 | 0.98 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x3.0 | 36 % | 81.1 % | 0.1140 | 1.55 | 1.15 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 25 % | x1.5 | 48 % | 70.2 % | 0.0757 | 2.15 | 1.11 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x1.5 | 44 % | 70.2 % | 0.0859 | 2.44 | 1.26 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 25 % | x2.0 | 56 % | 79.9 % | 0.0617 | 1.75 | 0.90 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x2.0 | 48 % | 79.9 % | 0.0814 | 2.31 | 1.19 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 25 % | x3.0 | 73 % | 92.6 % | 0.0449 | 1.28 | 0.66 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x3.0 | 56 % | 92.6 % | 0.0738 | 2.10 | 1.08 |

quick run: stopped here, 4758 s

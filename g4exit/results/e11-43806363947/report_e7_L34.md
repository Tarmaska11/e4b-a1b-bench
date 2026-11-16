# E7 -- structure of the FFN of layer 34 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 53312, validation 5922, held-out test 11478; split by record
instrument check (numpy FFN vs the engine's ffn_out): 4.70e-04

## I. the linear part
one linear map of x: error 0.2396 (dir 0.4148). Reduced rank: r=64 0.2830, r=256 0.2541, r=512 0.2446, r=1024 0.2402

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 18.0 % of the variance (coverage 58 %), previous token 7.4 % (coverage 57 %)
- FFN output y: current token explains 20.7 % of the variance (coverage 58 %), previous token 7.5 % (coverage 57 %)
- nonlinear residual y - Lx: current token explains 0.0 % of the variance (coverage 58 %), previous token -1.6 % (coverage 57 %)
- output side: the top 256 / 512 / 1024 directions hold 31.1 / 48.1 / 71.1 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 1190/3042/6015; x64 309/760/2238; x256 44/183/664; tok256 3/99/3573. Token-table coverage of test: 86 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2396 | 0.4148 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0427 | 0.0728 | 0.1325 |
| dynamic oracle 12.5 % | 12.5 % | 0.0342 | 0.0581 | 0.0739 |
| dynamic oracle 25 % | 25.0 % | 0.0203 | 0.0344 | 0.0292 |
| dynamic oracle 50 % | 50.0 % | 0.0048 | 0.0081 | 0.0049 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3450) | wiki_en (n=2346) | wiki_ru (n=3701) |
|---|---:|---:|---:|---:|
| linear map alone | 0.1881 | 0.2125 | 0.2703 | 0.2756 |
| dynamic oracle 6.25 % + bypass | 0.0405 | 0.0367 | 0.0490 | 0.0457 |
| dynamic oracle 12.5 % + bypass | 0.0303 | 0.0310 | 0.0369 | 0.0379 |
| dynamic oracle 25 % + bypass | 0.0166 | 0.0194 | 0.0202 | 0.0234 |
| dynamic oracle 50 % + bypass | 0.0035 | 0.0049 | 0.0042 | 0.0058 |

## E. cascade: a 1-bit scan of all neurons picks a shortlist, exact scores of the shortlist pick the neurons to compute

bytes are a share of the dense three-matrix FFN at Q4_K: a 1-bit matrix = 1/13.5; exact rows of the shortlist; down rows of the final set

reference on the same positions: 12.5 %: oracle 0.0342, exact-gate selector 0.0350 (bytes 42 %); 25 %: oracle 0.0203, exact-gate selector 0.0297 (bytes 50 %)

| 1-bit scan | final choice by | computed | shortlist | bytes read | oracle's set inside the shortlist | error with bypass | x oracle | x exact-gate selector |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x1.5 | 24 % | 43.6 % | 0.0574 | 1.68 | 1.64 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x1.5 | 22 % | 43.6 % | 0.0587 | 1.72 | 1.68 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x2.0 | 28 % | 50.6 % | 0.0560 | 1.64 | 1.60 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x2.0 | 24 % | 50.6 % | 0.0568 | 1.66 | 1.63 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x3.0 | 37 % | 62.6 % | 0.0538 | 1.57 | 1.54 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x3.0 | 28 % | 62.6 % | 0.0538 | 1.57 | 1.54 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 25 % | x1.5 | 41 % | 54.8 % | 0.0496 | 2.44 | 1.67 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x1.5 | 37 % | 54.8 % | 0.0519 | 2.55 | 1.75 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 25 % | x2.0 | 49 % | 66.8 % | 0.0440 | 2.16 | 1.48 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x2.0 | 41 % | 66.8 % | 0.0483 | 2.38 | 1.63 |
| gate only, x calibrated |u| | oracle score (exact gate and up rows of the shortlist) | 25 % | x3.0 | 66 % | 86.1 % | 0.0323 | 1.59 | 1.09 |
| gate only, x calibrated |u| | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x3.0 | 49 % | 86.1 % | 0.0400 | 1.97 | 1.35 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x1.5 | 31 % | 58.7 % | 0.0624 | 1.82 | 1.78 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x1.5 | 29 % | 58.7 % | 0.0633 | 1.85 | 1.81 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x2.0 | 36 % | 66.8 % | 0.0578 | 1.69 | 1.65 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x2.0 | 31 % | 66.8 % | 0.0591 | 1.73 | 1.69 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 12.5 % | x3.0 | 44 % | 77.8 % | 0.0513 | 1.50 | 1.47 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 12.5 % | x3.0 | 36 % | 77.8 % | 0.0525 | 1.53 | 1.50 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 25 % | x1.5 | 48 % | 67.6 % | 0.0432 | 2.13 | 1.46 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x1.5 | 44 % | 67.6 % | 0.0472 | 2.32 | 1.59 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 25 % | x2.0 | 56 % | 77.5 % | 0.0364 | 1.79 | 1.23 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x2.0 | 48 % | 77.5 % | 0.0433 | 2.13 | 1.46 |
| gate and up | oracle score (exact gate and up rows of the shortlist) | 25 % | x3.0 | 73 % | 91.1 % | 0.0269 | 1.32 | 0.91 |
| gate and up | exact gate x calibrated |u| (exact gate rows of the shortlist) | 25 % | x3.0 | 56 % | 91.1 % | 0.0358 | 1.76 | 1.21 |

quick run: stopped here, 6427 s

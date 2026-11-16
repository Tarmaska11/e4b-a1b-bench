# E7 -- structure of the FFN of layer 6 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 52638, validation 5983, held-out test 11464; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.66e-04

## I. the linear part
one linear map of x: error 0.3325 (dir 0.6371). Reduced rank: r=64 0.3734, r=256 0.3390, r=512 0.3322, r=1024 0.3314

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 27.8 % of the variance (coverage 57 %), previous token 11.3 % (coverage 56 %)
- FFN output y: current token explains 35.6 % of the variance (coverage 57 %), previous token 21.7 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains 0.8 % of the variance (coverage 57 %), previous token -1.3 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 43.2 / 59.6 / 79.1 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 1397/3092/4841; x64 278/780/1653; x256 32/178/609; tok256 16/111/5614. Token-table coverage of test: 85 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.3325 | 0.6371 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0048 | 0.0096 | 0.0353 |
| dynamic oracle 12.5 % | 12.5 % | 0.0025 | 0.0050 | 0.0126 |
| dynamic oracle 25 % | 25.0 % | 0.0007 | 0.0015 | 0.0023 |
| dynamic oracle 50 % | 50.0 % | 0.0000 | 0.0001 | 0.0001 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3450) | wiki_en (n=2427) | wiki_ru (n=3606) |
|---|---:|---:|---:|---:|
| linear map alone | 0.2400 | 0.2938 | 0.3875 | 0.4158 |
| dynamic oracle 6.25 % + bypass | 0.0032 | 0.0045 | 0.0047 | 0.0065 |
| dynamic oracle 12.5 % + bypass | 0.0017 | 0.0023 | 0.0025 | 0.0034 |
| dynamic oracle 25 % + bypass | 0.0005 | 0.0007 | 0.0007 | 0.0010 |
| dynamic oracle 50 % + bypass | 0.0000 | 0.0000 | 0.0000 | 0.0001 |

## D. blocks of co-active neurons (a file layout could read them as contiguous runs), and a variable number of neurons per token

instrument check: blocks of ONE neuron at 25 % give 0.0007; the neuron-level oracle above gave 0.0007

| neuron order | block size | mean correlation inside a block | computed exactly | error with bypass | dir | x the neuron-level oracle |
|---|---:|---:|---:|---:|---:|---:|
| co-activation clusters | 10 | 0.062 | 12.5 % | 0.0116 | 0.0226 | 4.73 |
| co-activation clusters | 10 | 0.062 | 25 % | 0.0051 | 0.0105 | 7.19 |
| co-activation clusters | 10 | 0.062 | 50 % | 0.0018 | 0.0039 | 48.22 |
| co-activation clusters | 40 | 0.052 | 12.5 % | 0.0336 | 0.0639 | 13.68 |
| co-activation clusters | 40 | 0.052 | 25 % | 0.0135 | 0.0261 | 18.84 |
| co-activation clusters | 40 | 0.052 | 50 % | 0.0038 | 0.0079 | 101.40 |
| co-activation clusters | 160 | 0.041 | 12.5 % | 0.0713 | 0.1365 | 28.99 |
| co-activation clusters | 160 | 0.041 | 25 % | 0.0352 | 0.0675 | 49.27 |
| co-activation clusters | 160 | 0.041 | 50 % | 0.0105 | 0.0204 | 278.46 |
| random order (control) | 40 | -0.000 | 12.5 % | 0.0538 | 0.1003 | 21.87 |
| random order (control) | 40 | -0.000 | 25 % | 0.0261 | 0.0487 | 36.56 |
| random order (control) | 40 | -0.000 | 50 % | 0.0082 | 0.0157 | 218.51 |

variable k: one threshold for every token instead of a fixed count (same MEAN share; hard tokens get more neurons)

| rule | mean share computed | neurons per token, 10th / 50th / 90th percentile | error with bypass | dir | x fixed top-k at that share |
|---|---:|---:|---:|---:|---:|
| absolute threshold on the score | 12.45 % | 1011 / 1281 / 1537 | 0.0024 | 0.0051 | 0.97 |
| threshold relative to the token's own rms score | 12.46 % | 567 / 1275 / 1873 | 0.0034 | 0.0052 | 1.37 |
| absolute threshold on the score | 24.93 % | 2122 / 2575 / 2950 | 0.0007 | 0.0014 | 0.94 |
| threshold relative to the token's own rms score | 24.88 % | 1470 / 2601 / 3398 | 0.0010 | 0.0014 | 1.40 |
| absolute threshold on the score | 49.92 % | 4374 / 5148 / 5764 | 0.0000 | 0.0001 | 0.77 |
| threshold relative to the token's own rms score | 49.85 % | 3854 / 5196 / 6112 | 0.0000 | 0.0001 | 1.19 |

quick run: stopped here, 5480 s

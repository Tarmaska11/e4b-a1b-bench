# E7 -- structure of the FFN of layer 16 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 53312, validation 5922, held-out test 11478; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.15e-04

## I. the linear part
one linear map of x: error 0.2643 (dir 0.3890). Reduced rank: r=64 0.3680, r=256 0.3030, r=512 0.2793, r=1024 0.2668

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 10.2 % of the variance (coverage 58 %), previous token 6.4 % (coverage 57 %)
- FFN output y: current token explains 10.5 % of the variance (coverage 58 %), previous token 6.5 % (coverage 57 %)
- nonlinear residual y - Lx: current token explains -1.4 % of the variance (coverage 58 %), previous token -1.8 % (coverage 57 %)
- output side: the top 256 / 512 / 1024 directions hold 31.6 / 49.6 / 73.2 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 729/3334/5086; x64 373/799/1365; x256 57/204/641; tok256 4/109/5103. Token-table coverage of test: 86 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2643 | 0.3890 | |
| dynamic oracle 6.25 % | 6.3 % | 0.1057 | 0.1493 | 0.2474 |
| dynamic oracle 12.5 % | 12.5 % | 0.0718 | 0.1009 | 0.1239 |
| dynamic oracle 25 % | 25.0 % | 0.0345 | 0.0482 | 0.0420 |
| dynamic oracle 50 % | 50.0 % | 0.0061 | 0.0085 | 0.0058 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3450) | wiki_en (n=2346) | wiki_ru (n=3701) |
|---|---:|---:|---:|---:|
| linear map alone | 0.2682 | 0.2709 | 0.2790 | 0.2484 |
| dynamic oracle 6.25 % + bypass | 0.1062 | 0.1076 | 0.1128 | 0.1000 |
| dynamic oracle 12.5 % + bypass | 0.0720 | 0.0728 | 0.0771 | 0.0680 |
| dynamic oracle 25 % + bypass | 0.0347 | 0.0348 | 0.0370 | 0.0327 |
| dynamic oracle 50 % + bypass | 0.0063 | 0.0062 | 0.0065 | 0.0058 |

## D. blocks of co-active neurons (a file layout could read them as contiguous runs), and a variable number of neurons per token

instrument check: blocks of ONE neuron at 25 % give 0.0345; the neuron-level oracle above gave 0.0345

| neuron order | block size | mean correlation inside a block | computed exactly | error with bypass | dir | x the neuron-level oracle |
|---|---:|---:|---:|---:|---:|---:|
| co-activation clusters | 10 | 0.044 | 12.5 % | 0.1498 | 0.2138 | 2.09 |
| co-activation clusters | 10 | 0.044 | 25 % | 0.1047 | 0.1490 | 3.04 |
| co-activation clusters | 10 | 0.044 | 50 % | 0.0508 | 0.0719 | 8.30 |
| co-activation clusters | 40 | 0.036 | 12.5 % | 0.1804 | 0.2600 | 2.51 |
| co-activation clusters | 40 | 0.036 | 25 % | 0.1382 | 0.1980 | 4.01 |
| co-activation clusters | 40 | 0.036 | 50 % | 0.0752 | 0.1071 | 12.29 |
| co-activation clusters | 160 | 0.028 | 12.5 % | 0.2017 | 0.2925 | 2.81 |
| co-activation clusters | 160 | 0.028 | 25 % | 0.1609 | 0.2316 | 4.67 |
| co-activation clusters | 160 | 0.028 | 50 % | 0.0949 | 0.1354 | 15.51 |
| random order (control) | 40 | -0.000 | 12.5 % | 0.1986 | 0.2829 | 2.76 |
| random order (control) | 40 | -0.000 | 25 % | 0.1641 | 0.2299 | 4.76 |
| random order (control) | 40 | -0.000 | 50 % | 0.1003 | 0.1385 | 16.39 |

variable k: one threshold for every token instead of a fixed count (same MEAN share; hard tokens get more neurons)

| rule | mean share computed | neurons per token, 10th / 50th / 90th percentile | error with bypass | dir | x fixed top-k at that share |
|---|---:|---:|---:|---:|---:|
| absolute threshold on the score | 12.43 % | 961 / 1244 / 1627 | 0.0709 | 0.1010 | 0.99 |
| threshold relative to the token's own rms score | 12.52 % | 1110 / 1302 / 1434 | 0.0720 | 0.1000 | 1.00 |
| absolute threshold on the score | 24.90 % | 2070 / 2526 / 3056 | 0.0337 | 0.0480 | 0.98 |
| threshold relative to the token's own rms score | 25.03 % | 2294 / 2600 / 2797 | 0.0345 | 0.0475 | 1.00 |
| absolute threshold on the score | 49.89 % | 4585 / 5107 / 5641 | 0.0059 | 0.0084 | 0.96 |
| threshold relative to the token's own rms score | 50.03 % | 4837 / 5168 / 5378 | 0.0061 | 0.0083 | 1.00 |

quick run: stopped here, 4545 s

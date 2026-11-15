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

## D. blocks of co-active neurons (a file layout could read them as contiguous runs), and a variable number of neurons per token

instrument check: blocks of ONE neuron at 25 % give 0.0207; the neuron-level oracle above gave 0.0207

| neuron order | block size | mean correlation inside a block | computed exactly | error with bypass | dir | x the neuron-level oracle |
|---|---:|---:|---:|---:|---:|---:|
| co-activation clusters | 10 | 0.046 | 12.5 % | 0.0605 | 0.1034 | 1.72 |
| co-activation clusters | 10 | 0.046 | 25 % | 0.0448 | 0.0768 | 2.16 |
| co-activation clusters | 10 | 0.046 | 50 % | 0.0267 | 0.0457 | 5.51 |
| co-activation clusters | 40 | 0.040 | 12.5 % | 0.0888 | 0.1508 | 2.53 |
| co-activation clusters | 40 | 0.040 | 25 % | 0.0632 | 0.1075 | 3.05 |
| co-activation clusters | 40 | 0.040 | 50 % | 0.0370 | 0.0630 | 7.62 |
| co-activation clusters | 160 | 0.033 | 12.5 % | 0.1190 | 0.2001 | 3.39 |
| co-activation clusters | 160 | 0.033 | 25 % | 0.0877 | 0.1477 | 4.24 |
| co-activation clusters | 160 | 0.033 | 50 % | 0.0499 | 0.0843 | 10.29 |
| random order (control) | 40 | -0.000 | 12.5 % | 0.0961 | 0.1629 | 2.74 |
| random order (control) | 40 | -0.000 | 25 % | 0.0708 | 0.1199 | 3.42 |
| random order (control) | 40 | -0.000 | 50 % | 0.0432 | 0.0731 | 8.92 |

variable k: one threshold for every token instead of a fixed count (same MEAN share; hard tokens get more neurons)

| rule | mean share computed | neurons per token, 10th / 50th / 90th percentile | error with bypass | dir | x fixed top-k at that share |
|---|---:|---:|---:|---:|---:|
| absolute threshold on the score | 12.58 % | 847 / 1234 / 1874 | 0.0336 | 0.0572 | 0.96 |
| threshold relative to the token's own rms score | 12.59 % | 785 / 1304 / 1770 | 0.0362 | 0.0584 | 1.03 |
| absolute threshold on the score | 25.15 % | 1944 / 2585 / 3309 | 0.0193 | 0.0329 | 0.93 |
| threshold relative to the token's own rms score | 25.14 % | 1859 / 2632 / 3239 | 0.0214 | 0.0339 | 1.03 |
| absolute threshold on the score | 50.18 % | 4447 / 5275 / 5853 | 0.0043 | 0.0073 | 0.89 |
| threshold relative to the token's own rms score | 50.17 % | 4386 / 5203 / 5854 | 0.0049 | 0.0076 | 1.02 |

quick run: stopped here, 3311 s

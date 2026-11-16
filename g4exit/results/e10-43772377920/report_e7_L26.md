# E7 -- structure of the FFN of layer 26 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 53312, validation 5922, held-out test 11478; split by record
instrument check (numpy FFN vs the engine's ffn_out): 3.59e-04

## I. the linear part
one linear map of x: error 0.1631 (dir 0.2547). Reduced rank: r=64 0.2153, r=256 0.1821, r=512 0.1707, r=1024 0.1645

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 16.5 % of the variance (coverage 58 %), previous token 10.7 % (coverage 57 %)
- FFN output y: current token explains 21.2 % of the variance (coverage 58 %), previous token 12.9 % (coverage 57 %)
- nonlinear residual y - Lx: current token explains -0.8 % of the variance (coverage 58 %), previous token -1.4 % (coverage 57 %)
- output side: the top 256 / 512 / 1024 directions hold 31.4 / 48.8 / 71.9 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 1784/3194/5248; x64 254/832/1265; x256 52/199/600; tok256 4/106/3087. Token-table coverage of test: 86 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.1631 | 0.2547 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0522 | 0.0801 | 0.2229 |
| dynamic oracle 12.5 % | 12.5 % | 0.0397 | 0.0603 | 0.1110 |
| dynamic oracle 25 % | 25.0 % | 0.0216 | 0.0326 | 0.0353 |
| dynamic oracle 50 % | 50.0 % | 0.0044 | 0.0066 | 0.0045 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3450) | wiki_en (n=2346) | wiki_ru (n=3701) |
|---|---:|---:|---:|---:|
| linear map alone | 0.1784 | 0.1431 | 0.2090 | 0.1516 |
| dynamic oracle 6.25 % + bypass | 0.0590 | 0.0477 | 0.0639 | 0.0475 |
| dynamic oracle 12.5 % + bypass | 0.0439 | 0.0363 | 0.0481 | 0.0368 |
| dynamic oracle 25 % + bypass | 0.0235 | 0.0198 | 0.0259 | 0.0204 |
| dynamic oracle 50 % + bypass | 0.0047 | 0.0040 | 0.0051 | 0.0042 |

## D. blocks of co-active neurons (a file layout could read them as contiguous runs), and a variable number of neurons per token

instrument check: blocks of ONE neuron at 25 % give 0.0216; the neuron-level oracle above gave 0.0216

| neuron order | block size | mean correlation inside a block | computed exactly | error with bypass | dir | x the neuron-level oracle |
|---|---:|---:|---:|---:|---:|---:|
| co-activation clusters | 10 | 0.046 | 12.5 % | 0.0715 | 0.1109 | 1.80 |
| co-activation clusters | 10 | 0.046 | 25 % | 0.0517 | 0.0800 | 2.39 |
| co-activation clusters | 10 | 0.046 | 50 % | 0.0288 | 0.0441 | 6.55 |
| co-activation clusters | 40 | 0.039 | 12.5 % | 0.0961 | 0.1487 | 2.42 |
| co-activation clusters | 40 | 0.039 | 25 % | 0.0706 | 0.1095 | 3.27 |
| co-activation clusters | 40 | 0.039 | 50 % | 0.0399 | 0.0618 | 9.08 |
| co-activation clusters | 160 | 0.031 | 12.5 % | 0.1161 | 0.1786 | 2.93 |
| co-activation clusters | 160 | 0.031 | 25 % | 0.0898 | 0.1386 | 4.16 |
| co-activation clusters | 160 | 0.031 | 50 % | 0.0529 | 0.0816 | 12.04 |
| random order (control) | 40 | -0.000 | 12.5 % | 0.1058 | 0.1619 | 2.67 |
| random order (control) | 40 | -0.000 | 25 % | 0.0834 | 0.1270 | 3.86 |
| random order (control) | 40 | -0.000 | 50 % | 0.0512 | 0.0774 | 11.65 |

variable k: one threshold for every token instead of a fixed count (same MEAN share; hard tokens get more neurons)

| rule | mean share computed | neurons per token, 10th / 50th / 90th percentile | error with bypass | dir | x fixed top-k at that share |
|---|---:|---:|---:|---:|---:|
| absolute threshold on the score | 12.37 % | 962 / 1244 / 1562 | 0.0397 | 0.0608 | 1.00 |
| threshold relative to the token's own rms score | 12.62 % | 954 / 1337 / 1577 | 0.0404 | 0.0595 | 1.02 |
| absolute threshold on the score | 24.83 % | 2122 / 2532 / 2966 | 0.0215 | 0.0329 | 0.99 |
| threshold relative to the token's own rms score | 25.19 % | 2100 / 2661 / 2969 | 0.0220 | 0.0318 | 1.02 |
| absolute threshold on the score | 49.83 % | 4647 / 5109 / 5543 | 0.0043 | 0.0066 | 0.98 |
| threshold relative to the token's own rms score | 50.18 % | 4692 / 5243 / 5520 | 0.0045 | 0.0064 | 1.02 |

quick run: stopped here, 5558 s

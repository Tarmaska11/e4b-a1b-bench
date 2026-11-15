# E7 -- structure of the FFN of layer 16 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 52638, validation 5983, held-out test 11464; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.15e-04

## I. the linear part
one linear map of x: error 0.2698 (dir 0.3984). Reduced rank: r=64 0.3690, r=256 0.3060, r=512 0.2835, r=1024 0.2720

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 10.3 % of the variance (coverage 57 %), previous token 6.4 % (coverage 56 %)
- FFN output y: current token explains 10.5 % of the variance (coverage 57 %), previous token 6.6 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains -1.3 % of the variance (coverage 57 %), previous token -1.7 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 31.4 / 49.4 / 73.1 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 381/3595/4266; x64 358/775/1909; x256 38/186/599; tok256 3/106/4937. Token-table coverage of test: 85 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2698 | 0.3984 | |
| dynamic oracle 6.25 % | 6.2 % | 0.1068 | 0.1512 | 0.2454 |
| dynamic oracle 12.5 % | 12.5 % | 0.0723 | 0.1018 | 0.1228 |
| dynamic oracle 25 % | 25.0 % | 0.0345 | 0.0484 | 0.0417 |
| dynamic oracle 50 % | 50.0 % | 0.0061 | 0.0085 | 0.0057 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=1981) | mmlu_ru (n=3450) | wiki_en (n=2427) | wiki_ru (n=3606) |
|---|---:|---:|---:|---:|
| linear map alone | 0.2691 | 0.2705 | 0.2842 | 0.2603 |
| dynamic oracle 6.25 % + bypass | 0.1064 | 0.1077 | 0.1115 | 0.1033 |
| dynamic oracle 12.5 % + bypass | 0.0722 | 0.0728 | 0.0753 | 0.0699 |
| dynamic oracle 25 % + bypass | 0.0348 | 0.0348 | 0.0356 | 0.0334 |
| dynamic oracle 50 % + bypass | 0.0063 | 0.0062 | 0.0062 | 0.0059 |

quick run: stopped here, 4987 s

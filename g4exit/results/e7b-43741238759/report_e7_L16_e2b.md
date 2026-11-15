# E7 -- structure of the FFN of layer 16 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 20975, validation 2428, held-out test 5488; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.83e-04

## I. the linear part
one linear map of x: error 0.1789 (dir 0.3761). Reduced rank: r=64 0.2300, r=256 0.1885, r=512 0.1792, r=1024 0.1773

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 31.5 % of the variance (coverage 56 %), previous token 26.7 % (coverage 56 %)
- FFN output y: current token explains 45.7 % of the variance (coverage 56 %), previous token 30.9 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains -0.5 % of the variance (coverage 56 %), previous token -0.6 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 32.4 / 49.8 / 72.8 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 541/1306/1940; x64 108/301/641; x256 15/76/227; tok256 4/33/1623. Token-table coverage of test: 77 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.1789 | 0.3761 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0691 | 0.1391 | 0.1686 |
| dynamic oracle 12.5 % | 12.5 % | 0.0466 | 0.0933 | 0.0851 |
| dynamic oracle 25 % | 25.0 % | 0.0223 | 0.0445 | 0.0291 |
| dynamic oracle 50 % | 50.0 % | 0.0040 | 0.0079 | 0.0040 |

the same arms by kind of position (held-out; bypass fitted on all kinds):

| arm | mmlu_en (n=2205) | mmlu_ru (n=3031) |
|---|---:|---:|
| linear map alone | 0.1672 | 0.2152 |
| dynamic oracle 6.25 % + bypass | 0.0652 | 0.0821 |
| dynamic oracle 12.5 % + bypass | 0.0440 | 0.0551 |
| dynamic oracle 25 % + bypass | 0.0211 | 0.0262 |
| dynamic oracle 50 % + bypass | 0.0038 | 0.0046 |

quick run: stopped here, 1728 s

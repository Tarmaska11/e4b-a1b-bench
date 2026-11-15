# E7 -- structure of the FFN of layer 6 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51028, validation 5905, held-out test 11278; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.73e-04

## I. the linear part
one linear map of x: error 0.3359 (dir 0.6342). Reduced rank: r=64 0.3788, r=256 0.3440, r=512 0.3365, r=1024 0.3350

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 28.0 % of the variance (coverage 56 %), previous token 11.3 % (coverage 55 %)
- FFN output y: current token explains 35.5 % of the variance (coverage 56 %), previous token 21.7 % (coverage 55 %)
- nonlinear residual y - Lx: current token explains 1.0 % of the variance (coverage 56 %), previous token -1.4 % (coverage 55 %)
- output side: the top 256 / 512 / 1024 directions hold 42.5 / 59.2 / 78.9 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 355/3062/6863; x64 127/842/1341; x256 45/180/646; tok256 8/106/5386. Token-table coverage of test: 84 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.3359 | 0.6342 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0047 | 0.0094 | 0.0345 |
| dynamic oracle 12.5 % | 12.5 % | 0.0024 | 0.0049 | 0.0123 |
| dynamic oracle 25 % | 25.0 % | 0.0007 | 0.0014 | 0.0022 |
| dynamic oracle 50 % | 50.0 % | 0.0000 | 0.0001 | 0.0001 |
| one global static set 12.5 % | 12.5 % | 0.1701 | 0.3310 | 0.3145 |
| one global static set 25 % | 25.0 % | 0.1024 | 0.1943 | 0.1824 |
| one global static set 50 % | 50.0 % | 0.0367 | 0.0654 | 0.0528 |
| static set per region, x16, 12.5 % (pool not budgeted) | 12.5 % | 0.1100 | 0.2126 | 0.2065 |
| static set per region, x16, 25 % (pool not budgeted) | 25.0 % | 0.0642 | 0.1201 | 0.1050 |
| static set per region, x16, 50 % (pool not budgeted) | 50.0 % | 0.0223 | 0.0403 | 0.0276 |
| static set per region, x64, 12.5 % (pool not budgeted) | 12.5 % | 0.0858 | 0.1672 | 0.1627 |
| static set per region, x64, 25 % (pool not budgeted) | 25.0 % | 0.0498 | 0.0940 | 0.0784 |
| static set per region, x64, 50 % (pool not budgeted) | 50.0 % | 0.0205 | 0.0362 | 0.0238 |
| static set per region, x256, 12.5 % (pool not budgeted) | 12.5 % | 0.0682 | 0.1339 | 0.1308 |
| static set per region, x256, 25 % (pool not budgeted) | 25.0 % | 0.0387 | 0.0750 | 0.0587 |
| static set per region, x256, 50 % (pool not budgeted) | 50.0 % | 0.0152 | 0.0286 | 0.0172 |
| static set per region, tok256, 12.5 % (pool not budgeted) | 12.5 % | 0.1336 | 0.2541 | 0.2227 |
| static set per region, tok256, 25 % (pool not budgeted) | 25.0 % | 0.0840 | 0.1561 | 0.1183 |
| static set per region, tok256, 50 % (pool not budgeted) | 50.0 % | 0.0344 | 0.0623 | 0.0385 |

budgeted block emulation: shared set + the K nearest regions' expert sets, weights copied from E4B, pool <= 2.25x

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| 12.5 % nominal: shared 640 + 4 x 160 of 64 experts, router x64 (pool 1.06x) | 11.5 % | 0.1020 | 0.1992 | 0.2228 |
| 12.5 % nominal: shared 640 + 16 x 40 of 256 experts, router x256 (pool 1.06x) | 11.1 % | 0.1029 | 0.1996 | 0.2310 |
| 25 % nominal: shared 1280 + 1 x 1280 of 16 experts, router x16 (pool 2.12x) | 25.0 % | 0.0691 | 0.1311 | 0.1213 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router x64 (pool 2.12x) | 22.2 % | 0.0569 | 0.1102 | 0.1240 |
| 25 % nominal: shared 1280 + 16 x 80 of 256 experts, router x256 (pool 2.12x) | 21.3 % | 0.0551 | 0.1046 | 0.1316 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router tok>x64 (pool 2.12x) | 22.2 % | 0.0981 | 0.1853 | 0.1723 |
| 37.5 % nominal: shared 1280 + 8 x 320 of 64 experts, router x64 (pool 2.12x) | 28.6 % | 0.0429 | 0.0824 | 0.0963 |
| 50 % nominal: shared 2560 + 2 x 1280 of 16 experts, router x16 (pool 2.25x) | 44.6 % | 0.0298 | 0.0544 | 0.0452 |
| 50 % nominal: shared 2560 + 8 x 320 of 64 experts, router x64 (pool 2.25x) | 41.0 % | 0.0263 | 0.0493 | 0.0544 |

## B. gate and up pre-activations through a rank-r bottleneck (all neurons kept), + linear correction
centred x: the top 128 / 256 / 512 / 1024 principal directions hold 41.3 / 54.6 / 70.5 / 86.9 % of its variance

| rank r | pre-activation energy kept | gate+up bytes vs dense | error with correction | dir | error without |
|---:|---:|---:|---:|---:|---:|
| 256 | 89.6 % | 11.2 % | 0.2180 | 0.4273 | 0.2789 |
| 512 | 93.5 % | 22.5 % | 0.1430 | 0.2735 | 0.1716 |
| 1024 | 97.3 % | 45.0 % | 0.0595 | 0.1068 | 0.0659 |
| 1536 | 99.0 % | 67.5 % | 0.0215 | 0.0381 | 0.0227 |

done in 5441 s

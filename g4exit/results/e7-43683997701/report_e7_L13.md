# E7 -- structure of the FFN of layer 13 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 52521, validation 5979, held-out test 11561; split by record
instrument check (numpy FFN vs the engine's ffn_out): 3.63e-04

## I. the linear part
one linear map of x: error 0.3858 (dir 0.6007). Reduced rank: r=64 0.4380, r=256 0.3953, r=512 0.3861, r=1024 0.3848

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 8.8 % of the variance (coverage 57 %), previous token 5.4 % (coverage 56 %)
- FFN output y: current token explains 15.4 % of the variance (coverage 57 %), previous token 11.2 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains -0.9 % of the variance (coverage 57 %), previous token -1.4 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 30.5 / 47.9 / 71.2 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 1836/3213/4932; x64 160/795/1954; x256 50/182/615; tok256 5/112/3553. Token-table coverage of test: 84 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.3858 | 0.6007 | |
| dynamic oracle 6.25 % | 6.2 % | 0.0355 | 0.0511 | 0.1824 |
| dynamic oracle 12.5 % | 12.5 % | 0.0273 | 0.0392 | 0.0998 |
| dynamic oracle 25 % | 25.0 % | 0.0156 | 0.0224 | 0.0342 |
| dynamic oracle 50 % | 50.0 % | 0.0035 | 0.0050 | 0.0042 |
| one global static set 12.5 % | 12.5 % | 0.2605 | 0.4036 | 0.4823 |
| one global static set 25 % | 25.0 % | 0.1816 | 0.2758 | 0.3185 |
| one global static set 50 % | 50.0 % | 0.0848 | 0.1241 | 0.1340 |
| static set per region, x16, 12.5 % (pool not budgeted) | 12.5 % | 0.1724 | 0.2620 | 0.3475 |
| static set per region, x16, 25 % (pool not budgeted) | 25.0 % | 0.1043 | 0.1551 | 0.2075 |
| static set per region, x16, 50 % (pool not budgeted) | 50.0 % | 0.0454 | 0.0662 | 0.0777 |
| static set per region, x64, 12.5 % (pool not budgeted) | 12.5 % | 0.1327 | 0.1985 | 0.2964 |
| static set per region, x64, 25 % (pool not budgeted) | 25.0 % | 0.0845 | 0.1251 | 0.1769 |
| static set per region, x64, 50 % (pool not budgeted) | 50.0 % | 0.0388 | 0.0568 | 0.0656 |
| static set per region, x256, 12.5 % (pool not budgeted) | 12.5 % | 0.1170 | 0.1752 | 0.2703 |
| static set per region, x256, 25 % (pool not budgeted) | 25.0 % | 0.0761 | 0.1128 | 0.1571 |
| static set per region, x256, 50 % (pool not budgeted) | 50.0 % | 0.0380 | 0.0555 | 0.0600 |
| static set per region, tok256, 12.5 % (pool not budgeted) | 12.5 % | 0.2650 | 0.4132 | 0.4479 |
| static set per region, tok256, 25 % (pool not budgeted) | 25.0 % | 0.1968 | 0.3015 | 0.3008 |
| static set per region, tok256, 50 % (pool not budgeted) | 50.0 % | 0.1036 | 0.1526 | 0.1345 |

budgeted block emulation: shared set + the K nearest regions' expert sets, weights copied from E4B, pool <= 2.25x

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| 12.5 % nominal: shared 640 + 4 x 160 of 64 experts, router x64 (pool 1.06x) | 11.7 % | 0.1691 | 0.2562 | 0.3698 |
| 12.5 % nominal: shared 640 + 16 x 40 of 256 experts, router x256 (pool 1.06x) | 11.0 % | 0.1713 | 0.2594 | 0.3806 |
| 25 % nominal: shared 1280 + 1 x 1280 of 16 experts, router x16 (pool 2.12x) | 25.0 % | 0.1213 | 0.1811 | 0.2373 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router x64 (pool 2.12x) | 22.7 % | 0.1082 | 0.1607 | 0.2417 |
| 25 % nominal: shared 1280 + 16 x 80 of 256 experts, router x256 (pool 2.12x) | 21.4 % | 0.1125 | 0.1666 | 0.2528 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router tok>x64 (pool 2.12x) | 22.8 % | 0.1975 | 0.3022 | 0.3361 |
| 37.5 % nominal: shared 1280 + 8 x 320 of 64 experts, router x64 (pool 2.12x) | 30.5 % | 0.0859 | 0.1264 | 0.1915 |
| 50 % nominal: shared 2560 + 2 x 1280 of 16 experts, router x16 (pool 2.25x) | 46.1 % | 0.0593 | 0.0870 | 0.1105 |
| 50 % nominal: shared 2560 + 8 x 320 of 64 experts, router x64 (pool 2.25x) | 43.0 % | 0.0609 | 0.0891 | 0.1257 |

## B. gate and up pre-activations through a rank-r bottleneck (all neurons kept), + linear correction
centred x: the top 128 / 256 / 512 / 1024 principal directions hold 48.8 / 61.2 / 74.7 / 88.1 % of its variance

| rank r | pre-activation energy kept | gate+up bytes vs dense | error with correction | dir | error without |
|---:|---:|---:|---:|---:|---:|
| 256 | 86.0 % | 11.2 % | 0.2726 | 0.4248 | 0.3448 |
| 512 | 90.9 % | 22.5 % | 0.1926 | 0.2868 | 0.2300 |
| 1024 | 95.8 % | 45.0 % | 0.0941 | 0.1297 | 0.1068 |
| 1536 | 98.2 % | 67.5 % | 0.0400 | 0.0540 | 0.0445 |

done in 4617 s

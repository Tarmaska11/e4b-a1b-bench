# E7 -- structure of the FFN of layer 34 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51904, validation 6326, held-out test 11342; split by record
instrument check (numpy FFN vs the engine's ffn_out): 4.59e-04

## I. the linear part
one linear map of x: error 0.2505 (dir 0.4352). Reduced rank: r=64 0.2917, r=256 0.2641, r=512 0.2548, r=1024 0.2508

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 17.9 % of the variance (coverage 56 %), previous token 7.4 % (coverage 56 %)
- FFN output y: current token explains 20.7 % of the variance (coverage 56 %), previous token 7.6 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains 0.1 % of the variance (coverage 56 %), previous token -1.6 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 30.9 / 47.9 / 71.0 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 1605/2957/6540; x64 121/754/1615; x256 34/181/621; tok256 5/101/4909. Token-table coverage of test: 84 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2505 | 0.4352 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0441 | 0.0751 | 0.1326 |
| dynamic oracle 12.5 % | 12.5 % | 0.0353 | 0.0600 | 0.0746 |
| dynamic oracle 25 % | 25.0 % | 0.0209 | 0.0355 | 0.0297 |
| dynamic oracle 50 % | 50.0 % | 0.0049 | 0.0084 | 0.0050 |
| one global static set 12.5 % | 12.5 % | 0.1527 | 0.2667 | 0.2663 |
| one global static set 25 % | 25.0 % | 0.1082 | 0.1874 | 0.1696 |
| one global static set 50 % | 50.0 % | 0.0558 | 0.0959 | 0.0820 |
| static set per region, x16, 12.5 % (pool not budgeted) | 12.5 % | 0.1341 | 0.2335 | 0.2276 |
| static set per region, x16, 25 % (pool not budgeted) | 25.0 % | 0.0952 | 0.1637 | 0.1408 |
| static set per region, x16, 50 % (pool not budgeted) | 50.0 % | 0.0510 | 0.0867 | 0.0665 |
| static set per region, x64, 12.5 % (pool not budgeted) | 12.5 % | 0.1181 | 0.2044 | 0.2012 |
| static set per region, x64, 25 % (pool not budgeted) | 25.0 % | 0.0850 | 0.1456 | 0.1249 |
| static set per region, x64, 50 % (pool not budgeted) | 50.0 % | 0.0499 | 0.0846 | 0.0617 |
| static set per region, x256, 12.5 % (pool not budgeted) | 12.5 % | 0.1037 | 0.1793 | 0.1798 |
| static set per region, x256, 25 % (pool not budgeted) | 25.0 % | 0.0776 | 0.1331 | 0.1132 |
| static set per region, x256, 50 % (pool not budgeted) | 50.0 % | 0.0472 | 0.0808 | 0.0567 |
| static set per region, tok256, 12.5 % (pool not budgeted) | 12.5 % | 0.1404 | 0.2447 | 0.2331 |
| static set per region, tok256, 25 % (pool not budgeted) | 25.0 % | 0.1045 | 0.1797 | 0.1499 |
| static set per region, tok256, 50 % (pool not budgeted) | 50.0 % | 0.0614 | 0.1036 | 0.0749 |

budgeted block emulation: shared set + the K nearest regions' expert sets, weights copied from E4B, pool <= 2.25x

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| 12.5 % nominal: shared 640 + 4 x 160 of 64 experts, router x64 (pool 1.06x) | 11.5 % | 0.1279 | 0.2226 | 0.2428 |
| 12.5 % nominal: shared 640 + 16 x 40 of 256 experts, router x256 (pool 1.06x) | 11.1 % | 0.1211 | 0.2111 | 0.2447 |
| 25 % nominal: shared 1280 + 1 x 1280 of 16 experts, router x16 (pool 2.12x) | 25.0 % | 0.0955 | 0.1648 | 0.1443 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router x64 (pool 2.12x) | 22.5 % | 0.0928 | 0.1602 | 0.1553 |
| 25 % nominal: shared 1280 + 16 x 80 of 256 experts, router x256 (pool 2.12x) | 21.6 % | 0.0869 | 0.1502 | 0.1597 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router tok>x64 (pool 2.12x) | 22.5 % | 0.1135 | 0.1962 | 0.1759 |
| 37.5 % nominal: shared 1280 + 8 x 320 of 64 experts, router x64 (pool 2.12x) | 29.6 % | 0.0782 | 0.1346 | 0.1259 |
| 50 % nominal: shared 2560 + 2 x 1280 of 16 experts, router x16 (pool 2.25x) | 45.3 % | 0.0562 | 0.0963 | 0.0789 |
| 50 % nominal: shared 2560 + 8 x 320 of 64 experts, router x64 (pool 2.25x) | 42.0 % | 0.0581 | 0.0994 | 0.0865 |

## B. gate and up pre-activations through a rank-r bottleneck (all neurons kept), + linear correction
centred x: the top 128 / 256 / 512 / 1024 principal directions hold 48.4 / 60.4 / 73.9 / 87.5 % of its variance

| rank r | pre-activation energy kept | gate+up bytes vs dense | error with correction | dir | error without |
|---:|---:|---:|---:|---:|---:|
| 256 | 83.3 % | 11.2 % | 0.1951 | 0.3360 | 0.2411 |
| 512 | 88.7 % | 22.5 % | 0.1494 | 0.2531 | 0.1806 |
| 1024 | 94.5 % | 45.0 % | 0.0806 | 0.1312 | 0.0969 |
| 1536 | 97.5 % | 67.5 % | 0.0369 | 0.0592 | 0.0452 |

done in 4619 s

# E7 -- structure of the FFN of layer 16 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51210, validation 5284, held-out test 11324; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.16e-04

## I. the linear part
one linear map of x: error 0.2718 (dir 0.3989). Reduced rank: r=64 0.3715, r=256 0.3083, r=512 0.2855, r=1024 0.2739

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 10.2 % of the variance (coverage 57 %), previous token 6.3 % (coverage 56 %)
- FFN output y: current token explains 10.5 % of the variance (coverage 57 %), previous token 6.5 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains -1.4 % of the variance (coverage 57 %), previous token -1.8 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 31.5 / 49.4 / 73.1 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 2176/3080/4295; x64 375/781/1900; x256 61/188/467; tok256 4/102/3255. Token-table coverage of test: 85 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.2718 | 0.3989 | |
| dynamic oracle 6.25 % | 6.3 % | 0.1084 | 0.1527 | 0.2482 |
| dynamic oracle 12.5 % | 12.5 % | 0.0734 | 0.1029 | 0.1246 |
| dynamic oracle 25 % | 25.0 % | 0.0351 | 0.0490 | 0.0423 |
| dynamic oracle 50 % | 50.0 % | 0.0062 | 0.0086 | 0.0058 |
| one global static set 12.5 % | 12.5 % | 0.2148 | 0.3209 | 0.4333 |
| one global static set 25 % | 25.0 % | 0.1713 | 0.2558 | 0.2925 |
| one global static set 50 % | 50.0 % | 0.0993 | 0.1473 | 0.1424 |
| static set per region, x16, 12.5 % (pool not budgeted) | 12.5 % | 0.2000 | 0.2956 | 0.3765 |
| static set per region, x16, 25 % (pool not budgeted) | 25.0 % | 0.1523 | 0.2251 | 0.2397 |
| static set per region, x16, 50 % (pool not budgeted) | 50.0 % | 0.0852 | 0.1249 | 0.1077 |
| static set per region, x64, 12.5 % (pool not budgeted) | 12.5 % | 0.1910 | 0.2813 | 0.3528 |
| static set per region, x64, 25 % (pool not budgeted) | 25.0 % | 0.1439 | 0.2119 | 0.2192 |
| static set per region, x64, 50 % (pool not budgeted) | 50.0 % | 0.0802 | 0.1171 | 0.0972 |
| static set per region, x256, 12.5 % (pool not budgeted) | 12.5 % | 0.1862 | 0.2738 | 0.3376 |
| static set per region, x256, 25 % (pool not budgeted) | 25.0 % | 0.1402 | 0.2062 | 0.2066 |
| static set per region, x256, 50 % (pool not budgeted) | 50.0 % | 0.0788 | 0.1150 | 0.0922 |
| static set per region, tok256, 12.5 % (pool not budgeted) | 12.5 % | 0.2199 | 0.3240 | 0.4028 |
| static set per region, tok256, 25 % (pool not budgeted) | 25.0 % | 0.1753 | 0.2582 | 0.2668 |
| static set per region, tok256, 50 % (pool not budgeted) | 50.0 % | 0.1042 | 0.1522 | 0.1278 |

budgeted block emulation: shared set + the K nearest regions' expert sets, weights copied from E4B, pool <= 2.25x

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| 12.5 % nominal: shared 640 + 4 x 160 of 64 experts, router x64 (pool 1.06x) | 11.4 % | 0.1985 | 0.2941 | 0.4107 |
| 12.5 % nominal: shared 640 + 16 x 40 of 256 experts, router x256 (pool 1.06x) | 10.9 % | 0.1991 | 0.2941 | 0.4225 |
| 25 % nominal: shared 1280 + 1 x 1280 of 16 experts, router x16 (pool 2.12x) | 25.0 % | 0.1553 | 0.2303 | 0.2508 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router x64 (pool 2.12x) | 22.3 % | 0.1546 | 0.2292 | 0.2741 |
| 25 % nominal: shared 1280 + 16 x 80 of 256 experts, router x256 (pool 2.12x) | 21.2 % | 0.1562 | 0.2309 | 0.2914 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router tok>x64 (pool 2.12x) | 22.4 % | 0.1801 | 0.2659 | 0.3078 |
| 37.5 % nominal: shared 1280 + 8 x 320 of 64 experts, router x64 (pool 2.12x) | 29.1 % | 0.1339 | 0.1979 | 0.2304 |
| 50 % nominal: shared 2560 + 2 x 1280 of 16 experts, router x16 (pool 2.25x) | 45.8 % | 0.0959 | 0.1417 | 0.1338 |
| 50 % nominal: shared 2560 + 8 x 320 of 64 experts, router x64 (pool 2.25x) | 42.0 % | 0.1008 | 0.1488 | 0.1541 |

## B. gate and up pre-activations through a rank-r bottleneck (all neurons kept), + linear correction
centred x: the top 128 / 256 / 512 / 1024 principal directions hold 48.3 / 60.9 / 74.6 / 88.0 % of its variance

| rank r | pre-activation energy kept | gate+up bytes vs dense | error with correction | dir | error without |
|---:|---:|---:|---:|---:|---:|
| 256 | 80.3 % | 11.2 % | 0.1762 | 0.2538 | 0.2471 |
| 512 | 87.2 % | 22.5 % | 0.1304 | 0.1847 | 0.1734 |
| 1024 | 93.9 % | 45.0 % | 0.0738 | 0.1025 | 0.0930 |
| 1536 | 97.2 % | 67.5 % | 0.0376 | 0.0519 | 0.0463 |

done in 5405 s

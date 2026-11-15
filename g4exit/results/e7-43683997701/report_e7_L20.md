# E7 -- structure of the FFN of layer 20 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51797, validation 6293, held-out test 11474; split by record
instrument check (numpy FFN vs the engine's ffn_out): 2.00e-04

## I. the linear part
one linear map of x: error 0.1464 (dir 0.1929). Reduced rank: r=64 0.2723, r=256 0.1984, r=512 0.1686, r=1024 0.1509

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 17.9 % of the variance (coverage 57 %), previous token 10.8 % (coverage 56 %)
- FFN output y: current token explains 19.9 % of the variance (coverage 57 %), previous token 12.4 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains 0.2 % of the variance (coverage 57 %), previous token -0.6 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 37.2 / 55.8 / 77.8 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 1217/3337/4978; x64 380/762/1558; x256 36/184/911; tok256 3/92/3373. Token-table coverage of test: 85 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.1464 | 0.1929 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0684 | 0.0899 | 0.2843 |
| dynamic oracle 12.5 % | 12.5 % | 0.0473 | 0.0620 | 0.1365 |
| dynamic oracle 25 % | 25.0 % | 0.0233 | 0.0305 | 0.0399 |
| dynamic oracle 50 % | 50.0 % | 0.0043 | 0.0056 | 0.0044 |
| one global static set 12.5 % | 12.5 % | 0.1082 | 0.1447 | 0.3879 |
| one global static set 25 % | 25.0 % | 0.0824 | 0.1104 | 0.2326 |
| one global static set 50 % | 50.0 % | 0.0461 | 0.0620 | 0.0913 |
| static set per region, x16, 12.5 % (pool not budgeted) | 12.5 % | 0.1017 | 0.1354 | 0.3371 |
| static set per region, x16, 25 % (pool not budgeted) | 25.0 % | 0.0755 | 0.1005 | 0.1851 |
| static set per region, x16, 50 % (pool not budgeted) | 50.0 % | 0.0405 | 0.0539 | 0.0658 |
| static set per region, x64, 12.5 % (pool not budgeted) | 12.5 % | 0.0969 | 0.1287 | 0.3146 |
| static set per region, x64, 25 % (pool not budgeted) | 25.0 % | 0.0713 | 0.0944 | 0.1674 |
| static set per region, x64, 50 % (pool not budgeted) | 50.0 % | 0.0379 | 0.0502 | 0.0575 |
| static set per region, x256, 12.5 % (pool not budgeted) | 12.5 % | 0.0946 | 0.1254 | 0.2984 |
| static set per region, x256, 25 % (pool not budgeted) | 25.0 % | 0.0693 | 0.0917 | 0.1553 |
| static set per region, x256, 50 % (pool not budgeted) | 50.0 % | 0.0368 | 0.0486 | 0.0524 |
| static set per region, tok256, 12.5 % (pool not budgeted) | 12.5 % | 0.1107 | 0.1457 | 0.3492 |
| static set per region, tok256, 25 % (pool not budgeted) | 25.0 % | 0.0841 | 0.1112 | 0.1966 |
| static set per region, tok256, 50 % (pool not budgeted) | 50.0 % | 0.0477 | 0.0631 | 0.0738 |

budgeted block emulation: shared set + the K nearest regions' expert sets, weights copied from E4B, pool <= 2.25x

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| 12.5 % nominal: shared 640 + 4 x 160 of 64 experts, router x64 (pool 1.06x) | 10.9 % | 0.0968 | 0.1288 | 0.3833 |
| 12.5 % nominal: shared 640 + 16 x 40 of 256 experts, router x256 (pool 1.06x) | 10.0 % | 0.0973 | 0.1294 | 0.4188 |
| 25 % nominal: shared 1280 + 1 x 1280 of 16 experts, router x16 (pool 2.12x) | 25.0 % | 0.0757 | 0.1009 | 0.1907 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router x64 (pool 2.12x) | 21.3 % | 0.0755 | 0.1003 | 0.2289 |
| 25 % nominal: shared 1280 + 16 x 80 of 256 experts, router x256 (pool 2.12x) | 19.6 % | 0.0769 | 0.1021 | 0.2626 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router tok>x64 (pool 2.12x) | 21.4 % | 0.0871 | 0.1152 | 0.2452 |
| 37.5 % nominal: shared 1280 + 8 x 320 of 64 experts, router x64 (pool 2.12x) | 26.9 % | 0.0670 | 0.0888 | 0.1919 |
| 50 % nominal: shared 2560 + 2 x 1280 of 16 experts, router x16 (pool 2.25x) | 44.8 % | 0.0463 | 0.0617 | 0.0893 |
| 50 % nominal: shared 2560 + 8 x 320 of 64 experts, router x64 (pool 2.25x) | 39.9 % | 0.0499 | 0.0660 | 0.1159 |

## B. gate and up pre-activations through a rank-r bottleneck (all neurons kept), + linear correction
centred x: the top 128 / 256 / 512 / 1024 principal directions hold 53.5 / 65.8 / 78.6 / 90.5 % of its variance

| rank r | pre-activation energy kept | gate+up bytes vs dense | error with correction | dir | error without |
|---:|---:|---:|---:|---:|---:|
| 256 | 86.0 % | 11.2 % | 0.0843 | 0.1067 | 0.1506 |
| 512 | 91.1 % | 22.5 % | 0.0575 | 0.0721 | 0.0961 |
| 1024 | 95.9 % | 45.0 % | 0.0299 | 0.0372 | 0.0462 |
| 1536 | 98.2 % | 67.5 % | 0.0148 | 0.0185 | 0.0219 |

done in 5585 s

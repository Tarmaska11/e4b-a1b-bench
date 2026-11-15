# E7 -- structure of the FFN of layer 26 (Q4_K_M weights, closed form, no gradient training)

positions (distinct token prefixes): train 51797, validation 6293, held-out test 11474; split by record
instrument check (numpy FFN vs the engine's ffn_out): 3.64e-04

## I. the linear part
one linear map of x: error 0.1738 (dir 0.2697). Reduced rank: r=64 0.2253, r=256 0.1918, r=512 0.1805, r=1024 0.1749

## A1. how much the token type explains (held-out positions of types seen >= 8 times in train)
- FFN input x: current token explains 16.5 % of the variance (coverage 57 %), previous token 10.8 % (coverage 56 %)
- FFN output y: current token explains 20.8 % of the variance (coverage 57 %), previous token 12.8 % (coverage 56 %)
- nonlinear residual y - Lx: current token explains -0.6 % of the variance (coverage 57 %), previous token -1.2 % (coverage 56 %)
- output side: the top 256 / 512 / 1024 directions hold 31.5 / 48.8 / 71.8 % of the nonlinear residual's energy

## A2. region structure

regions: x16, x64, x256, tok256. Sizes (train positions per region, min / median / max): x16 999/3368/4903; x64 205/778/2321; x256 50/185/909; tok256 6/101/7097. Token-table coverage of test: 85 %

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| linear map alone | 0 | 0.1738 | 0.2697 | |
| dynamic oracle 6.25 % | 6.3 % | 0.0538 | 0.0824 | 0.2219 |
| dynamic oracle 12.5 % | 12.5 % | 0.0407 | 0.0618 | 0.1112 |
| dynamic oracle 25 % | 25.0 % | 0.0221 | 0.0332 | 0.0356 |
| dynamic oracle 50 % | 50.0 % | 0.0045 | 0.0067 | 0.0045 |
| one global static set 12.5 % | 12.5 % | 0.1181 | 0.1900 | 0.2946 |
| one global static set 25 % | 25.0 % | 0.0900 | 0.1457 | 0.1804 |
| one global static set 50 % | 50.0 % | 0.0503 | 0.0817 | 0.0824 |
| static set per region, x16, 12.5 % (pool not budgeted) | 12.5 % | 0.1057 | 0.1691 | 0.2703 |
| static set per region, x16, 25 % (pool not budgeted) | 25.0 % | 0.0800 | 0.1282 | 0.1561 |
| static set per region, x16, 50 % (pool not budgeted) | 50.0 % | 0.0464 | 0.0742 | 0.0669 |
| static set per region, x64, 12.5 % (pool not budgeted) | 12.5 % | 0.0996 | 0.1591 | 0.2577 |
| static set per region, x64, 25 % (pool not budgeted) | 25.0 % | 0.0753 | 0.1205 | 0.1442 |
| static set per region, x64, 50 % (pool not budgeted) | 50.0 % | 0.0447 | 0.0714 | 0.0611 |
| static set per region, x256, 12.5 % (pool not budgeted) | 12.5 % | 0.0962 | 0.1534 | 0.2443 |
| static set per region, x256, 25 % (pool not budgeted) | 25.0 % | 0.0746 | 0.1191 | 0.1359 |
| static set per region, x256, 50 % (pool not budgeted) | 50.0 % | 0.0450 | 0.0715 | 0.0577 |
| static set per region, tok256, 12.5 % (pool not budgeted) | 12.5 % | 0.1164 | 0.1850 | 0.2875 |
| static set per region, tok256, 25 % (pool not budgeted) | 25.0 % | 0.0915 | 0.1460 | 0.1723 |
| static set per region, tok256, 50 % (pool not budgeted) | 50.0 % | 0.0552 | 0.0879 | 0.0774 |

budgeted block emulation: shared set + the K nearest regions' expert sets, weights copied from E4B, pool <= 2.25x

| arm | active neurons | error with bypass | dir | error without bypass |
|---|---:|---:|---:|---:|
| 12.5 % nominal: shared 640 + 4 x 160 of 64 experts, router x64 (pool 1.06x) | 11.4 % | 0.1054 | 0.1692 | 0.3102 |
| 12.5 % nominal: shared 640 + 16 x 40 of 256 experts, router x256 (pool 1.06x) | 10.8 % | 0.1050 | 0.1683 | 0.3231 |
| 25 % nominal: shared 1280 + 1 x 1280 of 16 experts, router x16 (pool 2.12x) | 25.0 % | 0.0809 | 0.1305 | 0.1615 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router x64 (pool 2.12x) | 22.3 % | 0.0794 | 0.1281 | 0.1887 |
| 25 % nominal: shared 1280 + 16 x 80 of 256 experts, router x256 (pool 2.12x) | 21.2 % | 0.0793 | 0.1277 | 0.2059 |
| 25 % nominal: shared 1280 + 4 x 320 of 64 experts, router tok>x64 (pool 2.12x) | 22.6 % | 0.0946 | 0.1515 | 0.2012 |
| 37.5 % nominal: shared 1280 + 8 x 320 of 64 experts, router x64 (pool 2.12x) | 29.4 % | 0.0684 | 0.1102 | 0.1548 |
| 50 % nominal: shared 2560 + 2 x 1280 of 16 experts, router x16 (pool 2.25x) | 45.7 % | 0.0505 | 0.0814 | 0.0824 |
| 50 % nominal: shared 2560 + 8 x 320 of 64 experts, router x64 (pool 2.25x) | 42.4 % | 0.0512 | 0.0825 | 0.0960 |

## B. gate and up pre-activations through a rank-r bottleneck (all neurons kept), + linear correction
centred x: the top 128 / 256 / 512 / 1024 principal directions hold 55.7 / 67.6 / 79.8 / 90.9 % of its variance

| rank r | pre-activation energy kept | gate+up bytes vs dense | error with correction | dir | error without |
|---:|---:|---:|---:|---:|---:|
| 256 | 83.8 % | 11.2 % | 0.1104 | 0.1708 | 0.1430 |
| 512 | 89.7 % | 22.5 % | 0.0775 | 0.1192 | 0.0962 |
| 1024 | 95.2 % | 45.0 % | 0.0414 | 0.0633 | 0.0503 |
| 1536 | 97.8 % | 67.5 % | 0.0208 | 0.0317 | 0.0250 |

done in 5543 s

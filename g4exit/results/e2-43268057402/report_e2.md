# E2 — re-building E4B's FFN in closed form (Q4_K_M weights, cached activations, no gradient training)

tokens: train 20975, validation 2428, held-out test 5488 (split by prompt). Lower is better; 0 = identical to the dense FFN.

## layer 6  (10240 neurons)

instrument check — numpy FFN vs the engine's own FFN output: rel. error 1.93e-04
share of total contribution mass carried by the top 6.25 / 12.5 / 25 / 50 % of neurons: 60.6 % / 73.2 % / 87.2 % / 97.8 %

| neurons kept per token | oracle | oracle + bypass | gate-selected + bypass | static + refit | 8 regions + refit |
|---:|---:|---:|---:|---:|---:|
| 0 (a linear map alone) | | 0.1695 | | | |
| 6.25 % | 0.0199 | 0.0024 | 0.0032 | 0.1153 | 0.1219 |
| 12.50 % | 0.0070 | 0.0012 | 0.0022 | 0.0991 | 0.1018 |
| 25.00 % | 0.0013 | 0.0003 | 0.0012 | 0.0742 |  |
| 50.00 % | 0.0000 | 0.0000 | 0.0001 |  |  |

## layer 16  (10240 neurons)

instrument check — numpy FFN vs the engine's own FFN output: rel. error 2.84e-04
share of total contribution mass carried by the top 6.25 / 12.5 / 25 / 50 % of neurons: 32.5 % / 47.7 % / 67.6 % / 89.1 %

| neurons kept per token | oracle | oracle + bypass | gate-selected + bypass | static + refit | 8 regions + refit |
|---:|---:|---:|---:|---:|---:|
| 0 (a linear map alone) | | 0.1605 | | | |
| 6.25 % | 0.1686 | 0.0613 | 0.0774 | 0.1471 | 0.1577 |
| 12.50 % | 0.0850 | 0.0412 | 0.0570 | 0.1442 | 0.1531 |
| 25.00 % | 0.0291 | 0.0196 | 0.0411 | 0.1240 |  |
| 50.00 % | 0.0040 | 0.0035 | 0.0174 |  |  |

## layer 34  (10240 neurons)

instrument check — numpy FFN vs the engine's own FFN output: rel. error 4.71e-04
share of total contribution mass carried by the top 6.25 / 12.5 / 25 / 50 % of neurons: 31.8 % / 45.4 % / 64.5 % / 87.0 %

| neurons kept per token | oracle | oracle + bypass | gate-selected + bypass | static + refit | 8 regions + refit |
|---:|---:|---:|---:|---:|---:|
| 0 (a linear map alone) | | 0.1680 | | | |
| 6.25 % | 0.1231 | 0.0328 | 0.0342 | 0.1413 | 0.1423 |
| 12.50 % | 0.0686 | 0.0256 | 0.0284 | 0.1298 | 0.1314 |
| 25.00 % | 0.0266 | 0.0148 | 0.0243 | 0.1132 |  |
| 50.00 % | 0.0043 | 0.0034 | 0.0151 |  |  |

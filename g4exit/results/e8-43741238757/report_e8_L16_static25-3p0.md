# E8 -- gradient pilot of one rebuilt block, layer 16 (L16_static25-3p0)

real positions (distinct token prefixes): train 86587, validation 4281, held-out test 9894; mixed-input share of a batch 50 %
instrument check (dense FFN recomputed vs the engine's ffn_out): 2.03e-04

## arm static25: 2560 shared + 0 of 0 experts x 0 = 25.0 % of the neurons active, pool 0.25x, budget 3.47 h
closed-form start: error 0.1630 (dir 0.2501). References on the same positions: linear map alone 0.2592; dynamic oracle + bypass at the same active share 0.0390 (dir 0.0556)
after 9207 steps (9.43 M samples, 54.4 passes over the real positions, 1.4 s/step): **held-out error 0.1414 (dir 0.2120)**, on train positions 0.0882
curve (hours: held-out / train): 0.00: 0.1630 / 0.1103, 0.25: 0.1402 / 0.1014, 0.50: 0.1381 / 0.0971, 0.74: 0.1383 / 0.0947, 0.99: 0.1388 / 0.0930, 1.24: 0.1394 / 0.0917, 1.49: 0.1398 / 0.0907, 1.73: 0.1403 / 0.0901, 1.98: 0.1407 / 0.0895, 2.23: 0.1410 / 0.0891, 2.48: 0.1412 / 0.0888, 2.73: 0.1413 / 0.0885, 2.97: 0.1414 / 0.0884, 3.22: 0.1416 / 0.0883

done in 18972 s

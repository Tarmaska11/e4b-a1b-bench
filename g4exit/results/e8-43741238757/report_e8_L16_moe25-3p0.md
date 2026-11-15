# E8 -- gradient pilot of one rebuilt block, layer 16 (L16_moe25-3p0)

real positions (distinct token prefixes): train 89807, validation 4776, held-out test 9805; mixed-input share of a batch 50 %
instrument check (dense FFN recomputed vs the engine's ffn_out): 2.21e-04

## arm moe25: 1280 shared + 4 of 64 experts x 320 = 25.0 % of the neurons active, pool 2.12x, budget 3.43 h
closed-form start: error 0.1614 (dir 0.2421). References on the same positions: linear map alone 0.2597; dynamic oracle + bypass at the same active share 0.0389 (dir 0.0553)
after 855 steps (0.88 M samples, 4.9 passes over the real positions, 14.5 s/step): **held-out error 0.1238 (dir 0.1861)**, on train positions 0.0788
curve (hours: held-out / train): 0.00: 0.1614 / 0.1132, 0.25: 0.1473 / 0.1044, 0.49: 0.1401 / 0.0989, 0.74: 0.1355 / 0.0945, 0.99: 0.1322 / 0.0909, 1.24: 0.1299 / 0.0883, 1.49: 0.1283 / 0.0859, 1.73: 0.1269 / 0.0840, 1.98: 0.1261 / 0.0826, 2.22: 0.1253 / 0.0814, 2.47: 0.1248 / 0.0804, 2.72: 0.1245 / 0.0798, 2.97: 0.1242 / 0.0794, 3.22: 0.1240 / 0.0790

done in 18982 s

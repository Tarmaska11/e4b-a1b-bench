# E8 -- gradient pilot of one rebuilt block, layer 16 (L16_topk25-3p0)

real positions (distinct token prefixes): train 86587, validation 4281, held-out test 9894; mixed-input share of a batch 50 %
instrument check (dense FFN recomputed vs the engine's ffn_out): 2.03e-04

## arm topk25: all 10240 neurons kept in the file, per token only the top 25.0 % (by their own exact score) are used, + bypass; every weight trained with that rule in the loop; budget 3.39 h
closed-form start: error 0.0334 (dir 0.0476). References on the same positions: linear map alone 0.2592; dynamic oracle + bypass at the same active share 0.0390 (dir 0.0556)
after 5802 steps (5.94 M samples, 34.3 passes over the real positions, 2.1 s/step): **held-out error 0.0211 (dir 0.0289)**, on train positions 0.0090
curve (hours: held-out / train): 0.00: 0.0334 / 0.0252, 0.24: 0.0244 / 0.0177, 0.48: 0.0220 / 0.0138, 0.73: 0.0215 / 0.0123, 0.97: 0.0212 / 0.0114, 1.21: 0.0212 / 0.0107, 1.45: 0.0212 / 0.0103, 1.70: 0.0211 / 0.0100, 1.94: 0.0211 / 0.0097, 2.18: 0.0211 / 0.0095, 2.42: 0.0211 / 0.0093, 2.67: 0.0211 / 0.0092, 2.91: 0.0211 / 0.0091, 3.15: 0.0211 / 0.0090

done in 18967 s

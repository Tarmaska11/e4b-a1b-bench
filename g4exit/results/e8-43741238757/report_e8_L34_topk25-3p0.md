# E8 -- gradient pilot of one rebuilt block, layer 34 (L34_topk25-3p0)

real positions (distinct token prefixes): train 84738, validation 4181, held-out test 9570; mixed-input share of a batch 50 %
instrument check (dense FFN recomputed vs the engine's ffn_out): 4.63e-04

## arm topk25: all 10240 neurons kept in the file, per token only the top 25.0 % (by their own exact score) are used, + bypass; every weight trained with that rule in the loop; budget 3.50 h
closed-form start: error 0.0195 (dir 0.0333). References on the same positions: linear map alone 0.2292; dynamic oracle + bypass at the same active share 0.0233 (dir 0.0397)
after 3625 steps (3.71 M samples, 21.9 passes over the real positions, 3.5 s/step): **held-out error 0.0128 (dir 0.0193)**, on train positions 0.0060
curve (hours: held-out / train): 0.00: 0.0195 / 0.0146, 0.25: 0.0164 / 0.0124, 0.50: 0.0149 / 0.0103, 0.75: 0.0141 / 0.0090, 1.00: 0.0137 / 0.0082, 1.25: 0.0134 / 0.0076, 1.50: 0.0133 / 0.0071, 1.75: 0.0131 / 0.0068, 2.00: 0.0131 / 0.0066, 2.25: 0.0130 / 0.0064, 2.51: 0.0129 / 0.0062, 2.76: 0.0129 / 0.0061, 3.01: 0.0129 / 0.0061, 3.26: 0.0129 / 0.0060

done in 19013 s

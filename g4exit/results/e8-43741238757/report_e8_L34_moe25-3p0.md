# E8 -- gradient pilot of one rebuilt block, layer 34 (L34_moe25-3p0)

real positions (distinct token prefixes): train 89020, validation 4457, held-out test 9654; mixed-input share of a batch 50 %
instrument check (dense FFN recomputed vs the engine's ffn_out): 4.69e-04

## arm moe25: 1280 shared + 4 of 64 experts x 320 = 25.0 % of the neurons active, pool 2.12x, budget 3.45 h
closed-form start: error 0.0942 (dir 0.1598). References on the same positions: linear map alone 0.2227; dynamic oracle + bypass at the same active share 0.0230 (dir 0.0392)
after 873 steps (0.89 M samples, 5.0 passes over the real positions, 14.2 s/step): **held-out error 0.0749 (dir 0.1272)**, on train positions 0.0362
curve (hours: held-out / train): 0.00: 0.0942 / 0.0560, 0.25: 0.0875 / 0.0496, 0.49: 0.0837 / 0.0462, 0.74: 0.0810 / 0.0441, 0.99: 0.0791 / 0.0422, 1.24: 0.0780 / 0.0409, 1.49: 0.0771 / 0.0398, 1.74: 0.0764 / 0.0388, 1.98: 0.0760 / 0.0381, 2.23: 0.0755 / 0.0375, 2.48: 0.0753 / 0.0371, 2.72: 0.0750 / 0.0368, 2.97: 0.0749 / 0.0366, 3.22: 0.0748 / 0.0364

done in 18990 s

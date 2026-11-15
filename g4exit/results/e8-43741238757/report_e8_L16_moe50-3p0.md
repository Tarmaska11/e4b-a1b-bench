# E8 -- gradient pilot of one rebuilt block, layer 16 (L16_moe50-3p0)

real positions (distinct token prefixes): train 89807, validation 4776, held-out test 9805; mixed-input share of a batch 50 %
instrument check (dense FFN recomputed vs the engine's ffn_out): 2.21e-04

## arm moe50: 2560 shared + 8 of 64 experts x 320 = 50.0 % of the neurons active, pool 2.25x, budget 3.42 h
closed-form start: error 0.1109 (dir 0.1655). References on the same positions: linear map alone 0.2597; dynamic oracle + bypass at the same active share 0.0069 (dir 0.0098)
after 766 steps (0.78 M samples, 4.4 passes over the real positions, 16.1 s/step): **held-out error 0.0816 (dir 0.1224)**, on train positions 0.0494
curve (hours: held-out / train): 0.00: 0.1109 / 0.0779, 0.25: 0.0992 / 0.0701, 0.49: 0.0936 / 0.0654, 0.74: 0.0902 / 0.0617, 0.98: 0.0879 / 0.0592, 1.23: 0.0861 / 0.0569, 1.48: 0.0848 / 0.0550, 1.73: 0.0839 / 0.0536, 1.97: 0.0831 / 0.0524, 2.22: 0.0827 / 0.0515, 2.47: 0.0823 / 0.0508, 2.72: 0.0819 / 0.0502, 2.96: 0.0818 / 0.0499, 3.21: 0.0817 / 0.0496

done in 18998 s

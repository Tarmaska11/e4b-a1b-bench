# E8 -- gradient pilot of one rebuilt block, layer 16 (L16_control-0p45_topk125-2p5)

real positions (distinct token prefixes): train 86219, validation 4558, held-out test 9821; mixed-input share of a batch 50 %
instrument check (dense FFN recomputed vs the engine's ffn_out): 2.05e-04

## arm control: 10240 shared + 0 of 0 experts x 0 = 100.0 % of the neurons active, pool 1.00x, budget 0.51 h
closed-form start: error 0.0060 (dir 0.0085). References on the same positions: linear map alone 0.2542; dynamic oracle + bypass at the same active share 0.0000 (dir 0.0000)
after 538 steps (0.55 M samples, 3.2 passes over the real positions, 3.5 s/step): **held-out error 0.0013 (dir 0.0017)**, on train positions 0.0006
curve (hours: held-out / train): 0.00: 0.0060 / 0.0043, 0.07: 0.0036 / 0.0025, 0.14: 0.0025 / 0.0016, 0.20: 0.0020 / 0.0011, 0.27: 0.0017 / 0.0009, 0.34: 0.0015 / 0.0007, 0.40: 0.0014 / 0.0006, 0.47: 0.0013 / 0.0006
control verdict: start 0.0060 -> 0.0013 (NOT recovered)

## arm topk125: all 10240 neurons kept in the file, per token only the top 12.5 % (by their own exact score) are used, + bypass; every weight trained with that rule in the loop; budget 2.85 h
closed-form start: error 0.0687 (dir 0.0990). References on the same positions: linear map alone 0.2542; dynamic oracle + bypass at the same active share 0.0802 (dir 0.1153)
after 2980 steps (3.05 M samples, 17.7 passes over the real positions, 3.5 s/step): **held-out error 0.0418 (dir 0.0573)**, on train positions 0.0231
curve (hours: held-out / train): 0.00: 0.0687 / 0.0516, 0.20: 0.0558 / 0.0448, 0.41: 0.0489 / 0.0369, 0.61: 0.0454 / 0.0320, 0.82: 0.0438 / 0.0292, 1.02: 0.0430 / 0.0275, 1.22: 0.0425 / 0.0262, 1.43: 0.0423 / 0.0253, 1.63: 0.0422 / 0.0246, 1.84: 0.0420 / 0.0241, 2.04: 0.0420 / 0.0238, 2.25: 0.0419 / 0.0235, 2.45: 0.0419 / 0.0234, 2.65: 0.0419 / 0.0232

done in 18668 s

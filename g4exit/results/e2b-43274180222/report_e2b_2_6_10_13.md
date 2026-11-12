# E2b — selector and bypass design, layers 2,6,10,13

tokens: train 20975, validation 2428, held-out test 5488 (split by prompt)

## layer 2

instrument check (numpy vs engine FFN output): 2.25e-04
**Q4 noise floor: 0.0006** (bf16 original vs this Q4_K_M file, same inputs). By matrix: gate 0.0003, up 0.0002, down 0.0001. Weight-space error: gate 0.0057, up 0.0056, down 0.0004.
top 6.25 / 12.5 / 25 / 50 % of neurons carry: contribution mass 58 % / 69 % / 83 % / 96 %; energy 97.7 % / 98.8 % / 99.6 % / 100.0 %. Share of neurons with |gelu(g)| < 0.02: 23.0 %.
W_up spectrum: the top 128 / 512 of 2560 singular directions hold 17.6 / 44.6 % of its energy

| b | oracle | +mean | +lin256 | +lin | gate | gate-cal | gate+u128 | gate+u512 | pc-full | pc-r256 | pc-real128 | pc-real512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | 0.1649 | | | | | | | | |
| 6.25 % | 0.0330 | 0.0070 | 0.0028 | 0.0025 | 0.0030 | 0.0033 | 0.0092 | 0.0047 | 0.0235 | 0.0204 | 0.0307 | 0.0239 |
| 12.50 % | 0.0141 | 0.0033 | 0.0015 | 0.0014 | 0.0022 | 0.0019 | 0.0058 | 0.0029 | 0.0119 | 0.0096 | 0.0177 | 0.0126 |
| 25.00 % | 0.0033 | 0.0010 | 0.0006 | 0.0006 | 0.0016 | 0.0011 | 0.0033 | 0.0015 | 0.0039 | 0.0030 | 0.0084 | 0.0051 |
| 50.00 % | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0004 | 0.0003 | 0.0012 | 0.0005 | 0.0005 | 0.0004 | 0.0028 | 0.0014 |

## layer 6

instrument check (numpy vs engine FFN output): 1.94e-04
**Q4 noise floor: 0.0024** (bf16 original vs this Q4_K_M file, same inputs). By matrix: gate 0.0005, up 0.0004, down 0.0014. Weight-space error: gate 0.0057, up 0.0056, down 0.0061.
top 6.25 / 12.5 / 25 / 50 % of neurons carry: contribution mass 61 % / 73 % / 87 % / 98 %; energy 97.9 % / 99.0 % / 99.7 % / 100.0 %. Share of neurons with |gelu(g)| < 0.02: 41.6 %.
W_up spectrum: the top 128 / 512 of 2560 singular directions hold 14.2 / 41.2 % of its energy

| b | oracle | +mean | +lin256 | +lin | gate | gate-cal | gate+u128 | gate+u512 | pc-full | pc-r256 | pc-real128 | pc-real512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | 0.1696 | | | | | | | | |
| 6.25 % | 0.0199 | 0.0051 | 0.0027 | 0.0024 | 0.0032 | 0.0032 | 0.0129 | 0.0074 | 0.0185 | 0.0125 | 0.0279 | 0.0198 |
| 12.50 % | 0.0070 | 0.0022 | 0.0013 | 0.0012 | 0.0022 | 0.0020 | 0.0080 | 0.0046 | 0.0094 | 0.0061 | 0.0171 | 0.0115 |
| 25.00 % | 0.0013 | 0.0005 | 0.0003 | 0.0003 | 0.0012 | 0.0009 | 0.0041 | 0.0024 | 0.0033 | 0.0021 | 0.0090 | 0.0056 |
| 50.00 % | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0001 | 0.0011 | 0.0006 | 0.0005 | 0.0003 | 0.0035 | 0.0020 |

## layer 10

instrument check (numpy vs engine FFN output): 2.25e-04
**Q4 noise floor: 0.0007** (bf16 original vs this Q4_K_M file, same inputs). By matrix: gate 0.0003, up 0.0003, down 0.0001. Weight-space error: gate 0.0057, up 0.0058, down 0.0004.
top 6.25 / 12.5 / 25 / 50 % of neurons carry: contribution mass 37 % / 51 % / 70 % / 89 %; energy 82.8 % / 89.9 % / 95.7 % / 99.3 %. Share of neurons with |gelu(g)| < 0.02: 6.4 %.
W_up spectrum: the top 128 / 512 of 2560 singular directions hold 14.0 / 41.1 % of its energy

| b | oracle | +mean | +lin256 | +lin | gate | gate-cal | gate+u128 | gate+u512 | pc-full | pc-r256 | pc-real128 | pc-real512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | 0.0751 | | | | | | | | |
| 6.25 % | 0.0992 | 0.0277 | 0.0136 | 0.0120 | 0.0171 | 0.0158 | 0.0244 | 0.0177 | 0.0195 | 0.0223 | 0.0384 | 0.0297 |
| 12.50 % | 0.0453 | 0.0147 | 0.0083 | 0.0077 | 0.0105 | 0.0108 | 0.0187 | 0.0129 | 0.0122 | 0.0137 | 0.0287 | 0.0210 |
| 25.00 % | 0.0117 | 0.0056 | 0.0038 | 0.0037 | 0.0067 | 0.0064 | 0.0129 | 0.0083 | 0.0056 | 0.0061 | 0.0193 | 0.0129 |
| 50.00 % | 0.0011 | 0.0009 | 0.0007 | 0.0007 | 0.0037 | 0.0030 | 0.0067 | 0.0039 | 0.0012 | 0.0012 | 0.0100 | 0.0061 |

## layer 13

instrument check (numpy vs engine FFN output): 1.86e-04
**Q4 noise floor: 0.0009** (bf16 original vs this Q4_K_M file, same inputs). By matrix: gate 0.0004, up 0.0004, down 0.0001. Weight-space error: gate 0.0053, up 0.0053, down 0.0004.
top 6.25 / 12.5 / 25 / 50 % of neurons carry: contribution mass 36 % / 49 % / 67 % / 88 %; energy 86.1 % / 91.1 % / 95.9 % / 99.2 %. Share of neurons with |gelu(g)| < 0.02: 5.7 %.
W_up spectrum: the top 128 / 512 of 2560 singular directions hold 13.1 / 39.8 % of its energy

| b | oracle | +mean | +lin256 | +lin | gate | gate-cal | gate+u128 | gate+u512 | pc-full | pc-r256 | pc-real128 | pc-real512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | | | | 0.0999 | | | | | | | | |
| 6.25 % | 0.0620 | 0.0217 | 0.0126 | 0.0098 | 0.0104 | 0.0108 | 0.0237 | 0.0169 | 0.0228 | 0.0200 | 0.0342 | 0.0276 |
| 12.50 % | 0.0334 | 0.0136 | 0.0087 | 0.0074 | 0.0083 | 0.0080 | 0.0190 | 0.0136 | 0.0143 | 0.0126 | 0.0267 | 0.0206 |
| 25.00 % | 0.0113 | 0.0060 | 0.0044 | 0.0042 | 0.0069 | 0.0064 | 0.0141 | 0.0099 | 0.0066 | 0.0059 | 0.0190 | 0.0137 |
| 50.00 % | 0.0014 | 0.0011 | 0.0009 | 0.0009 | 0.0040 | 0.0037 | 0.0079 | 0.0053 | 0.0014 | 0.0013 | 0.0106 | 0.0069 |

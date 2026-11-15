# E7H -- LM head: cheap sketch of every row + exact re-score of a shortlist

head 262144 x 2560 (Q4_K, dequantised = the truth). Positions: mmlu_last 160, mmlu_q 1000, prose_en 2400, prose_ru 2400
instrument check, exact top-1 = the actual next token: mmlu_q 58.7 %, prose_en 53.5 %, prose_ru 58.1 %
weight-space error of the sketches: 1 bit 0.368, 2 bits 0.123. Exact head: 1.84 s per 192 positions on this runner.

## sketch s1

| positions | rank of exact top-1: median / 99.9 % / max | rank of the worst of the exact top-64: 99 % / max | N=1024: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=4096: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=8192: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=16384: top-1 in, top-64 all in, missed top-64 mass (mean / max) |
|---|---|---|---|---|---|---|
| mmlu_last | 0 / 1 / 1 | 24496 / 31960 | 100.00 %, 1.9 %, 5.3e-08 / 9.4e-07 | 100.00 %, 48.1 %, 1.8e-09 / 6.5e-08 | 100.00 %, 80.0 %, 1.5e-10 / 1.3e-08 | 100.00 %, 95.6 %, 1.0e-12 / 1.2e-10 |
| mmlu_q | 0 / 158 / 706 | 31800 / 83703 | 100.00 %, 19.7 %, 1.6e-03 / 1.6e-01 | 100.00 %, 57.1 %, 1.0e-04 / 1.4e-02 | 100.00 %, 79.5 %, 2.5e-05 / 7.4e-03 | 100.00 %, 93.6 %, 3.6e-06 / 1.5e-03 |
| prose_en | 0 / 40 / 70 | 26810 / 85410 | 100.00 %, 45.8 %, 1.2e-04 / 4.9e-02 | 100.00 %, 82.7 %, 6.7e-06 / 4.5e-03 | 100.00 %, 91.5 %, 3.2e-06 / 4.3e-03 | 100.00 %, 97.4 %, 5.5e-07 / 7.2e-04 |
| prose_ru | 0 / 557 / 1035 | 38019 / 70211 | 99.96 %, 3.9 %, 3.0e-03 / 2.3e-01 | 100.00 %, 32.8 %, 2.8e-04 / 2.8e-02 | 100.00 %, 65.9 %, 4.5e-05 / 9.1e-03 | 100.00 %, 89.0 %, 5.0e-06 / 3.9e-03 |

## sketch s1q8

| positions | rank of exact top-1: median / 99.9 % / max | rank of the worst of the exact top-64: 99 % / max | N=1024: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=4096: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=8192: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=16384: top-1 in, top-64 all in, missed top-64 mass (mean / max) |
|---|---|---|---|---|---|---|
| mmlu_last | 0 / 1 / 1 | 24591 / 32752 | 100.00 %, 2.5 %, 5.2e-08 / 9.4e-07 | 100.00 %, 47.5 %, 1.8e-09 / 6.5e-08 | 100.00 %, 80.0 %, 1.5e-10 / 1.3e-08 | 100.00 %, 96.2 %, 9.3e-13 / 1.2e-10 |
| mmlu_q | 0 / 166 / 679 | 31928 / 82434 | 100.00 %, 19.3 %, 1.6e-03 / 1.6e-01 | 100.00 %, 57.2 %, 1.0e-04 / 1.4e-02 | 100.00 %, 79.9 %, 2.3e-05 / 7.4e-03 | 100.00 %, 93.4 %, 4.1e-06 / 1.5e-03 |
| prose_en | 0 / 41 / 73 | 27361 / 86236 | 100.00 %, 45.3 %, 1.2e-04 / 4.9e-02 | 100.00 %, 82.6 %, 6.7e-06 / 4.5e-03 | 100.00 %, 91.5 %, 2.7e-06 / 4.3e-03 | 100.00 %, 97.4 %, 5.5e-07 / 7.2e-04 |
| prose_ru | 0 / 546 / 1027 | 38713 / 83591 | 99.96 %, 3.8 %, 3.0e-03 / 2.3e-01 | 100.00 %, 33.3 %, 2.7e-04 / 2.8e-02 | 100.00 %, 66.1 %, 4.5e-05 / 9.1e-03 | 100.00 %, 89.2 %, 4.1e-06 / 3.9e-03 |

## sketch s1q4

| positions | rank of exact top-1: median / 99.9 % / max | rank of the worst of the exact top-64: 99 % / max | N=1024: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=4096: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=8192: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=16384: top-1 in, top-64 all in, missed top-64 mass (mean / max) |
|---|---|---|---|---|---|---|
| mmlu_last | 0 / 1 / 1 | 25818 / 35631 | 100.00 %, 1.9 %, 8.7e-08 / 3.2e-06 | 100.00 %, 44.4 %, 2.6e-09 / 6.5e-08 | 100.00 %, 70.6 %, 4.0e-10 / 1.3e-08 | 100.00 %, 92.5 %, 1.8e-11 / 2.2e-09 |
| mmlu_q | 0 / 307 / 1456 | 123379 / 246386 | 99.90 %, 15.3 %, 3.6e-03 / 3.0e-01 | 100.00 %, 39.1 %, 1.4e-03 / 1.6e-01 | 100.00 %, 54.7 %, 5.0e-04 / 6.6e-02 | 100.00 %, 68.9 %, 1.3e-04 / 2.1e-02 |
| prose_en | 0 / 37 / 103 | 29432 / 101673 | 100.00 %, 41.9 %, 1.9e-04 / 1.0e-01 | 100.00 %, 79.9 %, 8.7e-06 / 4.3e-03 | 100.00 %, 89.2 %, 3.0e-06 / 4.3e-03 | 100.00 %, 96.3 %, 5.0e-07 / 5.7e-04 |
| prose_ru | 0 / 2511 / 3956 | 146212 / 249115 | 99.75 %, 1.7 %, 8.9e-03 / 9.4e-01 | 100.00 %, 8.1 %, 2.4e-03 / 1.4e-01 | 100.00 %, 18.0 %, 1.1e-03 / 1.4e-01 | 100.00 %, 35.9 %, 5.1e-04 / 8.4e-02 |

## sketch s2

| positions | rank of exact top-1: median / 99.9 % / max | rank of the worst of the exact top-64: 99 % / max | N=1024: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=4096: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=8192: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=16384: top-1 in, top-64 all in, missed top-64 mass (mean / max) |
|---|---|---|---|---|---|---|
| mmlu_last | 0 / 1 / 1 | 1353 / 1778 | 100.00 %, 98.1 %, 2.9e-13 / 3.6e-11 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 |
| mmlu_q | 0 / 24 / 39 | 4488 / 9948 | 100.00 %, 76.4 %, 6.8e-05 / 9.5e-03 | 100.00 %, 98.3 %, 3.6e-07 / 2.0e-04 | 100.00 %, 99.8 %, 5.1e-10 / 4.7e-07 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 |
| prose_en | 0 / 8 / 12 | 1818 / 19535 | 100.00 %, 96.7 %, 3.3e-07 / 2.9e-04 | 100.00 %, 99.8 %, 1.0e-08 / 1.8e-05 | 100.00 %, 100.0 %, 5.0e-10 / 1.2e-06 | 100.00 %, 100.0 %, 5.0e-10 / 1.2e-06 |
| prose_ru | 0 / 83 / 107 | 5776 / 14576 | 100.00 %, 53.3 %, 2.9e-04 / 2.6e-02 | 100.00 %, 96.6 %, 2.9e-06 / 2.0e-03 | 100.00 %, 99.8 %, 5.4e-09 / 1.2e-05 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 |

done in 2484 s

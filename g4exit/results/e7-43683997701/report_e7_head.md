# E7H -- LM head: cheap sketch of every row + exact re-score of a shortlist

head 262144 x 2560 (Q4_K, dequantised = the truth). Positions: mmlu_last 160, mmlu_q 1000, prose_en 2400, prose_ru 2400
instrument check, exact top-1 = the actual next token: mmlu_q 57.2 %, prose_en 51.6 %, prose_ru 56.4 %
weight-space error of the sketches: 1 bit 0.368, 2 bits 0.123. Exact head: 0.86 s per 192 positions on this runner.

## sketch s1

| positions | rank of exact top-1: median / 99.9 % / max | rank of the worst of the exact top-64: 99 % / max | N=1024: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=4096: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=8192: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=16384: top-1 in, top-64 all in, missed top-64 mass (mean / max) |
|---|---|---|---|---|---|---|
| mmlu_last | 0 / 1 / 1 | 24496 / 31959 | 100.00 %, 1.9 %, 5.3e-08 / 9.4e-07 | 100.00 %, 48.1 %, 1.8e-09 / 6.5e-08 | 100.00 %, 80.0 %, 1.5e-10 / 1.3e-08 | 100.00 %, 95.6 %, 1.0e-12 / 1.2e-10 |
| mmlu_q | 0 / 41 / 70 | 35847 / 73936 | 100.00 %, 16.7 %, 1.3e-03 / 5.7e-02 | 100.00 %, 55.8 %, 1.2e-04 / 2.1e-02 | 100.00 %, 78.7 %, 3.9e-05 / 7.5e-03 | 100.00 %, 92.7 %, 1.8e-06 / 9.4e-04 |
| prose_en | 0 / 150 / 1127 | 24534 / 71240 | 99.96 %, 46.0 %, 5.9e-04 / 9.8e-01 | 100.00 %, 82.5 %, 1.8e-05 / 1.2e-02 | 100.00 %, 91.9 %, 2.4e-06 / 4.3e-03 | 100.00 %, 97.2 %, 3.3e-07 / 7.2e-04 |
| prose_ru | 0 / 297 / 653 | 44123 / 89152 | 100.00 %, 5.0 %, 3.6e-03 / 2.0e-01 | 100.00 %, 31.5 %, 3.6e-04 / 4.4e-02 | 100.00 %, 65.2 %, 4.6e-05 / 8.6e-03 | 100.00 %, 87.9 %, 9.4e-06 / 8.6e-03 |

## sketch s1q4

| positions | rank of exact top-1: median / 99.9 % / max | rank of the worst of the exact top-64: 99 % / max | N=1024: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=4096: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=8192: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=16384: top-1 in, top-64 all in, missed top-64 mass (mean / max) |
|---|---|---|---|---|---|---|
| mmlu_last | 0 / 1 / 1 | 25818 / 35631 | 100.00 %, 1.9 %, 8.7e-08 / 3.2e-06 | 100.00 %, 44.4 %, 2.6e-09 / 6.5e-08 | 100.00 %, 70.6 %, 4.0e-10 / 1.3e-08 | 100.00 %, 92.5 %, 1.8e-11 / 2.2e-09 |
| mmlu_q | 0 / 56 / 58 | 128107 / 186766 | 100.00 %, 13.6 %, 2.7e-03 / 1.1e-01 | 100.00 %, 38.0 %, 8.2e-04 / 5.7e-02 | 100.00 %, 51.8 %, 3.8e-04 / 5.7e-02 | 100.00 %, 67.4 %, 1.2e-04 / 2.1e-02 |
| prose_en | 0 / 94 / 1242 | 28870 / 75031 | 99.96 %, 42.2 %, 6.9e-04 / 9.8e-01 | 100.00 %, 79.2 %, 2.1e-05 / 1.2e-02 | 100.00 %, 90.0 %, 4.6e-06 / 5.3e-03 | 100.00 %, 96.5 %, 2.8e-07 / 5.7e-04 |
| prose_ru | 0 / 894 / 2189 | 150084 / 240385 | 99.92 %, 2.1 %, 9.6e-03 / 4.2e-01 | 100.00 %, 9.0 %, 3.0e-03 / 2.1e-01 | 100.00 %, 17.7 %, 1.5e-03 / 1.7e-01 | 100.00 %, 35.0 %, 5.9e-04 / 7.6e-02 |

## sketch s2

| positions | rank of exact top-1: median / 99.9 % / max | rank of the worst of the exact top-64: 99 % / max | N=1024: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=4096: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=8192: top-1 in, top-64 all in, missed top-64 mass (mean / max) | N=16384: top-1 in, top-64 all in, missed top-64 mass (mean / max) |
|---|---|---|---|---|---|---|
| mmlu_last | 0 / 1 / 1 | 1353 / 1778 | 100.00 %, 98.1 %, 2.9e-13 / 3.6e-11 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 |
| mmlu_q | 0 / 10 / 11 | 4369 / 12478 | 100.00 %, 76.8 %, 1.1e-04 / 1.2e-02 | 100.00 %, 98.1 %, 1.2e-06 / 8.7e-04 | 100.00 %, 99.8 %, 6.4e-10 / 6.4e-07 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 |
| prose_en | 0 / 7 / 10 | 1545 / 6392 | 100.00 %, 97.0 %, 5.8e-07 / 3.9e-04 | 100.00 %, 99.9 %, 1.5e-09 / 3.3e-06 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 |
| prose_ru | 0 / 34 / 99 | 5788 / 9362 | 100.00 %, 54.0 %, 3.5e-04 / 3.1e-02 | 100.00 %, 96.0 %, 1.8e-06 / 1.6e-03 | 100.00 %, 99.8 %, 1.5e-08 / 3.2e-05 | 100.00 %, 100.0 %, 0.0e+00 / 0.0e+00 |

done in 1417 s

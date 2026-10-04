**(2a / 2b) cam0 s1 policy fine-tuned on N look 2 demos** (success_once at 0, 1k, 2k, 3k, 4k (100 ep.) | 5k (250 ep.))

| arm | N | curve | final | encoder NC1 / eff. rank | residual: identity / affine paired / pca16 (N demos) |
|---|---|---|---|---|---|
| end-to-end | 5 | 0.02 0.03 0.05 0.05 0.01 | **0.040** | 11.11 / 14.6 | 1.88 / 0.67 / 0.87 |
| end-to-end | 10 | 0.03 0.02 0.03 0.07 0.04 | **0.048** | 10.07 / 13.9 | 1.67 / 0.65 / 0.85 |
| end-to-end | 25 | 0.02 0.03 0.05 0.05 0.04 | **0.060** | 11.11 / 14.3 | 1.69 / 0.66 / 0.83 |
| encoder only | 5 | 0.02 0.11 0.09 0.03 0.10 | **0.084** | 3.30 / 5.0 | 1.40 / 0.55 / 0.79 |
| encoder only | 10 | 0.05 0.08 0.10 0.05 0.08 | **0.108** | 2.80 / 4.4 | 1.65 / 0.48 / 0.73 |
| encoder only | 25 | 0.02 0.05 0.25 0.18 0.15 | **0.180** | 2.17 / 4.5 | 1.74 / 0.42 / 0.60 |

**(2c) goal shift: look 2 default encoder s2 → cam0 goal controller s2, goal variant in look 2** (reference: goal look 2 oracle .685)

| arm | N | curve 0–4k (100 ep.) | 5k (250 ep.) |
|---|---|---|---|
| (i) stitched, map only | 10 | 0.36 0.31 0.47 0.43 0.45 | **0.452** |
| (ii) stitched, all | 10 | 0.29 0.21 0.22 0.17 0.27 | **0.184** |
| (iii) identity map, all | 10 | 0.13 0.20 0.29 0.12 0.19 | **0.172** |
| (iv) from scratch, 5k | 10 | 0.00 0.00 0.01 0.00 0.00 | **0.004** |
| (iv) from scratch, 50k (every 5k, 250 ep.) | 10 | 0.00 0.00 0.00 0.00 0.00 0.01 0.00 0.02 0.01 0.00 0.01 | final 0.012 (last-3 0.009) |
| (i) stitched, map only | 25 | 0.45 0.43 0.64 0.65 0.50 | **0.528** |
| (ii) stitched, all | 25 | 0.29 0.32 0.44 0.41 0.45 | **0.384** |
| (iii) identity map, all | 25 | 0.05 0.43 0.43 0.48 0.29 | **0.380** |
| (iv) from scratch, 5k | 25 | 0.00 0.01 0.01 0.03 0.01 | **0.012** |
| (iv) from scratch, 50k (every 5k, 250 ep.) | 25 | 0.00 0.02 0.06 0.07 0.08 0.10 0.06 0.08 0.08 0.10 0.08 | final 0.076 (last-3 0.085) |

# Q1 sparse-path benchmark

Host: macOS-15.7.9-x86_64-i386-64bit, x86_64. Python/NumPy/SciPy and BLAS thread configuration are recorded in sparse-benchmark.json.

All 18 cases completed and passed the paired objective check. Timings include initialization and KKT certification. Compare the same final absolute lambda: the event solver computes the prefix from lambda_max, while DR solves the final point. DR reuses the R1 iteration with the Q1 mean-shift adapter and matrix-free resolvents. DR KKT target is 1e-8; event endpoints reach roughly machine precision.

For n=80,p=500 and n=100,p=2000, values are medians of three repetitions. The n=100,p=5000 stress cases have one run each; those ratios are exploratory single-run measurements. Import/startup time is excluded. The source requests one BLAS thread.

| p | tau | regime | path s | DR s | DR/path | event steps | support | repeats |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 500 | 0.2 | large-lambda | 0.0055 | 0.2323 | 42.44 | 4 | 2 | 3 |
| 500 | 0.2 | support-8 | 0.0476 | 0.1358 | 2.85 | 33 | 8 | 3 |
| 500 | 0.5 | large-lambda | 0.0181 | 0.1610 | 8.90 | 11 | 2 | 3 |
| 500 | 0.5 | support-8 | 0.0950 | 0.0988 | 1.04 | 61 | 8 | 3 |
| 500 | 0.8 | large-lambda | 0.0164 | 0.2307 | 14.04 | 11 | 2 | 3 |
| 500 | 0.8 | support-8 | 0.0578 | 0.1635 | 2.83 | 36 | 8 | 3 |
| 2000 | 0.2 | large-lambda | 0.1046 | 2.9213 | 27.92 | 19 | 3 | 3 |
| 2000 | 0.2 | support-8 | 0.1944 | 0.6365 | 3.27 | 52 | 8 | 3 |
| 2000 | 0.5 | large-lambda | 0.0567 | 0.8933 | 15.76 | 21 | 2 | 3 |
| 2000 | 0.5 | support-8 | 0.1688 | 0.4425 | 2.62 | 59 | 8 | 3 |
| 2000 | 0.8 | large-lambda | 0.0323 | 1.5914 | 49.20 | 9 | 2 | 3 |
| 2000 | 0.8 | support-8 | 0.1065 | 0.7327 | 6.88 | 36 | 8 | 3 |
| 5000 | 0.2 | large-lambda | 0.1188 | 9.3794 | 78.93 | 10 | 2 | 1 |
| 5000 | 0.2 | support-8 | 0.4854 | 4.9093 | 10.11 | 48 | 8 | 1 |
| 5000 | 0.5 | large-lambda | 0.1977 | 9.4050 | 47.57 | 16 | 2 | 1 |
| 5000 | 0.5 | support-8 | 0.6497 | 2.7373 | 4.21 | 57 | 8 | 1 |
| 5000 | 0.8 | large-lambda | 0.2254 | 18.9650 | 84.14 | 14 | 2 | 1 |
| 5000 | 0.8 | support-8 | 0.6696 | 8.4893 | 12.68 | 46 | 8 | 1 |

Event steps include the requested endpoint, which can truncate an affine segment. Support budgets exclude the intercept. No flat-face pivots were required by these sparse-prefix cases.

large-lambda: DR/path range 8.90–84.14; event steps 4–21.
support-8: DR/path range 1.04–12.68; event steps 33–61.

Maximum paired objective difference: 1.506e-08. Maximum event KKT: 3.941e-16; maximum DR KKT: 9.999e-09.

These observations support fast large-lambda prefixes. They do not imply a uniform large speedup for all support-limited fits: the smallest measured ratio is close to parity, and even eight slopes can require dozens of residual-threshold events.

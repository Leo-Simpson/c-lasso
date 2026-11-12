# Q1: matched speed comparison of all four backends

Path-following and DR are substantially faster than the two PDS schemes in these tested settings. P-PDS is consistently faster than PF-PDS. DR can match or slightly beat path-following at some modest sparse endpoints; path-following leads clearly at large lambda.

| n | p | tau | workload | Path s | DR s | P-PDS s | PF-PDS s | repeats |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 40 | 9 | 0.2 | 14-point-path | 0.0379 | 0.1244 | 1.4868 | 1.9771 | 3 |
| 40 | 9 | 0.5 | 14-point-path | 0.0456 | 0.1539 | 0.7379 | 1.0374 | 3 |
| 40 | 9 | 0.8 | 14-point-path | 0.0357 | 0.1038 | 0.9605 | 1.2883 | 3 |
| 80 | 500 | 0.2 | large-lambda | 0.0058 | 0.2035 | 0.8254 | 1.2033 | 3 |
| 80 | 500 | 0.2 | support-8 | 0.0378 | 0.1027 | 0.4191 | 0.6302 | 3 |
| 80 | 500 | 0.5 | large-lambda | 0.0133 | 0.1253 | 0.4710 | 0.7048 | 3 |
| 80 | 500 | 0.5 | support-8 | 0.0769 | 0.0754 | 0.2765 | 0.4078 | 3 |
| 80 | 500 | 0.8 | large-lambda | 0.0140 | 0.1654 | 0.6776 | 0.9827 | 3 |
| 80 | 500 | 0.8 | support-8 | 0.0413 | 0.1050 | 0.4285 | 0.6532 | 3 |
| 100 | 2000 | 0.5 | large-lambda | 0.0755 | 0.8093 | 4.0338 | 6.5377 | 1 |
| 100 | 2000 | 0.5 | support-8 | 0.2401 | 0.4282 | 1.8376 | 2.8372 | 1 |

The 40x9 cases time a complete 14-lambda path (.5 down to .01), including warm starts. The larger cases time a cold solve at one fixed lambda: either .9 of the constrained lambda_max, or the lambda of an eight-slope event-path solution. Target selection is untimed and common to all methods. The event solver still traces from its own lambda_max in each timed call. The eight-slope label describes the reference solution, not an automatic support search by the iterative schemes.

Times include solver initialization, matrix/operator setup, factorization or spectral calculation, and KKT certification. Python imports are excluded through untimed warmup. Backend order rotates across repetitions. The p=9 and p=500 cases show medians of three repetitions; p=2000 values are single stress runs. There are 116 timed solver calls over 11 matched cases.

All methods use requested KKT tolerance 1e-8. Event residuals can be much smaller than that tolerance. Every timed call checks its lambdas, KKT residual, and objective against the event reference. This timing experiment does not replace the separate independent Clarabel validation.

P-PDS: 3.67–11.95 times the DR runtime across these cases.
PF-PDS: 5.41–15.89 times the DR runtime across these cases.

PF-PDS takes 1.33–1.62 times the P-PDS runtime. Its inherited R1 iteration evaluates the augmented quadratic gradient twice, versus once for P-PDS. Iteration counts and per-run timings are in method-speed.json.

Maximum objective difference between any timed fit and the event reference: 1.083e-08. Maximum KKT: 9.999e-09.

## Thread control and earlier measurements

This run explicitly sets OMP_NUM_THREADS and OPENBLAS_NUM_THREADS to 1 and applies threadpoolctl limits. Runtime inspection verifies one thread for both the OpenMP OpenBLAS pool and the OpenMP runtime; actual pool metadata is saved in method-speed.json. The benchmark environment is macOS-15.7.9-x86_64-i386-64bit, x86_64.

The older sparse benchmark and an initial four-method run requested one thread only through OPENBLAS_NUM_THREADS. This build uses OpenMP, so that request did not reliably control its thread pool. Those historical timings are preserved, with the initial four-method run in method-speed-requested-only.*. Use the explicitly controlled table above for the present comparison. The older benchmark script is also corrected for future runs; its historical results were not overwritten.

## Reproduction

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark/benchmark_Q1_methods.py

Run from the Q1 checkout with its source first on PYTHONPATH. NumPy, SciPy, and threadpoolctl are required for this benchmark; production solvers do not depend on threadpoolctl. The original 352-test regression suite and independent PDS validation remain unchanged by this benchmark-only change.

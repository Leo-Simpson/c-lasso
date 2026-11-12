# Q1 PDS completion

Both P-PDS and PF-PDS reuse Classo_R1 through the Q1 mean-shift adapter.

| Method | Independent paths | Requested fits | Max absolute objective gap | Max KKT |
|---|---:|---:|---:|---:|
| P-PDS | 24/24 | 336 | 5.045e-11 | 9.9995e-9 |
| PF-PDS | 24/24 | 336 | 5.287e-11 | 9.99993e-9 |

Forty-six paths completed at the initial 100000 per-point limit. Both methods
hit that limit on seed 7, tau=.5. They passed independent retries at the
standard R1 budget of 500000 per point, without relaxing the KKT tolerance.
Total iterations across that 14-point path were 273380 and 273450.
The new default is 500000; the public diagnostic API accepts pds_max_iter.

pds-validation.json combines the 46 completed original paths and the two
independently checked retries, with an iteration_budget_provenance field.
The original failure records and retry results are retained separately.
These are accuracy validations, not isolated performance benchmarks.

Reproduce all cases from the checkout with the parent development environment:

    python benchmark/validate_Q1.py --methods P-PDS PF-PDS --output pds-validation.json

Both methods are available through Classo/pathlasso with typ='Q1' and their
usual meth names. n_active stopping remains specific to Path-Alg.

Full suite: 352 passed, including 42 Q1 tests. See pds-full-tests.txt.

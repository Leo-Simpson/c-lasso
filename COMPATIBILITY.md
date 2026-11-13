# Compatibility and numerical testing

The maintenance line retains Python **3.8–3.15** compatibility. Python 3.8–3.10
remain in the matrix for existing scientific environments; current installations
should use a currently maintained Python release. Raising the Python floor is a
separate release-policy decision, not an incidental consequence of a solver fix.
The matrix uses the latest patch available for each minor, except the deliberately
frozen Python 3.8.19 reproduction job.

The tested dependency floors are:

| Dependency | Minimum |
| --- | --- |
| NumPy | 1.24.4 |
| SciPy | 1.10.1 |
| Matplotlib | 3.7.5 |
| pandas | 2.0.3 |

These bounds live in `setup.py`. Higher versions are permitted; runtime packages
are not capped at whichever release happened to be installed on a developer's
machine. Pip selects versions compatible with the chosen Python, so an older
Python does not necessarily install the globally newest SciPy release.

## Required CI coverage

Every pull request runs the **same complete `classo/tests` suite** in ten
independent environments, with `fail-fast: false` so one failure cannot hide the
remaining results:

| Environment | Dependency selection | Purpose |
| --- | --- | --- |
| Linux / Python 3.8.19 | `ci/constraints/minimum.txt` | Declared floors and the reported student numerical stack |
| Linux / Python 3.9–3.15, one job per minor | Latest compatible stable releases | Compatibility with current dependency resolution |
| macOS and Windows / Python 3.15 | Latest compatible stable releases | Platform and numerical-library coverage |

Python 3.8–3.10 jobs use Ubuntu 22.04, which supplies these legacy interpreter
builds; newer Python jobs use the current Ubuntu runner. The minimum/reproduction
job pins NumPy 1.24.4 and SciPy 1.10.1, with compatible
plotting/dataframe/test packages. It serves two explicit purposes: it tests the
supported floor and preserves the originally reported numerical combination.
This is not an assertion that every transitive dependency matches the students'
original machine. Exact resolved dependencies and BLAS builds are recorded.

Latest-stable jobs upgrade dependencies without `--pre`. Numerical libraries must
have wheels; missing wheels fail the job instead of silently changing to a local
source build. No matrix failure is allowed to pass as an expected failure. The
existing `build` status aggregates all matrix jobs and lint.

The workflow runs on pull requests and pushes targeting `master`, `multi-class`,
`ridge` and `quantile`, and weekly on the default branch to catch dependency
updates without waiting for a code change. Feature branches acquire this policy
when this workflow is integrated into them; changing master does not rewrite
those branch snapshots. The same suite command includes each branch's additional
tests, rather than selecting a smaller compatibility subset.

Release publication also calls the compatibility workflow on the release commit;
PyPI publishing requires both its success and the distribution build/wheel checks.
Pull requests and manual builds never publish.

## Reproducibility and diagnosis

Each test job uploads:

- `environment.json`: Python/package versions, commit, operating system,
  architecture and thread settings;
- `pip-freeze.txt`: the exact resolved environment;
- `blas-lapack.txt`: NumPy/SciPy build configuration;
- JUnit test results and coverage XML.

To record a local environment:

```sh
python ci/report_environment.py --output environment-record
```

A historical environment is useful for reproducing a bug; latest-compatible
jobs are useful for detecting dependency regressions. Neither substitutes for
the other. Nonunique coefficient representations are compared using scores,
original objectives and KKT conditions where appropriate; tolerances are not
loosened simply to make a new dependency version pass.

## Runtime comparisons

Compatibility tests are not cross-machine speed benchmarks. Compare solver
revisions in **one fixed Python/dependency environment**, on the same machine,
with the same data, lambda endpoint, sparsity budget, BLAS build and thread count.

`ci/constraints/benchmark.txt` records the numerical stack used for the November
performance comparison (Python 3.12.13). For example, in a clean environment:

```sh
python -m pip install -c ci/constraints/benchmark.txt -e .
python ci/report_environment.py --output benchmark-environment
```

Keep the full resolved environment with each measurement and serialize timing
runs. CI sets BLAS/OpenMP thread variables to one for numerical reproducibility,
but its heterogeneous runners and shared hardware are not a performance baseline.

## Updating the policy

Add new stable Python minors after compatible dependency wheels are available.
Review floors and classifiers deliberately, update this document and the
constraint files together, and require the full matrix to pass. A frozen
reproduction profile should not drift automatically with latest-stable jobs.

References: [Python release status](https://www.python.org/downloads/),
[SciPy's compatibility roadmap](https://docs.scipy.org/doc/scipy/dev/toolchain.html),
and [NumPy's downstream testing guidance](https://numpy.org/doc/stable/dev/depending_on_numpy.html).

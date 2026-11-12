# Changelog

## 1.0.12 — unreleased

Maintenance release; the existing regression and binary classification interfaces
remain unchanged. Multiclass MC1/MC2 functionality is planned for 1.1.0 and is not
part of this release.

- Declare matplotlib and pandas as runtime dependencies and remove unused h5py (#24).
- Raise the intended ValueError for unknown path formulations (#25).
- Restore a regression test hidden by a duplicate function name (#26).
- Replace removed NumPy infinity aliases for NumPy 2 compatibility (#27).
- Correct C1 intercept margin tracking and add KKT regression checks (#28).
- Continue C1/C2 paths through flat loss regions (#29).
- Correct C1 boundary entry signs and zero-coefficient directions (#30).
- Update repository links and restore documentation builds (#31).
- Modernize release validation and use PyPI Trusted Publishing.

- Correct R3 concomitant path curvature, interpolation, and scale units.
- Correct R4 intercept constraints, symmetric scale initialization, zero-scale
  boundaries, and requested path method dispatch.
- Safeguard concomitant proximal roots and require DR splitting residual convergence.

- Continue tied C1/C2 paths with reduced dual events and certified optimal-face pivots.

- Preserve positive R1/R2 penalties and the Huber objective at zero penalty.
- Use one rank-aware constraint projection across regression solvers; support
  equivalent rescaled/redundant equality rows consistently in all formulations.
- Compute R2 lambda normalization at the Huber intercept optimum and retain
  weak design directions in unpenalized constrained least squares.
- Validate penalty grids and concomitant scale penalties; handle zero-gradient
  absolute paths without division by zero.

"""Convert the bundled Central Park Soil R data to NumPy files, without R."""

from pathlib import Path
import warnings

import numpy as np
import rdata
from scipy.sparse import csc_matrix

DATA_DIR = Path(__file__).resolve().parent

A_path = DATA_DIR / "A.npy"
A_path.unlink(missing_ok=True)

cps_path = DATA_DIR / "cps.npz"
cps_path.unlink(missing_ok=True)

with warnings.catch_warnings():
    # rdata exposes unsupported R classes as their underlying slots.
    warnings.filterwarnings("ignore", message="Missing constructor for R class")
    data = rdata.read_rds(DATA_DIR / "cps_pH_processed.RDS")

y = data["y"]
x = data["x"].values

# R's dgCMatrix slots use the same compressed-column layout as SciPy.
rA = data["A"]
A = csc_matrix((rA.x, rA.i, rA.p), shape=tuple(rA.Dim)).toarray()
label_nodes = np.asarray(rA.Dimnames[1], dtype=str)


np.save(A_path, A)
print(f"Saved A.npy,  {A.shape=}")


np.savez(cps_path, y=y, x=x, label=label_nodes)
print(f"Saved cps.npz: y {y.shape}, x {x.shape}, A {A.shape}")

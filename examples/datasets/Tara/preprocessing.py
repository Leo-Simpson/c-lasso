"""Convert the bundled Tara R files to A.npy and tara.npz (requires rdata)."""

from pathlib import Path
import warnings

import numpy as np
import rdata
from scipy.sparse import csc_matrix

DATA_DIR = Path(__file__).resolve().parent

A_path = DATA_DIR / "A.npy"
A_path.unlink(missing_ok=True)

tara_path = DATA_DIR / "tara.npz"
tara_path.unlink(missing_ok=True)

with warnings.catch_warnings():
    # rdata exposes unsupported R classes as their underlying slots.
    warnings.filterwarnings("ignore", message="Missing constructor for R class")
    data = rdata.read_rds(DATA_DIR / "tara_sal_processed.RDS")
    workspace = rdata.read_rda(DATA_DIR / "tara_sal_trac.Rdata")

y = data["y"]
x = data["x"].values

# A is an R dgCMatrix: its slots use the same CSC layout as SciPy.
rA = data["A"]
A = csc_matrix((rA.x, rA.i, rA.p), shape=tuple(rA.Dim)).toarray()
label_nodes = np.asarray(rA.Dimnames[1], dtype=str)

# Convert R's one-based training indices to Python's zero-based indices.
tr = workspace["tr"] - 1


np.save(A_path, A)
print(f"Saved A.npy,  {A.shape=}")
np.savez(tara_path, tr=tr, y=y, x=x, label=label_nodes)
print(f"Saved tara.npz: {tr.shape=}, {y.shape=}, {x.shape=}, {label_nodes.shape=}")

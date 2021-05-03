import numpy as np
import matplotlib.pyplot as plt

import os

my_path = os.path.dirname(__file__)

output = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


data = np.load(os.path.join(my_path, "bm-R1.npz"))

ticks = [str(tuple(size)) for size in data["SIZES"]]

algo = ["Path-Alg", "P-PDS", "DR", "CVX"]

fig = plt.figure()

for i, name in enumerate(["T_pa", "T_pds", "T_dr", "T_cvx"]):
    T = data[name]
    N_size, N_data = T.shape
    mean = np.mean(T, axis=1)
    std = np.std(T, axis=1) / np.sqrt(N_data)
    label = algo[i]
    plt.errorbar(range(N_size), mean, yerr=std, label=label)

plt.yscale("log")
plt.title("Running time (in seconds)")
plt.xticks(range(N_size), ticks)
plt.xlabel("(n, d)")
plt.ylabel("running time (in seconds)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output, "bm-R1-times.png"))
plt.close()

ref = np.min(
    [np.mean(data[name], axis=1) for name in ["L_pa", "L_pds", "L_dr", "L_cvx"]], axis=0
)
for i, name in enumerate(["L_pa", "L_pds", "L_dr", "L_cvx"]):
    L = data[name]
    N_size, N_data = L.shape
    mean = np.mean(L, axis=1)
    label = algo[i]
    plt.plot(range(N_size), mean - ref, label=label)

plt.title("Loss - minimum(loss)")
plt.xticks(range(N_size), ticks)
plt.xlabel("(n, d)")
plt.ylabel("loss function")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output, "bm-R1-losses.png"))
plt.close()

for i, name in enumerate(["C_pa", "C_pds", "C_dr", "C_cvx"]):
    C = data[name]
    N_size, N_data = C.shape
    mean = np.mean(C, axis=1)
    std = np.std(C, axis=1) / np.sqrt(N_data)
    label = algo[i]
    plt.plot(range(N_size), mean, label=label)

plt.title("Value of norm(Cbeta)")
plt.xticks(range(N_size), ticks)
plt.xlabel("(n, d)")
plt.ylabel("norm(Cbeta)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output, "bm-R1-constraint.png"))
plt.close()

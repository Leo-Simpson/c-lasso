import numpy as np
import matplotlib.pyplot as plt

import os
my_path = os.path.dirname(__file__)

output = os.path.join(my_path, "output/")


data = np.load(os.path.join(my_path, 'bm-C1.npz'))

labels = [str(size) for size in data["SIZES"]]


fig = plt.figure()

for name in ["T_pa", "T_cvx"]:
    T = data[name]
    N_size, N_data = T.shape
    mean = np.mean(T, axis=1)
    std = np.std(T, axis=1)/np.sqrt(N_data)
    label = name[2:]
    plt.errorbar(range(N_size), mean, yerr=std, label=label)

plt.title("Running time")
plt.xticks(range(N_size), labels)
plt.legend()
plt.savefig(os.path.join(output, "bm-C1-times.png"))
plt.show()


for name in ["L_pa", "L_cvx"]:
    L = data[name] - data["L_pa"]
    N_size, N_data = L.shape
    mean = np.mean(L, axis=1)
    std = np.std(L, axis=1)/np.sqrt(N_data)
    label = name[2:]
    plt.errorbar(range(N_size), mean, yerr=std, label=label)

plt.title("Value of loss function")
plt.xticks(range(N_size), labels)
plt.legend()
plt.savefig(os.path.join(output, "bm-C1-losses.png"))
plt.show()

for name in ["C_pa", "C_cvx"]:
    L = data[name]
    N_size, N_data = L.shape
    mean = np.mean(L, axis=1)
    std = np.std(L, axis=1)/np.sqrt(N_data)
    label = name[2:]
    plt.errorbar(range(N_size), mean, yerr=std, label=label)

plt.title("Value of norm(Cbeta)")
plt.xticks(range(N_size), labels)
plt.legend()
plt.savefig(os.path.join(output, "bm-C1-constraint.png"))
plt.show()
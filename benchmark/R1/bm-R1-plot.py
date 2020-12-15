import numpy as np
import matplotlib.pyplot as plt

data = np.load('bm-R1.npz')

print(data["SIZES"])
labels = [str(size) for size in data["SIZES"]]
print(labels)


fig = plt.figure()

for name in ["T_pa", "T_pds", "T_dr", "T_cvx"]:
    T = data[name]
    N_size, N_data = T.shape
    mean = np.mean(T, axis=1)
    std = np.std(T, axis=1)/np.sqrt(N_data)
    label = name[2:]
    plt.errorbar(range(N_size), mean, yerr=std, label=label)

plt.xticks(range(N_size), labels)
plt.legend()
plt.savefig("bm-R1-times.png")
plt.show()


for name in ["L_pa", "L_pds", "L_dr", "L_cvx"]:
    L = data[name] - data["L_pa"]
    N_size, N_data = L.shape
    mean = np.mean(L, axis=1)
    std = np.std(L, axis=1)/np.sqrt(N_data)
    label = name[2:]
    plt.errorbar(range(N_size), mean, yerr=std, label=label)

plt.xticks(range(N_size), labels)
plt.legend()
plt.savefig("bm-R1-losses.png")
plt.show()

for name in ["C_pa", "C_pds", "C_dr", "Cs_cvx"]:
    L = data[name] - data["L_pa"]
    N_size, N_data = L.shape
    mean = np.mean(L, axis=1)
    std = np.std(L, axis=1)/np.sqrt(N_data)
    label = name[2:]
    plt.errorbar(range(N_size), mean, yerr=std, label=label)

plt.title("Value of norm(Cbeta)")
plt.xticks(range(N_size), labels)
plt.legend()
plt.savefig("bm-R1-constraint.png")
plt.show()
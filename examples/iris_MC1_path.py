"""Reproduce the small Iris MC1 path benchmark (requires scikit-learn).

Run after activating the development environment:
python iris_MC1_path.py --output /absolute/path/to/outputs/iris-mc1
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from classo import Classo
from classo.solve_MC1 import problem_MC1
from classo.mc1_path_engine import solve_path_MC1


def reference(pb, lam):
    p, K = pb.init.shape
    N = p*K
    T = np.repeat(np.eye(p), K, axis=0)
    J = np.vstack([np.c_[-np.eye(N), T], np.c_[np.eye(N), T]])

    def objective(z):
        loss, grad = pb.loss_gradient(z[:N].reshape(p, K))
        return ((loss+lam*z[N:].sum())/pb.dim[0],
                np.r_[grad.ravel(), np.full(p, lam)]/pb.dim[0])

    fit = minimize(objective, np.zeros(N+p), jac=True, method="SLSQP",
                   bounds=[(None, None)]*N+[(0, None)]*p,
                   constraints=[dict(type="ineq", fun=lambda z: J.dot(z), jac=lambda z: J)],
                   options=dict(ftol=1e-12, maxiter=2000))
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x[:N].reshape(p, K)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    data = load_iris()
    reports = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for col, K in enumerate((2, 3)):
        chosen = data.target < K
        train, test = train_test_split(np.flatnonzero(chosen), test_size=.2,
                                      stratify=data.target[chosen], random_state=13)
        mean, scale = data.data[train].mean(0), data.data[train].std(0)
        X, Xtest = (data.data[train]-mean)/scale, (data.data[test]-mean)/scale
        y, ytest = data.target[train], data.target[test]
        pb = problem_MC1((X, None, y))
        start = perf_counter()
        BETA, knots = solve_path_MC1(pb, .1)
        elapsed = perf_counter()-start
        coefficients = np.asarray(BETA)
        comparisons = []
        for relative in (.8, .4, .1):
            B = np.array([[np.interp(relative, knots[::-1], coefficients[::-1, j, k])
                           for k in range(K)] for j in range(4)])
            lam = relative*pb.lambdamax
            ref = reference(pb, lam)
            value = lambda b: pb.loss_gradient(b)[0]+lam*np.max(np.abs(b), axis=1).sum()
            item = dict(relative_lambda=relative,
                        reference_objective_difference=float(value(B)-value(ref)),
                        train_accuracy=float(np.mean(np.argmax(X.dot(B), 1) == y)),
                        test_accuracy=float(np.mean(np.argmax(Xtest.dot(B), 1) == ytest)))
            if K == 2:
                beta = Classo((X, np.empty((0, 4)), 2*y-1), lam/2,
                              typ="C1", meth="Path-Alg", true_lam=True)
                item['C1_max_coefficient_difference'] = float(np.max(np.abs(B-np.c_[-beta, beta])))
                binary_y = 2*y-1
                binary_value = lambda b: (np.sum(np.maximum(1-binary_y*X.dot(b), 0)**2)
                                          + (lam/2)*np.abs(b).sum())
                item['C1_unridged_objective_excess'] = float(binary_value(beta)-binary_value(B[:, 1]))
                item['C1_prediction_agreement'] = float(np.mean((X.dot(beta)>0) == (X.dot(B[:, 1])>0)))
                axes[col].plot([relative]*4, beta, "kx", markersize=5)
            comparisons.append(item)
        for j, name in enumerate(data.feature_names):
            curve = coefficients[:, j, 1] if K == 2 else np.max(np.abs(coefficients[:, j]), axis=1)
            axes[col].plot(knots, curve, label=name)
        axes[col].set(xlim=(1., .1), xlabel="lambda / lambda_max",
                      ylabel="Class 1 coefficient" if K == 2 else "Row infinity norm",
                      title="Two species (crosses: C1)" if K == 2 else "Three species (joint MC1)")
        axes[col].grid(alpha=.2)
        axes[col].legend(fontsize=8)
        reports.append(dict(classes=K, n_train=len(train), n_test=len(test),
                            knots=len(knots), seconds=elapsed,
                            max_endpoint_KKT_residual=float(max(pb.path_info['residuals'])),
                            comparisons=comparisons))
    fig.savefig(output/'iris-path.png', dpi=180)
    plt.close(fig)
    (output/'results.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)

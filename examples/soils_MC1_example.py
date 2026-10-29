"""88-soils example: continuous-pH R1 versus ten pH-quantile MC1 classes.

Exploratory feature comparison on all 88 samples; no predictive validation.
Requires pandas, scipy, matplotlib and biom-format in addition to c-lasso.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import biom

from classo.path_alg import solve_path
from classo.solve_MC1 import problem_MC1
from classo.mc1_path_engine import solve_path_MC1


def load_soils(repo):
    folder = repo/'examples/pH_data'
    data = loadmat(folder/'matlab/pHData.mat')
    X, y = data['X'], data['Y'].ravel()
    table = biom.load_table(str(folder/'qiime2/originals/238_otu_table.biom'))
    counts = table.matrix_data.toarray().T
    samples = list(table.ids())
    otu_ids = table.ids(axis='observation')
    metadata = pd.read_csv(folder/'qiime2/originals/88soils_metadata.txt', sep='\t', index_col=0)
    ph = pd.to_numeric(metadata.ph, errors='coerce')
    # The MATLAB table labels are opaque to SciPy. Recover column IDs by
    # matching exact count vectors, then independently verify every full row.
    pairs = []
    for i, value in enumerate(y):
        hits = np.flatnonzero(np.isclose(ph.values, value, atol=1e-8, rtol=0))
        if len(hits) == 1 and metadata.index[hits[0]] in samples:
            pairs.append((i, samples.index(metadata.index[hits[0]])))
    mat_rows, raw_rows = zip(*pairs)
    lookup = {}
    for j in range(counts.shape[1]):
        lookup.setdefault(tuple(counts[list(raw_rows), j]), []).append(j)
    raw = np.exp(X)
    mat_counts = np.where(raw < .75, 0, np.rint(raw))
    indices = []
    for j in range(X.shape[1]):
        hits = lookup.get(tuple(mat_counts[list(mat_rows), j]), [])
        if len(hits) != 1:
            raise ValueError('Ambiguous original OTU mapping for MATLAB feature '+str(j))
        indices.append(hits[0])
    sample_map = []
    for i in range(len(X)):
        hits = np.flatnonzero(np.all(counts[:, indices] == mat_counts[i], axis=1))
        if len(hits) != 1:
            raise ValueError('Ambiguous soil sample mapping')
        sample_map.append(samples[hits[0]])
    np.testing.assert_allclose(ph.loc[sample_map], y, atol=1e-8)
    features = []
    for j, original in enumerate(indices):
        identifier = otu_ids[original]
        taxonomy = table.metadata(identifier, axis='observation')['taxonomy']
        named = [name for name in taxonomy if '__' in name and name.split('__', 1)[1]]
        features.append(dict(feature_index=j, otu_id=identifier, taxonomy=';'.join(taxonomy),
                             display_taxon=named[-1] if named else identifier))
    return X, y, pd.DataFrame(features), sample_map


def intervals(BETA, knots, offset=0):
    rows = []
    for i in range(len(knots)-1):
        B = (BETA[i]+BETA[i+1])/2
        coef = B[offset:]
        strength = np.max(np.abs(coef), axis=1) if coef.ndim == 2 else np.abs(coef)
        selected = np.flatnonzero(strength > 1e-7)
        if len(selected):
            rows.append(dict(B=B, selected=selected, size=len(selected),
                             relative_lambda=(knots[i]+knots[i+1])/2))
    return rows


def pick(rows, budget):
    suitable = [row for row in rows if row['size'] <= budget]
    if not suitable:
        raise ValueError('No nonzero path segment under budget '+str(budget))
    largest = max(row['size'] for row in suitable)
    return next(row for row in suitable if row['size'] == largest)


def run(repo, output, cap=20):
    output.mkdir(parents=True, exist_ok=True)
    X, y, features, sample_ids = load_soils(repo)
    X = X-X.mean(axis=0)
    C, centered_y = np.ones((1, X.shape[1])), y-y.mean()
    edges = np.quantile(y, np.arange(1, 10)/10)
    labels = np.digitize(y, edges)
    bins = pd.DataFrame([dict(class_id=k, n=int(np.sum(labels == k)),
                              min_ph=float(y[labels == k].min()),
                              max_ph=float(y[labels == k].max())) for k in range(10)])
    bins.to_csv(output/'pH_classes.tsv', sep='\t', index=False)
    pd.DataFrame(dict(sample_id=sample_ids, pH=y, class_id=labels)).to_csv(
        output/'samples.tsv', sep='\t', index=False)
    features.to_csv(output/'feature_mapping.tsv', sep='\t', index=False)
    start = perf_counter()
    rb, rl = solve_path((X, C, centered_y), .01, cap+1, 0, 'R1')
    regression_seconds = perf_counter()-start
    pb = problem_MC1((X, C, labels), intercept=True)
    start = perf_counter()
    mb, ml = solve_path_MC1(pb, .01, n_active=cap)
    multiclass_seconds = perf_counter()-start
    print('paths done',len(rb),len(mb),'MC1 stop',pb.path_info['stop_reason'],flush=True)
    regression = intervals(rb, rl)
    multiclass = intervals(mb, ml, offset=1)
    comparison, selections = [], []
    fitted = {}
    for budget in (2, 5, 10, 15, 20):
        if budget > cap:
            continue
        r, m = pick(regression, budget), pick(multiclass, budget)
        R, M = set(r['selected']), set(m['selected'])
        shared, union = R & M, R | M
        row = dict(budget=budget, regression_features=len(R), MC1_features=len(M),
                   shared=len(shared), jaccard=len(shared)/len(union),
                   regression_relative_lambda=r['relative_lambda'],
                   MC1_relative_lambda=m['relative_lambda'])
        comparison.append(row)
        fitted[budget] = (r, m)
        for j in sorted(union):
            f = features.iloc[j].to_dict()
            f.update(budget=budget, selected_R1=j in R, selected_MC1=j in M,
                     regression_coefficient=float(r['B'][j]),
                     MC1_row_norm=float(np.max(np.abs(m['B'][j+1]))))
            selections.append(f)
    pd.DataFrame(comparison).to_csv(output/'support_overlap.tsv', sep='\t', index=False)
    pd.DataFrame(selections).to_csv(output/'selected_features.tsv', sep='\t', index=False)
    np.savez_compressed(output/'paths.npz', regression_B=np.asarray(rb), regression_lambda=rl,
                        MC1_B=np.asarray(mb), MC1_lambda=ml, class_edges=edges)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    report = pd.DataFrame(comparison)
    ax.plot(report.budget, report.regression_features, 'o-', label='R1 selected')
    ax.plot(report.budget, report.MC1_features, 's--', label='MC1 selected')
    ax.plot(report.budget, report.shared, 'o-', label='Shared OTUs')
    ax.set(xlabel='Maximum selected features', ylabel='OTU count', title='88 soils: pH regression vs 10 pH classes')
    ax.set_xticks(report.budget)
    ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'support-overlap.png', dpi=180);plt.close(fig)
    r, m = fitted[10 if cap >= 10 else max(fitted)]
    union = sorted(set(r['selected']) | set(m['selected']), key=lambda j: -abs(r['B'][j]))
    coef = m['B'][np.asarray(union)+1]
    limit = max(np.max(np.abs(coef)), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .32*len(union))),
                             gridspec_kw=dict(width_ratios=[1, 2]), constrained_layout=True)
    ticks = np.arange(len(union))
    axes[0].barh(ticks, r['B'][union])
    axes[0].set_yticks(ticks, [str(features.iloc[j].otu_id)+' '+str(features.iloc[j].display_taxon) for j in union], fontsize=8)
    axes[0].invert_yaxis();axes[0].axvline(0, color='black', lw=.7)
    axes[0].set(title='R1: continuous pH', xlabel='Regression coefficient')
    im=axes[1].imshow(coef, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[1].set_yticks(ticks, [])
    axes[1].set_xticks(range(10));axes[1].set(title='MC1: ten pH classes', xlabel='pH class (low to high)')
    fig.colorbar(im, ax=axes[1], label='Class-score coefficient')
    fig.savefig(output/'selected-features.png', dpi=180);plt.close(fig)
    summary = dict(n_samples=len(X), n_features=X.shape[1], classes=10,
                   class_sizes=bins.n.tolist(), cap=cap, regression_knots=len(rb),
                   MC1_knots=len(mb), regression_seconds=regression_seconds,
                   MC1_seconds=multiclass_seconds, MC1_stop=pb.path_info['stop_reason'],
                   MC1_max_stationarity_residual=float(max(pb.path_info['residuals'],default=0)),
                   feature_mapping='116/116 OTUs and 88/88 samples verified by exact counts',
                   comparison=comparison)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cap', type=int, default=20)
    args=parser.parse_args()
    run(args.repo,args.output,args.cap)

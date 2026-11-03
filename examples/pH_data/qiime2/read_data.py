"""marimo version of `read data.ipynb`.

QIIME 2 is distributed through conda and cannot be installed by ``uv``, so this
notebook has no sandbox header. Run it from an activated QIIME 2 environment
(with the ``gneiss`` plugin) that also has marimo installed::

    conda activate qiime2
    pip install marimo
    marimo edit examples/pH_data/qiime2/read_data.py

Inputs are read from ``originals/`` and outputs written to ``news/``, both next
to this file.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import subprocess

    import marimo as mo
    import pandas as pd
    import qiime2


@app.cell
def _():
    here = mo.notebook_dir()
    originals, news = here / "originals", here / "news"
    news.mkdir(exist_ok=True)

    def qiime(*args):
        """Run a QIIME 2 command from this folder, failing loudly on errors."""
        subprocess.run(["qiime", *args], cwd=here, check=True)

    return news, originals, qiime


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 88 soils: pH data

    ## Metadata
    """)
    return


@app.cell
def _(originals):
    metadata = pd.read_table(originals / "88soils_modified_metadata.txt", index_col=0)
    y = metadata["ph"].values
    mo.vstack([metadata, mo.md(f"y: `{y.shape}`")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Import the OTU table and the taxonomy
    """)
    return


@app.cell
def _(news, qiime):
    qiime(
        "tools", "import",
        "--input-path", "originals/238_otu_table.biom",
        "--output-path", "news/88soils.biom.qza",
        "--type", "FeatureTable[Frequency]",
    )
    qiime(
        "tools", "import",
        "--input-path", "originals/88soils_taxonomy.txt",
        "--output-path", "news/88soils_taxonomy.qza",
        "--type", "FeatureData[Taxonomy]",
    )
    table = qiime2.Artifact.load(str(news / "88soils.biom.qza")).view(pd.DataFrame)
    table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Filter, cluster along the pH gradient and compute balances
    """)
    return


@app.cell
def _(news, qiime):
    qiime(
        "feature-table", "filter-features",
        "--i-table", "news/88soils.biom.qza",
        "--o-filtered-table", "news/88soils_filt100.biom.qza",
        "--p-min-frequency", "100",
    )
    qiime(
        "gneiss", "gradient-clustering",
        "--i-table", "news/88soils_filt100.biom.qza",
        "--m-gradient-file", "originals/88soils_metadata.txt",
        "--m-gradient-column", "ph",
        "--o-clustering", "news/ph_tree.nwk.qza",
        "--p-no-weighted",
    )
    qiime(
        "composition", "add-pseudocount",
        "--i-table", "news/88soils_filt100.biom.qza",
        "--p-pseudocount", "1",
        "--o-composition-table", "news/88soils_composition.biom.qza",
    )
    qiime(
        "gneiss", "ilr-hierarchical",
        "--i-table", "news/88soils_composition.biom.qza",
        "--i-tree", "news/ph_tree.nwk.qza",
        "--o-balances", "news/88soils_balances.qza",
    )
    balances = qiime2.Artifact.load(str(news / "88soils_balances.qza")).view(
        pd.DataFrame
    )
    balances.to_csv(news / "table.csv")
    balances
    return


@app.cell
def _(originals):
    tax = pd.read_table(originals / "88soils_taxonomy.txt", index_col=0)
    tax
    return


if __name__ == "__main__":
    app.run()

"""Record the resolved CI/benchmark environment; compatible with Python 3.8+."""
import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ('c-lasso', 'numpy', 'scipy', 'matplotlib', 'pandas',
                 'pytest', 'pytest-cov', 'pip', 'setuptools'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = dict(python=sys.version, executable=sys.executable,
                  platform=platform.platform(), machine=platform.machine(),
                  commit=os.environ.get('GITHUB_SHA'), packages=packages,
                  threads={key: os.environ.get(key) for key in
                           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                            'NUMEXPR_NUM_THREADS')})
    contents = json.dumps(report, indent=2) + '\n'
    (args.output/'environment.json').write_text(contents, encoding='utf-8')
    print(contents)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                            capture_output=True, text=True, check=True)
    (args.output/'pip-freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        for name in ('numpy', 'scipy'):
            print('\n=== {} ==='.format(name))
            try:
                importlib.import_module(name).show_config()
            except ImportError as error:
                # Retain useful diagnostics even when installation failed.
                print(error)
    (args.output/'blas-lapack.txt').write_text(configuration.getvalue(), encoding='utf-8')
    print(configuration.getvalue())


if __name__ == '__main__':
    main()

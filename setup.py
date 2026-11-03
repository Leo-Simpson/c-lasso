# Project metadata lives in pyproject.toml; this file only wires in versioneer.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import versioneer  # noqa: E402
from setuptools import setup  # noqa: E402

setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())

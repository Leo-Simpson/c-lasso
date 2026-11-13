from setuptools import setup, find_packages
import versioneer

setup(
    name="c-lasso",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    author="Leo Simpson",
    url="https://github.com/bio-datascience/c-lasso",
    author_email="leo.bill.simpson@gmail.com",
    description="Algorithms for constrained Lasso problems",
    packages=["classo"],
    python_requires=">=3.8",
    install_requires=["numpy>=1.24.4", "scipy>=1.10.1", "matplotlib>=3.7.5", "pandas>=2.0.3"],
    extras_require={
        "tests": ["pytest>=8.3.5", "pytest-cov>=5.0.0"],
        "docs": [
            "sphinx",
            "sphinx-gallery",
            "sphinx_rtd_theme",
            "numpydoc",
            "matplotlib",
            "pandas",
            "rdata"
        ],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
    ],
    long_description_content_type="text/markdown",
    long_description=open("README.md", encoding="utf-8").read(),
    zip_safe=False,
)

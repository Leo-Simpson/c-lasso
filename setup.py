from setuptools import setup, find_packages
import versioneer

setup(name='c-lasso',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='MIT',
      author='Leo Simpson',
      url='https://github.com/Leo-Simpson/CLasso',
      author_email='leo.bill.simpson@gmail.com',
      description='Algorithms for constrained Lasso problems',
      packages=['classo'],
      install_requires = [
                          'numpy',
                          'h5py',
                          'scipy'
                    ],
      extras_require = {
            'tests': [
                  'pytest',
                  'pytest-cov'],
            'docs': [
                  'sphinx',
                  'sphinx-gallery',
                  'sphinx_rtd_theme',
                  'numpydoc',
                  'matplotlib',
                  'pandas'
            ]
      },
      classifiers=['Intended Audience :: Science/Research',
                  'Operating System :: Microsoft :: Windows',
                  'Operating System :: POSIX',
                  'Operating System :: Unix',
                  'Operating System :: MacOS',
                  'Programming Language :: Python :: 3',
                  'Programming Language :: Python :: 3.6',
                  'Programming Language :: Python :: 3.7',
                  'Programming Language :: Python :: 3.8',
                  'Programming Language :: Python :: 3.9',
                  ],
      long_description_content_type = 'text/markdown',
      long_description=open('README.md').read(),
      zip_safe=False)

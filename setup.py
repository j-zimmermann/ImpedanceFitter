import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))

about = {}
print(here)
with open(os.path.join(here, 'impedancefitter', '__version__.py'), mode='r', encoding='utf-8') as f:
    exec(f.read(), about)

setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    author="Leonard Thiele, Julius Zimmermann",
    author_email="leonard.thiele@uni-rostock.de, julius.zimmermann@uni-rostock.de",
    description="Library for fitting impedance data to equivalent circuit models.",
    license="GPL-3.0-only",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={"": ["LICENSE"]},
    python_requires='>=3.7',
    install_requires=['pyyaml==5.4.1', 'xlrd==2.0.1',
                      'openpyxl==3.0.7', 'pandas>=1.0.1',
                      'openturns>=1.17', 'lmfit==1.0.3',
                      'numdifftools==0.9.39', 'corner==2.0.1',
                      'emcee==3.0.1', 'tqdm>=4.42.1', 'pyparsing',
                      'schemdraw==0.6.0; python_version < "3.7.0"',
                      'schemdraw==0.8; python_version >= "3.7.0"',
                      'packaging'],
)

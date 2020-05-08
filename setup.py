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
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=['pyyaml==5.3.1', 'xlrd==1.2.0', 'pandas==1.0.1',
                      'openturns==1.13', 'lmfit==1.0.1', 'corner==2.0.1',
                      'emcee==3.0.1', 'tqdm==4.42.1', 'pyparsing', 'xlsxwriter'],
)

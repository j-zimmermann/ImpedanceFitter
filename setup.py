import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impedancefitter",
    version="1.0.1",
    author="Leonard Thiele, Julius Zimmermann",
    author_email="leonard.thiele@uni-rostock.de, julius.zimmermann@uni-rostock.de",
    description="Library for fitting impedance data to models like single-shell oder double-shell model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=['pyyaml', 'numpy==1.17', 'xlrd==1.2.0',
                      'openturns==1.13', 'scipy==1.3.1', 'lmfit==0.9.15rc1', 'corner==2.0.1', 'emcee==3.0.1'],
)

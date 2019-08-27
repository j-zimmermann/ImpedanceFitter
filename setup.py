import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impedancefitter",
    version="1.0.0",
    author="Leonard Thiele, Julius Zimmermann",
    author_email="leonard.thiele@uni-rostock.de, julius.zimmermann@uni-rostock.de",
    description="Library for fitting impedance data to models like single-shell oder double-shell model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    dependency_links=['https://github.com/lmfit/lmfit-py/archive/41667df8052d2e9102076c49eea008e03a20ca1b.zip#egg=lmfit-0.9.13+git.41667df'],
    install_requires=['pyyaml', 'lmfit==0.9.13+git.41667df',
                      'numpy>=1.16', 'xlrd>=1.2.0', 'openturns>=1.12'],
)

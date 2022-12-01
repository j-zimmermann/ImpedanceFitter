import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

here = os.path.abspath(os.path.dirname(__file__))

about = {}
print(here)
with open(os.path.join(here, 'impedancefitter', '__version__.py'), mode='r', encoding='utf-8') as f:
    exec(f.read(), about)

with open('requirements.txt') as fp:
    install_requires = fp.read()

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
    install_requires=install_requires,
    package_data={"": ["LICENSE", "requirements.txt"]},
    python_requires='>=3.7',
)

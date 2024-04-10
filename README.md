## Original repository

This package is a fork of [j-zimmermann/ImpedanceFitter](https://github.com/j-zimmermann/ImpedanceFitter)
implementing pandas data frame as new input format for the impedance data.

[![DOI](https://zenodo.org/badge/297969672.svg)](https://zenodo.org/badge/latestdoi/297969672)


## DF Input Format

Data frame format must be indicated like this.

```python
freq_name = ...
real_name = ...
imag_name = ...
df_format = f'DF {freq_name}-{real_name}-{imag_name}'
```

Below is an example for the following data frame.

```python
import pandas as pd
from impedancefitter import get_equivalent_circuit_model, Fitter

text = """\
Freq,Im1,Re1,Im2,Re2
12,100,100,100,100
28,100,100,100,100
32,100,100,100,100
36,100,100,100,100
44,100,100,100,100
68,100,100,100,100
84,100,100,100,100
108,100,100,100,100
136,100,100,100,100
196,100,100,100,100
256,100,100,100,100
"""

df = pd.read_csv(pd.compat.StringIO(text))

model = 'ColeColeR_s1 + ColeColeR_s2'
model_object = get_equivalent_circuit_model(model)

fitter = Fitter('DF Freq-Re1-Im1', df=df)

# your code...
```
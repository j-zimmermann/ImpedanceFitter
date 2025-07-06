import os

import numpy as np
import pandas as pd

from impedancefitter.RC import RC_model
from impedancefitter.readin import (
    readin_Data_from_collection,
    readin_Data_from_csv_E4980AL,
    readin_Data_from_dataframe,
    readin_Data_from_TXT_file,
)

f = np.logspace(1, 8, num=20)
omega = 2.0 * np.pi * f
R = 100
C = 1e-4
Z = RC_model(omega, R, C)
d = {"freq": f, "real": Z.real, "imag": Z.imag}
data = pd.DataFrame(data=d)


def test_xslx():
    """Test reading from XLSX."""
    data.to_excel("test.xlsx", index=False)
    omega_read, Z_array_read = readin_Data_from_collection("test.xlsx", "XLSX")
    os.remove("test.xlsx")
    assert np.all(np.isclose(omega_read, omega)) and np.all(
        np.isclose(Z_array_read[0], Z)
    )


def test_csv():
    """Test reading from CSV."""
    data.to_csv("test.csv", index=False)
    omega_read, Z_array_read = readin_Data_from_collection("test.csv", "CSV")
    os.remove("test.csv")
    assert np.all(np.isclose(omega_read, omega)) and np.all(
        np.isclose(Z_array_read[0], Z)
    )


def test_csv_delimiter():
    """Test reading from TSV."""
    data.to_csv("test.csv", sep="\t", index=False)
    omega_read, Z_array_read = readin_Data_from_collection(
        "test.csv", "CSV", delimiter="\t"
    )
    os.remove("test.csv")
    assert np.all(np.isclose(omega_read, omega)) and np.all(
        np.isclose(Z_array_read[0], Z)
    )


def test_txt():
    """Test reading from TXT."""
    data.to_csv("test.txt", index=False, sep="\t")
    omega_read, Z_array_read = readin_Data_from_TXT_file("test.txt", 1)
    os.remove("test.txt")
    assert np.all(np.isclose(omega_read, omega)) and np.all(
        np.isclose(Z_array_read[0], Z)
    )


def test_csv_E4980AL():
    """Test reading from E4980AL format."""
    d2 = d.copy()
    d2["v"] = f
    d2["a"] = np.full(f.shape, 1e-5)
    data2 = pd.DataFrame(data=d2)
    data2.to_csv("test.csv", index=False)
    omega_read, Z_array_read = readin_Data_from_csv_E4980AL(
        "test.csv", current_threshold=1e-5
    )
    os.remove("test.csv")
    assert np.all(np.isclose(omega_read, omega)) and np.all(
        np.isclose(Z_array_read[0], Z)
    )


def test_df():
    """Test reading from DataFrame."""
    data2 = pd.DataFrame(data=data)
    omega_read, Z_array_read = readin_Data_from_dataframe(
        data2, df_freq_column="freq", df_real_column="real", df_imag_column="imag"
    )
    assert np.all(np.isclose(omega_read, omega)) and np.all(
        np.isclose(Z_array_read[0], Z)
    )


# TODO implement tests for Gamry etc.

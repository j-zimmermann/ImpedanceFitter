"""
Plot the impedance measured with a frequency response analyser.
For that, we first convert the data measured with a MokuGo and
a Rohde&Schwarz oscilloscpe.
Both devices have an input impedance of 1MOhm.
We wanted to measure a 47Ohm resistor.
Thus, we connected a 100Om resistor in series with the 47Ohm resistor.
This needs to be accounted for in the conversion by the `R_device` keyword.
"""

import impedancefitter

omega, Z = impedancefitter.bode_csv_to_impedance(
    "MokuGo_47R_100R_shunt.csv", "MokuGo", R_device=100
)
impedancefitter.plot_impedance(omega, Z)
impedancefitter.plot_bode(omega, Z)
omega, Z = impedancefitter.bode_csv_to_impedance(
    "RohdeSchwarz_47R_100R_shunt.csv", "R&S", R_device=100
)
impedancefitter.plot_impedance(omega, Z)
impedancefitter.plot_bode(omega, Z)

# -*- coding: utf-8 -*-
e0 = 8.854e-12
length = 5e-3    # length of electrodes unit(m)
width = 1.6e-3   # width of electrodes
S = length * width  # area of electrodes
thickness = 2e-3  # gap between electrodes
c0 = S * e0 / thickness  # capacitance of air, the electrode is a parallel capacitor = 3.54116e-14
c0 = 2.41974648880026e-13    # air capacitance# different than in formula
cf = 2.42532194241202e-13    # stray capacitance

Rc = 9.05e-6  # unit(m). cell radius in buffer, this should be changed according to the chosed cell line
dm = 7e-9    # thinckness of cell membrane m
ecp = 60     # permittivity for cytoplasm
p = 0.15     # volumn fraction (*100*100%). p = 3/4*pi*Rc^3*n_cell/Volumn
#Matlab: 
#p = 0.075
#Rc = 5.8e-6


Rn = (0.6)**(1. / 3.) * Rc  # 0.84*Rc, see Feldman 1999
dn = 40e-9   # thickness of nulear membrane
enp = 120    # Permittivity for nucleoplasm
#K = 7.28788242e-06
#alpha = 7.75831954e-01
ksup = 0.735

v1 = (1-dm/Rc)**3
v2 = (Rn/(Rc-dm))**3
v3 =  (1-dn/Rn)**3    

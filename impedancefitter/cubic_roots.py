from cmath import sqrt
import numpy as np


def get_cubic_roots(a, b, c):
    Q = (a * a - 3. * b) / 9.
    R = (2. * a * a * a - 9. * a * b + 27. * c) / 54.
    A = -(R + sqrt(R * R - Q * Q * Q))**(1. / 3.)
    if np.isclose(abs(A), 0):
        B = 0
    else:
        B = Q / A
    # first root
    x1 = (A + B) - a / 3.
    x2 = -0.5 * (A + B) - a / 3. + 1j * 0.5 * sqrt(3.) * (A - B)
    x3 = -0.5 * (A + B) - a / 3. - 1j * 0.5 * sqrt(3.) * (A - B)
    return x1, x2, x3

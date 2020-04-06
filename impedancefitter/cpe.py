from .elements import Z_CPE, Z_w
from .utils import add_additions, parallel


def cpe_model(omega, k, alpha, L=None, C=None, R=None, cf=None):
    Zs_fit = Z_CPE(omega, k, alpha)
    Z_fit = add_additions(omega, Zs_fit, None, None, L, C, R, cf)
    return Z_fit


def cpe_ct_model(omega, k, alpha, Rct, L=None, C=None, R=None, cf=None):
    Zs_fit = parallel(Z_CPE(omega, k, alpha), Rct)
    Z_fit = add_additions(omega, Zs_fit, None, None, L, C, R, cf)
    return Z_fit


def cpe_ct_w_model(omega, k, alpha, Rct, Aw, L=None, C=None, R=None, cf=None):
    Z_par = Rct + Z_w(omega, Aw)
    Zs_fit = parallel(Z_CPE(omega, k, alpha), Z_par)
    Z_fit = add_additions(omega, Zs_fit, None, None, L, C, R, cf)
    return Z_fit

from .elements import Z_CPE, Z_w, parallel


def cpe_model(omega, k, alpha):
    Z_fit = Z_CPE(omega, k, alpha)
    return Z_fit


def cpe_ct_model(omega, k, alpha, Rct):
    Z_fit = parallel(Z_CPE(omega, k, alpha), Rct)
    return Z_fit


def cpe_ct_w_model(omega, k, alpha, Rct, Aw):
    Z_par = Rct + Z_w(omega, Aw)
    Z_fit = parallel(Z_CPE(omega, k, alpha), Z_par)
    return Z_fit

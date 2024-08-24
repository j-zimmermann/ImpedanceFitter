import pytest
from impedancefitter.utils import draw_scheme

# those were used for testing the code
# they do not work with LMFIT!!!
# they were just there to check that everything is alright

working = ["RC + R + parallel(R, C) + parallel(R, C)",
           "parallel(R + R, C + C)",
           "parallel(R, C)",
           "parallel(R + C, parallel(R, C))",
           "parallel(C + R, R_f1)",
           "RC + R + parallel(R + C, C + R) + parallel(R, C)",
           "parallel(R + C + C, C + R)",
           "parallel(R + C, parallel(R, C))",
           "RC + R + parallel(R + C, C + R) + parallel(R, C)",
           "R + C + L + CPE",
           "RC + R + parallel(R, parallel(R, CPE))",
           "RC + R + parallel(parallel(R,C), parallel(R, C))",
           "parallel(parallel(R,C), parallel(R, C)) + RC + R",
           "RC + R + parallel(R + C, parallel(R, C))",
           "parallel(RC_f1 + parallel(R_f3, C_f4),  Cstray)",
           "parallel(parallel(R, R), C)",
           "parallel(RC_f1 + parallel(R_f3, C_f4),  Cstray) + R + L"]


def test_scheme():
    for w in working:
        draw_scheme(w, show=False)
    assert True


def test_draw_exception():
    model = "parallel(R_f1, C + R)"
    with pytest.raises(Exception):
        draw_scheme(model)

import os

from impedancefitter.utils import draw_scheme

# those were used for testing the code
# they do not work with LMFIT!!!
# they were just there to check that everything is alright

working = [
    "R + C + L + CPE",
    "RC + R + parallel(R, C) + parallel(R, C)",
    "parallel(R + R, C + C)",
    "parallel(R, C)",
    "parallel(R + C, parallel(R, C))",
    "parallel(C + R, R_f1)",
    "RC + R + parallel(R + C, C + R) + parallel(R, C)",
    "parallel(R + C + C, C + R)",
    "parallel(R + C, parallel(R, C))",
    "RC + R + parallel(R + C, C + R) + parallel(R, C)",
    "RC + R + parallel(R, parallel(R, CPE))",
    "RC + R + parallel(parallel(R,C), parallel(R, C))",
    "parallel(parallel(R,C), parallel(R, C)) + RC + R",
    "RC + R + parallel(R + C, parallel(R, C))",
    "parallel(RC_f1 + parallel(R_f3, C_f4),  Cstray)",
    "parallel(parallel(R, R), C)",
    "parallel(RC_f1 + parallel(R_f3, C_f4),  Cstray) + R + L",
]


for idx, w in enumerate(working):
    print(w)
    draw_scheme(w, show=False, save=True)
    os.rename("scheme.svg", f"scheme_{idx}.svg")

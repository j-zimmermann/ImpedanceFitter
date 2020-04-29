from impedancefitter.utils import get_comp_model

models = ["R + C + parallel(ColeCole, DRC)",
"R_1 + C_1 + parallel(ColeCole_2, DRC_2)",
"parallel(ColeCole + R, DRC)",
"parallel(ColeCole, DRC)",
"R + parallel(ColeCole, DRC) + C",
"parallel(ColeCole, DRC) + R + C",
"parallel(parallel(ColeCole, L), DRC) + R + C",
"parallel(ColeCole + L, DRC) + R + C",
"R + C",
         ]
for model in models:
    print(model)
    m = get_comp_model(model)
    print(m)
    print("\n")

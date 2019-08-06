from impedancefitter import Fitter

fitter = Fitter(mode=None, model='SingleShell', LogLevel='DEBUG', solvername='nelder-mead', inputformat='TXT', minimumFrequency=1, maximumFrequency=2e7)
fitter.main()

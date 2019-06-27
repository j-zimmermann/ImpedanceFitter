from impedancefitter import Fitter

fitter = Fitter(mode=None, model='DoubleShell', LogLevel='INFO', solvername='nelder-mead', inputformat='TXT')
fitter.main()

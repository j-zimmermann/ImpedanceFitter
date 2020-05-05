import pprint as pp
from impedancefitter import Fitter

fitter = Fitter('TXT', LogLevel='DEBUG', minimumFrequency=1, maximumFrequency=2e7)
fitter.run('parallel(SingleShell + CPE, Cstray)', solver='nelder-mead')
pp.pprint(fitter.fit_data)

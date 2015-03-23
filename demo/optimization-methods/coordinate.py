# Optimize Rosenbrock function with coordinate search optimizer. 

from pyopus.optimizer.coordinate import CoordinateSearch
from pyopus.optimizer.mgh import Rosenbrock
from numpy import array

if __name__=='__main__':
	prob=Rosenbrock()
	opt=CoordinateSearch(prob.f, debug=1, maxiter=100000, 
			step0=1e-1, minstep=1e-6) 
	opt.reset(prob.initial)
	opt.run()
	
	print("x=%s f=%e" % (str(opt.x), opt.f))
	
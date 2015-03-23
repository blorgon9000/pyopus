# Optimize Rosenbrock function with Hooke-Jeeves optimizer. 

from pyopus.optimizer.hj import HookeJeeves
from pyopus.optimizer.mgh import Rosenbrock

if __name__=='__main__':
	prob=Rosenbrock()
	opt=HookeJeeves(prob.f, debug=1, maxiter=100000, step0=1e-1, minstep=1e-6)
	opt.reset(prob.initial)
	opt.run()

	print("x=%s f=%e" % (str(opt.x), opt.f))

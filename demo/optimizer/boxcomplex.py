# Optimize Rosenbrock function using Box complex optimizer. 

from pyopus.optimizer.boxcomplex import BoxComplex
from pyopus.problems.mgh import Rosenbrock 

if __name__=='__main__':
	prob=Rosenbrock()
	opt=BoxComplex(prob.f, [-10,-10], [10,10], debug=1, maxiter=100000)
	opt.reset(prob.initial)
	opt.run()

	print("x=%s f=%e" % (opt.x, opt.f))
	
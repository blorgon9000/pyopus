# Demonstration of CUTEr use - unconstrained problem (ROSENBR)
# (C)2011 Arpad Buermen
# Licensed under LGPL V2.1

from pyopus.problems import cutermgr
from numpy import array

if __name__ == '__main__':
	# Clear cache - this is not neccessary because the cache entry is removed
	# by prepareProblem() before the problem interface is built. 
	cutermgr.clearCache('ROSENBR')
	
	# Prepare two problems
	cutermgr.prepareProblem('ROSENBR')

	# These two are equivalent and demonstrate the use of sifParams and sifOptions. 
	# prepareProblem("LUBRIFC", sifParams={'NN': 10})
	# prepareProblem("LUBRIFC", sifOptions=['-param', 'NN=10'])

	# Import ROSENBR (unconstrained problem)
	# prepareProblem() returns a reference to the imported module.
	# Use this only if the problem interface is already available in the cache. 
	uproblem=cutermgr.importProblem('ROSENBR')
		
	# ROSENBR
	#
	# f = (1 - x1)^2 + 100 (x2 - x1^2)^2
	# 
	# g = [ -2 (1 - x1) - 400 (x2 - x1^2) x1    200 (x2 - x1^2) ]
	#
	#     [ 2 - 400 (x2 - x1^2) + 800 x1^2    -400 x1 ]
	# H = [                                           ]
	#     [ -400 x1                            200    ]
	#
	#
	# at x0 = [ -1.2  1.0 ]
	#
	#   f = 24.2
	#   g = [ -215.6  -88.0 ]
	#
	#       [ 1330.0  480.0 ]
	#   H = [               ]
	#       [ 480.0   200.0 ]

	print "Unconstrained problem demo"
	info=uproblem.getinfo()
	print "Problem name        : ", info['name']
	print "Problem size        : ", info['n']
	print "Constraint count    : ", info['m']
	print "NNZ in UT Hessian   : ", info['nnzh']
	print "Lower bounds        : ", info['bl']
	print "Upper bounds        : ", info['bu']
	print "Initial point       : ", info['x']
	print "Variable type       : ", info['vartype']
	print "Variable names      : ", uproblem.varnames()
	print "Sifdecode params    : ", info['sifparams']
	print "Sifdecode options   : ", info['sifoptions']
	print "Nonl. vars. first   : ", info['nvfirst']
	
	x0=info['x']
	
	print "\nEvaluating objective at x0"
	f=uproblem.obj(x0)
	print "f(x0)=", f
	
	print "\nEvaluating function and gradient"
	(f, g)=uproblem.obj(x0, True)
	print "f(x0)=", f
	print "g(x0)=", g
	
	print "\nEvaluating function and constraints"
	(f, c)=uproblem.objcons(x0)
	print "f(x0)=", f
	print "c(x0)=", c
	
	print "\nEvaluating constraints"
	c=uproblem.cons(x0)
	print "c(x0)=", c
	
	print "\nEvaluating constraints and Jacobian"
	(c, J)=uproblem.cons(x0, True)
	print "c(x0)=", c
	print "J(x0)=", J
	
	print "\nEvaluating function gradient and constraints Jacobian"
	(g, J)=uproblem.lagjac(x0)
	print "g(x0)=", g
	print "J(x0)=", J
	
	print "\nEvaluating Hessian of objective for unconstrained problem"
	H=uproblem.hess(x0)
	print "H(x0)=", H
	
	print "\nEvaluating Hessian of objective"
	H=uproblem.ihess(x0)
	print "H(x0)=", H
	
	print "\nEvaluating Hessian at x0 times [2.0, 2.0]"
	r=uproblem.hprod(array([2.0, 2.0]), x0)
	print "H(x0)*[2.0, 2.0]=", r
	
	print "\nEvaluating previous Hessian times [2.0, 2.0]"
	r=uproblem.hprod(array([2.0, 2.0]))
	print "H*[2.0, 2.0]=", r
	
	print "\nEvaluating gradient of objective at x0 and Hessian"
	(g, H)=uproblem.gradhess(x0)
	print "g(x0)=", g
	print "H(x0)=", H
	
	# Sparse Jacobian cannot be obtained for unconstrained problems because sparse
	# matrices of size 0-by-n are not supported in SciPy.
	
	print "\nEvaluating sparse Hessian of objective for unconstrained problem"
	H=uproblem.sphess(x0)
	print "H(x0)=", H.todense()
	
	print "\nEvaluating sparse Hessian of objective"
	H=uproblem.isphess(x0)
	print "H(x0)=", H.todense()
	
	print "\nEvaluating gradient and sparse Hessian of objective"
	(g, J)=uproblem.gradsphess(x0)
	print "g(x0)=", g 
	print "J(x0)=", J.todense()
	
	print "\nCollecting report"
	rep=uproblem.report()
	print "Setup time        : ", rep['tsetup']
	print "Run time          : ", rep['trun']
	print "Num. of f eval    : ", rep['f']
	print "Num. of g eval    : ", rep['g']
	print "Num. of H eval    : ", rep['H']
	print "Num. of H prod    : ", rep['Hprod']

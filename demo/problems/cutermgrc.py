# Demonstration of PyCUTEr use - constrained problem (HS71)
# (C)2011 Arpad Buermen
# Licensed under LGPL V2.1

from pyopus.problems import cutermgr
from numpy import array

if __name__ == '__main__':
	# Clear cache - this is not neccessary because the cache entry is removed
	# by prepareProblem() before the problem interface is built. 
	cutermgr.clearCache('HS71')

	# Prepare two problems
	cutermgr.prepareProblem('HS71')

	# Import HS71 (unconstrained problem)
	# prepareProblem() returns a reference to the imported module.
	# Use this only if the problem interface is already available in the cache. 
	cproblem=cutermgr.importProblem('HS71')
	
	# HS71
	#
	# f = x1 x4 (x1 + x2 + x3) + x3
	# subject to
	# (C1)     x1 x2 x3 x4 - 25 >= 0
	# (C2)     x1^2 + x2^2 + x3^2 + x4^2 - 40 = 0
	# (bounds) 1 <= x1, x2, x3, x4 <= 5
	#
	# g = [ x4 (x1 + x2 + x3) + x1 x4    x1 x4    x1 x4 + 1    x1 (x1 + x2 + x3) ]
	#
	#     [ 2 x4              x4    x4    2 x1 + x2 + x3 ]
	#     [                                              ]
	#     [ x4                0     0     x1             ]
	# H = [                                              ]
	#     [ x4                0     0     x1             ]
	#     [                                              ]
	#     [ 2 x1 + x2 + x3    x1    x1    0              ]
	#
	#     [ x2 x3 x4    x1 x3 x4    x1 x2 x4    x1 x2 x3 ]
	# J = [                                              ]
	#     [ 2*x1        2 x2        2 x3        2 x4     ]
	#
	#       [ 0        x3 x4    x2 x4    x2 x3 ]
	#       [                                  ]
	#       [ x3 x4    0        x1 x4    x1 x3 ]
	# HC1 = [                                  ]
	#       [ x2 x4    x1 x4    0        x1 x2 ]
	#       [                                  ]
	#       [ x2 x3    x1 x3    x1 x2    0     ]
	#
	#       [ 2    0    0    0 ]
	#       [                  ]
	#       [ 0    2    0    0 ]
	# HC2 = [                  ]
	#       [ 0    0    2    0 ]
	#       [                  ]
	#       [ 0    0    0    2 ]
	#
	# at x0 = [ 1.0  5.0  5.0  1.0 ]
	#
	#   f   = 16.0
	#   c1  = 0.0
	#   c2  = 12.0
	#  
	#   g   = [ 12.0    1.0    2.0    11.0 ] 
	#
	#         [ 2.0     1.0    1.0    12.0 ]
	#         [                            ]
	#         [ 1.0     0.0    0.0    1.0  ]
	#   H   = [                            ]
	#         [ 1.0     0.0    0.0    1.0  ]
	#         [                            ]
	#         [ 12.0    1.0    1.0    0.0  ]
	#
	#         [ 25.0    5.0     5.0     25.0 ]
	#   J   = [                              ]
	#         [ 2.0     10.0    10.0    2.0  ]
	#
	#         [ 0.0     5.0    5.0    25.0 ]
	#         [                            ]
	#         [ 5.0     0.0    1.0    5.0  ]
	#   HC1 = [                            ]
	#         [ 5.0     1.0    0.0    5.0  ]
	#         [                            ]
	#         [ 25.0    5.0    5.0    0.0  ]


	print "Constrained problem demo"
	info=cproblem.getinfo()
	print "Problem name        : ", info['name']
	print "Problem size        : ", info['n']
	print "Constraint count    : ", info['m']
	print "NNZ in UT Hessian   : ", info['nnzh']
	print "Lower bounds        : ", info['bl']
	print "Upper bounds        : ", info['bu']
	print "Initial point       : ", info['x']
	print "Variable type       : ", info['vartype']
	print "Variable names      : ", cproblem.varnames()
	print "Sifdecode params    : ", info['sifparams']
	print "Sifdecode options   : ", info['sifoptions']
	print "Constraint names    : ", cproblem.connames()
	print "Eq. constr. first   : ", info['efirst']
	print "Lin. constr. first  : ", info['lfirst']
	print "NNZ in Jacobian     : ", info['nnzj']
	print "Equality constr.    : ", info['equatn']
	print "Linear constr.      : ", info['linear']
	print "Lower constraint    : ", info['cl']
	print "Upper constraint    : ", info['cu']
	print "Init. Lagr. mult.   : ", info['v']
	
	x0=info['x']
	
	print "\nEvaluating objective at x0"
	f=cproblem.obj(x0)
	print "f(x0)=", f
	
	print "\nEvaluating function and gradient"
	(f, g)=cproblem.obj(x0, True)
	print "f(x0)=", f
	print "g(x0)=", g
	
	print "\nEvaluating function and constraints"
	(f, c)=cproblem.objcons(x0)
	print "f(x0)=", f
	print "c(x0)=", c
	
	print "\nEvaluating constraints"
	c=cproblem.cons(x0)
	print "c(x0)=", c
	
	print "\nEvaluating single constraint (0)"
	c0=cproblem.cons(x0, False, 0)
	print "c0(x0)      : ", c0
		
	print "\nEvaluating constraints and Jacobian"
	(c, J)=cproblem.cons(x0, True)
	print "c(x0)=", c
	print "J(x0)=", J
	
	print "\nEvaluating single constraint (0) and its gradient"
	(c0, gc0)=cproblem.cons(x0, True, 0)
	print "c0(x0)=", c0
	print "gc0(x0)=", gc0
	
	print "\nEvaluating function gradient and constraints Jacobian"
	(g, J)=cproblem.lagjac(x0)
	print "g(x0)=", g
	print "J(x0)=", J
	
	print "\nEvaluating gradient of Lagrangian at v=[1, 1] and constraints Jacobian"
	(g, J)=cproblem.lagjac(x0, array([1.0, 1.0]))
	print "gl(x0,[1.0, 1.0])=", g
	print "J(x0)=", J

	print "\nEvaluating product of previous J and [1, 1, 1, 1]"
	r=cproblem.jprod(False, array([1.0, 1.0, 1.0, 1.0]))
	print "J*[1.0, 1.0, 1.0, 1.0]=", r

	print "\nEvaluating product of J at x0 and [1, 1, 1, 1]"
	r=cproblem.jprod(False, array([1.0, 1.0, 1.0, 1.0]), x0)
	print "J(x0)*[1.0, 1.0, 1.0, 1.0]=", r

	print "\nEvaluating product of previous transposed J and [1, 1]"
	r=cproblem.jprod(True, array([1.0, 1.0]))
	print "JT*[1.0, 1.0]=", r

	print "\nEvaluating product of transposed J at x0 and [1, 1]"
	r=cproblem.jprod(True, array([1.0, 1.0]), x0)
	print "JT(x0)*[1.0, 1.0]=", r
	
	print "\nEvaluating Hessian of Lagrangian at [1, 1] for constrained problem"
	Hl=cproblem.hess(x0, array([1.0, 1.0]))
	print "Hl(x0, [1.0, 1.0])-", Hl
	
	print "\nEvaluating Hessian of objective"
	H=cproblem.ihess(x0)
	print "H(x0)=", H
	
	print "\nEvaluating Hessian of constraint (0)"
	H0=cproblem.ihess(x0, 0)
	print "H0(x0)=", H0
	
	print "\nEvaluating Hessian of constraint (1)"
	H1=cproblem.ihess(x0, 1)
	print "H1(x0)=", H1
	
	print "\nEvaluating Hessian of the Lagrangian at (x0, [1.0, 1.0]) times [2.0, 2.0, 2.0, 2.0]"
	r=cproblem.hprod(array([2.0, 2.0, 2.0, 2.0]), x0, array([1.0, 1.0]))
	print "Hl(x0, [1.0, 1.0])*[2.0, 2.0, 2.0, 2.0]=", r
	
	print "\nEvaluating previous Hess. of the Lagr. times [2.0, 2.0, 2.0, 2.0]"
	r=cproblem.hprod(array([2.0, 2.0, 2.0, 2.0]))
	print "Hl*[2.0, 2.0, 2.0, 2.0]=", r
	
	print "\nEvaluating gradient of Lagrangian at (x0, [1.0, 1.0]), Jacobian, and Hessian of Lagrangian"
	(gl, J, Hl)=cproblem.gradhess(x0, array([1.0, 1.0]), True)
	print "gl(x0, [1.0, 1.0])=", gl
	print "J(x0)=", J
	print "Hl(x0, [1.0, 1.0])=", Hl
	
	print "\nEvaluating grad. of obj. at x0, Jacobian, and Hessian of Lagrangian"
	(g, J, Hl)=cproblem.gradhess(x0, array([1.0, 1.0]), False)
	print "g(x0)=", g
	print "J(x0)=", J
	print "Hl(x0)=", Hl
	
	print "\nEvaluating constraints and sparse Jacobian"
	(c, J)=cproblem.scons(x0)
	print "c(x0)=", c
	print "J(x0)=", J.todense()
	
	print "\nEvaluating constraint (0) and its sparse gradient"
	(c0, gc0)=cproblem.scons(x0, 0)
	print "c0(x0)=", c0
	print "gc0(x0)=", gc0.todense()
	
	print "\nEvaluating sparse objective gradient and sparse Jacobian"
	(g, J)=cproblem.slagjac(x0)
	print "g(x0)=", g.todense()
	print "J(x0)=", J.todense()
	
	print "\nEvaluating sparse Lagrangian gradient and sparse Jacobian at (x0, [1.0, 1.0])"
	(gl, J)=cproblem.slagjac(x0, array([1.0, 01.0]))
	print "gl(x0)=", gl.todense()
	print "J(x0)=", J.todense()
	
	print "\nEvaluating sparse Hessian of Lagrangian at (x0, [1, 1]) for constrained problem"
	Hl=cproblem.sphess(x0, array([1.0, 1.0]))
	print "Hl(x0, [1.0, 1.0])=", Hl.todense()
	
	print "\nEvaluating sparse Hessian of objective"
	H=cproblem.isphess(x0)
	print "H(x0)=", H.todense()
	
	print "\nEvaluating sparse Hessian of single constraint (0)"
	H0=cproblem.isphess(x0, 0)
	print "H0(x0)=", H0.todense()
	
	print "\nEvaluating sparse Hessian of single constraint (1)"
	H1=cproblem.isphess(x0, 1)
	print "H0(x0)=", H1.todense()
	
	print "\nEvaluating grad. of obj., constr. Jac., and Hess. of Lagr. at (x0, [0.0, 0.0])"
	(g, J, Hl)=cproblem.gradsphess(x0, array([1.0, 1.0]))
	print "g(x0)=", g.todense()
	print "J(x0)=", J.todense()
	print "Hl(x0, [1.0, 1.0])=", Hl.todense()
	
	print "\nEvaluating grad. of Lagr., constr. Jac., and Hess. of Lagr. at (x0, [0.0, 0.0])"
	(gl, J, Hl)=cproblem.gradsphess(x0, array([1.0, 1.0]), True)
	print "gl(x0)=", gl.todense()
	print "J(x0)=", J.todense()
	print "Hl(x0, [1.0, 1.0])=", Hl.todense()
	
	print "\nCollecting report"
	rep=cproblem.report()
	print "Setup time        : ", rep['tsetup']
	print "Run time          : ", rep['trun']
	print "Num. of f eval    : ", rep['f']
	print "Num. of g eval    : ", rep['g']
	print "Num. of H eval    : ", rep['H']
	print "Num. of H prod    : ", rep['Hprod']
	print "Num. of c eval    : ", rep['c']
	print "Num. of cg eval   : ", rep['cg']
	print "Num. of cH eval   : ", rep['cH']
	
# -*- coding: UTF-8 -*-
# Lukšan-Vlček nonsmooth problems test suites

from pyopus.problems.lvns import *

if __name__=='__main__':
	print "Unconstrained minimax problems"
	for ii in range(len(UMM.names)):
		prob=UMM(number=ii)
		
		# These problems need setting up before they are used. 
		# Only one problem can be uset at a time. 
		prob.setup()
		
		print "%2d: %20s n=%2d: f0=%e" % (ii, prob.name, prob.n, prob.f(prob.initial))
	print
	
	print "Unconstrained nonsmooth problems"
	for ii in range(len(UNS.names)):
		prob=UNS(number=ii)
		
		# These problems need setting up before they are used. 
		# Only one problem can be uset at a time. 
		prob.setup()
		
		print "%2d: %20s n=%2d: f0=%e" % (ii, prob.name, prob.n, prob.f(prob.initial))
	print
	
	print "Linearly constrained minimax problems"
	for ii in range(len(LCMM.names)):
		prob=LCMM(number=ii)
		
		# These problems need setting up before they are used. 
		# Only one problem can be uset at a time. 
		prob.setup()
		
		print "%2d: %20s n=%2d: f0=%e" % (ii, prob.name, prob.n, prob.f(prob.initial))
	print



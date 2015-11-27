# -*- coding: UTF-8 -*-
# Lukšan-Vlček unconstrained problems test suite

from pyopus.problems.lvu import *

if __name__=='__main__':
	print "Unconstrained problems, n=50"
	for ii in range(len(LVU.names)):
		prob=LVU(number=ii, n=50)
		
		# These problems need setting up before they are used. 
		# Only one problem can be uset at a time. 
		prob.setup()
		
		print "%2d: %40s: f0=%e" % (ii, prob.name, prob.f(prob.initial))
	print


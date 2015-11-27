# Karmitsa test suite

from pyopus.problems.karmitsa import *

if __name__=='__main__':
	print "Unconstrained, n=50"
	for ii in range(10):
		prob=LSNSU(number=ii, n=50)
		print "%d: %25s: f0=%e" % (ii, prob.name, prob.f(prob.initial))
	print
	
	print "Bound constrained, n=50"
	for ii in range(10):
		prob=LSNSB(number=ii, n=50)
		print "%d: %25s: f0=%e" % (ii, prob.name, prob.f(prob.initial))
	print
	
	print "Inequality constrained, n=10"
	for ii in range(10):
		for jj in range(8):
			prob=LSNSI(number=ii, cnumber=jj, n=10)
			c=prob.c(prob.initial)
			feas=((c>prob.cl) & (c<prob.ch)).all()
			print "%d, %d: %30s: f0=%12.4e feas=%d" % (ii,jj, prob.name, prob.f(prob.initial), feas)
	print
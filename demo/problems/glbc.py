# Global optimization problems

from pyopus.problems.glbc import GlobalBCsuite

if __name__=='__main__':
	print "Global optimization problems (bound constrained, initial point at 1/4 range)"
	for ii in range(len(GlobalBCsuite)):
		prob=GlobalBCsuite[ii]()
		x0=prob.xl*0.25+prob.xh*0.75
		print "%2d: %40s n=%2d: f0=%e" % (ii, prob.name, prob.n, prob(x0))
	print


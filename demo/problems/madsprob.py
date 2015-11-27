# Global optimization problems

from pyopus.problems.madsprob import MADSPROBsuite

if __name__=='__main__':
	print "MADS problems"
	for ii in range(len(MADSPROBsuite)):
		prob=MADSPROBsuite[ii]()
		fx, cx = prob.fc(prob.initial)
		print "%2d: %20s n=%2d: f0=%e" % (ii, prob.name, prob.n, fx)
	print


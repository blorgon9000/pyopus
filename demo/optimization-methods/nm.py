# Optimize MGH unconstrained test suite with Nelder-Mead optimizer. 

from pyopus.optimizer.nm import NelderMead
from pyopus.optimizer.base import Reporter
from pyopus.optimizer.mgh import *
from numpy import array, sqrt
from platform import system
if system()=='Windows':
	# clock() is the most precise timer in Windows
	from time import clock as timer
else:
	# time() is the most precise timer in Linux
	from time import time as timer
from sys import stdout

# Custom reporter that prints dots 
class MyReporter(Reporter):
	def __init__(self, name, iterSpacing=1, concise=False, printImprovement=True):
		Reporter.__init__(self)
		self.name=name
		self.iterSpacing=iterSpacing
		self.concise=concise
		self.printImprovement=printImprovement
		self.fbest=None
		self.lastPrintout=None
	
	def reset(self):
		fbest=None
		
	def __call__(self, x, f, opt, annot):
		report=False
		
		if self.fbest is None:
			self.lastPrintout=opt.niter
			self.fbest=f
			report=True
		
		if self.fbest>f and self.printImprovement:
			self.fbest=f
			report=True
		
		if opt.niter-self.lastPrintout>=self.iterSpacing: 
			report=True
		
		if report:
			if self.concise:
				stdout.write(".")
				stdout.flush()
			else:
				print("%30s (%2d) iter=%-6d f=%12.4e fbest=%12.4e" % (self.name[:30], x.size, opt.niter, f, self.fbest))
			self.lastPrintout=opt.niter
		

if __name__=='__main__':
	suite=[ 
		[ Rosenbrock() ],
		[ FreudensteinAndRoth() ], 
		[ PowellBadlyScaled() ], 
		[ BrownBadlyScaled() ], 
		[ Beale() ], 
		[ JennrichAndSampson() ], 
		[ McKinnon() ], 
		[ McKinnon(), array([[0.0, 0.0], [1.0, 1.0], [(1.0+sqrt(33.0))/8.0, (1.0-sqrt(33.0))/8.0]]) ], 
		[ HelicalValley() ], 
		[ Bard() ], 
		[ Gaussian() ],  
		[ Meyer() ], 
		[ GulfResearchAndDevelopement() ], 
		[ Box3D() ], 
		[ PowellSingular() ], 
		[ Wood() ], 
		[ KowalikAndOsborne() ], 
		[ BrownAndDennis() ], 
		[ Quadratic(4) ], 
		[ PenaltyI(4) ], 
		[ PenaltyII(4) ], 
		[ Osborne1() ], 
		[ BrownAlmostLinear(5) ], 
		[ BiggsEXP6() ], 
		[ ExtendedRosenbrock(6) ], 
		[ BrownAlmostLinear(7) ], 
		[ Quadratic(8) ], 
		[ ExtendedRosenbrock(8) ], 
		[ VariablyDimensioned(8) ], 
		[ ExtendedPowellSingular(8) ], 
		[ Watson() ], 
		[ ExtendedRosenbrock(10) ], 
		[ PenaltyI(10) ], 
		[ PenaltyII(10) ],
		[ Trigonometric() ], 
		[ Osborne2() ],
		[ ExtendedPowellSingular(12) ],
		[ Quadratic(16) ], 
		[ Quadratic(24) ], 
	]

	t1=timer()

	results=[]

	for probdef in suite:
		# Take problem function. 
		prob=probdef[0]
		
		# Write a message. 
		print("\nProcessing: "+prob.name+" ("+str(prob.n)+") ") 
		
		# Create optimizer object. 
		opt=NelderMead(prob.f, maxiter=100000,  reltol=1e-16, ftol=1e-16, xtol=1e-9)
		
		# Install custom reporter plugin. Print dot for 1000 evaluations. 
		opt.installPlugin(MyReporter(prob.name, 1000, concise=True, printImprovement=False))
		
		# If problem has a custom initial simplex, set it. 
		if len(probdef)==1:
			opt.reset(prob.initial)
		else:
			# Custom simplex (for McKinnon) 
			opt.reset(probdef[1])
		
		# Start timing, run, measure time. 
		dt=timer()
		opt.run()
		dt=timer()-dt
		
		# Write number of function evaluations. 
		print(" %d evaluations" % opt.niter)
		
		# Calculate initial and final gradient
		gini=prob.g(prob.initial)
		gend=prob.g(opt.x)
		
		# Store results for this problem. 
		result={ 
			'i': opt.niter, 
			'x': opt.x, 
			'f': opt.f, 
			'gi': sqrt((gini**2).sum()), 
			'ge': sqrt((gend**2).sum()), 
			't': dt
		}
		results.append(result)

	# Print summary. Last column is iterations per second. 
	print("\n")
	for i in range(0, len(suite)):
		prob=suite[i][0]
		print("%30s (%2d) : ni=%6d f=%16.8e gi=%11.3e ge=%11.3e  it/s=%11.3e" % (
				prob.name[:30], 
				prob.initial.size, 
				results[i]['i'], 
				results[i]['f'], 
				results[i]['gi'], 
				results[i]['ge'], 
				results[i]['i']/results[i]['t']
			)
		)

	t2=timer()

	print("\ntime=%f" % (t2-t1))

		

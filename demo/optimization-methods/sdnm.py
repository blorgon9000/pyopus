# Optimize MGH suite with sufficient descent Nelder-Mead optimizer. 

from pyopus.optimizer.sdnm import SDNelderMead
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

	results=[]

	# Sub-suites of problems
	# mysuite=suite[7:8]	# McKinnon (alt)
	# mysuite=suite[0:8] # First 8 functions
	# mysuite=suite[1:2]	# Freudenstein and Roth
	# mysuite=[suite[3], suite[6], suite[8], suite[10], suite[16], suite[34]] # brown, mckin, helical, gaussian, kowalik, trig
	mysuite=[suite[0]] 
	mysuite=suite

	for probdef in mysuite:
		# Take problem function. 
		prob=probdef[0]
		
		# Write a message. 
		print("\nProcessing: "+prob.name+" ("+str(prob.n)+") ") 
		
		# Create optimizer object.
		opt=SDNelderMead(prob.f, debug=0, maxiter=100000)
		
		# Install custom reporter plugin. Print dot for 500 evaluations. 
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

	# Print summary. Last column is initial/final gradient.   
	print("\n")
	for i in range(0, len(mysuite)):
		prob=mysuite[i][0]
		print "%2d: %30s (%2d): ni=%6d f=%16.8e gradient: %9.1e -> %9.1e : r=%9.1e" % (
			i, 
			prob.name[:30], 
			prob.initial.size, 
			results[i]['i'], 
			results[i]['f'], 
			results[i]['gi'], 
			results[i]['ge'], 
			results[i]['gi']/results[i]['ge'], 
		)

		

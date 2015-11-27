# Demo of the Common Problem Interface

from pyopus.problems import mgh, glbc, lvns, mwbm
import numpy as np

if __name__ == '__main__':
	# Build a list of problems from various collections
	problems=[
		# A More-Garbow-Hillstrom problem
		mgh.ExtendedRosenbrock(n=10), 
		# Luksan-Vlcek problems
		lvns.UNS("ElAttar"), 
		lvns.LCMM("PENTAGON"), 
		# A global optimization problem
		glbc.Rastrigin(n=30), 
		# All variants of the More-Wild problem 9
		mwbm.MWBM(9, problemType=0), # Smooth
		mwbm.MWBM(9, problemType=1), # Piecewise-smooth
		mwbm.MWBM(9, problemType=2, epsilon=1e-3), # Deterministically noisy
		mwbm.MWBM(9, problemType=3, epsilon=1e-3), # Stochastically noisy
	]

	# Try adding some CUTEr problems
	try:
		from pyopus.problems.cuter import CUTEr
		
		# Set these to True if you want to rebuild problems and do it quietly
		rebuild=True
		quiet=True
	  
		problems.extend([
			CUTEr("HS71",  forceRebuild=rebuild, quiet=quiet), 
			CUTEr("HS80",  forceRebuild=rebuild, quiet=quiet), 
			CUTEr("ROSENBR",  forceRebuild=rebuild, quiet=quiet), 
			# Store as ARWHEAD_10 in cache, use N=10
			CUTEr("ARWHEAD",  "ARWHEAD_10",  sifParams={ "N": 10 },   forceRebuild=rebuild, quiet=quiet), 
		])
  
	except:
		print("CUTEr not available")
  
	for problem in problems:
		# Get common problem interface
		cpi=problem.cpi()
		
		# Some problems reuqire an initialization 
		# (call the 'setup' member of the cpi structure)
		if cpi['setup'] is not None:
			cpi['setup']()
		
		# get dimension and number of constraints
		n=cpi['n']
		m=cpi['m']
		
		# Get initial point
		if cpi['x0'] is not None:
			x0=cpi['x0']
		else:
			# Use the center of bounds on x (e.g. for glbc) as x0 when
			# x0 is not available
			x0=(cpi['xlo']+cpi['xhi'])/2
		
		# Evaluate f at initial point
		f0=cpi['f'](x0)
		
		# Evaluate constraints
		if m>0:
			c0=cpi['c'](x0)
			
			# Get bounds on constraint functions
			cl=cpi['clo']
			ch=cpi['chi']
			
			# Cumulative constraint violation
			h0=(np.where(c0<cl, cl-c0, 0.0)+np.where(c0>ch, c0-ch, 0.0)).sum()
		
		# Print summary
		print "%25s n=%2d: f0=%12.3e" % (cpi['name'], n, f0),
		if m>0:
			print " constrained, m=%2d, h0=%12.3e" % (m, h0)
		else:
			print
		
		
		

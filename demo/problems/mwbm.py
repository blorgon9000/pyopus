# Demo of the More-Wild problems

from pyopus.problems.mwbm import MWBM
import numpy as np

if __name__ == '__main__':
	# Every problem is described by an entry in the list of descriptors. 
	# We use this list to get a problem count. 
	for ii in range(len(MWBM.descriptors)):
		# Create the default (smooth) problem
		problem=MWBM(ii, problemType=0)
		# Set problemType to
		#  1 for piecewise-smooth problems
		#  2 for deterministically noisy problems 
		#  3 for stochastically noisy problems
		cpi=problem.cpi()
		print(
			"%2d: %25s n=%2d m=%2d s=%2d f0=%e" % 
			(
				ii, 
				cpi['name'], 
				cpi['n'], 
				cpi['info']['m'], 
				cpi['info']['s'], 
				cpi['f'](cpi['x0'])
			)
		)
		
		
		

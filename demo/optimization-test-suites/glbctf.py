# Global bound constrained optimization test suite. 

from pyopus.optimizer.glbctf import *

if __name__=='__main__':
	# Create functions
	functions=[
		Quadratic(30), 
		SchwefelA(30), 
		SchwefelB(30), 
		SchwefelC(30), 
		Rosenbrock(30), 
		Step(30),
		QuarticNoisy(30), 
		SchwefelD(30), 
		Rastrigin(30), 
		Ackley(30), 
		Griewank(30), 
		Penalty1(30), 
		Penalty2(30), 
		ShekelFoxholes(2), 
		Kowalik(4), 
		SixHump(2), 
		Branin(2), 
		GoldsteinPrice(2), 
		Hartman(3), 
		Hartman(6), 
		Shekel(4, m=5), 
		Shekel(4, m=7), 
		Shekel(4, m=10)
	]

	print("")
	# Go through all functions, check global minimum (f value)
	for func in functions:
		xglob=func.xmin
		fglob=func.fmin
		f=func(xglob)
		print(" %30s (%2d): fglob=%14.5e f=%14.5e" % (func.name, func.n, fglob, f))
		
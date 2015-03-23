# Delaying the evaluation of a test function. 

from pyopus.optimizer.base import RandomDelay
from pyopus.optimizer.mgh import Rosenbrock, MGHf, MGHg
from numpy import where, zeros, sqrt
from platform import system
if system()=='Windows':
	# clock() is the most precise timer in Windows
	from time import clock as timer
else:
	# time() is the most precise timer in Linux
	from time import time as timer

def myfunc(x):
	return 2.0*x
	
if __name__=='__main__':
	callable=myfunc
	xini=10.0
	
	# Delay is chonen randomly from [1.0, 2.0] with uniform distribution. 
	delayedCallable=RandomDelay(callable, [1.0, 2.0])
	
	print("Starting evaluation without delay.")
	t1=timer()
	f=callable(xini)
	dt=timer()-t1
	print("Evaluation took %.3fs, f=%e." % (dt, f))
	
	print("\nStarting evaluation with delay.")
	t1=timer()
	f=delayedCallable(xini)
	dt=timer()-t1
	print("Evaluation took %.3fs, f=%e." % (dt, f))
	
	# Calling a test suite function evaluates both f and g. To delay
	# the call to f() or g() method, use wrapper objects. 
	rosenbrockObject=Rosenbrock()
	xini=rosenbrockObject.initial
	# Delaying function evaluation
	delayedF=RandomDelay(MGHf(rosenbrockObject), [1.0, 2.0])
	print("\n\nStarting Rosenbrock f evaluation with delay.")
	t1=timer()
	f=delayedF(xini)
	dt=timer()-t1
	print("Evaluation took %.3fs, f=%e" % (dt, f))
	
	# Delaying gradient evaluation
	delayedG=RandomDelay(MGHg(rosenbrockObject), [1.0, 2.0])
	print("\nStarting Rosenbrock g evaluation with delay.")
	t1=timer()
	g=delayedG(xini)
	dt=timer()-t1
	print("Evaluation took %.3fs, g=%s" % (dt, str(g)))
	
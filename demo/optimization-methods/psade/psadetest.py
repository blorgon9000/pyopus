from pyopus.optimizer.base import RandomDelay, Reporter
from pyopus.optimizer.psade import ParallelSADE
from pyopus.optimizer.glbctf import *
from pickle import dump, load
from numpy import arange, array, random, ones, zeros
from platform import system
if system()=='Windows':
	# clock() is the most precise timer in Windows
	from time import clock as timer
else:
	# time() is the most precise timer in Linux
	from time import time as timer
import sys


# Runs PSADE on function f with nw workers. nw=0 means the run is local (no PVM needed). 
# If nw=0 the run is stopped after maxiter evaluations, otherwise the run stops when
# function value becomes smaller than fstop. 
# If nw<0 maxiter function evaluations at optimum are performed. Useful for timing. 
# Return tuple (number of evaluations, time needed). 
def runFunction(f, maxiter, fstop, nw, xlo, xhi, vm):
	# Is the virtual machine alive? 
	if vm is None:
		if nw>1:
			raise Exception, "No virtual machine. Runs with nw>1 are not possible."
		else:
			vm=None
	
	# Local run for less than minslaves=2 workers (nw=0 and nw=1). 
	opt=ParallelSADE(f, xlo, xhi, debug=0, maxiter=maxiter, fstop=fstop, 
			vm=vm, maxSlaves=nw, minSlaves=2)
	
	# Reporter plugin
	opt.installPlugin(Reporter(onImprovement=False, onIterStep=10000))

	# Reset optimizer
	opt.reset()
	
	# Run and time
	t1=timer()
	opt.run()
	t2=timer()
	
	# Build results tuple (x, f, niter, time)
	results=(opt.x, opt.f, opt.niter, t2-t1)
	
	return results
		
# delay 10m..20m
# Run suite of functions. 
def runSuite(nWorkers, nTrials, delay=None, indices=None, vm=None):
	# Default is all functions
	if indices is None:
		indices=range(len(problems))
	
	# Summaries across trials. 
	fres=zeros((nTrials, len(problems)))
	ni=zeros((nTrials, len(problems)))
	dt=zeros((nTrials, len(problems)))
	
	# Loop - trials
	for trial in range(nTrials):
		# Loop - functions
		for ndx in indices:
			# Seed random generator
			random.seed(trial)
			
			# Get function
			(function, maxiter, fstop) = problems[ndx]
			
			# 0 workers... don't stop at fmin
			# 1 or more workers... stop at fmin
			if nWorkers==0:
				fstop=None
			
			# Install delay from interval delay=[min, max] in function evaluation (if required). 
			if delay is None:
				f=function
			else:
				f=RandomDelay(function, delay)
			
			# Get limits
			xlo=function.xl
			xhi=function.xh
			
			# Run function
			print("\n\nRunning ("+str(ndx+1)+") "+function.name+" : trial "+str(trial+1))
			(xres, fres[trial, ndx], ni[trial, ndx], dt[trial, ndx])=runFunction(f, maxiter, fstop, nWorkers, xlo, xhi, vm)
			print("%2d: %30s %02dD fmin=%-14.5e ni=%6d f=%-14.5e   %.1f iter/s" % 
				(ndx+1, function.name, function.n, function.fmin, ni[trial, ndx], fres[trial, ndx], ni[trial, ndx]*1.0/dt[trial, ndx]))
	
	# Save summary
	summary={
		'indices': indices, 
		'fres': fres, 
		'ni': ni,
		'dt': dt
	}
	dump(summary, open('summary'+str(nWorkers)+'.dat', 'w'))

# Analyzes results stored in .dat files produced by optimization runs. 
def analyze(nWorkers):
	summary=load(open('summary'+str(nWorkers)+'.dat', 'r'))
	
	print("")
	for ndx in summary['indices']:
		(function, maxiter, fstop) = problems[ndx]
		print("%2d: %30s %02dD f min=%9.2e stop=%9.2e  avg=%9.2e %10.2e..%9.2e  OK=%02d/%02d" % \
			(
				ndx+1, function.name, function.n, function.fmin, fstop, 
				summary['fres'][:, ndx].mean(), summary['fres'][:, ndx].min(), summary['fres'][:, ndx].max(), 
				len((summary['fres'][:, ndx]<=fstop).nonzero()[0]), len(summary['fres'][:, ndx])
			)
		)

# Analyzes speedup from results stored in .dat files. 
# nWorkersRef is the number of workers in the run to which we are comparing for speedup. 
def speedup(nWorkers, nWorkersRef=1):
	summary=load(open('summary'+str(nWorkers)+'.dat', 'r'))
	summaryRef=load(open('summary'+str(nWorkersRef)+'.dat', 'r'))
	
	print("")
	# Average iterations, time
	for ndx in summaryRef['indices']:
		# Load reference information on problem. 
		(function, maxiter, fstop) = problems[ndx]
		
		# Process reference run (nWorkersRef). 
		# Count number of trials. 
		allRunsRef=len(summaryRef['ni'][:, ndx])
		# Calculate how many trials reached fstop. 
		okRunsRef=(summaryRef['fres'][:, ndx]<=fstop).nonzero()[0]
		# Calculate mean number of iterations for those who reached fstop. 
		meanIterRef=(summaryRef['ni'][okRunsRef, ndx]).mean()
		# Calculate mean run time for those who reached fstop. 
		meanTimeRef=(summaryRef['dt'][okRunsRef, ndx]).mean()
		
		# Do the same for the run with nWorkers. 
		allRuns=len(summary['ni'][:, ndx])
		okRuns=(summary['fres'][:, ndx]<=fstop).nonzero()[0]
		meanIter=(summary['ni'][okRuns, ndx]).mean()
		meanTime=(summary['dt'][okRuns, ndx]).mean()
		
		# Calculate speedup. 
		speedup=meanTimeRef/meanTime
		
		# Calculate parallel efficiency (actual/theoretical speedup). 
		# Expected to be around 1. <1 means bad parallel algorithm. 
		# >1 is also possible if parallel approach is more efficient than serial approach. 
		# >1 can result from random fluctuations caused by the asynchronous nature of the algorithm. 
		efficiency=speedup/(nWorkers/nWorkersRef)
		
		print("%2d: %30s %02dD %2dW : ni=%5d t=%8.2e %02d/%02d : %2dW ni=%5d t=%8.2e %2d/%2d : %5.2f e=%.3f" % \
			(
				ndx+1, function.name, function.n, 
				nWorkers, meanIter, meanTime, len(okRuns), allRuns, 
				nWorkersRef, meanIterRef, meanTimeRef, len(okRunsRef), allRunsRef, 
				speedup, efficiency
			)
		)

			
problems=[
		(Quadratic(30), 100000, 1e-10), 		# 1
		(SchwefelA(30), 100000, 1e-2), 			# 2
		(SchwefelB(30), 100000, 1e-3), 			# 3
		(SchwefelC(30), 100000, 0.5), 			# 4
		(Rosenbrock(30), 100000, 20), 			# 5
		(Step(30), 100000, 0.0), 				# 6
		(QuarticNoisy(30), 100000, 1e-2), 		# 7
		(SchwefelD(30), 100000, -12569.45), 	# 8
		(Rastrigin(30), 100000, 1e-10), 		# 9
		(Ackley(30), 100000, 1e-8), 			# 10
		(Griewank(30), 100000, 1e-10), 			# 11
		(Penalty1(30), 100000, 1e-10), 			# 12
		(Penalty2(30), 100000, 1e-10), 			# 13
		(ShekelFoxholes(2), 10000, 1.0), 		# 14
		(Kowalik(4), 30000, 3.075e-4), 			# 15
		(SixHump(2), 20000, -1.031628), 		# 16
		(Branin(2), 20000, 0.398), 				# 17
		(GoldsteinPrice(2), 20000, 3.0), 		# 18
		(Hartman(3), 20000, -3.86), 			# 19
		(Hartman(6), 20000, -3.3218), 			# 20
		(Shekel(4, m=5), 20000, -10.153), 		# 21
		(Shekel(4, m=7), 20000, -10.402), 		# 22
		(Shekel(4, m=10), 20000, -10.536)		# 23
]
	
if __name__=='__main__':
	try:
		from pyopus.parallel.pvm import PVM
		vm=PVM(debug=0)
		vm.spawnerBarrier()
	except:
		print("Failed to initialize PVM. Only local runs possible.")
		vm=None
		
	if len(sys.argv)<2:
		print("\nNot enough arguments. Use help to get help.")
		sys.exit(1)
	
	action=sys.argv[1]
	
	if action=="run":
		if len(sys.argv)>=3:
			nw=eval(sys.argv[2])
		else:
			nw=0
		
		if len(sys.argv)>=4:
			ntrials=int(sys.argv[3])
		else:
			ntrials=1
		
		if len(sys.argv)>=5:
			delay=eval(sys.argv[4])
		else:
			delay=None
			
		if len(sys.argv)>=6:
			indices=eval(sys.argv[5])
		else:
			indices=None
		
		runSuite(nw, ntrials, delay, indices, vm)
	elif action=="analyze":
		if len(sys.argv)>=3:
			nw=int(sys.argv[2])
		else:
			nw=0
			
		analyze(nw)
	elif action=="speedup":
		if len(sys.argv)>=3:
			nw=int(sys.argv[2])
		else:
			nw=1
		
		if len(sys.argv)>=4:
			nwr=int(sys.argv[3])
		else:
			nwr=1
		
		speedup(nw, nwr)
	else:
		if action!="help":
			print("Bad option.")
		
		print("""
Syntax: 
  psadetest.py run Nworkers Ntrials delayExpr indicesExpr
    Runs witn given number of workers and trials, 
    saves results to summaryNworkers.dat
      Default: Ntrials=1, Nworkers=0
      Nworkers=None fills all free slots in the VM
    delayExpr evaluates to a list of two floats 
	(min and max delay in seconds)
      If ommited or None no delay is inserted. 
    indicesExpr evaluates to a list of integers 
    representing problem numbers. 
      If ommited all problems are run. 
    fstop is used if Nworkers>0. With Nworkers=0 the 
    algorithm runs until maxiter iterations. 
    A realistic delay expression is "[0.000625, 0.00125]"
  	
  psadetest.py analyze Nworkers
    Analyzes summaryNworkers.dat
  	
  psadetest.py speedup Nworkers Nworkers_reference
    Analyzes average speedups for Nworkers wrt Nworkers_reference. 
    Takes into account only those runs where global minimum was 
    found (below fmin). 
""")
			
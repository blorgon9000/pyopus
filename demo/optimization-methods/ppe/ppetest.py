from pyopus.optimizer.base import Reporter, RandomDelay
from pyopus.optimizer.ppe import ParallelPointEvaluator, GenPointSet
from pyopus.optimizer.mgh import Rosenbrock, MGHf
from numpy import array, random, mgrid, concatenate
from time import time
import sys

def testalg(f, maxiter, nw, chunkSize, vm):
	# 2D Rosenbrock, evaluate points on a grid covering [-10,10)x[-10,10), 
	# stepx=stepy=0.25
	gr=mgrid[-10:10:0.1,-10:10:0.1]
	xgr=gr[0]
	ygr=gr[1]
	shp=xgr.shape
	# Reshape x and y grid from 2D array to 1D array
	nshp=(xgr.size,1)
	# Create 1D array of points on the grid
	pts=concatenate((xgr.reshape(nshp),ygr.reshape(nshp)),1)
	generator=GenPointSet(pts)
	
	# Other generators
	# generator=GenUniform(array([-10.0, -10.0]), array([10.0, 10.0]))
	# generator=GenNormal(array([0.0, 0.0]), array([10.0, 10.0]))
	
	ppe=ParallelPointEvaluator(f, [generator], maxSlaves=nw, vm=vm, debug=0)
	ppe.installPlugin(Reporter())
	
	ppe.maxiter=maxiter
	ppe.chunkSize=chunkSize
	
	# No need for reset (no initial point)
	# generator is reset when master loop is entered
	ppe.run()
	
	# Look at internal timer and stats
	print("Evaluations: "+str(ppe.runIter)+" time: "+str(ppe.tRun))
	print(str(ppe.runIter/ppe.tRun)+" evaluations/s")
	

if __name__=='__main__':
	try:
		from pyopus.parallel.pvm import PVM
		vm=PVM(debug=0)
		vm.spawnerBarrier()
	except:
		print("Failed to initialize PVM. Only local runs possible.")
		vm=None

	random.seed(0)
	
	if len(sys.argv)<2:
		print("\nNot enough arguments. Use help to get help.")
		sys.exit(1)
	
	action=sys.argv[1]
	
	if action=="run":
		# Number of workers
		if len(sys.argv)>2:
			nw=eval(sys.argv[2])
		else:
			# Default is all workers
			nw=1000
		
		# Number of iterations
		if len(sys.argv)>3:
			maxiter=int(sys.argv[3])
		else:
			# Default is 1000
			maxiter=1000
		
		# Chunk size
		if len(sys.argv)>4:
			chunkSize=int(sys.argv[4])
		else:
			# Default is 1
			chunkSize=1
		
		# Delay expression
		if len(sys.argv)>5:
			delay=eval(sys.argv[5])
		else:
			delay=None
			
		if delay is not None:
			f=RandomDelay(MGHf(Rosenbrock()), delay)
		else:
			f=MGHf(Rosenbrock())
				
		testalg(f, maxiter, nw, chunkSize, vm)
		
	else:
		if action!="help":
			print("Bad option.")
		
		print("""
Syntax: 
  ppetest.py run Nworkers Maxiter ChunkSize DelayExpr
    Runs with given number of workers, generates Maxiter points from a grid 
	spanning the [-10, 10)x[-10, 10) range with step equal to 0.1.
	
	Maxiter can be at most (20*10)**2=40000. 
	
	The points are processed in chunks of size ChunkSize. 
    
	Every point evaluation takes some time. 
	
	delayExpr=[min,max] is the interval for the random evaluation delay. 

    Default Nworkers=None, Maxiter=1000, ChunkSize=1, DelayExpr=None

    A realistic delay expression is "[0.000625, 0.00125]"
""")
	
# Random delay unif(max/2,max)
# ATHLON XP
# Rosenbrock 2D
# max delay		0.0		0.02	0.01		0.005		0.0025		0.00125
# 
# itps(-1)		3878.6	59.51	103.41		208.64		249.58		249.33
# itps(0)		1581.1	59.32	104.39		176.74		250.04		249.72
# itps(5)		527.18	286.93	445.83		472.78		498.76		503.56

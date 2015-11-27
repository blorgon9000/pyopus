# Dispatches a fixed number of tasks to computational nodes
#  mpirun -n 4 python 03-dispatch.py

from pyopus.parallel.cooperative import cOS
from pyopus.parallel.mpi import MPI
from funclib import jobGenerator, jobProcessor, jobCollector

if __name__=='__main__':
	# Set up MPI
	cOS.setVM(MPI())

	results=cOS.dispatch(
		jobList=((jobProcessor, [value]) for value in xrange(100)), 
		remote=True
	)

	print("Results: "+str(results))

	# Finish, need to do this if MPI is used
	cOS.finalize()

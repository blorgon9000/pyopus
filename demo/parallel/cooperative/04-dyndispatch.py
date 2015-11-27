# Dispatches tasks to computational nodes until specified result is reached or exceeded
# This example also demonstrates the use of a collector. 
#  mpirun -n 4 python 04-dyndispatch.py

from pyopus.parallel.cooperative import cOS
from pyopus.parallel.mpi import MPI
from funclib import dynJobGenerator, jobProcessor, jobCollector

if __name__=='__main__':
	# Set up MPI
	cOS.setVM(MPI())

	# This list will be filled with results
	results=[]

	cOS.dispatch(
		jobList=dynJobGenerator(start=0, step=1), # Start at 0, increase by one
		collector=jobCollector(results, stopAtResult=150), 
		remote=True
	)

	print("Results: "+str(results))

	# Finish, need to do this if MPI is used
	cOS.finalize()

# This demo does the same as the dyndispatch demo, except that a
# custom dispatcher loop is used. This is how asynchronous parallel 
# optimization algorithms like DE and PSADE are implemented. 
#  mpirun -n 4 python 05-asyncloop.py

from pyopus.parallel.cooperative import cOS
from pyopus.parallel.mpi import MPI
from funclib import jobProcessor

# Result at which we stop
stopAtResult=150

# Minimal and maximal number of parallel tasks
# The maximal number of parallel tasks can be infinite (set maxTasks to None)
minTasks=1
maxTasks=1000

if __name__=='__main__':
	# Set up MPI
	cOS.setVM(MPI())

	# Thsi list will hold the jobs (values that are doubled)
	jobs=[]

	# This list will be filled with results
	results=[]

	# Stop the loop
	stop=False

	# Running task status storage
	running={} 

	# Job index of next job
	atJob=0

	# Main loop
	# Run until stop flag set and all tasks are joined
	while not (stop and len(running)==0):
		# Spawn tasks if slots are available and maximal number of tasks is not reached
		# Spawn one task if there are no tasks
		while (
			# Spawn 
			not stop and (
				# no tasks running, need at least one task, spawn
				len(running)==0 or 
				# too few slaves in a parallel environment (number of slots > 0), 
				# force spawn regardless of the number of free slots
				(cOS.slots()>0 and len(running)<minTasks) or 
				# free slots available and less than maximal slaves, spawn
				(cOS.freeSlots()>0 and (maxTasks is None or len(running)<maxTasks)) 
			)
		):
			# Job (value to double)
			job=atJob
			
			# Spawn a global search task
			tid=cOS.Spawn(jobProcessor, args=[job], remote=True, block=True)
			
			print "Spawned task", tid, "for job", job
			
			# Store the job
			running[tid]={
				'index': atJob, 
				'job': job, 
			}
			
			# Go to next job
			atJob+=1
			
		# Join jobs
		tid,result = cOS.Join(block=True).popitem()
		
		print "Received", result, "from", tid
		
		# Get status and remove it from the dictionarz of running jobs
		status=running[tid]
		del running[tid]
		index=status['index']
		
		# Make space for the result
		if index>=len(results):
			results.extend([None]*(index+1-len(results)))
			
		
		# Store result
		results[index]=result
		
		# Check if we need to stop
		if result>=stopAtResult and not stop:
			stop=True
			
			print "Spawning no more tasks"
			
	print("Results: "+str(results))

	# Finish, need to do this if MPI is used
	cOS.finalize()

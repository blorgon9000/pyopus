from pyopus.parallel.cooperative import cOS
from pyopus.parallel.mpi import MPI

__all__=[ 'printMsg', 'printMsgMPI', 'jobGenerator', 'jobProcessor', 'jobCollector' ]

# Prints a message n times, allows a context switch after every printed line
# Context switch takes place at every OS system call. 
def printMsg(msg, n):
	for ii in range(n):
		print(msg+" : "+str(ii))
		cOS.Yield()
	return n

# Same as previous, except that prints the host and the task before the message
def printMsgMPI(msg, n):
	hostID=MPI.hostID()
	taskID=MPI.taskID()
	
	for ii in range(n):
		print("h="+str(hostID)+" t="+str(taskID)+": "+msg+" : "+str(ii))
		cOS.Yield()
	return n

# Process a job (value), return result (multiply value by 2)
def jobProcessor(value):
	hostID=MPI.hostID()
	taskID=MPI.taskID()
	
	print("Processing "+str(value)+ " on "+ str(hostID)+" "+str(taskID))
	return 2*value

# Collect results
def jobCollector(resultStorage, stopAtResult=None):
	global stopFlag
	try:
		while True:
			(index, value, result)=yield
			print("Result for value="+str(value)+" is "+str(result))
			
			# Make space
			if len(resultStorage)<=index:
				resultStorage.extend([None]*(index+1-len(resultStorage)))
			
			resultStorage[index]=result
			
			# This is used only in example 04
			# Set stop flag if stopAt specified and result reaches stopFlag
			# This stops the job generator
			if stopAtResult is not None and result>=stopAtResult and stopFlag is False:
				print "Result", stopAtResult, "reached, stopping generator."
				stopFlag=True
			
	except GeneratorExit:
		print("Collector finished")

# Stop flag
stopFlag=False

# Generate jobs
def dynJobGenerator(start, step=1):
	global stopFlag
	
	# Reset stop flag
	stopFlag=False
	
	ii=start
	while not stopFlag:
		yield (jobProcessor, [ii])
		ii+=step
	
	print "Generator finished."

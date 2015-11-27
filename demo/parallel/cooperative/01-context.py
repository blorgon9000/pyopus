# Context switching

from pyopus.parallel.cooperative import cOS
from funclib import printMsg

if __name__=='__main__':
	# Spawn two tasks
	print("Spawning tasks")
	tidA=cOS.Spawn(printMsg, kwargs={'msg': 'Hello A', 'n': 10})
	tidB=cOS.Spawn(printMsg, kwargs={'msg': 'Hello B', 'n': 20})

	# IDs of running tasks
	running=set([tidA,tidB])
	print("Running tasks: "+str(running))

	# Wait for all tasks to finish
	while len(running)>0:
		# Wait for any task
		retval=cOS.Join()
		# Wait for specified tasks
		# retval=cOS.Join(running)
		
		# Remove IDs of finished tasks
		for tid in retval.keys():
			print("Task "+str(tid)+" finished, return value: "+str(retval[tid]))
			running.remove(tid)

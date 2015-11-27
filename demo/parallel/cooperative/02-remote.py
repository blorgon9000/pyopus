# Outsources tasks, run this example as
#  mpirun -n 4 python 02-remote.py

from pyopus.parallel.cooperative import cOS
from pyopus.parallel.mpi import MPI
from funclib import printMsgMPI

if __name__=='__main__':
	# Set up MPI
	cOS.setVM(MPI())

	# Spawn two tasks (locally)
	tidA=cOS.Spawn(printMsgMPI, kwargs={'msg': 'Hello A', 'n': 10})
	tidB=cOS.Spawn(printMsgMPI, kwargs={'msg': 'Hello B', 'n': 20})

	# Spawn two remote tasks
	tidC=cOS.Spawn(printMsgMPI, kwargs={'msg': 'Hello C', 'n': 15}, remote=True)
	tidD=cOS.Spawn(printMsgMPI, kwargs={'msg': 'Hello D', 'n': 18}, remote=True)

	# IDs of running tasks
	running=set([tidA,tidB,tidC,tidD])

	# Wait for all tasks to finish
	while len(running)>0:
		# Wait for any task
		retval=cOS.Join()
		# Wait for specified tasks
		# retval=cOS.Join(running)
		
		# Remove IDs of finished tasks
		for tid in retval.keys():
			print("Task: "+str(tid)+" finished, return value: "+str(retval[tid]))
			running.remove(tid)

	# Finish, need to do this if MPI is used
	cOS.finalize()

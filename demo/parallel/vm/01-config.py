# Print statistics

import sys

# Starting a task with mpirun starts multiple identical processes. 
# If MPI is imported then the main program is executed only at slot 0. 
# If not, all slots execute the main program. 
from pyopus.parallel.mpi import MPI as VM

if __name__=='__main__':
	vm=VM(debug=2)
	
	# Print info
	print("---- Master")
	print("Host ID   : "+str(vm.hostID()))
	print("Task ID   : "+str(vm.taskID()))
	print("Parent ID : "+str(vm.parentTaskID()))
	
	# Print hosts and processes
	print("---- Hosts and tasks\n"+vm.formatSpawnerConfig()+"----")
	
	# Print process slot info
	print("Total process slots: "+str(vm.slots()))
	print("Free process slots : "+str(vm.freeSlots()))
	
	vm.finalize()
	
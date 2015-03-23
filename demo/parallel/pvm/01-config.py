from pyopus.parallel.pvm import PVM

if __name__=='__main__':
	vm=PVM(debug=0)
	
	# Is PVM alive?
	if vm.alive():
		# Set up this task as spawner, 60s timeout.  
		vm.spawnerBarrier(timeout=60.0)
		
		# Print info
		print("Host ID   : "+str(vm.hostID()))
		print("Task ID   : "+str(vm.taskID()))
		print("Parent ID : "+str(vm.parentTaskID()))
		
		txt=""
		for hostID in vm.hosts():
			txt+=" "+str(hostID)
		print("Hosts     : "+txt)
		
		# Print hosts and processes
		print("")
		print(vm.formatSpawnerConfig())
		
		# Print process slot info
		print("Total process slots: "+str(vm.slots()))
		print("Free process slots : "+str(vm.freeSlots()))
		
	else:
		print("Something is wrong with the virtual machine.")

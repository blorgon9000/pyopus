from pyopus.parallel.pvm import PVM
import funclib
import os

if __name__=='__main__':
	vm=PVM(debug=2)
	
	# Is PVM alive
	if vm.alive():
		# Set up this task as spawner, 60s timeout.  
		vm.spawnerBarrier(timeout=60.0) 

		# Clean up local storage on all machines in the virtual machine. 
		print("\nCleaning up.")
		vm.clearLocalStorage(timeout=6.0)
		
	else:
		print("Something is wrong with the virtual machine.")

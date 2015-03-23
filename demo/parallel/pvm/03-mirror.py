from pyopus.parallel.pvm import PVM
import funclib
import os

if __name__=='__main__':
	vm=PVM(debug=2)
	
	# Is PVM alive
	if vm.alive():
		# Set up this task as spawner, 60s timeout.  
		vm.spawnerBarrier(timeout=60.0) 
		
		# Startup dir must be the same as the one where funclib is located 
		# so we can import it (funclib is not in PYTHONPATH). 
		# Mirror current dir on spawner to workers local storage. 
		# Startupdir is by default the local storage dir. 
		vm.setSpawn(mirrorMap={'*':'.'})
		
		# Spawn 1 task anywhere, send vm as argument with name 'vm'.  
		# The spawned function must be defined in an importable module outside main .py file. 
		# Print some status information and local storage layout. 
		print("\nSpawning task.")
		taskIDs=vm.spawnFunction(funclib.helloLs, kwargs={'vm': vm}, count=1)
		print("Spawned: "+funclib.idList2str(taskIDs))
		print("Collecting stdout ...")
		
		# Receive any message from anyone so we collect the forwarded stdout from workers. 
		# No message is sent so we actually wait for 2s timeout and collect workers' stdout. 
		funclib.receiveMessages(vm, 2.0)
		
	else:
		print("Something is wrong with the virtual machine.")

# Demonstrates file mirroring

import sys
from pyopus.parallel.mpi import MPI as VM
	
import funclib
import os

if __name__=='__main__':
	# Startup dir must contain funclib so we can import it on a worker 
	# (funclib is not in PYTHONPATH). 
	# Mirror current dir on spawner to workers local storage. 
	# Startupdir is by default the created local storage dir. 
	vm=VM(mirrorMap={'*':'.'}, debug=2)
	
	# Spawn 1 task anywhere, send vm as argument with name 'vm'.  
	# The spawned function must be defined in an importable module outside main .py file. 
	# Print some status information and local storage layout. 
	print("\nSpawning task.")
	taskIDs=vm.spawnFunction(funclib.helloLs, kwargs={'vm': vm}, count=1)
	print taskIDs
	print("Spawned: "+str(taskIDs[0]))
	print("Collecting stdout ...")
	
	# Wait for a message, e.g. TaskExit
	vm.receiveMessage()
	
	vm.finalize()

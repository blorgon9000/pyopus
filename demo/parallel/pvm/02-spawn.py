from pyopus.parallel.pvm import PVM
import funclib 
import os, time

if __name__=='__main__':
	vm=PVM(debug=2)
	
	# Is PVM alive
	if vm.alive():
		# Set up this task as spawner, 60s timeout.  
		vm.spawnerBarrier(timeout=60.0)
		
		# Get host list. 
		hostIDs=vm.hosts()
		print("Hosts: "+funclib.idList2str(hostIDs))
		print("Free slots: "+str(vm.freeSlots()))
		
		# Startup dir must be the same as the one where funclib is located 
		# so we can import it (funclib is not in PYTHONPATH). 
		vm.setSpawn(startupDir=os.getcwd())
		
		
		# Spawn 2 tasks anywhere, send vm as argument with name 'vm'.  
		# The spawned function must be defined in an importable module outside main .py file. 
		print("\nSpawning 2 tasks, anywhere.")
		taskIDs=vm.spawnFunction(funclib.hello, kwargs={'vm': vm}, count=2)
		print("Spawned: "+funclib.idList2str(taskIDs))
		print("Collecting stdout ...")
		
		# Receive any message from anyone so we collect the forwarded stdout from workers. 
		# No message is sent so we actually wait for 2s timeout and collect workers' stdout. 
		funclib.receiveMessages(vm, 2.0)
		print("Free slots: "+str(vm.freeSlots()))
		
		# Spawn tasks that fill all free slots on given host, send vm as argument with name 'vm'.  
		# The spawned function must be defined in an importable module outside main .py file. 
		hostID=hostIDs[0]
		print("\nSpawning tasks on host "+str(hostID)+". Fill all free slots.")
		taskIDs=vm.spawnFunction(funclib.hello, kwargs={'vm': vm}, targetList=[hostID])
		print("Spawned: "+funclib.idList2str(taskIDs))
		print("Collecting stdout ...")
		
		# Receive any message from anyone so we collect the forwarded stdout from workers. 
		# No message is sent so we actually wait for 2s timeout and collect workers' stdout. 
		funclib.receiveMessages(vm, 2.0)
		print("Free slots: "+str(vm.freeSlots()))
		
	else:
		print("Something is wrong with the virtual machine.")

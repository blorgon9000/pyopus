# Spawns a remote task, handles messages (collects results, detects exit)

import sys
from pyopus.parallel.mpi import MPI as VM
# By default stdout is forwarded to the mpirun terminal

import funclib 
from pyopus.parallel.base import MsgTaskExit, MsgTaskResult
import os, time

if __name__=='__main__':
	# Startup dir must be the same as the one where funclib is located 
	# so we can import it (funclib is not in PYTHONPATH). 
	# MPI guarantees this by default, while PVM does not. 
	vm=VM(startupDir=os.getcwd(), debug=2)
	
	# Get host list. 
	hostIDs=vm.hosts()
	initialFreeSlots=vm.freeSlots()
	print("Hosts: ")
	for hostID in hostIDs:
		print("  "+str(hostID))
	print("Free slots: "+str(initialFreeSlots))
	
	# Spawn 2 tasks anywhere, send vm as argument with name 'vm'.  
	# The spawned function must be defined in an importable module outside main .py file. 
	print("\nSpawning 2 tasks, anywhere.")
	taskIDs=vm.spawnFunction(funclib.hello, kwargs={'vm': vm}, count=2)
	print("Spawned: ")
	for task in taskIDs:
		print "  ", str(task)
	print("Free slots: "+str(vm.freeSlots())+"\n")
	
	print("----\n"+vm.formatSpawnerConfig()+"----")
	
	# Blocking receive 4 messages (2 return values and 2 exit)
	while vm.freeSlots()!=initialFreeSlots:
		received=vm.receiveMessage()
		
		# Handle error (None) and timeout (empty tuple)
		if received is None or len(received)==0:
			continue
		
		# Unpack
		(fromId, msg)=received
		
		# Note that the received message may be comming from a dead worker. 
		# Verify that the message is comming from one of our workers. 
		if fromId not in taskIDs:
			continue
		
		# Handle message
		if type(msg) is MsgTaskResult:
			print("Received from "+str(fromId)+" TaskResult success="+str(msg.success)+"\n  "+str(msg.returnValue))
		elif type(msg) is MsgTaskExit:
			print("Received from "+str(fromId)+" TaskExit")
		
		print("Free slots: "+str(vm.freeSlots())+"\n")
	
	print("----\n"+vm.formatSpawnerConfig()+"----")
	
	vm.finalize()

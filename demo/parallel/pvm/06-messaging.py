from pyopus.parallel.pvm import PVM
from pyopus.parallel.base import MsgTaskExit, MsgTaskResult

import funclib
import os, time

if __name__=='__main__':
	vm=PVM(debug=0)
	
	# Is PVM alive
	if vm.alive():
		# Set up this task as spawner, 60s timeout.  
		vm.spawnerBarrier(timeout=60.0) 
		
		# Get hosts, find a non-local host
		myHostID=vm.hostID()
		for hostID in vm.hosts():
			if hostID!=myHostID:
				break
		
		# See if we have at least one remote host. 
		if hostID==myHostID:
			print("\nWarning. Measuring local communication speed.")
		
		# Set work direcotry on worker to be the same as on the spawner. 
		vm.setSpawn(startupDir=os.getcwd())
		
		# Prepare data sizes
		dataSizes=[0, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
		
		# Spawn bounceBack()
		taskIDs=vm.spawnFunction(funclib.bounceBack, kwargs={'vm': vm}, count=1)
		
		# Check if it succeeded
		if len(taskIDs)<1:
			print("Failed to spawn bounceBack().")
			exit(-1)
			
		taskID=taskIDs[0]
		
		print("\nMeasuring message delivery time and data throughput to "+str(hostID)+".")
		
		# Go through data sizes
		total_time=0
		for dataSize in dataSizes:
			# Create data
			data="a"*dataSize
			
			# One send and receive to get things started
			vm.sendMessage(taskID, data)
			(srcID, msg)=vm.receiveMessage()
			
			# How many times do we need to cycle send/receive for runtime=1s? 
			# Initially use 1000 repeats. 
			if total_time>0 and oldDataSize>0:
				# Calculate new repeats for 1 secons run
				repeats=int(repeats/total_time*1.0*oldDataSize/dataSize)
				if repeats==0:
					repeats=1
			else:
				repeats=1000
			# Send and receive
			mark=time.time()
			for count in range(repeats):
				vm.sendMessage(taskID, data)
				(srcID, msg)=vm.receiveMessage()
			total_time=time.time()-mark
			dt=total_time/2.0/repeats
			oldDataSize=dataSize
			
			# Print result
			print("Data size %9.2fkB, iterations=%5d, time=%7.2fms, speed=%9.1fkB/s" % (dataSize/1e3, repeats, dt*1000, dataSize/dt/1e3))
			
		# Send None (will make bounceBack() exit. 
		vm.sendMessage(taskID, None)
		
		# Wait for MsgTaskExit message
		while True:
			(src, msg)=vm.receiveMessage()
			if type(msg) is MsgTaskExit:
				break
		
	else:
		print("Something is wrong with the virtual machine.")

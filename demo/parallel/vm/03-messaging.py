# Measures the message delay and average transfer speed for messages of various sizes 

import sys
from pyopus.parallel.mpi import MPI as VM
from pyopus.parallel.base import MsgTaskExit, MsgTaskResult
import funclib
import os, time
import numpy as np

if __name__=='__main__':
	# Set work direcotry on worker to be the same as on the spawner. 
	vm=VM(startupDir=os.getcwd(), debug=1)
	
	# Get hosts, find a non-local host
	myHostID=vm.hostID()
	
	# Find a remote host
	for hostID in vm.hosts():
		if hostID!=myHostID:
			break
	
	# See if we have at least one remote host. 
	if hostID==myHostID:
		print("\nWarning. Measuring local communication speed.")
	
	# Prepare data sizes
	dataSizes=[0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
	
	# Spawn bounceBack()
	taskIDs=vm.spawnFunction(funclib.bounceBack, kwargs={'vm': vm}, targetList=[hostID], count=1)
	
	# Check if it succeeded
	if len(taskIDs)<1:
		print("Failed to spawn bounceBack().")
		exit(-1)
		
	taskID=taskIDs[0]
	
	print "Task layout:"
	print vm.formatSpawnerConfig()
	
	print("Measuring message delivery time and data throughput to "+str(hostID)+".")
	print("Bounce back task: "+str(taskID))
	
	# Go through data sizes
	total_time=0
	for dataSize in dataSizes:
		# Create data
		data=np.random.randint(0, 256, size=dataSize).astype(np.uint8)
		
		# How many times do we need to cycle send/receive for runtime=1s? 
		if total_time>0 and oldDataSize>0:
			# Calculate new repeats for 1 secons run
			repeats=int(repeats/total_time*2.0*oldDataSize/dataSize)
			if repeats==0:
				repeats=1
		else:
			# Initial repeats
			repeats=10000
		
		# Warm up
		for count in range(repeats):
			vm.sendMessage(taskID, data)
			dummy=vm.receiveMessage()
		
		# Time
		mark=time.time()
		for count in range(repeats):
			vm.sendMessage(taskID, data)
			dummy=vm.receiveMessage()
		total_time=time.time()-mark
		
		# Evaluate
		dt=total_time/2.0/repeats
		oldDataSize=dataSize
		tp=dataSize*8/dt
		
		# Print result
		print("Data size %9.3fkB, iterations=%6d, time=%7.0fus, speed=%5.3fMb/s" % (dataSize/1000.0, repeats, dt*1e6, tp/1e6))
		
	# Send None (will make bounceBack() exit. 
	vm.sendMessage(taskID, None)
	
	# Wait for MsgTaskExit message
	while True:
		(src, msg)=vm.receiveMessage()
		
		if type(msg) is MsgTaskExit:
			break
	
	vm.finalize()

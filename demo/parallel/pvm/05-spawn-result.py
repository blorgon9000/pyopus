from pyopus.parallel.pvm import PVM
from pyopus.parallel.base import MsgTaskExit, MsgTaskResult

import funclib
import os

if __name__=='__main__':
	vm=PVM(debug=0)
	
	# Is PVM alive
	if vm.alive():
		# Set up this task as spawner, 60s timeout.  
		vm.spawnerBarrier(timeout=60.0) 

		# Set work directory on worker to be the same as on the spawner. 
		vm.setSpawn(startupDir=os.getcwd())
		
		# Prepare expressions
		exprList=["1+1", "5*5", "2**7", "bla*bla"]
		
		# Create expression to taskID map, initialize values to None
		expr2taskID={}
		expr2taskID.fromkeys(exprList)
		
		# Spawn evaluators that send MsgTaskResult messages with return value (sendBack=True). 
		taskIDList=[]
		for expr in exprList:
			print("Spawning ebvaluator for: "+expr)
			taskIDs=vm.spawnFunction(funclib.pyEvaluator, kwargs={'vm': vm, 'expr': expr}, count=1, sendBack=True)
			if len(taskIDs)>0:
				# Spawn OK
				taskIDList.extend(taskIDs)
				expr2taskID[expr]=taskIDs[0]
		
		# Collect results from successfully spawned workers and wait for them to exit. 
		running=set(taskIDList)
		results={}
		while len(running)>0 and len(results)<len(taskIDList):
			# Receive message, block
			recv=vm.receiveMessage(-1)
			# Process it
			if recv is not None and len(recv)==2:
				(srcID, msg)=recv
				# Process received result
				if type(msg) is MsgTaskExit:
					running.remove(srcID)
				elif type(msg) is MsgTaskResult:
					results[srcID]=msg.returnValue
		
		# Print results
		for (expr, taskID) in expr2taskID.iteritems():
			if taskID is None:
				print("%s=[FAILED TO SPAWN]" % expr)
			else:
				result=results[taskID]
				if result is None:
					print("%s=[EVAL ERROR]" % expr)
				else:
					print("%s=%s" % (expr, str(result)))
	else:
		print("Something is wrong with the virtual machine.")

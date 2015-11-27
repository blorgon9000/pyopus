# Asynchronous spawning of multiple tasks

import sys
from pyopus.parallel.mpi import MPI as VM
	
from pyopus.parallel.base import MsgTaskExit, MsgTaskResult

import funclib
import os

if __name__=='__main__':
	vm=VM(startupDir=os.getcwd(), debug=0)
	
	# Prepare expressions
	exprList=["1+1", "5*5", "bla*bla", "2**7"]
	
	# Create expression to taskID map, initialize values to None
	expr2taskID={}
	expr2taskID.fromkeys(exprList)
	
	# Spawn evaluators that send MsgTaskResult messages with return value (sendBack=True). 
	taskIDList=[]
	taskCount=0
	for expr in exprList:
		print("Spawning evaluator for: "+expr)
		taskIDs=vm.spawnFunction(funclib.pyEvaluator, kwargs={'vm': vm, 'expr': expr}, count=1, sendBack=True)
		if len(taskIDs)>0:
			# Spawn OK
			taskIDList.extend(taskIDs)
			expr2taskID[expr]=taskIDs[0]
			taskCount+=1
			print("  Task ID: %s" % str(taskIDs[0]))
		else:
			taskIDList.append(None)
			print("  Not spawned")
	
	print 
	
	# Collect results from successfully spawned workers and wait for them to exit. 
	running=set(taskIDList)
	results={}
	while len(running)>0 and len(results)<taskCount:
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
	for ii in range(len(exprList)):
		expr=exprList[ii]
		taskID=taskIDList[ii]
		if taskID is not None:
			result=results[taskID]
			if result is None:
				print("%s=[EVAL ERROR]" % expr)
			else:
				print("%s=%s" % (expr, str(result)))
		else:
			print("%s=[NOT SPAWNED]" % expr)
	
	vm.finalize()
	
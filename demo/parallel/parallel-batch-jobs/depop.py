# Runs differential evolution on all problems in glbctf for population 
# sizes from 10 to 100. Every run is repeated 50 times. 
# The runs are performed in parallel by means of PVM library. 
# 
# The cost function history is pickled in files named fhist_fx_py_rz.pck 
# where x is the function index, y is the population size and z is the run 
# number. The file contains a pickled1 1D array of cost function values. 
#
# The summary is pickled in fsummary.pck. It contains a dictionary 
# with the following members: 
# * funcIndices - list of function indices from global test problem suite
# * names - list of function names
# * dims - list of function dimension 
# * populationSizes - population sizes list 
# * finalf - final (best) function value array 
#     first index - function index 
#     second index - population size index 
#     third index - run index
# 
# Every run evaluates 75000 candidates or evolves 1500 generations, depending 
# on the nGen and maxIter settings. 
# Produces a large number of fhist*.pck files (around 60GB) if writeFhist is 
# set to True. 

import os
from numpy import array, zeros, arange
from cPickle import dump
from pyopus.optimizer import glbctf 
from pyopus.parallel.pvm import PVM
from pyopus.parallel.base import MsgTaskExit, MsgTaskResult
import funclib

if __name__=='__main__':
	# Settings 
	funcIndicesList=range(len(glbctf.GlobalBCsuite))
	nRun=50
	nGen=None # or 1500
	maxIter=75000 # or None
	popSizeList=range(10, 101) 
	writeFhist=True
	
	vm=PVM(debug=0)
	
	# Is PVM alive
	if not vm.alive():
		raise Exception, "Something is wrong with the virtual machine."
	
	# Set up this task as spawner, 60s timeout.  
	vm.spawnerBarrier(timeout=60.0) 

	# Set work directory on worker to be the same as on the spawner. 
	vm.setSpawn(startupDir=os.getcwd())
	
	# Prepare results storage
	finalF=zeros((len(funcIndicesList), len(popSizeList), nRun))
	
	# Prepare jobs storage, TaskID is the key, (ifunc, popsize, irun) is the value
	jobs={}
	
	# Where are we just now
	atFunc=0
	atRun=0
	atPopSize=0
	
	# Main loop
	finished=False
	while not (len(jobs)==0 and finished):
		# Hand out jobs
		while vm.freeSlots()>0 and not finished:
			# Prepare 
			prob=glbctf.GlobalBCsuite[funcIndicesList[atFunc]]()
			popSize=popSizeList[atPopSize]
			
			# Spawn
			taskIDs=vm.spawnFunction(funclib.deRun, 
				kwargs={'prob': prob, 'popSize': popSize, 'maxiter': maxIter, 'maxGen': nGen}, 
				count=1, sendBack=True)
			taskID=taskIDs[0]
			
			# Store job
			jobs[taskID]=(atFunc, atPopSize, atRun)
			
			# Debug message
			print "Sent run", atRun+1, " with population size", popSize, " to task", str(taskID)
			
			# Next job
			atRun+=1
			if atRun>=nRun:
				atRun=0
				atPopSize+=1
				if atPopSize>=len(popSizeList):
					atPopSize=0
					atFunc+=1
					if atFunc>=len(funcIndicesList):
						finished=True
		
		# Wait for messages, block
		recv=vm.receiveMessage(-1)
		
		if recv is not None and len(recv)==2:
			# Got something
			(srcID, msg)=recv
			
			# Is it a result?
			if type(msg) is MsgTaskResult:
				# Yes, unpack it
				(f, fHistory)=msg.returnValue
				
				# Get job info, remove from job list
				(iFunc, iPopSize, iRun)=jobs[srcID]
				del jobs[srcID]
				
				# Debug message
				print "Received func %2d, run %2d/%2d, popsize %3d from %s" % \
					(funcIndicesList[iFunc], iRun+1, nRun, popSizeList[iPopSize], str(srcID))
				
				# Store final f
				finalF[iFunc][iPopSize][iRun]=f
				
				# Store f history
				#  and iRun%10==0 # za vsako 10. 
				if writeFhist:
					fp=open("fhist_f%d_p%d_r%d.pck" % (funcIndicesList[iFunc], popSizeList[iPopSize], iRun), "wb")
					dump(fHistory, fp, protocol=-1)
					fp.close()
	
	# Prepare function names
	names=[]
	dims=[]
	for i in funcIndicesList:
		prob=glbctf.GlobalBCsuite[i]()
		names.append(prob.name)
		dims.append(prob.n)
	
	# Store summary
	summary={
		'funcIndices': funcIndicesList, 
		'names': names, 
		'dims': dims, 
		'populationSizes': popSizeList, 
		'finalF': finalF
	}
	fp=open("fsummary.pck", "wb")
	dump(summary, fp, protocol=-1)
	fp.close()
	

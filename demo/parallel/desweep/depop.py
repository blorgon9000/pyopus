# Runs differential evolution on problems in glbc for population 
# sizes from 10 to 100. Every run is repeated 50 times. 
# The runs are performed in parallel by means of MPI library. 
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
#  mpirun -n 8 python depop.py

import os, sys
from numpy import array, zeros, arange
from cPickle import dump
from pyopus.problems import glbc
from pyopus.parallel.cooperative import cOS 
from pyopus.parallel.mpi import MPI
import funclib

# Settings 
funcIndicesList=[0,1,2] # range(len(glbc.GlobalBCsuite))
nRun=3 #50
nGen=None # or 1500
maxIter=75000 # or None
popSizeList=[10, 20] # range(10, 101, 40) 
writeFhist=True

def jobGenerator():
	for atFunc in range(len(funcIndicesList)):
		for atPopSize in range(len(popSizeList)):
			for atRun in range(nRun):
				yield (
					funclib.deRun, 
					[], 
					{
						'prob': glbc.GlobalBCsuite[funcIndicesList[atFunc]](), 
						'popSize': popSizeList[atPopSize], 
						'runIndex': atRun, 
						'maxiter': maxIter, 
						'maxGen': nGen, 
					},
					# Extra data not passed to deRun
					(atFunc, atPopSize, atRun)
				)

def jobCollector(finalF):
	try:
		while True:
			index, job, result = yield
			iFunc, iPopSize, iRun = job[3] # Get extra data
			fBest, fHistory = result
			
			print "Received func=%2d, run=%2d, popsize=%3d "
			
			if writeFhist:
				fp=open("fhist_f%d_p%d_r%d.pck" % (funcIndicesList[iFunc], popSizeList[iPopSize], iRun), "wb")
				dump(fHistory, fp, protocol=-1)
				fp.close()
			
			finalF[iFunc][iPopSize][iRun]=fBest
	except:
		print "Finished"
	
	
	
if __name__=='__main__':
	cOS.setVM(MPI(startupDir=os.getcwd()))
	
	# Prepare results storage
	finalF=zeros((len(funcIndicesList), len(popSizeList), nRun))
	
	# Dispatch jobs
	cOS.dispatch(
		jobList=jobGenerator(), 
		collector=jobCollector(finalF), 
		remote=True 
	)
	
	# Prepare function names
	names=[]
	dims=[]
	for i in funcIndicesList:
		prob=glbc.GlobalBCsuite[i]()
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
	
	cOS.finalize()

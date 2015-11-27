"""
.. inheritance-diagram:: pyopus.optimizer.de
    :parts: 1

**Box constrained differential evolution global optimizer 
(PyOPUS subsystem name: DEOPT)**

Published first in [de]_. 

.. [de] Storn R., Price K.: Differential evolution - a simple and efficient heuristic 
        for global optimization over continuous spaces. Journal of Global Optimization 
        vol. 11, pp. 341--359, 1997.
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import BoxConstrainedOptimizer
from ..parallel.cooperative import cOS

from numpy import array, concatenate, arange, zeros, random, where, isinf
from numpy.random import permutation, random
from time import sleep, time

__all__ = [ 'DifferentialEvolution' ] 


class DifferentialEvolution(BoxConstrainedOptimizer):
	"""
	Box constrained differential evolution global optimizer class
	
	If *debug* is above 0, debugging messages are printed. 
	
	The lower and upper bound (*xlo* and *xhi*) must both be finite. 
	
	*maxGen* is the maximal number of generations. It sets an upper bound on 
	the number of iterations to be the greater of the following two values:
	*maxiter* and :math:`(maxGen+1) \\cdot populationSize`. 
	
	*populationSize* is the number of inidividuals (points) in the population. 
	
	*w* is the differential weight between 0 and 2. 
	
	*pc* is the crossover probability between 0 and 1. 
	
	If *spawnerLevel* is not greater than 1, evaluations are distributed across 
	available computing nodes (that is unless task distribution takes place at 
	a higher level). 
	
	Infinite bounds are allowed. This means that the algorithm behaves as 
	an unconstrained algorithm if lower bounds are :math:`-\\infty` and 
	upper bounds are :math:`+\\infty`. In this case the initial population 
	must be defined with the :meth:`reset` method.
	
	See the :class:`~pyopus.optimizer.base.BoxConstrainedOptimizer` for more 
	information. 
	"""
	def __init__(self, function, xlo, xhi, debug=0, fstop=None, maxiter=None,
			maxSlaves=None, minSlaves=0, 
			maxGen=None, spawnerLevel=1, populationSize=100, w=0.5, pc=0.3):
		
		if maxGen is not None:
			# maxGen puts a limit on maxiter
			# maxIter=max(maxIter, (maxGen+1)*populationSize)
			maxiter=max(maxiter, (maxGen+1)*populationSize)
			
		BoxConstrainedOptimizer.__init__(self, function, xlo, xhi, debug, fstop, maxiter)
		
		# Pupulation size
		self.Np=populationSize
		
		# Differential weight
		self.w=w
		
		# Crossover probability
		self.pc=pc
		
		# Population
		self.population=None
		self.fpopulation=None
		
		# Point status (set of points to be evaluated and points being evaluated)
		# Used only when initial population is being evaluated
		self.pointsForEvaluation=set()
		self.pointsBeingEvaluated=set()
		
		# Which population point to use as next parent. Used by master. 
		self.ip=None
		
		self.maxSlaves=maxSlaves
		self.minSlaves=minSlaves
		self.spawnerLevel=spawnerLevel
		
	def initialPopulation(self):
		"""
		Constructs and returns the initial population. 
		
		Fails if any bound is infinite. 
		"""
		if isinf(self.xlo).any() or isinf(self.xhi).any():
			raise Exception, DbgMsg("DEOPT", "Infinite bounds detected. Need initial population.")
		
		# Random permutations of Np subintervals for every variable
		# One column is one variable (it has Np rows)
		perm=zeros([self.Np, self.ndim])
		for i in range(self.ndim):
			perm[:,i]=(permutation(self.Np))
		
		# Random relative interval coordinates (0,1)
		randNum=random((self.Np, self.ndim))
		
		# Build Np points from random subintervals
		return self.denormalize((perm+randNum)/self.Np)
		
	def generatePoint(self, i):
		"""
		Generates a new point through mutation and crossover. 
		"""
		# Select 3 distinct indices different from i between 0 and np-1
		indices=arange(self.Np)
		indices=concatenate((indices[:i], indices[i:]), axis=0)
		indices=permutation(indices)
		a, b, c = indices[0], indices[1], indices[2]
		
		# Get the point (x)
		x=self.population[i]
		
		# Generate mutant (y)
		y=self.population[a]+self.w*(self.population[b]-self.population[c])
		
		# Place it inside the box
		self.bound(y)
		
		# Generate ndim random numbers from 0..1
		uvec=random(self.ndim)
		
		# Crossover x and y, use components of y with probability self.pc
		yind=where(uvec<self.pc)[0]
		z=x.copy()
		z[yind]=y[yind]
		
		return z
	
	# Generate evaluators for points from the initial population
	def initPopJobGen(self):
		for ii in range(self.population.shape[0]):
			if self.debug:
				DbgMsgOut("DEOPT", "Inital point evaluation #%d" % ii)
			x=self.population[ii,:]
			yield self.getEvaluator(x)
	
	# Handle the result of an initial population point evaluation
	def initPopJobCol(self):
		while True:
			index, job, retval = (yield)
			evf, args = job
			x=args[0]
			if self.debug:
				DbgMsgOut("DEOPT", "Inital point evaluation result received #%d" % index)
			f, annot = retval
			self.newResult(x, f, annot)
			self.fpopulation[index]=f
		
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		BoxConstrainedOptimizer.check(self)
		
		# We require box constraints
		if (self.xlo is None):
			raise Exception, DbgMsg("DEOPT", "Lower bound is not defined.")
		
		if (self.xhi is None):
			raise Exception, DbgMsg("DEOPT", "Upper bound is not defined.")
		
		# Check if constraints are finite
		if (~isfinite(self.xlo)).any() or (~isfinite(self.xhi)).any():
			raise Exception, DbgMsg("DEOPT", "Bounds must be finite.")
		
		# Check population size
		if self.Np<3:
			raise Exception, DbgMsg("DEOPT", "Population must have at least 3 points.")
			
		# Check differential weight
		if self.w<0.0 or self.w>2.0:
			raise Exception, DbgMsg("DEOPT", "Differential weight must be from [0,2] interval.")
			
		# Check crossover probability
		if self.pc<0.0 or self.pc>1.0:
			raise Exception, DbgMsg("DEOPT", "Crossover probability must be from [0,1] interval.")
	
	def reset(self, x0=None):
		"""
		Puts the optimizer in its initial state. 
		
		*x* is eiter the initial point (which must be within bounds and is 
		ignored) or the initial population in the form of a 2-dimensional 
		array or list where the first index is the population member index 
		while the second index is the component index. The initial population 
		must lie within bounds *xlo* and *xhi* and have *populationSize* 
		members. 
		
		If *x* is ``None`` the initial population is generated automatically. 
		In this case *xlo* and *xhi* determine the dimension of the problem. 
		See the :meth:`initialPopulation` method. 
		"""
		if x0 is None:
			# Generate a point within bounds, use xlo and xhi for dimension
			x0=zeros(len(self.xlo))
			self.bound(x0)
		else:
			# Make it an array
			x0=array(x0)
		
		if len(x0.shape)==2:
			# Initial population
			if x0.shape[0]!=self.Np:
				raise Exception, DbgMsg("DEOPT", "Initial population has must have %d members." % self.Np)
			
			# Take first point to get the dimension
			BoxConstrainedOptimizer.reset(self, x0[0])
			
			# Check if the population is within bounds
			if (x0<self.xlo).any() or (x0>self.xhi).any():
				raise Exception, DbgMsg("DEOPT", "Initial population is outside bounds.")
			
			# Set initial population
			self.population=x0.copy()
			self.fpopulation=zeros(self.Np)
		elif len(x0.shape)==1:
			# Use the point to get the dimension of the problem. 
			# Generate initial population. 
			BoxConstrainedOptimizer.reset(self, x0)
			
			# Initialize population
			self.population=self.initialPopulation()
			self.fpopulation=zeros(self.Np)
		else:
			raise Exception, DbgMsg("DEOPT", "Only initial point (1D) or population (2D) can be set.")
		
	def run(self):
		"""
		Run the algorithm. 
		"""
		# Reset stop flag of the Optimizer class
		self.stop=False
				
		# Evaluate initial population in parallel
		if self.debug:
			DbgMsgOut("DEOPT", "Evaluating initial population")
			
		cOS.dispatch(
			jobList=self.initPopJobGen(), 
			collector=self.initPopJobCol(), 
			remote=self.spawnerLevel<=1
		)
		
		# Index of point being processed
		self.ip=0
		
		# Main loop
		tidStatus={} # Status storage
		# Run until stop flag set and all tasks are joined
		while not (self.stop and len(tidStatus)==0):
			# Spawn tasks if slots are available and maximal number of tasks is not reached
			# Spawn one task if there are no tasks
			while (
				# Spawn global search if stop flag not set
				not self.stop and (
					# no tasks running, need at least one task, spawn
					len(tidStatus)==0 or 
					# too few slaves in a parallel environment, force spawn regardless of free slots
					(cOS.slots()>0 and len(tidStatus)<self.minSlaves) or 
					# free slots available and less than maximal slaves, spawn
					(cOS.freeSlots()>0 and (self.maxSlaves is None or len(tidStatus)<self.maxSlaves)) 
				)
			):
				# Generate offspring in round-robin fashion
				
				# We are trying to replace point ip
				z=self.generatePoint(self.ip)
				
				# Prepare evaluator
				evaluator=self.getEvaluator(z)
				
				# Spawn a global search task
				tid=cOS.Spawn(evaluator[0], args=evaluator[1], remote=self.spawnerLevel<=1, block=True)
				
				# Store the job
				tidStatus[tid]={
					'ip': self.ip, 
					'z': z.copy(), 
					'job': evaluator, 
				}
				
				# Go to next parent
				self.ip = (self.ip + 1) % self.Np
				
				if self.debug:
					DbgMsgOut("DEOPT", "Started point evaluation, task "+str(tid))
					
				# If there are no free slots left, stop spawning
				if cOS.freeSlots()<=0:
					break
			
			# Join task
			tid,retval = cOS.Join(block=True).popitem()
			st=tidStatus[tid]
			del tidStatus[tid]
			
			# Get index and point
			ip=st['ip']
			x=st['z']
			
			if self.debug:
				DbgMsgOut("DEOPT", "Received point "+str(ip)+" from task "+str(tid))
		
			# Get result, register it
			f, annot = retval
			self.newResult(x, f, annot)
			
			
			# Store in population if new point is better
			if self.fpopulation[ip]>f:
				self.population[ip]=x
				self.fpopulation[ip]=f
			
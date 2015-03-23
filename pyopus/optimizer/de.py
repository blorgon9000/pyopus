# -*- coding: cp1250 -*-

"""
.. inheritance-diagram:: pyopus.optimizer.de
    :parts: 1

**Box constrained differential evolution global optimizer 
(PyOPUS subsystem name: DEOPT)**

Published first in 

Storn R., Price K.: Differential evolution - a simple and efficient heuristic 
for global optimization over continuous spaces. Journal of Global Optimization 
vol. 11, pp. 341–359, 1997.
"""

#"""
#The algorithm was published in 
#
#Olenšek J., Tuma T., Puhan J., Buermen A.: A new asynchronous parallel 
#global optimization meth od based on simulated annealing and differential 
#evolution. Applied Soft Computing Journal, vol. 11, pp. 1481-1489, 2011. 
#"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import BoxConstrainedOptimizer
from ..parallel.evtdrvms import EventDrivenMS, MsgSlaveStarted, MsgStartup, MsgIdle
from ..parallel.base import Msg, MsgTaskExit

from numpy import array, concatenate, arange, zeros, random, where, isinf
from numpy.random import permutation, random
from time import sleep, time

__all__ = [ 'MsgEvaluatePoint', 'MsgResult', 'DifferentialEvolution' ] 

class MsgEvaluatePoint(Msg):
	"""
	A message requesting the evaluation of a point *x*. 
	"""
	def __init__(self, x):
		Msg.__init__(self)
		self.x=x

class MsgResult(Msg):
	"""
	A message holding the result of the evaluation of a point *x*. 
	
	*f* is the result of the cost function evaluation while *annotations* 
	holds the corresponding annotations. 
	"""
	def __init__(self, x, f, annotations=None):
		Msg.__init__(self)
		self.x=x
		self.f=f
		self.annotations=annotations


class DifferentialEvolution(BoxConstrainedOptimizer, EventDrivenMS):
	"""
	Box constrained differential evolution global optimizer class
	
	If *debug* is above 1, the *debug* option of the :class:`EventDrivenMS` 
	class is set to 1. 
	
	The lower and upper bound (*xlo* and *xhi*) must both be finite. 
	
	*maxGen* is the maximal number of generations. It sets an upper bound on 
	the number of iterations to be the greater of the following two values:
	*maxiter* and :math:`(maxGen+1) \\cdot populationSize`. 
	
	*populationSize* is the number of inidividuals (points) in the population. 
	
	*w* is the differential weight between 0 and 2. 
	
	*pc* is the crossover probability between 0 and 1. 
	
	If a virtual machine object is passed in the *vm* argument the algorithm 
	runs in parallel on the virtual machine represented by *vm*. The algorithm 
	can also run locally (*vm* set to ``None``). 
	
	The algorithm is capable of handling notification messages received when a 
	slave fails or a new host joins the virtual machine. In the first case the 
	work that was performed by the failed slave is reassigned. In the second 
	case new slaves are spawned. The latter is performed by the handler in the 
	:class:`EventDrivenMS` class. Work is assigned to new slaves when the 
	algorithm detects that they are idle. 
	
	Infinite bounds are allowed. This means that the algorithm behaves as 
	an unconstrained algorithm if lower bounds are :math:`-\\infty` and 
	upper bounds are :math:`+\\infty`. In this case the initial population 
	must be defined with the :meth:`reset` method.
	
	See the :class:`~pyopus.optimizer.base.BoxConstrainedOptimizer` and the 
	:class:`~pyopus.parallel.evtdrvms.EventDrivenMS` classes for more 
	information. 
	"""
	def __init__(self, function, xlo, xhi, debug=0, fstop=None, maxiter=None,
					vm=None, maxSlaves=None, minSlaves=0, 
					maxGen=None, populationSize=100, w=0.5, pc=0.3):
		
		if maxGen is not None:
			# maxGen puts a limit on maxiter
			# maxIter=max(maxIter, (maxGen+1)*populationSize)
			maxiter=max(maxiter, (maxGen+1)*populationSize)
			
		BoxConstrainedOptimizer.__init__(self, function, xlo, xhi, debug, fstop, maxiter)
		
		if debug>1:
			debugEvt=1
		else:
			debugEvt=0
			
		# This will call the fillHandlerTable() method of ParallelSADE. 
		EventDrivenMS.__init__(self, vm=vm, maxSlaves=maxSlaves, minSlaves=minSlaves, 
								debug=debugEvt, 
								slaveIdleMessages=True, localIdleMessages=True)

		# Number of variables is the number of dimensions
		
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
	
	# For pickling
	def __getstate__(self):
		state=self.__dict__.copy()
		del state['handler']
		
		return state
	
	# For unpickling
	def __setstate__(self, state):
		self.__dict__.update(state)
		
		self.handler={}
		self.fillHandlerTable()
		
	def fillHandlerTable(self):
		"""
		Fills the handler table of the 
		:class:`~pyopus.parallel.evtdrvms.EventDrivenMS` object 
		with the handlers that take care of the PSADE's messages. 
		"""
		# Parent class' messages
		EventDrivenMS.fillHandlerTable(self)
		
		# Idle message is listed in EventDrivenMS.fillHandlerTable()
		# Here we only override its handler. 
		
		# Other messages (master), allow only from started tasks
		self.addHandler(MsgResult, self.handleResult, self.allowStarted)
				
		# Slave's messages, allow only from all (default)
		self.addHandler(MsgEvaluatePoint, self.handleEvaluatePoint)
		
	#
	# Message handlers
	# 
	
	def handleSlaveStarted(self, source, message): 
		"""
		Handles the MsgSlaveStarted message received by the master from a 
		freshly started slave. Does nothing special, just calls the inherited 
		handler from the :class:`EventDrivenMS` class. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("DEOPT", "Task "+str(source)+" is up.") 
		
		# Call parent's method
		return EventDrivenMS.handleSlaveStarted(self, source, message)
	
	def handleTaskExit(self, source, message):
		"""
		Handles a notification message reporting that a slave has failed 
		(is down). 
		
		Stores the work that was performed at the time the slave failed so it 
		can later be reassigned to some other slave. 
		"""
		# Get ID
		id=message.taskID
		
		# Are we evaluating the initial population
		if len(self.pointsForEvaluation)+len(self.pointsBeingEvaluated)>0:
			# Mark point as unevaluated, remove it from the set of points being evaluated
			taskStorage=self.taskStorage(id)
			if taskStorage is not None and 'ip' in taskStorage: 
				ip=taskStorage['ip']
				self.pointsBeingEvaluated.discard(ip)
				self.pointsForEvaluation.add(ip)
			
			# Debug message
			if self.debug:
				DbgMsgOut("DEOPT", "Task "+str(id)+" down while evaluating initial population. Reassigned point "+str(ip))
			
		# Call parent's method that will remove the task's structures
		EventDrivenMS.handleTaskExit(self, source, message)
	
	def handleStartup(self, source, message): 
		"""
		Handles a :class:`MsgStartup` message received by master/slave when the 
		event loop is entered. 
		
		Performs some initializations on the master and prevents the plugins 
		in the slave from printing messages. 
		"""
		# Are we the master
		if message.isMaster:
			# Master
			# Initialize sets
			self.pointsForEvaluation=set(range(self.Np))
			self.pointsBeingEvaluated=set()
			
			if self.debug:
				DbgMsgOut("DEOPT", "Master started.") 
		else:
			# Slave
			# Make plugins quiet - we are the worker, we only forward stuff to the master
			for plugin in self.plugins:
				plugin.setQuiet(True)
			
			# Debug message
			if self.debug:
				DbgMsgOut("DEOPT", "Slave started.") 
		
		# Call parent's method
		return EventDrivenMS.handleStartup(self, source, message)
	
	def handleIdle(self, source, message):
		"""
		Handles a :class:`MsgIdle` message. 
		
		Distributes work to the slaves. 
		
		In the beginning the initial population is evaluated in parallel 
		(if possible) by sending class:`MsgEvaluatePoint` messages to slaves. 
		
		After the initial population was evaluated the main part of the 
		algorithm starts to produce offspring and evaluates them on slaves. 
		"""
		# Are we in initial mode
		if len(self.pointsForEvaluation)+len(self.pointsBeingEvaluated)>0:
			# Evaluating initial population
			
			# Find an unevaluated point (status 0=unevaluated, 1=in evaluation, 2=evaluated)
			if len(self.pointsForEvaluation)>0:
				# Points available, pop one
				ip=self.pointsForEvaluation.pop()
				
				# Mark it being evaluated
				self.pointsBeingEvaluated.add(ip)
				
				# Note that slave is processing point 'ip'
				self.taskStorage(message.taskID)['ip']=ip
				
				# Take the point
				x=self.population[ip,:]
				
				# Debug message
				if self.debug:
					DbgMsgOut("DEOPT", "Sending initial population point "+str(ip)+" to task "+str(message.taskID))
				
				# Prepare EvaluatePoint message
				return [(message.taskID, MsgEvaluatePoint(x))]
		else:
			# Algorithm running, generate offspring in round-robin fashion
			
			# We are trying to replace point ip
			z=self.generatePoint(self.ip)
			
			# Remember what the worker is evaluating 
			self.taskStorage(message.taskID)['ip']=self.ip
			
			# Proceed to next parent point 
			self.ip+=1
			if self.ip>=self.Np:
				self.ip=0
			
			# If a slave fails the point is simply not evaluated and does 
			# not compete with the original point in the population
			
			# Debug message
			if self.debug:
				DbgMsgOut("DEOPT", "Sending point to task "+str(message.taskID))
			
			# Send point to worker 
			return [(message.taskID, MsgEvaluatePoint(z))]
			
	def handleEvaluatePoint(self, source, message):
		"""
		Handles :class:`MsgEvaluatePoint` messages received by slaves. 
		Evaluates the received point and sends back a :class:`MsgResult` 
		message. 
		"""
		# Evaluate
		x=message.x
		f=self.fun(x)
		return [(source, MsgResult(x, f, self.annotations))]
	
	def handleResult(self, source, message): 
		"""
		Handles a :class:`MsgResult` message and takes care of the evaluation 
		result. 
		
		Marks the slave that sent the message as idle so that new work can be 
		assigned to it by the :class:`MsgIdle` message handler. 
		"""
		x=message.x
		f=message.f
		annotations=message.annotations
		
		ip=self.taskStorage(source)['ip']
			
		if self.debug:
			DbgMsgOut("DEOPT", "Received point "+str(ip)+" from task "+str(source))
			
		# If the message was sent from None (local mode) this stuff got handled in self.fun()
		if source is not None:
			# Handle new result
			self.newResult(x, f, annotations)
		
		# Are we in initial mode
		if len(self.pointsForEvaluation)+len(self.pointsBeingEvaluated)>0:
			# Put the result in fpopulation
			self.fpopulation[ip]=f
		
			# Remove point from the set of points being evaluated
			self.pointsBeingEvaluated.discard(ip)
		
			# Are we done with the initial population
			if len(self.pointsForEvaluation)+len(self.pointsBeingEvaluated)==0:
				# Yes, set the parent point index to 0
				self.ip=0
		elif self.fpopulation[ip]>f:
				# Accept point only if it improves the parent point
				self.population[ip]=x
				self.fpopulation[ip]=f
		
		# Exit loop if stopping condition satisfied
		if self.stop:
			self.exitLoop=True
			
		# Mark slave as idle - end message chain
		self.markSlaveIdle(source, True)
		
		# No response to this message
		return []
	
		
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
		Runs the optimization algorithm. 
		"""
		# Reset stop flag of the Optimizer class
		self.stop=False
		
		# Start master event loop
		EventDrivenMS.masterEventLoop(self)

	

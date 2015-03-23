# -*- coding: cp1250 -*-

"""
.. inheritance-diagram:: pyopus.optimizer.psade
    :parts: 1

**Box constrained parallel SADE global optimizer 
(PyOPUS subsystem name: PSADEOPT)**

SADE stands for Simulated Annealing with Differential Evolution. 

A provably convergent (parallel) global optimization algorithm. 

The algorithm was published in 

Olenšek J., Tuma T., Puhan J., Buermen A.: A new asynchronous parallel 
global optimization meth od based on simulated annealing and differential 
evolution. Applied Soft Computing Journal, vol. 11, pp. 1481-1489, 2011. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import BoxConstrainedOptimizer
from ..parallel.evtdrvms import EventDrivenMS, MsgSlaveStarted, MsgStartup, MsgIdle
from ..parallel.base import Msg, MsgTaskExit

from numpy import array, concatenate, arange, zeros, random, isfinite, log, exp, tan, pi
from numpy.random import permutation, rand
from time import sleep, time

__all__ = [ 'MsgEvaluatePoint', 'MsgEvaluateGlobal', 'MsgEvaluateLocal', 
			'MsgResult', 'MsgGlobalResult', 'MsgLocalResult', 'ParallelSADE' ] 

class MsgEvaluatePoint(Msg):
	"""
	A message requesting the evaluation of a normalized point *x*. 
	"""
	def __init__(self, x):
		Msg.__init__(self)
		self.x=x

class MsgResult(Msg):
	"""
	A message holding the result of the evaluation of a normalized point *x*. 
	
	*f* is the result of the cost function evaluation while *annotations* 
	holds the corresponding annotations. 
	"""
	def __init__(self, x, f, annotations=None):
		Msg.__init__(self)
		self.x=x
		self.f=f
		self.annotations=annotations

class MsgEvaluateGlobal(Msg):
	"""
	A message requesting the evaluation of a global search step. 
	
	*xip* is the parent normalized point, *xi1* is the first of the two 
	normalized points that competed for better T and R parameters. *delta1* and 
	*delta2* are the two differential vectors used by the differential 
	evolution operator. *R* is the range parameter used for generating 
	the random step. *w* and *px* are the differential operator weight and the 
	crossover probability. 
	"""
	def __init__(self, xip, xi1, delta1, delta2, R, w, px):
		Msg.__init__(self)
		self.xip=xip
		self.xi1=xi1
		self.delta1=delta1
		self.delta2=delta2
		self.R=R
		self.w=w
		self.px=px

class MsgEvaluateLocal(Msg):
	"""
	A message requesting the evaluation of a local search step. 
	
	*xa* and *fa* are a normalized point and its corresponding cost function 
	value. *delta* is the search direction. 
	"""
	def __init__(self, xa, fa, delta):
		Msg.__init__(self)
		self.xa=xa
		self.fa=fa
		self.delta=delta

class MsgGlobalResult(MsgResult):
	"""
	A message holding the results of a global step. 
	
	*x* is the evaluated normalized point and *f* is the corresponding cost 
	function value. *annotations* (if not ``None``) holds the annotations 
	corresponding to the evaluated point. 
	"""
	
	def __init__(self, x, f, annotations=None):
		Msg.__init__(self)
		self.x=x
		self.f=f
		self.annotations=annotations
		
class MsgLocalResult(Msg):
	"""
	A message holding the results of a local step. 
	*x* is a tuple holding the evaluated normalized points and *f* is a tuple 
	of corresponding cost function values. *annotations* (if not ``None``) is a 
	tuple holding the annotations corresponding to the evaluated points. All 
	three tuples must be of the same length. 
	"""
	def __init__(self, x, f, annotations=None):
		Msg.__init__(self)
		self.x=x
		self.f=f
		self.annotations=annotations		


class ParallelSADE(BoxConstrainedOptimizer, EventDrivenMS):
	"""
	Parallel SADE global optimizer class
	
	If *debug* is above 1, the *debug* option of the :class:`EventDrivenMS` 
	class is set to 1. 
	
	The lower and upper bound (*xlo* and *xhi*) must both be finite. 
	
	*populationSize* is the number of inidividuals (points) in the population. 
	
	*pLocal* is the probability of performing a local step. 
	
	*Tmin* is the minimal temperature of the annealers. 
	
	*Rmin* and *Rmax* are the lower and upper bound on the range parameter of 
	the annealers. 
	
	*wmin*, *wmax*, *pxmin*, and *pxmax* are the lower and upper bounds for 
	the differential evolution's weight and crossover probability parameters. 
	
	If a virtual machine object is passed in the *vm* argument the algorithm 
	runs in parallel on the virtual machine represented by *vm*. The algorithm 
	can also run locally (*vm* set to ``None``). 
	
	The algorithm is capable of handling notification messages received when a 
	slave fails or a new host joins the virtual machine. In the first case the 
	work that was performed by the failed slave is reassigned. In the second 
	case new slaves are spawned. The latter is performed by the handler in the 
	:class:`EventDrivenMS` class. Work is assigned to new slaves when the 
	algorithm detects that they are idle. 
	
	All operations are performed on normalized points ([0,1] interval 
	corresponding to the range defined by the bounds). 
	
	See the :class:`~pyopus.optimizer.base.BoxConstrainedOptimizer` and the 
	:class:`~pyopus.parallel.evtdrvms.EventDrivenMS` classes for more 
	information. 
	"""
	def __init__(self, function, xlo, xhi, debug=0, fstop=None, maxiter=None, 
					vm=None, maxSlaves=None, minSlaves=0, 
					populationSize=20, pLocal=0.01, Tmin=1e-10, Rmin=1e-10, Rmax=1.0, 
					wmin=0.5, wmax=1.5, pxmin=0.1, pxmax=0.9):
		
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
		
		# Local step probability
		self.pLocal=pLocal
		
		# Minimal temperature
		self.Tmin=Tmin
		
		# Minimal range
		self.Rmin=Rmin
		
		# Maximal range
		self.Rmax=Rmax
		
		# Differential operator weight bounds
		self.wmin=wmin
		self.wmax=wmax
		
		# Crossover probability bounds
		self.pxmin=pxmin
		self.pxmax=pxmax
		
		# Population (normalized points)
		self.population=None
		self.fpopulation=None
		
		# Temperatures and ranges
		self.T=None
		self.R=None
		
		# Differential operator weights and 
		# Crossover probabilities
		self.w=None
		self.px=None
		
		# Indices of temperatures corresponding to population members
		self.indices=None
				
		# Stats
		self.accBetter=None
		self.accWorst=None
		self.rej=None
		self.localAcc=None
		self.localRej=None
		self.parentCount=None
		
		# Point status (set of points to be evaluated and points being evaluated)
		self.pointsForEvaluation=set()
		self.pointsBeingEvaluated=set()
		
		# Which population point to send out next. This is for master's use. 
		self.ip=None
	
	def initialPopulation(self):
		"""
		Constructs and returns the initial population. 
		"""
		# Random permutations of Np subintervals for every variable
		# One column is one variable (it has Np rows)
		perm=zeros([self.Np, self.ndim])
		for i in range(self.ndim):
			perm[:,i]=(permutation(self.Np))
		
		# Random relative interval coordinates (0,1)
		randNum=rand(self.Np, self.ndim)
		
		# Build Np points from random subintervals
		return self.denormalize((perm+randNum)/self.Np)
		
	def initialTempRange(self):
		"""
		Chooses the values of the range and temperature parameters for the 
		annealers. 
		"""
		# Maximal temperature
		Tmax=-(self.fpopulation.max()-self.fpopulation.min())/log(0.9)
		
		# Exponential constants
		cT=1.0/(self.Np-1)*log(Tmax/self.Tmin)
		cR=1.0/(self.Np-1)*log(self.Rmax/self.Rmin)
		cw=1.0/(self.Np-1)*log(self.wmax/self.wmin)
		cpx=1.0/(self.Np-1)*log(self.pxmax/self.pxmin)
		
		# Temperatures
		self.T=Tmax*exp(-cT*arange(self.Np))
		
		# Ranges
		self.R=self.Rmax*exp(-cR*arange(self.Np))
		
		# Differential operator weights
		self.w=self.wmax*exp(-cw*arange(self.Np))
		
		# Crossover probabilities
		self.px=self.pxmax*exp(-cpx*arange(self.Np))
		
		# Stats
		self.accBetter=zeros(self.Np)
		self.accWorse=zeros(self.Np)
		self.rej=zeros(self.Np)
		self.localAcc=0
		self.localRej=0
		self.parentCount=zeros(self.Np)
		
	def contest(self, ip):
		"""
		Performs a contest between two random points in the population for 
		better values of the temperature and range parameter. The first point's 
		index is *ip*. The second point is chosen randomly. 
		"""
		# Select two random points
		rp=permutation(self.Np)
		i1=ip
		i2=rp[0]
		if i2==i1:
			i2=rp[1]
		
		# Function values
		f1=self.fpopulation[i1]
		f2=self.fpopulation[i2]
		
		# Temperature indices
		it1=self.indices[i1]
		it2=self.indices[i2]
		
		# Temperatures
		T1=self.T[it1]
		T2=self.T[it2]

		# Calculate PT
		PT=min(1, exp((f1-f2)*(1/T1-1/T2)))
		
		# Random number, is it lower than PT
		if rand(1)[0]<PT:
			# Yes, swap T, R, w, and px by swapping the indices
			self.indices[i1]=it2
			self.indices[i2]=it1
			
	def selectControlParameters(self):
		"""
		Selects the point (annealer) whose range, temperature, differential 
		operator weight and crossover probability will be used in the global 
		step. 
		
		Returns the index of the point. 
		"""
		# Sort cost function values (lowest first) - get indices
		ndx=self.fpopulation.argsort()
		
		# Selection probabilities
		rank=zeros(self.Np)
		rank[ndx]=arange(self.Np)
		probs=exp(-rank)
		probs/=probs.sum()
		
		# Cumulative probability
		cumprobs=probs.cumsum()
		cumprobs=concatenate((array([0.0]),cumprobs))
		
		# Select random number
		nr=rand(1)[0]
		
		# Find the interval to which it belongs
		inInterval=(cumprobs[:-1]<=nr) & (nr<cumprobs[1:])
		iR=inInterval.nonzero()[0][0]
		
		# Find the corresponding temperature index
		itR=self.indices[iR]
		
		return itR
	
	def generateTrialPrerequisites(self):
		"""
		Generates all the prerequisites for the generation of a trial point. 
		
		Choosed 5 random normalized points (xi1..xi5) from the population. 
		
		Returns a tuple comprising the normalized point xi1, and two 
		differential vectors xi2-xi3, xi4-xi5. 
		"""
		# Generate random permutation
		rp=permutation(self.Np)
		i1=rp[0]
		i2=rp[1]
		i3=rp[2]
		i4=rp[3]
		i5=rp[4]
		
		# Points
		xi1=self.population[i1,:]
		xi2=self.population[i2,:]
		xi3=self.population[i3,:]
		xi4=self.population[i4,:]
		xi5=self.population[i5,:]
		
		return (xi1, xi2-xi3, xi4-xi5)
		
	def generateTrial(self, xip, xi1, delta1, delta2, R, w, px):
		"""
		Generates a normalized trial point for the global search step. 
		
		A mutated normalized point is generated as 

		``xi1 + delta1*w*random1 + delta2*w*random2``
		
		where *random1* and *random2* are two random numbers from the [0,1] 
		interval. 
		
		A component-wise crossover of the mutated point and *xip* is performed 
		with the crossover probability *px*. Then every component of the 
		resulting point is changed by a random value generated from the Cauchy 
		probalility distribution with parameter *R*. 

		Finally the bounds are enforced by selecting a random value between 
		*xip* and the violated bound for every component of the generated point 
		that violates a bound. 
		
		Returns a normalized point. 
		"""
		# Mutated point
		xm=xi1+delta1*w*rand(1)[0]+delta2*w*rand(1)[0]
		
		# Crossover
		mask=(rand(self.ndim)<px)
		indices=mask.nonzero()[0]
		xt=xip.copy()
		xt[indices]=xm[indices]
		
		# Random step (Cauchy)
		xt=xt+R*tan(pi*(rand(self.ndim)-0.5))
		
		# Lower bound violated, fix it
		mask=xt<0.0
		indices=mask.nonzero()[0]
		if len(indices)>0:
			xt[indices]=xip[indices]+rand(len(indices))*(0.0-xip[indices])
		
		# Upper bound violated, fix it
		mask=xt>1.0
		indices=mask.nonzero()[0]
		if len(indices)>0:
			xt[indices]=xip[indices]+rand(len(indices))*(1.0-xip[indices])
		
		return xt
		
	def accept(self, xt, ft, ip, itR):
		"""
		Decides if a normalized point *xt* should be accepted. *ft* is the 
		corresponding cost function value. *ip* is the index of the best point 
		in the population. *itR* is the index of the point (annealer) whose 
		temperature is used in the Metropolis criterion. 
		
		Returns a tuple (*accepted*, *bestReplaced*) where *accepted* is 
		``True`` if the point should be accpeted and *bestReplaced* is ``True`` 
		if accepting *xt* will replace the best point in the population. 
		"""
		
		# w and px adaptation is not implemented 
		
		# Acceptance probability (Metropolis)
		fp=self.fpopulation[ip]
		PM=min(1.0, exp(-(ft-fp)/self.T[itR]))
		
		# Is ip the best point in the population
		ipIsBest=(self.fpopulation[ip]==self.fpopulation.min())
					
		# Test acceptance
		if rand(1)[0]<PM:
			# Are we trying to replace best point with a worse one
			if ft>=fp and ipIsBest:
				# Yes, but won't do it
				pass
			else:
				# Replace parent
				self.population[ip,:]=xt
				self.fpopulation[ip]=ft
				
				# Stats
				if ft<fp:
					self.accBetter[itR]+=1
				else:
					self.accWorse[itR]+=1
				
				return True, ipIsBest
		
		# Stats
		self.rej[itR]+=1
				
		return False, ipIsBest
	
	def localStep(self, xa, fa, d):
		"""
		Performs a local step starting at normalized point *xa* with the 
		corresponding cost function value *fa* in direction *d*. 
		
		The local step is performed with the help of a quadratic model. Two or 
		three additional points are evaluated. 
		
		The return value is a tuple of three tuples. The furst tuple lists the 
		evaluated normalized point, the second one lists the corresponding cost 
		function values and the third one the corresponding annotations. All 
		three tuples must have the same size. 
		
		Returns ``None`` if something goes wrong (like a failure to move a 
		point within bounds). 
		"""
		# Relative position of xb and xb
		db=rand(1)[0]
		xb=xa+d*db
		
		# Force xb inside bounds
		count=0
		while (xb<0.0).any() or (xb>1.0).any():
			db/=2.0
			xb=xa+d*db
			if count>10:
				return None
			count+=1
		
		# Evaluate f(xb), store annotations
		fb=self.fun(self.denormalize(xb))
		ab=self.annotations
		
		# Direction of decrease
		if fb<fa:
			# Origin in xb
			doffs=db
			dc=2*rand(1)[0]*db
		else:
			# Origin in xa
			doffs=0
			dc=-2*rand(1)[0]*db
		
		# Third point
		xc=xa+d*(doffs+dc)
		
		# Force xc inside bounds
		count=0
		while (xc<0.0).any() or (xc>1.0).any():
			dc/=2.0
			xc=xa+d*(doffs+dc)
			if count>10:
				# Giving up
				return ((xb, ), array([fb]), (ab, ))
			count+=1
		
		# Fix dc so that the origin is in xa
		dc=dc+doffs
		
		# Evaluate f(xc)
		fc=self.fun(self.denormalize(xc))
		ac=self.annotations
		
		# Quadratic model
		# dd:  0  db dc
		# f:   fa fb fc
		# f(dd) = coefA * dd^2 + coefB * dd + coefC
		if db==0 or dc==0 or db==dc:
			# Can't calculate model, giving up
			f=zeros(2)
			f[0]=fb
			f[1]=fc
			return ((xb, xc), f, (ab, ac))
			
		coefC=fa
		coefA=((fb-fa)/db-(fc-fa)/dc)/(db-dc)
		coefB=((fb-fa)/db*dc-(fc-fa)/dc*db)/(dc-db)
		
		# Is the model convex
		if coefA>0:
			# Minimum
			dmin=-coefB/(2*coefA)
			xd=xa+d*dmin
		
			# Is minimum inside bounds?
			if (xd<0.0).any() or (xd>1.0).any():
				# Minimum outside bounds
				# Force xd inside bounds
				count=0
				while (xd<0.0).any() or (xd>1.0).any():
					dmin/=2.0
					xd=xa+d*dmin
					if count>10:
						# Giving up
						f=zeros(2)
						f[0]=fb
						f[1]=fc
						return ((xb, xc), f, (ab, ac))
					count+=1
			
			# Evaluate f(xd)
			fd=self.fun(self.denormalize(xd))
			ad=self.annotations
			
			# Return evaluated points
			f=zeros(3)
			f[0]=fb
			f[1]=fc
			f[2]=fd
			return ((xb, xc, xd), f, (ab, ac, ad))
		else:
			# Return evaluated points
			f=zeros(2)
			f[0]=fb
			f[1]=fc
			return ((xb, xc), f, (ab, ac))
	
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
		self.addHandler(MsgGlobalResult, self.handleGlobalResult, self.allowStarted)
		self.addHandler(MsgLocalResult, self.handleLocalResult, self.allowStarted)
				
		# Slave's messages, allow only from all (default)
		self.addHandler(MsgEvaluatePoint, self.handleEvaluatePoint)
		self.addHandler(MsgEvaluateGlobal, self.handleEvaluateGlobal)
		self.addHandler(MsgEvaluateLocal, self.handleEvaluateLocal)
	
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
			DbgMsgOut("PSADEOPT", "Task "+str(source)+" is up.") 
		
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
				DbgMsgOut("PSADEOPT", "Task "+str(id)+" down while evaluating initial population. Reassigned point "+str(ip))
			
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
				DbgMsgOut("PSADEOPT", "Master started.") 
		else:
			# Slave
			# Make plugins quiet - we are the worker, we only forward stuff to the master
			for plugin in self.plugins:
				plugin.setQuiet(True)
			
			# Debug message
			if self.debug:
				DbgMsgOut("PSADEOPT", "Slave started.") 
		
		# Call parent's method
		return EventDrivenMS.handleStartup(self, source, message)
	
	def handleIdle(self, source, message):
		"""
		Handles a :class:`MsgIdle` message. 
		
		Distributes work to the slaves. 
		
		In the beginning the initial population is evaluated in parallel 
		(if possible) by sending class:`MsgEvaluatePoint` messages to slaves. 
		
		After the initial population was evaluated it distributes 
		:class:`MsgEvaluateGlobal` messages to slaves. 
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
					DbgMsgOut("PSADEOPT", "Sending initial population point "+str(ip)+" to task "+str(message.taskID))
				
				# Prepare EvaluatePoint message
				return [(message.taskID, MsgEvaluatePoint(x))]
			else:
				# Nothing more to evaluate
				return []
		else:
			# Algorithm running, request a global search point evaluation
			
			# Contest for better temperature and range parameters
			self.contest(self.ip)
			
			# Choose control parameters
			itR = self.selectControlParameters()
						
			# Get parent point
			xip=self.population[self.ip,:]
			
			# Remember control parameters index and parent index
			taskStorage=self.taskStorage(message.taskID)
			taskStorage['itR']=itR
			taskStorage['ip']=self.ip
			
			# Generate trial point prerequisites
			(xi1, delta1, delta2) = self.generateTrialPrerequisites()
			
			# Go to next parent
			self.ip = (self.ip + 1) % self.Np
			
			# Debug message
			if self.debug:
				DbgMsgOut("PSADEOPT", "Sending global search point to task "+str(message.taskID))
			
			# Generate a point and send it to worker
			return [(message.taskID, MsgEvaluateGlobal(xip, xi1, delta1, delta2, self.R[itR], self.w[itR], self.px[itR]))]
			
	def handleEvaluatePoint(self, source, message):
		"""
		Handles :class:`MsgEvaluatePoint` messages received by slaves during 
		the evaluation of the initial population. Evaluates the received point 
		and sends back a :class:`MsgResult` message. 
		"""
		# Evaluate
		x=message.x
		f=self.fun(self.denormalize(x))
		return [(source, MsgResult(x, f, self.annotations))]
	
	def handleEvaluateGlobal(self, source, message):
		"""
		Handles :class:`MsgEvaluateGlobal` messages reeived by slaves when the 
		algorithm is running. Performs a global step and returns a 
		:class:`MsgGlobalResult` message with the evaluated trial point. 
		"""
		# Generate global search point
		xt=self.generateTrial(message.xip, message.xi1, message.delta1, message.delta2, message.R, message.w, message.px)
		
		# Evaluate
		ft=self.fun(self.denormalize(xt))
		return [(source, MsgGlobalResult(xt, ft, self.annotations))]
		
	def handleEvaluateLocal(self, source, message):
		"""
		Handles :class:`MsgEvaluateLocal` points received by slaves when the 
		algorithm is running. Performs a local step and returns a 
		:class:`MsgLocalResult` message with the evaluated points. If the local 
		step is a failure, the points, cost function values, and annotations 
		are all ``None``. 
		"""
		# Do local search
		xa=message.xa
		fa=message.fa
		delta=message.delta
		
		# Do local step (perform evaluations)
		localResults=self.localStep(xa, fa, delta)
		
		# Send result message
		if localResults is not None:
			return [(source, MsgLocalResult(*localResults))]
		else:
			return [(source, MsgLocalResult(None, None, None))]
	
	def handleResult(self, source, message): 
		"""
		Handles a :class:`MsgResult` message and takes care of the evaluation 
		result. These messages are used only during the evaluation of the 
		initial population. 
		
		Marks the slave that sent the message as idle so that new work can be 
		assigned to it by the :class:`MsgIdle` message handler. 
		"""
		x=message.x
		f=message.f
		annotations=message.annotations
		
		ip=self.taskStorage(source)['ip']
			
		if self.debug:
			DbgMsgOut("PSADEOPT", "Received initial population point "+str(ip)+" from task "+str(source))
			
		# If the message was sent from None (local mode) this stuff got handled in self.fun()
		if source is not None:
			# Handle new result
			self.newResult(self.denormalize(x), f, annotations)
		
		# Put the result in fpopulation
		self.fpopulation[ip]=f
		
		# Remove point from the set of points being evaluated
		self.pointsBeingEvaluated.discard(ip)
		
		# Are we done with the initial population
		if len(self.pointsForEvaluation)+len(self.pointsBeingEvaluated)==0:
			# Yes, set the parent point index to 0
			self.ip=0
			
			# Initialize temperatures and range parameters
			self.initialTempRange()
		
		# Exit loop if stopping condition satisfied
		if self.stop:
			self.exitLoop=True
			
		# Mark slave as idle - end message chain
		self.markSlaveIdle(source, True)
		
		# No response to this message
		return []
	
	def handleGlobalResult(self, source, message): 
		"""
		Handles a :class:`MsgGlobalresult` with the result of a global step. 
		Takes care of accepting the point into teh population and decides 
		whether a local step should be taken. In case a local step is needed it 
		responds with a :class:`MsgEvaluateLocal` message sent to the slave 
		that was performing the global search. 
		
		If no local step is taken marks the slave as idle so that new work can 
		be assigned to it by the :class:`MsgIdle` message handler. 
		"""
		x=message.x
		f=message.f
		annotations=message.annotations
		
		# Debug message
		if self.debug:
			DbgMsgOut("PSADEOPT", "Received global search point from task "+str(source))
		
		# Get parent, temperature, wS, and pS
		taskStorage=self.taskStorage(source)
		ip=taskStorage['ip']
		itR=taskStorage['itR']
			
		# If the message was sent from None (local mode) this stuff got handled in self.fun()
		if source is not None:
			# Handle new result
			self.newResult(self.denormalize(x), f, annotations)
		
		# Accept point
		(accepted, ipIsBest)=self.accept(x, f, ip, itR)
		
		self.parentCount[ip]+=1
		
		# Debug message
		if self.debug and accepted:
			DbgMsgOut("PSADEOPT", "Global search point accepted.")
		
		# Exit loop if stopping condition satisfied
		if self.stop:
			self.exitLoop=True
		
		# Do we want local search
		if accepted or ipIsBest or rand(1)[0]<self.pLocal:
			# Local search
			
			# Debug message
			if self.debug: 
				DbgMsgOut("PSADEOPT", "Starting local search.")
			
			# Choose two random points
			rp=permutation(self.Np)
			i1=rp[0]
			i2=rp[1]
			
			# Points
			xi1=self.population[i1,:]
			xi2=self.population[i2,:]
			
			# Difference vector
			delta=(xi1-xi2)
			
			# Origin
			xa=self.population[ip,:]
			fa=self.fpopulation[ip]
			
			# Send message
			return [(source, MsgEvaluateLocal(xa, fa, delta))]
		else:
			# No local search
			
			# Mark slave as idle - end message chain
			self.markSlaveIdle(source, True)
			
			# No response to this message
			return []
	
	def handleLocalResult(self, source, message): 
		"""
		Handles a :class:`MsgLocalresult` message with the results of a local 
		step. Takes care of accepting the point in the population and marks the 
		slave that sent the message as idle so that new work can be assigned to 
		it by the :class:`MsgIdle` message handler. 
		"""
		f=message.f
		x=message.x
		annotations=message.annotations
		
		# Was local step successfull
		if f is None:
			# Local step failed
			
			# Debug message
			if self.debug:
				DbgMsgOut("PSADEOPT", "Local step failed, task "+str(source))
		else:
			# Handle results
				
			# Debug message
			if self.debug:
				DbgMsgOut("PSADEOPT", "Received local search points from task "+str(source))
			
			# Get parent
			ip=self.taskStorage(source)['ip']
			xip=self.population[ip,:]
			fip=self.fpopulation[ip]
		
			# Sort function values (lowest f last), get indices
			ndx=(f.argsort())[-1::-1]
		
			# If the message was sent from None (local mode) this stuff got handled in self.fun()
			if source is not None:
				for i in ndx:
					# Handle new result
					self.newResult(self.denormalize(x[i]), f[i], annotations[i])
			
			# Is the best point better than parent
			ibest=ndx[-1]
			if f[ibest]<fip:
				# Yes, replace parent
				self.population[ip,:]=x[ibest]
				self.fpopulation[ip]=f[ibest]
				
				self.localAcc+=1
				
				# Debug message
				if self.debug:
					DbgMsgOut("PSADEOPT", "Replacing parent with local step result.")
			else:
				self.localRej+=1
			
		# Mark slave as idle - end message chain
		self.markSlaveIdle(source, True)
		
		# Exit loop if stopping condition satisfied
		if self.stop:
			self.exitLoop=True
			
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
			raise Exception, DbgMsg("PSADEOPT", "Lower bound is not defined.")
		
		if (self.xhi is None):
			raise Exception, DbgMsg("PSADEOPT", "Upper bound is not defined.")
		
		# Check if constraints are finite
		if (~isfinite(self.xlo)).any() or (~isfinite(self.xhi)).any():
			raise Exception, DbgMsg("PSADEOPT", "Bounds must be finite.")
	
	def reset(self, x0=None):
		"""
		Puts the optimizer in its initial state. 
		
		If the initial point *x* is a 1-dimensional array or list, it is 
		ignored. It must, however, match the dimension of the bounds. 
		
		If it is a 2-dimensional array or list the first index is the initial 
		population member index while the second index is the component index. 
		The initial population must lie within bounds *xlo* and *xhi* and have 
		*populationSize* members. 
		
		If *x* is ``None`` the initial population is generated automatically. 
		See the :meth:`initialPopulation` method. 
		"""
		if x0 is None:
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
			self.population=self.normalize(x0.copy())
			
			# Build functiom values vector
			self.fpopulation=zeros(self.Np)
			
			# Build indices
			self.indices=permutation(self.Np)
		elif len(x0.shape)==1:
			BoxConstrainedOptimizer.reset(self, x0)
			
			# Initialize population
			self.population=self.normalize(self.initialPopulation())
			
			# Build functiom values vector
			self.fpopulation=zeros(self.Np)
		
			# Build indices
			self.indices=permutation(self.Np)
		else:
			raise Exception, DbgMsg("PSADEOPT", "Only initial point (1D) or population (2D) can be set.")
			
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Reset stop flag of the Optimizer class
		self.stop=False
		
		# Start master event loop
		EventDrivenMS.masterEventLoop(self)

	

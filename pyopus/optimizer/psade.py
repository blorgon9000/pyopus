# -*- coding: UTF-8 -*-

"""
.. inheritance-diagram:: pyopus.optimizer.psade
    :parts: 1

**Box constrained parallel SADE global optimizer 
(PyOPUS subsystem name: PSADEOPT)**

SADE stands for Simulated Annealing with Differential Evolution. 

A provably convergent (parallel) global optimization algorithm. 

The algorithm was published in [psade]_. 

.. [psade] Olenšek J., Tuma T., Puhan J., Bűrmen Á.: A new asynchronous parallel 
           global optimization meth od based on simulated annealing and differential 
           evolution. Applied Soft Computing Journal, vol. 11, pp. 1481-1489, 2011. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import BoxConstrainedOptimizer, normalizer, denormalizer
from ..parallel.cooperative import cOS

from numpy import array, concatenate, arange, zeros, random, isfinite, log, exp, tan, pi, concatenate, reshape
from numpy.random import permutation, rand
from time import sleep, time

__all__ = [ 'ParallelSADE' ] 


class ParallelSADE(BoxConstrainedOptimizer):
	"""
	Parallel SADE global optimizer class
	
	If *debug* is above 0, debugging messages are printed. 
	
	The lower and upper bound (*xlo* and *xhi*) must both be finite. 
	
	*populationSize* is the number of inidividuals (points) in the population. 
	
	*pLocal* is the probability of performing a local step. 
	
	*Tmin* is the minimal temperature of the annealers. 
	
	*Rmin* and *Rmax* are the lower and upper bound on the range parameter of 
	the annealers. 
	
	*wmin*, *wmax*, *pxmin*, and *pxmax* are the lower and upper bounds for 
	the differential evolution's weight and crossover probability parameters. 
	
	All operations are performed on normalized points ([0,1] interval 
	corresponding to the range defined by the bounds). 
	
	If *spawnerLevel* is not greater than 1, evaluations are distributed across 
	available computing nodes (that is unless task distribution takes place at 
	a higher level). 
	
	See the :class:`~pyopus.optimizer.base.BoxConstrainedOptimizer` for more 
	information. 
	"""
	def __init__(self, function, xlo, xhi, debug=0, fstop=None, maxiter=None, 
		populationSize=20, pLocal=0.01, Tmin=1e-10, Rmin=1e-10, Rmax=1.0, 
		wmin=0.5, wmax=1.5, pxmin=0.1, pxmax=0.9, 
		minSlaves=1, maxSlaves=None, spawnerLevel=1
	):
		
		BoxConstrainedOptimizer.__init__(self, function, xlo, xhi, debug, fstop, maxiter)
		
		if debug>1:
			debugEvt=1
		else:
			debugEvt=0
			
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
		
		self.spawnerLevel=spawnerLevel
		self.minSlaves=minSlaves
		self.maxSlaves=maxSlaves
	
	def initialPopulation(self, Np):
		"""
		Constructs and returns the initial population with *Np* members. 
		"""
		# Random permutations of Np subintervals for every variable
		# One column is one variable (it has Np rows)
		perm=zeros([Np, self.ndim])
		for i in range(self.ndim):
			perm[:,i]=(permutation(Np))
		
		# Random relative interval coordinates (0,1)
		randNum=rand(Np, self.ndim)
		
		# Build Np points from random subintervals
		return self.denormalize((perm+randNum)/Np)
		
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
	
	@classmethod
	def localStep(cls, xa, fa, d, origin, scale, evf, args):
		"""
		Performs a local step starting at normalized point *xa* with the 
		corresponding cost function value *fa* in direction *d*. 
		Runs remotely. 
		
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
		# fb=self.fun(self.denormalize(xb))
		args[0]=denormalizer(xb, origin, scale)
		fb,ab=evf(*args)
		#fb=self.fun()
		#ab=self.annotations
		
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
		args[0]=denormalizer(xc, origin, scale)
		fc,ac=evf(*args)
		# fc=self.fun(self.denormalize(xc))
		# ac=self.annotations
		
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
			args[0]=denormalizer(xd, origin, scale)
			fd,ad=evf(*args)
			# fd=self.fun(self.denormalize(xd))
			# ad=self.annotations
			
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
	
	# Generate evaluators for points from the initial population
	def initPopJobGen(self):
		for ii in range(self.population.shape[0]):
			if self.debug:
				DbgMsgOut("PSADEOPT", "Inital point evaluation #%d" % ii)
			x=self.denormalize(self.population[ii,:])
			yield self.getEvaluator(x)
	
	# Handle the result of an initial population point evaluation
	def initPopJobCol(self):
		while True:
			index, job, retval = (yield)
			evf, args = job
			x=args[0]
			if self.debug:
				DbgMsgOut("PSADEOPT", "Inital point evaluation result received #%d" % index)
			f, annot = retval
			self.newResult(x, f, annot)
			self.fpopulation[index]=f
	
	def run(self):
		"""
		Run the algorithm. 
		"""
		# Reset stop flag of the Optimizer class
		self.stop=False
				
		# Evaluate initial population in parallel
		cOS.dispatch(
			jobList=self.initPopJobGen(), 
			collector=self.initPopJobCol(), 
			remote=self.spawnerLevel<=1
		)
		
		# Set the parent point index to 0
		self.ip=0
			
		# Initialize temperatures and range parameters
		self.initialTempRange()
		
		# Main loop
		tidStatus={} # Status storage
		# Run until stop flag set and all tasks are joined
		while not (self.stop and len(tidStatus)==0):
			# Spawn initial tasks if slots are available and maximal number of tasks is not reached
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
				# Contest for better temperature and range parameters
				self.contest(self.ip)

				# Choose control parameters
				itR = self.selectControlParameters()
							
				# Get parent point
				xip=self.population[self.ip,:]

				# Generate trial point prerequisites
				(xi1, delta1, delta2) = self.generateTrialPrerequisites()

				# Generate trial point
				xt=self.generateTrial(xip, xi1, delta1, delta2, self.R[itR], self.w[itR], self.px[itR])
				
				# Prepare evaluator
				evaluator=self.getEvaluator(self.denormalize(xt))
				
				# Spawn a global search task
				tid=cOS.Spawn(evaluator[0], args=evaluator[1], remote=self.spawnerLevel<=1, block=True)
				
				# Store the job
				tidStatus[tid]={
					'itR': itR, 
					'ip': self.ip, 
					'global': True, 
					'xt': xt.copy(), # normalized point
					'job': evaluator, 
				}
				
				# Go to next parent
				self.ip = (self.ip + 1) % self.Np
				
				if self.debug:
					DbgMsgOut("PSADEOPT", "Started global search, task "+str(tid))
					
				# If there are no free slots left, stop spawning
				if cOS.freeSlots()<=0:
					break

			
			# Join task
			tid,retval = cOS.Join(block=True).popitem()
			st=tidStatus[tid]
			del tidStatus[tid]
			
			# Get stored information
			itR=st['itR']
			ip=st['ip']
				
			# What was it running?
			if st['global']:
				# Global search finished
				evf, args = st['job']
				xdn=args[0] # denormalized point
				f, annot = retval
				xt=st['xt']
				
				if self.debug:
					DbgMsgOut("PSADEOPT", "Received global search result from task "+str(tid))
		
				# Register result
				self.newResult(xdn, f, annot)
				
				# Accept point
				(accepted, ipIsBest)=self.accept(xt, f, ip, itR)
				
				self.parentCount[ip]+=1
				
				# Debug message
				if self.debug and accepted:
					DbgMsgOut("PSADEOPT", "Global search point accepted, isBest=%d" % ipIsBest)
					
				# Do we want local search
				if accepted or ipIsBest or rand(1)[0]<self.pLocal:
					# Local search
			
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
					
					# Spawn a local search task
					tid=cOS.Spawn(
						self.localStep, 
						args=[xa, fa, delta, self.normOrigin, self.normScale, evf, args], 
						remote=self.spawnerLevel<=1, 
						block=True
					)
					
					# Store the job
					st['global']=False
					tidStatus[tid]=st
					
					# Debug message
					if self.debug: 
						DbgMsgOut("PSADEOPT", "Started local search, task "+str(tid))
			else:
				# Local search finished
				if retval is None:
					# Local step failed
					if self.debug:
						DbgMsgOut("PSADEOPT", "Local step failed, task "+str(tid))
				else:
					# Local step OK
					if self.debug:
						DbgMsgOut("PSADEOPT", "Received local search points from task "+str(tid))
				
					# Unpack results
					x,f,annot = retval
					
					# Get parent
					xip=self.population[ip,:]
					fip=self.fpopulation[ip]
					
					# Sort function values (lowest f last), get indices
					ndx=(f.argsort())[-1::-1]
					
					# Register results
					for ii in ndx:
						self.newResult(self.denormalize(x[ii]), f[ii], annot[ii])
		
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
		
		If it is a 2-dimensional array or list the first index is the initial 
		population member index while the second index is the component index. 
		The initial population must lie within bounds *xlo* and *xhi* and have 
		*populationSize* members. 
		
		If the initial point *x0* is a 1-dimensional array or list, Np-1 
		population members are generated. Point *x0* is the Np-th member. 
		See the :meth:`initialPopulation` method. 
		
		If *x0* is ``None`` the Np members of the initial population are 
		generated automatically. 
		"""
		if x0 is None:
			# No initial point
			noInitialPoint=True
			x0=zeros(len(self.xlo))
			self.bound(x0)
		else:
			# Initial point/population given
			noInitialPoint=False
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
			# No initial point or initial vector x0 given
			BoxConstrainedOptimizer.reset(self, x0)
			
			# Initialize population
			ngen=self.Np if noInitialPoint else self.Np-1
			self.population=self.normalize(self.initialPopulation(ngen))
			
			# Add initial point to population
			if not noInitialPoint:
				self.population=concatenate([
					self.population, 
					self.normalize(reshape(x0,(1,self.ndim)))
				])
			
			# Build functiom values vector
			self.fpopulation=zeros(self.Np)
		
			# Build indices
			self.indices=permutation(self.Np)
		else:
			raise Exception, DbgMsg("PSADEOPT", "Only initial point (1D) or population (2D) can be set.")
			

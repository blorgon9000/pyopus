# -*- coding: UTF-8 -*-
"""
.. inheritance-diagram:: pyopus.optimizer.qpmads
    :parts: 1

**Mesh Adaptive Direct Search with quadratic programming (PyOPUS subsystem name: MADSOPT)**

Mesh adaptive direct search based on [mads]_. The 2n version uses orthogonal 
directions [madsort]_. Handles infeasible initial points with the progressive 
barrier approach [madspb]_. 

Supports uniform distribution of the search directions according to [unimads]_. 

Builds and solves quadratic models based on [qpmads]_. 

.. [mads] Audet C., Dennis Jr. J.E., Mesh Adaptive Direct Search Algorithms for 
          Constrained Optimization. SIAM Journal on Optimization, vol. 17, 
          pp. 188-217, 2006. 

.. [madsort] Abramson M.A., Audet C., Dennis Jr. J.E., Le Digabel S.,
          ORTHO-MADS: A Deterministic MADS Instance with Orthogonal Directions.
          SIAM Journal on Optimization, vol. 20, pp.948-966,2009.

.. [madspb] Audet C., Dennis Jr. J.E., A Progressive Barrier for Derivative-Free 
          Nonlinear Programming. SIAM Journal on Optimization, vol. 20, 
          pp. 445-472, 2009. 

.. [unimads] Bűrmen Á., Tuma T., Generating Poll Directions for Mesh Adaptive Direct 
          Search with Realizations of a Uniformly Distributed Random Orthogonal 
          Matrix, Pacific Journal of Optimization, acepted for publication, 2015. 

.. [qpmads] Bűrmen Á., Olenšek J., Tuma T., Mesh Adaptive Direct Search with Second 
          Directional Derivative-Based Hessian Update, Computational Optimization 
          and Applications, DOI: 10.1007/s10589-015-9753-5, 2015. 

Not all member functions are documented because this module is under development. 
"""

# Some conclusions
# - quadratic models affect how high the data profile reaches
# - a larger set of regression points makes the algorithm approach the final value of data profile faster
# - linear update seems equivalent to powell update

from ..misc import ghalton, sobol
from ..misc.debug import DbgMsgOut, DbgMsg
from base import ConstrainedOptimizer
from filter import Filter
from numpy import max, abs, array
import numpy as np
import scipy.linalg
from cvxopt import matrix, solvers
from pprint import pprint

# Uncomment for ipopt
# from mpi4py import MPI
# import ipopt

# Uncomment for slsqp
# import scipy.optimize

__all__=[ 'QPMADS' ]

class IPOPTproblemWrapper(object):
	def __init__(self, g, H, c0, J):
		self.g=g 
		self.H=H 
		self.c0=c0
		self.J=J
	
	def objective(self, x):
		return (
			(self.g.reshape(self.g.size)*x.reshape(self.g.size)).sum()+
			0.5*np.dot(x.reshape((1,self.g.size)), np.dot(self.H, x.reshape((self.g.size,1))))
		)

	def gradient(self, x):
		return self.g.reshape(self.g.size)+np.dot(self.H, x.reshape((self.g.size,1))).reshape(self.g.size)
	
	def constraints(self, x):
		return self.c0.reshape(self.c0.size)+np.dot(self.J, x.reshape((x.size,1))).reshape(self.c0.size)
	
	def jacobian(self, x):
		return self.J



solvers.options['show_progress']=False

__all__ = [ 'QPMADS' ] 

def withinTol(x1, x2, abstol, reltol):
	ax1=np.abs(x1)
	ax2=np.abs(x2)
	tol=np.where(ax1>ax2, ax1, ax2)*reltol
	tol=np.where(tol<abstol, abstol, tol)
	return (np.abs(x2-x1)<=tol).all()


class QPMADS(ConstrainedOptimizer):
	"""
	Mesh Adaptive Direct Search
	
	See the :class:`~pyopus.optimizer.base.ConstrainedOptimizer` for the 
	explanation of *function*, *xlo*, *xhi*, *constraints*, *clo*, *chi*, 
	*fc*, *cache*, *debug*, *fstop*, and *maxiter*. 
	
	*fgH* specifies a function that evaluates the function gradient and 
	Hessian. The function should return them as a tuple with gradient as 
	the first component and the Hessian as the second component. 
	
	*cJ* specifies a function that computes the Jacobian of the nonlinear 
	constraints. The function returns a Numpy array with first index for 
	the constraint and the second index for the optimization variable. 
	
	*scaling* is the scaling of the problem. If it is scalar, it applies 
	to all components of the independent variable. If it is a vector 
	scaling can be specified independently for every component.
	
	*meshBase* is the base for constructing the mesh. Mesh densities are 
	powers of thin number. 
	
	*initialMeshDensity* specifies the density of the initial mesh. 
	The mesh size is obtained by dividing the power of the *meshBase* 
	with this number. 
	
	*stepbase* is the base for constructing the step size. Steps size 
	is computed as some constant times a power of this number. 
	
	*startStep* is the initial step size. The exponents of the initial 
	mesh and step size is chosen in such manner that the initial step 
	closely approximates this number. 
	
	*maxStep* is the upper limit for step size. 
	
	*stopStep* is the step size at which the algorithm stops. 
	
	*generator* is the direction generator method.
	
	* 0 -- OrthoMADS
	* 1 -- QRMADS 
	* 2 -- UniMADS (uniformly distributed directions)
	
	*unifmat* is the matrix construction method for QRMADS and UniMADS. 
	This matrix is used as the input of the QR decomposition. 
	nb is the smallest even number not greater than n. 
	
	* 0 -- n independent random samples from :math:`[0,1)^{nb}` 
	* 1 -- n samples from nb-dimensional Sobol sequence, advance generator by one sample
	* 2 -- 1 sample from nb-dimensional Sobol sequence and n random samples from :math:`[0,1)^{nb}` 
	* 3 -- n+1 samples from nb-dimensional Sobol sequence, 
	  advance generator by one sample
	* 4 -- n+1 independent random samples from :math:`[0,1)^{nb}` 
	* 5 -- one sample from nb(n+1)-dimensional Sobol sequence (take nb rows, n columns)
	* 6 -- one sample from nb(n+1)-dimensional Sobol sequence (take nb rows, n+1 columns)
	
	*qraugment* is the matrix augmentation method for QRMADS. The first 
	column is always transformed with the Box-Muller transformation 
	resulting in a vector that (when normalized) is uniformly distributed on 
	the unit sphere. The remaining columns are transformed in the following 
	manner:
	
	* 0 -- matrix elements transformed to uniform distribution over [-1,1]
	* 1 -- matrix elements transformed to uniform distribution over [0,1]
	* 2 -- matrix elements transformed to normal distribution :math:`N(0,1)`
	* 3 -- same as 2, except that the first column is fixed 
	  (the first element is 1, the remaining elements are 0)
	
	*qraugment* does not affect UniMADS. For UniMADS all matrix elements 
	are transformed to normal distribution :math:`N(0,1)` before QR 
	decomposition takes place. 
	
	*protoset* specifies the prototype set for UniMADS 
	
	* 0 -- regular simplex with n+1 points
	* 1 -- n orthogonal vectors and their negatives
	
	*rounding* specifies the rounding mode for QRMADS and UniMADS. 
	
	* 0 -- round to sphere
	* 1 -- round to hypercube
	
	Two generators are created for generating random/Halton/Sobol sequences. 
	The main generator is used for most iterations, while the auxiliary 
	generator is used for the iterations involving the finest-yet meshes. 
	If *sequencereset* is ``True`` the main generator is reset to the state 
	of the auxiliary generator whenever the finest-yet mesh is reached. 
	
	Setting *boundSlide* to ``True`` sets the components of a point that 
	violate bound to the value of the respective bound. This makes the 
	point feasible. 
	
	Setting *boundSnap* to ``True`` shortens the all steps that result in 
	bound violation to such extent that the resulting point lies on the 
	bound with largest violation. This makes the point feasible. 
	
	Setting *roundOnFinestMeshEntry* to ``True`` rounds the points examined 
	on the finest-yet mesh to the nearest mesh point. Otherwise these 
	points are not rounded. 
	
	*greedy* turns on greedy evaluation in the poll step (stops the poll 
	step as soon as a better point is found).
	
	*speculativePollSearch* enables the speculative step after a 
	successful poll step. 
	
	*fabstol* and *freltol* specify the absolute and the relative error 
	of the computed function value used for estimating the error of the 
	Hessian update. 
	
	*xabstol* and *xreltol* specify the absolute and the relative error 
	of the point position used for estimating the error of the Hessian 
	update. 
	
	*Habstol* and *Hreltol* specify the maximal absolute and relative 
	error of the Hessian update for which the update is applied. Setting 
	them to ``None`` applies the update regardless of the error. 
	
	*HinitialDiag* specifies the diagonal elements of the initial 
	hessian approximation. If this is a scalar all diagonal elements 
	of the initial approximation are identical. 
	
	Setting *model* to ``True`` turns on the construction of the 
	quadratic model of the function and constraints. 
	
	Setting *linearUpdate* to ``True`` turns on Hessian updating based 
	on three collinear points. This type of updating works well with 
	*protoset* set to 1. 
	
	Setting *simplicalUpdate* to ``True`` turns on Hessian updating based 
	on n+2 points. Works well for *protoset* set to 0. 
	
	*powellUpdate* turns on experimental Hessian updating based on 
	minimum Frobenius norm updating developed by Powell. 
	
	*forceRegression* forces the computation of the gradients based 
	on linear regression for the *powellUpdate*. For other updates 
	the gradients are computed using linear regression. 
	
	*boundStepMirroring* turns on mirroring of points that violate 
	bounds. 
	
	*linearUpdateExtend* evaluates an additional expansion point 
	if a point violates the bound. useful with *protoset* set to 1. 
	If direction d violates bounds and -d does not an additional point 
	along direction -2d is evaluated. This results in 3 collinear points 
	that make it possible to apply the *linearUpdate*. 
	
	*rho* specifies the distance within which points are collected for 
	regression. 
	
	*lastDirectionOrdering* enables the ordering of poll steps based on 
	the last successful direction. 
	
	*modelOrdering* enables the ordering of poll steps based on the 
	quadratic model. 
	
	*modelSearch* turns on quadratic model-based search (uses CVXOPT). 
	This search is also referred to as the QP search. 
	
	*beta* is the parameter used in the algorithm for finding a 
	positive definite Hessian. 
	
	*rhoNeg* is the trust region radius used when the Hessian is 
	not positive definite. 
	
	Setting *qpFeasibilityRestoration* to ``True`` shrinks the QP 
	computed step so that it satisfies the model constraints. 
	
	*speculativeModelSearch* turns on the speculative step after 
	a successful QP serach step. 
	
	Setting *hmax* to a nonzero value turns on filter-based point 
	acceptance which makes it possible to swart from an infeasible 
	point that violates the nonlinear constraints. 
	
	If *stepCutAffectsUpdate* is enabled a step cut caused by 
	bound sliding or bound snapping prevents the mesh and the 
	step size from increasing, even when the step is successful. 
	
	If *speculativeStepAffectsUpdate* is enabled a rejected 
	speculative step prevents the mesh and the step size from 
	increasing. 
	
	See the :class:`~pyopus.optimizer.base.Optimizer` class for 
	more information. 
	"""
	def __init__(self, function, xlo=None, xhi=None, 
			constraints=None, clo=None, chi=None, fc=None, cache=False, 
			fgH=None, cJ=None, 
			debug=0, fstop=None, maxiter=None, 
			scaling=1.0, meshBase=4.0, initialMeshDensity=1.0, 
			stepBase=2.0, startStep=1.0, maxStep=np.Inf, stopStep=1e-12, 
			generator=2, qraugment=0, unifmat=0, protoset=1, 
			rounding=1, sequenceReset=False, 
			boundSnap=False, boundSlide=False, 
			roundOnFinestMeshEntry=False, 
			greedy=True, speculativePollSearch=True, 
			fabstol=2**-1075, freltol=2**-53,
			xabstol=2**-1075, xreltol=2**-53, 
			# Habstol=2**-1075, Hreltol=1e-2, 
			Habstol=None, Hreltol=None, 
			HinitialDiag=0.0, 
			model=True, linearUpdate=True, simplicalUpdate=False, powellUpdate=False, 
			forceRegression=True, 
			boundStepMirroring=True, linearUpdateExtend=True, 
			rho=4.0, lastDirectionOrdering=False, modelOrdering=True, 
			modelSearch=True, beta=0.001, rhoNeg=np.Inf, 
			qpFeasibilityRestoration=False, 
			speculativeModelSearch=False, 
			stepCutAffectsUpdate=True, speculativeStepAffectsUpdate=False, 
			hmax=0.0, 
		):
		ConstrainedOptimizer.__init__(
			self, function, xlo, xhi, constraints, clo, chi, fc=fc, 
			debug=debug, fstop=fstop, maxiter=maxiter, nanBarrier=True, cache=cache
		)
		
		# Function gradient and Hessian, constraint Jacobian
		self.fgH=fgH
		self.cJ=cJ
		
		self.extremeBarrierBox=True
		self.generator=generator  # 0 = OrthoMADS, 1 = QRMADS, 2 = UniMADS 
		self.qraugment=qraugment  # QRMADS augmentation: 
								  # 0 = uniform from [-1,1], 
								  # 1 = uniform from [0,1], 
								  # 2 = normal from N(0,1)
								  # 3 = normal from N(0,1), fixed first vector 
		self.unifmat=unifmat      # QRMADS and UniMADS initial matrix columns:  
								  # 0 = n random(n), 
								  # 1 = n Sobol(n)
								  # 2 = 1 Sobol(n) and n random(n)
								  # 3 = (n+1) Sobol(n)
								  # 4 = (n+1) random(n)
								  # 5 = 1 Sobol(n(n+1))
		self.protoset=protoset    # UniMADS: 0 = n+1 regular simplex, 1=2n orthogonal with negatives, 2=n+1 orthogonal with normalized negative sum
		self.rounding=rounding    # QRMADS and UniMADS: 0 = simple, 1 = with hypercube projection
		self.sequenceReset=sequenceReset # True = reset main generator after finest-yet mesh reached
		self.speculativePollSearch=speculativePollSearch
								  # 0 = no, 1 = one step (OrthoMADS), >1 = maximal # of steps
		self.greedy=greedy        # default = True
		
		# OrthoMADS
		# generator=0, sequenceReset=True, qraugment=*, unifmat=*, protoset=*, rounding=*
		
		self.boundSnap=boundSnap # True=snap bounds violations to bounds
		self.boundSlide=boundSlide
		self.scaling=scaling
		self.meshBase=meshBase
		self.initialMeshDensity=initialMeshDensity
		self.stepBase=stepBase
		
		# Initial steo
		self.startStep=startStep
		
		# Maximal step
		self.maxStep=maxStep
		
		self.stopStep=stopStep
		
		# Stopping condition (poll step size)
		self.stopStep=stopStep
		
		# Precision
		self.fabstol=fabstol
		self.freltol=freltol
		self.xabstol=xabstol
		self.xreltol=xreltol
		self.Habstol=Habstol
		self.Hreltol=Hreltol
		
		# Initial Hessian
		self.HinitialDiag=HinitialDiag
		
		# Hessian modification, quadratic programming
		self.model=model
		self.linearUpdate=linearUpdate
		self.simplicalUpdate=simplicalUpdate
		self.powellUpdate=powellUpdate
		self.forceRegression=forceRegression
		self.boundStepMirroring=boundStepMirroring
		self.linearUpdateExtend=linearUpdateExtend
		self.rho=rho
		self.lastDirectionOrdering=lastDirectionOrdering
		self.modelOrdering=modelOrdering
		self.modelSearch=modelSearch
		self.beta=beta
		self.rhoNeg=rhoNeg
		self.speculativeModelSearch=speculativeModelSearch
		self.qpFeasibilityRestoration=qpFeasibilityRestoration
		self.stepCutAffectsUpdate=stepCutAffectsUpdate
		self.speculativeStepAffectsUpdate=speculativeStepAffectsUpdate
		self.roundOnFinestMeshEntry=roundOnFinestMeshEntry
		
		# Filtering (hmax>0 turns on filtering) - infeasible strating points are allowed in this mode
		# hmax is the initial maximal sum of squares of nonlinear constraint violations
		if debug>=2:
			self.filt=Filter(hmax, debug=1)
		else:
			self.filt=Filter(hmax)
		
		# Removed parameters declaration
		# speculativeSteps=1, 
		# modelSearchStepLimit=1, modelSearchShrinkFactor=2.0, 
		
		# Values of removed parameters
		"""
		Parameters that will be removed in future versions:
	
		Model search evaluates multiple trial points. It starts with the 
		step predicted by the QP solver. If it fails the step is scaled 
		with *modelSearchShrinkFactor* and retried. 
		
		*modelSearchStepLimit* specifies the number of trial points in 
		the model search before it gives up and fails. 
		
		*speculativeSteps* specifies the number of speculative steps tried 
		after a successful poll step. Every speculative step is twice longer 
		than the last successful step. 
		"""
		#self.modelSearchStepLimit=modelSearchStepLimit
		#self.modelSearchShrinkFactor=modelSearchShrinkFactor
		#self.speculativeSteps=speculativeSteps    
		self.modelSearchStepLimit=1
		self.modelSearchShrinkFactor=2.0
		self.speculativeSteps=1
			
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		ConstrainedOptimizer.check(self)
		
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		the 1-dimensional array *x0*. The length of the array becomes the 
		dimension of the optimization problem (:attr:`ndim` member). The length  
		of *x* must match that of *xlo* and *xhi*. 
		"""
		
		# Dimension of the problem
		self.ndim=np.array(x0).size
		
		self.scaling=np.array(self.scaling)
		if self.scaling.size==1:
			self.scaling=np.ones(self.ndim)*self.scaling
		
		# Create proto set
		if self.generator==1 or self.generator==2:
			# QRMADS or UniMADS
			if self.protoset==0:
				# n+1
				A=-1.0*np.ones((self.ndim,self.ndim))/self.ndim
				ii=np.arange(self.ndim)
				A[ii,ii]=1.0
				V=np.linalg.cholesky(A)
				self.Vproto=np.hstack((V.T, -V.T.sum(axis=1).reshape((self.ndim,1))))
			elif self.protoset==1:
				# 2n, return only first n
				self.Vproto=np.eye(self.ndim)
			else:
				# n orthogonal + normalized negative sum
				self.Vproto=np.zeros(self.ndim, self.ndim+1)
				self.Vproto[:,:-1]=np.eye(self.ndim)
				self.Vproto[:,-1]=-np.ones(self.ndim)/self.ndim**0.5
			
		# Create Halton sequence generators
		if self.generator>=1:
			# QRMADS, UniMADS
			self.bmDim=int(np.ceil(self.ndim/2.0)*2)
		else:
			# OrthoMADS
			self.bmDim=self.ndim
		
		# Halton generator (only for OrthoMADS)
		if self.generator==0:
			self.halton=ghalton.Halton(self.bmDim)
			self.haltonAlt=ghalton.Halton(self.bmDim)
			# Skip first bmDim values to avoid strongly correlated components
			self.halton.get(self.bmDim)
			self.haltonAlt.get(self.bmDim)		
		
		# Sobol sequence generator (used in QRMADS and UniMADS)
		self.sobol=sobol.Sobol(self.bmDim)
		self.sobolAlt=sobol.Sobol(self.bmDim)
		# Skip first 2 values (all zeros and all 0.5)
		self.sobol.skip(2)
		self.sobolAlt.skip(2)
		
		# Big Sobol sequence generator (QRMADS and UniMADS)
		if self.unifmat>=5:
			self.bsobol=sobol.Sobol(self.bmDim*(self.ndim+1))
			self.bsobolAlt=sobol.Sobol(self.bmDim*(self.ndim+1))
			# Skip first 2 values (all zeros and all 0.5)
			self.bsobol.skip(2)
			self.bsobolAlt.skip(2)
		
		# Random number generator (used in QRMADS and UniMADS)
		# 4n+1 .. 10
		# 2n+1 .. 1 - 
		#         10 - bad QRNS, reasonable MADSMODEL, good QRC 0.8 and QRS 1.0
		#         0 - MADSMODEL almost as good as nomadqp, QRC 0.75, last moment 0.8, QRNS bad
		self.gen=np.random.RandomState(0) # 0 default, 1, 10, 5
		self.genAlt=np.random.RandomState(0) # 0 default, 1, 10, 5

		# Debug message
		if self.debug:
			DbgMsgOut("MADSOPT", "Resetting MADS.")
		
		# Current mesh size exponent
		self.meshExp=int(np.round(np.log(self.startStep)/np.log(self.stepBase)))
		while True:
			self.newMeshStep()
			if self.step>self.maxStep:
				self.meshExp-=1
			else:
				break
		
		# Smallest mesh size exponent
		self.minMeshExp=0
				
		# Generate initial minimal mesh basis (alternative generator)
		self.minMeshSteps=self.generatePollSteps(reduction=True)
		
		# Skip one set of steps (main generator)
		self.generatePollSteps()
		
		# Last successfull direction
		self.pGood=None
		
		# Linear and simplical Hessian update
		if self.linearUpdate or self.simplicalUpdate:
			if np.array(self.HinitialDiag).size==1:
				self.H=np.eye(self.ndim)*self.HinitialDiag
			else:
				self.H=np.diag(np.array(self.HinitialDiag))
			
		if self.simplicalUpdate:
			# Q, R, V, dfc for simplical Hessian update
			self.Q=None
			self.R=None
			self.V=None
			self.dfc=None
		
		# No model
		self.xgo=None
		self.fo=None
		self.co=None
		self.ito=None
		self.modH=None
		self.modg=None
		self.modJ=None
		
		# Point set 
		self.pointSet={}
		
		# Increasing maxPoints improves speed, not final result
		# 2n, 4n is good
		# n, 2n
		# unimadssqp:nomadqp, no extend
		# 4n+1, qrc=23:12  madsmodel=35:35
		# 3n+1  qrc=20:13  madsmodel=31:35
		# 2n+1  qrc=22:12  madsmodel=34:35
		
		# extend all points that violate bounds
		# 4n+1  qrc=23:12  madsmodel=35:35
		
		# extend n/4 points that violate constraints
		# 4n+1  qrc=23:12  madsmodel=35:35
		
		# with simplical update, no extend
		# 2n+1  qrc=22:12  madsmodel=32:35
		# 4n+1  qrc=22:13  madsmodel=34:34
		
		
		# Maximal point set size
		# self.maxPoints=self.ndim+1
		self.maxPoints=2*self.ndim+1
		# self.maxPoints=2*self.ndim
		# self.maxPoints=3*self.ndim+1
		# self.maxPoints=4*self.ndim+1
		
		# Minimal number of points for regression
		self.minPoints=self.ndim+1
		# self.minPoints=2*self.ndim
		
		# Call constructor of parent class to set nc
		ConstrainedOptimizer.reset(self, x0)
		
		# Initialize Powell update structures
		if self.powellUpdate:
			self.pH=[]
			self.pg=[]
			for ii in range(self.nc+1):
				self.pH.append(np.zeros((self.ndim,self.ndim)))
				self.pg.append(np.zeros(self.ndim))
			self.px0=None
			self.pf0=None
			self.pc0=None
			
		# Sort bounds
		self.ndxbge=np.where(self.xlo>-np.inf)[0]
		self.ndxble=np.where(self.xhi<np.inf)[0]
		
		# Sort constraints
		if self.nc>0:
			self.ndxge=np.where((self.clo>-np.inf) & (self.clo!=self.chi))[0]
			self.ndxle=np.where((self.chi<np.inf) & (self.clo!=self.chi))[0]
			self.ndxeq=np.where(np.isfinite(self.chi) & (self.clo==self.chi))[0]
		else:
			self.ndxle=[]
			self.ndxge=[]
			self.ndxeq=[]
			
		# Prepare filter
		self.filt.reset()
	
	def gridRestrain(self, x, avoidOrigin=False):
		"""
		Returns a point restrained to the current grid. 
		If *avoidOrigin* is ``True`` rounds away from the origin. 
		
		The point is a normalized point (i.e. needs to be multiples by self.scaling
		to get the actual step). 
		"""
		if avoidOrigin:
			return np.sign(x)*np.ceil(np.abs(x)/self.mesh)*self.mesh
		else:
			return np.round(x/self.mesh)*self.mesh
	
	def restoreFeasibility(self, x0, d, H=None, g=None, J=None, c=None, boundCheck=True, boundSlide=False, rounding=True):
		"""
		Restore feasibility by shrinking *d* until *x0* + *d* * *self.scaling* 
		  satisfies bounds
		  reduces the model (if *H* and *g* are given)
		  satisfies linearized constraints (if *J* and *c* are given)
		
		Note that the model applies to the normalized step. 
		
		Applies rounding to *d* if *rounding* is set to ``True``. 
		
		Returns final scaled point, the used shrink factor *a*, and a flag 
		indicating sliding was used. 
		"""
		
		# Shrink until model is reduced and constraints are satisfied
		
		# If bounds are still violated shrink further
		
		# Prepare
		agood=0.0
		abad=None
		xgood=x0
		xbad=None
		while True:
			# New value for a
			if abad is not None:
				a=(agood+abad)/2
			else:
				a=1.0
				
			# Step
			if rounding:
				dr=self.gridRestrain(a*d)
			else:
				dr=a*d
				
			x=x0+dr*self.scaling
		
			# Assume the point is OK
			OK=True
			
			# Verify bounds
			slide=False
			if boundCheck and ((x-self.xlo<0).any() or (x-self.xhi>0).any()): 
				if boundSlide:
					x=np.where(x<self.xlo, self.xlo, x)
					x=np.where(x>self.xhi, self.xhi, x)
					dr=(x-x0)/self.scaling
					slide=True
				else:
					OK=False
			
			# Verify model reduction
			if H is not None:
				m=0.5*np.dot(dr.reshape((1,self.ndim)), np.dot(H, dr.reshape((self.ndim,1))))+(g*dr).sum()
				if m>=0:
					OK=False
					#print "Model not reduced"
					#1/0 
			
			# Verify linearized constraints 
			if J is not None:
				lc=np.dot(J, dr.reshape((self.ndim,1))).reshape(self.nc)+c.reshape(self.nc)
				if (lc>self.chi).any() or (lc<self.clo).any():
					OK=False
					#print "Constraints violated in model"
					#1/0 

			if abad is None:
				# This is the first try (a=1.0)
				if OK:
					# All requirements OK, we're done
					return x, a, slide
				else:
					# Store bad point, start bisection
					abad=a
					xbad=x
			elif OK:
				# Bisect, replace good point
				agood=a
				xgood=x
			else:
				# Bisect, replace bad point
				abad=a
				xbad=x
			
			# Are we done
			if (np.abs((agood-abad)*d)<self.mesh/2).all(): 
				if slide:
					return xgood, agood, slide
				else:
					return xgood, agood, slide
				
	def boxMuller(self, v):
		"""
		Box-Muller transformation of a vector / matrix columns. 
		"""
		if len(v.shape)==1:
			v1=v.reshape((v.shape[0], 1))
		else:
			v1=v
		
		# Box Muller method (n normal random numbers)
		# Need an even number of rows 
		x2k=v1[::2,:]
		x2k1=v1[1::2,:]
		
		s=(-2*np.log(x2k))**0.5
		y2k=s*np.cos(2*np.pi*x2k1)
		y2k1=s*np.sin(2*np.pi*x2k1)
		
		v2=np.zeros(v1.shape)
		v2[::2,:]=y2k
		v2[1::2,:]=y2k1
		
		return v2.reshape(v.shape)
	
	def orderSteps(self, p, refDir, reverse=False):
		"""
		Orders the steps in ascending angle order with respect to 
		refDir. 
		
		If *reverse* is `True`, steps are reversed when angle is 
		greater than pi/2. 
		
		Note that *steps* are normalized steps. Actual steps are 
		obtained by multiplying them with *self.scaling*. 
		"""
		
		# Normalization factors
		rn=(refDir**2).sum()**0.5
		pn=(p**2).sum(axis=0)**0.5
		
		# Cosine
		c=np.dot(p.T, refDir.reshape((self.ndim,1))).reshape((p.shape[1]))/rn/pn
		
		# Bound to [-1,1]
		c=np.where(c>1,1,c)
		c=np.where(c<-1,-1,c)
		
		# Select candidates
		pproc=p.copy()
		
		# Handle reversing
		if reverse:
			# If cosine is negative, reverse direction
			revndx=np.where(c<0)[0]
			c[revndx]=-c[revndx]
			pproc[:,revndx]=-pproc[:,revndx]
		
		# Get angle, order from smallest to largest angle
		angle=np.arccos(c)
		ii=np.argsort(angle)
		
		psort=pproc[:,ii]		
		
		return psort
		
	def modelOrderSteps(self, x0, p, H, g, J=None, c=None, reverse=False):
		"""
		Order steps *p* according to quadratic model given by *H* and *g*. 
		
		If *J* and *c* are given linearized constraints of the form 
		clo<=Jx+c<=chi
		are considered as the primary criterion for ordering. 
		
		If *reverse* is ``True`` considers pairs of the forms (p, -p) first. 
		After that the resulnting points are ordered. 
		
		Note that steps are normalized steps. The model applies to normalized 
		steps. Actual steps are obtained by scaling *p* with *self.scaling*. 
		"""
		
		# Model reduction (por p the model is t2+t1, members are values corresponding to steps)
		# WARNING: A really nasty bug removed - added 0.5*
		t2=0.5*(p*np.dot(H, p)).sum(axis=0).reshape((p.shape[1]))
		t1=np.dot(p.T, g.reshape((self.ndim, 1))).reshape((p.shape[1]))
		
		# Constraint violation
		if J is not None and self.nc>0:
			# Violation in direction of positive steps
			# Rows = constraints, columns = steps
			
			# Normalize violations by L1 norm of the corresponding row of J
			cvnorm=(np.abs(J).max(axis=1)).reshape((self.nc,1))
			cvnorm=np.where(cvnorm==0.0, 1.0, cvnorm)
			# Violations of hi constraint (>0 implies violation)
			cvhi=np.dot(J, p)+c.reshape((self.nc,1))-self.chi.reshape((self.nc,1))
			#print
			#print J
			#print cvnorm
			#print cvnorm
			cvhi=cvhi/cvnorm
			#print cvhi
			# Violations of lo constraint (>0 implies violation)
			cvlo=-(np.dot(J, p)+c.reshape((self.nc,1))-self.clo.reshape((self.nc,1)))
			cvlo=cvlo/cvnorm
			# Cumulative violations of both constraints
			cvp=(np.where(cvhi>0, cvhi, 0)**2).sum(axis=0)+(np.where(cvlo>0, cvlo, 0)**2).sum(axis=0)
			# Bounds
			#bvhi=p+((x0-self.xhi)/self.scaling)[:,None]
			#bvlo=-(p+((x0-self.xlo)/self.scaling)[:,None])
			# Add to cumulative violation
			#cvp+=(np.where(bvhi>0, bvhi, 0)**2).sum(axis=0)+(np.where(bvlo>0, bvlo, 0)**2).sum(axis=0)
			if reverse:
				# Now do the same for reverse directions
				cvhi=-np.dot(J, p)+c.reshape((self.nc,1))-self.chi.reshape((self.nc,1))
				cvhi=cvhi/cvnorm
				cvlo=-(-np.dot(J, p)+c.reshape((self.nc,1))-self.clo.reshape((self.nc,1)))
				cvlo=cvlo/cvnorm
				cvn=(np.where(cvhi>0, cvhi, 0)**2).sum(axis=0)+(np.where(cvlo>0, cvlo, 0)**2).sum(axis=0)
				#bvhi=-p+((x0-self.xhi)/self.scaling)[:,None]
				#bvlo=-(-p+((x0-self.xlo)/self.scaling)[:,None])
				#cvn+=(np.where(bvhi>0, bvhi, 0)**2).sum(axis=0)+(np.where(bvlo>0, bvlo, 0)**2).sum(axis=0)
		else:
			return p
		
		# Select candiates and model delta values
		
		# Model delta
		tp=t2+t1
		if reverse:
			# Include reverse directions
			# Model delta for reverse direction
			tn=t2-t1
			if J is not None and self.nc>0:
				# Selector based on model delta and constraint violation
				# First criterion is constraint violation (choose smaller)
				# Second criterion is model delta (choose smaller)
				psel=(cvp<cvn)|((cvp==cvn)&(tp<=tn))
				# Choose between p and -p according to 
				pproc=np.where(psel, p, -p)
				# Pick corresponding model deltas and constraint violations
				mdelta=np.where(psel, tp, tn)
				cv=np.where(psel, cvp, cvn)
			else:
				# Selector based on model delta
				psel=(tp<=tn)
				# Choose between p and -p
				pproc=np.where(psel, p, -p)
				# Pick corresponding model deltas
				mdelta=np.where(psel, tp, tn)
		else:
			# Handle the case without reverse directions
			pproc=p
			mdelta=tp
			if J is not None:
				cv=cvp
				
		# Order selected directions according to 
		# 1) constraint violation
		# 2) model delta
		if J is not None and self.nc>0:
			ii=np.lexsort((mdelta, cv))
		else:
			# Order from smallest to largest mdelta
			ii=np.argsort(mdelta)
		
		psort=pproc[:,ii]
		
		return psort	
		
	def generatePollSteps(self, reduction=False):
		"""
		Generates a scaled and rounded set of poll steps. 
		
		method can be
		  0 -- original OrthoMADS
		  1 -- QRMADS
		  2 -- UniMADS
		  3 -- 2 opposite vectors
		"""
		if self.generator==1 or self.generator==2: 
			# UniMADS and QRMADS
			# Build initial matrix
			while True:
				if self.unifmat==0:
					# n random
					if reduction:
						nm=self.genAlt.rand(self.bmDim, self.ndim)
					else:
						nm=self.gen.rand(self.bmDim, self.ndim)
				elif self.unifmat==1:
					# n Sobol
					# Use Sobol n times, but change its state only once
					if reduction:
						nm=self.sobolAlt.get(1).T
						state=self.sobolAlt.get_state()
						nm=np.hstack((nm, self.sobolAlt.get(self.ndim-1+0*self.ndim).T))
						self.sobolAlt.set_state(state)
					else:
						nm=self.sobol.get(1).T
						state=self.sobol.get_state()
						nm=np.hstack((nm, self.sobol.get(self.ndim-1+0*self.ndim).T))
						self.sobol.set_state(state)
				
				elif self.unifmat==2:
					# 1 Sobol, n random
					if reduction:
						nm=np.hstack((self.sobolAlt.get(1).T, self.genAlt.rand(self.bmDim, self.ndim)))
					else:
						nm=np.hstack((self.sobol.get(1).T, self.gen.rand(self.bmDim, self.ndim)))
				elif self.unifmat==3:
					# n+1 Sobol
					# Use Sobol n+1 times, but change its state only once
					if reduction:
						nm=self.sobolAlt.get(1).T
						state=self.sobolAlt.get_state()
						nm=np.hstack((nm, self.sobolAlt.get(self.ndim+0*self.ndim).T))
						self.sobolAlt.set_state(state)
					else:
						nm=self.sobol.get(1).T
						state=self.sobol.get_state()
						nm=np.hstack((nm, self.sobol.get(self.ndim+0*self.ndim).T))
						self.sobol.set_state(state)
				if self.unifmat==4:
					# n+1 random
					if reduction:
						nm=self.genAlt.rand(self.bmDim, self.ndim+1)
					else:
						nm=self.gen.rand(self.bmDim, self.ndim+1)
				elif self.unifmat==5:
					# n(n+1) Sobol
					# Use whole-matrix Sobol 
					if reduction:
						nm=self.bsobolAlt.get(1)[0,:(self.bmDim*self.ndim)].reshape((self.bmDim, self.ndim))
					else:
						nm=self.bsobol.get(1)[0,:(self.bmDim*self.ndim)].reshape((self.bmDim, self.ndim))
				elif self.unifmat==6:
					# (n+1)(n+1) Sobol
					# Use whole-matrix Sobol 
					if reduction:
						nm=self.bsobolAlt.get(1).reshape((self.bmDim, self.ndim+1))
					else:
						nm=self.bsobol.get(1).reshape((self.bmDim, self.ndim+1))
						
				if self.generator==2: 
					# UniMADS
					nm=self.boxMuller(nm)
					# Truncate
					nm=nm[:self.ndim,:]
				else:
					# QRMADS
					# Convert column 0 to N(0,1)
					# First column will become uniformly distributed on sphere after normalization
					nm[:,0]=self.boxMuller(nm[:,0])
					
					# Handle columns 1..
					if self.qraugment==0:
						# Convert to uniform distribution from [-1,1]
						nm[:,1:]=nm[:,1:]*2-1.0
					elif self.qraugment==1:
						# Convert to uniform distribution from [0,1]
						pass
					elif self.qraugment==2:
						# Convert to N(0,1)
						nm[:,1:]=self.boxMuller(nm[:,1:])
					else:
						# Convert to N(0,1), fixed first vector
						nm[:,0]=0
						nm[0,0]=1
						nm[:,1:]=self.boxMuller(nm[:,1:])
					
					# Truncate
					nm=nm[:self.ndim,:]
					
					# QR decompose the last n columns to obtain a random orthogonal matrix
					Q,R=np.linalg.qr(nm[:,1:]) 
					nm[:,1:]=Q
				
				# Scale columns to unit length
				# N(0,1) columns become uniformly distributed on sphere
				nm=np.dot(
					nm, 
					np.diag((nm**2).sum(axis=0)**(-0.5))
				)
				
				# QR decompose
				Q,R=np.linalg.qr(nm)
				
				# Scale columns of Q with 1 or -1
				if self.generator==2:
					# UniMADS
					d=np.diag(R)
					if np.abs(d).min()<np.abs(d).max()*1e-24:
						# raise Exception, "Weird"
						# print "Weird"
						pass
					#	continue
					D=np.diag(1.0-2*np.signbit(d))
					basis=np.dot(Q,D)
				else:
					# QRMADS
					basis=Q
				
				break
			
			# Rotate protoset
			p=np.dot(basis, self.Vproto)
			
			# Hypercube projection
			if self.rounding==1:
				# Project to hypercube [-1,1]^n
				linf=np.abs(p).max(axis=0)
				pproj=np.dot(
					p, np.diag(linf**(-1))
				)
			else:
				# No projection
				pproj=p.copy()
			
			# Scale
			pscaled=pproj*self.step
			
			# Grid restrain (round)
			pround=pscaled.copy()
			for ii in range(p.shape[1]):
				tmp=self.gridRestrain(pscaled[:,ii])
				pround[:,ii]=tmp
			
			# Build poll steps
			if self.protoset==1:
				# UniMADS 2n, QRMADS 2n
				# Interleave p and -p
				p1=np.zeros((self.ndim, self.ndim*2))
				p1[:,0::2]=pround
				p1[:,1::2]=-pround
			else:
				# UniMADS n+1, QRMADS n+1
				p1=pround
			
		else:
			# Original OrthoMADS generator, the basis is orthogonal
			if reduction:
				x=np.array(self.haltonAlt.get(1)[0])
			else:
				x=np.array(self.halton.get(1)[0])
			
			# Normalized direction
			x=2*x-1.0
			xl=((x**2).sum())**0.5
			v=x/xl
			
			v=v.reshape((self.ndim,1))
			
			# Scale with (step/mesh)**0.5
			vl=(self.step/self.mesh)**0.5
			v=v*vl
			
			# Find maximal alpha so that round(alpha*v) is not longer than vl
			alpha1=0.0
			alpha2=2*self.ndim**0.5
			ql1=(np.round(alpha1*v)**2).sum()**0.5
			ql2=(np.round(alpha2*v)**2).sum()**0.5
			
			alphaBest=0.0
			while alpha2-alpha1>1e-9:
				alpha=(alpha1+alpha2)/2
				ql=(np.round(alpha*v)**2).sum()**0.5
				if ql<=vl:
					alpha1=alpha
					if alpha>alphaBest:
						alphaBest=alpha
				if ql>vl:
					alpha2=alpha
			
			q=(np.round(alphaBest*v)).reshape((self.ndim,1))
			ql=(q**2).sum()**0.5
			
			# Basis vector length is approximately step/mesh
			basis=ql**2*np.eye(self.ndim)-2*np.dot(q, q.T)
			
			# Generate poll directions, interleave p, -p
			p=np.zeros((self.ndim, self.ndim*2))
			p[:,0::2]=basis
			p[:,1::2]=-basis
			
			# Scale basis with mesh size to obtain poll steps
			p1=p*self.mesh
		
		return p1
		
	def newMeshStep(self):
		# For spherical rounding
		#   gamma_regular_simplex = 0.5*n**1.5
		#   gamma_2n = 0.5*n
		#
		# Ratios
		#   gamma_hypercube/gama_spherical = (1-1/n)**0.5
		#   gamma_n_with_neg_sum/gamma_regular_simplex = (1+2*(1-1/n)/n**0.5)**0.5
		
		if self.generator==0:
			# OrthoMADS
			self.mesh=1.0*self.meshBase**self.meshExp
			if self.mesh>1.0:
				self.mesh=1.0
			self.step=1.0*self.stepBase**self.meshExp
			return
		else:
			# Handle protoset
			if self.protoset==0:
				# Regular simplex
				gamma=0.5*self.ndim**1.5
			elif self.protoset==1:
				# Orthogonal basis and its negative
				gamma=0.5*self.ndim
			else:
				# Orthogonal basis + negative sum 
				gamma=0.5*self.ndim**(1.5)*(1+2.0*(1-1.0/self.ndim)/self.ndim**0.5)**0.5
			
			# Handle rounding with hypercube projection
			if self.rounding==1 and self.ndim>1:
				gamma*=(1-1.0/self.ndim)**0.5
			
			
		# QRMADS, UniMADS
		#if self.protoset==0:
			## n+1 poll steps
			#if self.generator==2:
				## UniMADS
				#if self.rounding==0:
					## Simple
					#gamma=0.5*self.ndim**(1.5)
				#else:
					## Hypercube
					#gamma=0.5*self.ndim*(self.ndim-1.0)**0.5
			#else:
				## QRMADS
				#if self.rounding==0:
					## Simple
					#gamma=0.5*self.ndim**(1.5)*(1+2.0*(1-1.0/self.ndim)/self.ndim**0.5)**0.5
				#else:
					## Hypercube
					#gamma=0.5*self.ndim**(1.5)*(1+2.0*(1-1.0/self.ndim)/self.ndim**0.5)**0.5*(1-1.0/self.ndim)**0.5
		#else:
			## 2n poll steps
			#if self.rounding==0:
				## Simple
				#gamma=0.5*self.ndim
			#else:
				## Hypercube
				#gamma=0.5*(self.ndim**2-self.ndim)**0.5
			
		# Mesh size
		self.mesh=1.0*self.meshBase**(self.meshExp) 
		if self.mesh>1.0:
			self.mesh=1.0
		
		if self.rounding==0:
			# Increase gamma to gamma+1 - TODO: fix paper
			gamma=gamma+1
		else:
			# Hypercube rounding, round gamma+1 up because it needs to be an integer
			gamma=np.ceil(gamma+1)
			
		# Correct mesh size
		self.mesh/=gamma
		self.mesh/=self.initialMeshDensity
		
		# Step size
		self.step=1.0*self.stepBase**(self.meshExp)
		
	# Function error	
	def f_error(self, f):
		relerror=np.abs(f*self.freltol)
		if self.fabstol>relerror:
			return self.fabstol
		else:
			return relerror
	
	# Step error
	def d_error(self, x0, d, ap=1.0, an=-1.0):
		abse=d*0+self.xabstol
		rele0=np.abs(x0*self.xreltol)
		relep=np.abs((x0+ap*d)*self.xreltol)
		# abs((x0+an*d)*self.xreltol)
		return (
			np.where(abse>rele0, abse, rele0)
			+np.where(abse>relep, abse, relep)
			# +np.where(abse>relen, abse, relen)
		)
			
		relerror=np.abs(f*self.freltol)
		if self.fabstol>relerror:
			return self.fabstol
		else:
			return relerror

	# Approximate d.T A d around x0 from f(x)=0.5 x.T A x + g.T x + c
	# Equals the second directional derivative wrt lam of f(lam d)
	def approxd2(self, f0, fp, fn, f0tol=None, fptol=None, fntol=None, ap=1.0, an=-1.0):
		d2=2*((fp-f0)/ap - (fn-f0)/an)/(ap-an)
		
		# Error
		if f0tol is None:
			f0tol=self.f_error(f0)
		if fptol is None:
			fptol=self.f_error(fp)
		if fntol is None:
			fntol=self.f_error(fn)
			
		d2error = (
			(np.abs(fptol)+np.abs(f0tol))/np.abs(ap) + (np.abs(fntol)+np.abs(f0tol))/np.abs(an)
		)*2/np.abs(ap-an)
		
		return d2, d2error
	
	# H update error
	def error_deltaH(self, d2, d2_error, H, d, d_error):
		dnorm2=(d**2).sum()
		dnorm2_error=2*(np.abs(d)*d_error).sum()
		
		n=self.ndim
		a=np.abs(d2-np.dot(d.reshape((1,n)), np.dot(H, d.reshape((n,1)))))/dnorm2**2
		b=(
			d2_error
			+2*np.dot(d_error.reshape((1,n)), np.abs(np.dot(H, d.reshape((n,1)))))
		)/dnorm2**2 + 2*a*dnorm2_error/dnorm2
		
		# F norm upper bound
		return (
			 2*a**2*((d_error**2).sum()*dnorm2+(d_error*np.abs(d)).sum()**2)
			+b**2*dnorm2**2
			+4*a*b*dnorm2*(d_error*np.abs(d)).sum()
		)**0.5
		
		# Matrix
		tmpd=np.dot(d_error.reshape((n,1)), np.abs(d).reshape((1,n)))
		return (
			a*np.dot(d.reshape((n,1)), d.reshape((1,n)))
			+b*(tmpd+tmpd.T)
		)

		dnorm2=(d**2).sum()
		dnorm=(d**2).sum()**0.5
		abserror=d*0+self.xabstol
		relerror=np.abs(x0*self.xreltol)
		dn=d/dnorm
		dn_error=np.where(abserror>relerror, abserror, relerror)/dnorm
		
		dnTH=np.dot(dn.reshape((1,dn.size)), H)
		dnTHdn=np.dot(dnTH, dn.reshape((dn.size,1)))
		dnTHepsdn=np.dot(np.abs(dnTH), np.abs(dn_error.reshape((dn.size,1))))
		dnepsdnT=np.dot(dn.reshape((dn.size,1)), dn_error.reshape((1,dn.size)))
		dndnT=np.dot(dn.reshape((dn.size,1)), dn.reshape((1,dn.size)))
		
		return (
			np.abs(dndnT)*(np.abs(d2_error)/dnorm**2+2*np.abs(dnTHepsdn))+ 
			(np.abs(dnepsdnT)+np.abs(dnepsdnT.T))*np.abs(d2/dnorm**2-dnTHdn)
		)
	
	def updateHcore_linear(self, d, d_error, d2, d2_error):
		if not np.isfinite(d2):
			return False
		
		dTHd=np.dot(d.reshape((1,d.size)), np.dot(self.H, d.reshape((d.size,1))))
		#print d2, dTHd
		dn2=(d**2).sum()
		alpha=(d2-dTHd)/dn2**2
		#print "dn2", dn2
		#print "hmax", np.abs(self.H).max()
		#print "alpha", alpha
		
		dH=alpha*np.dot(d.reshape(d.size,1), d.reshape(1,d.size))
		
		if not np.isfinite(dH).all():
			return False
		
		newH=self.H+dH
		
		if not np.isfinite(newH).all():
			return False
		
		if self.Habstol is None or self.Hreltol is None:
			self.H=newH
			return True
		else:
			dHerr=self.error_deltaH(d2, d2_error, self.H, d, d_error)
			HF=(self.H**2).sum()**0.5
			dHF=(dH**2).sum()**0.5
			dHerrF=(dHerr**2).sum()**0.5
		
			if (dHerrF<=dHF*self.Hreltol+self.Habstol): 
				self.H=newH
				return True
			else:
				return False
			
	def updateH_linear(self, x0, d, f0, fp, fn, ap=1.0, an=-1.0):
		if np.isfinite(fp) and np.isfinite(fn) and np.isfinite(f0):
			d2, d2_error=self.approxd2(f0, fp, fn, ap=ap, an=an)
			d_error = self.d_error(x0, d, ap, an)
			self.updateHcore_linear(d, d_error, d2, d2_error)
			
	def simplicalBasis(self, x0, f0, c0, xn1):
		# Computes a new simplical basis centered at current best point
		# Leaves out point xn1
		# Returns True on success
		
		# Prepare vector, function, and constraint difference storage
		npts=len(self.pointSet)
		nc=self.c.size
		
		# Columns are vectors
		V=np.zeros((self.ndim, npts))
		# dfc columns correspond to function and constraint value differences to wrt at columns of V
		dfc=np.zeros((nc+1,npts))
		
		ii=0
		for age in self.pointSet.keys():
			# Skip origin
			#if age==self.bestIter:
			#	continue
			
			# Extract point	
			x,f,c=self.pointSet[age]
			
			# Skip points x0, xn1
			# TOOD: base criterion on mesh size
			if (
				(xn1 is not None and np.abs((x-xn1)/self.scaling).max()<=self.step*1e-12) or
				(x0 is not None and np.abs((x-x0)/self.scaling).max()<=self.step*1e-12)
			):
				continue
			
			# Compute vector and function difference
			V[:,ii]=(x-x0)/self.scaling
			dfc[0,ii]=f-f0
			dfc[1:,ii]=c-c0
			
			ii+=1
			
		# Are there enough points (at least n)
		if ii<self.ndim:
			# No, stop now
			return False
			
		# Truncate to ii points
		V=V[:,:ii]
		dfc=dfc[:,:ii]
		
		# QR decomposition
		try:
			# QR = VP
			# Integers in p denote the row number where 1 is found
			# in the permutation matrix P
			q,r,p=scipy.linalg.qr(V, pivoting=True)
		except:
			# QR decomposition failed
			return False
		
		# Take first n vectors as basis (use permutation matrix P)
		# Store Q, R, permuted V, permuted dfc, and origin iteration number
		p=p[:self.ndim]
		self.Q=q
		self.R=r[:,:self.ndim]
		self.V=V[:,p]
		self.dfc=dfc[:,p]
		
		# Remember origin
		self.sim_xorigin=x0
		self.sim_forigin=f0
		self.sim_corigin=c0
		
		return True
	
	def updateH_simplical_engine(self, xn1, fn1, cn1):
		# Solve for alpha vector for given vn1=xn1-sim_xorigin
		vn1=(xn1-self.sim_xorigin)/self.scaling
		dfn1=fn1-self.sim_forigin
		# dcn1=cn1-self.sim_corigin	
		
		# Solve for alpha, given last step xn1
		# V alpha = Q R alpha =  vn1
		#
		# alpha = R^-1 Q.T vn1
		failed=False
		try:
			alpha=scipy.linalg.solve_triangular(
				self.R, 
				np.dot(self.Q.T, vn1.reshape((self.ndim,1)))
			).reshape(self.ndim)
			#alpha=np.linalg.solve(self.V, vn1.reshape((self.ndim,1))).reshape(self.ndim)
		except:
			# Failed, stop
			failed=True
		
		# Verify alpha 
		#   - avoid cases when vn1 is close to a basis vector
		#     i.e. norm of alpha is 1 and max of abs is 1
		if not failed:
			# Verify alpha 
			#   - avoid cases when vn1 is close to a basis vector
			#     i.e. largest abs element is 1, second largest is 0
			#   - avoid cases when vn1 is close to origin
			#     i.e. largest abs element is 0
			alpha_sorted=np.sort(np.abs(alpha))
			if (
				(
					np.abs(alpha_sorted[-1]-1.0)<1e-4 and
					alpha_sorted.size>=2 and 
					np.abs(alpha_sorted[-2])<1e-4
				) or np.abs(alpha_sorted[-1])<1e-4
			):
				failed=True
		
		if not failed:
			# Compute a and A
			A=np.zeros((self.ndim,self.ndim))
			for ii in range(self.ndim):
				v=self.V[:,ii].reshape((self.ndim,1))
				A+=alpha[ii]*np.dot(v,v.T)
			A-=np.dot(vn1.reshape((self.ndim,1)), vn1.reshape((1,self.ndim)))
			a=2*((alpha*self.dfc[0,:]).sum()-dfn1)
			
			# Apply update
			num=(a-(A*self.H).sum())
			den=(A**2).sum()
			beta=num/den
			if np.isfinite(beta):
				dH=beta*A
				self.H+=beta*A
			else:
				failed=True
		
		return failed
			
	def updateH_simplical(self, x0, f0, c0, xn1=None, fn1=None, cn1=None):
		# x0, f0, c0 is the current origin
		# xn1, fn1, cn1 is the new point
		
		# Do we have a simplical basis
		if self.Q is None:
			# Build it, skip point xn1
			if not self.simplicalBasis(x0, f0, c0, xn1):
				# Failed, stop
				return
		
		if xn1 is not None:
			failed=self.updateH_simplical_engine(xn1, fn1, cn1)
		else:
			for jj in range(5):
				for age in self.pointSet.keys():
					x,f,c=self.pointSet[age]
					failed=self.updateH_simplical_engine(x, f, c)
		
		# Do we need a new simplical basis (new origin)
		if (x0!=self.sim_xorigin).any():
			self.simplicalBasis(x0, f0, c0, None)
		return failed
	
	def updateH_powell(self, x0, f0, c0):
		self.pH[0]=self.H
		
		npts=len(self.pointSet)
		
		# Prepare blank matrix, columns are vectors x-x0
		X=np.zeros((self.ndim, npts))
		
		# Prepare blank matrix, column 0 corresponds to f, columns 1..n to constrants
		# Rows correspond to vectors x-x0
		dfc=np.zeros((npts, self.nc+1))
		
		# Skip points equal to x0
		ii=0
		#flist=[]
		for age in self.pointSet.keys():
			# Extract point	
			x,f,c=self.pointSet[age]
			
			
			
			# Skip x0
			# TODO: base criterion on mesh not step
			if (
				(self.px0 is not None and np.abs((x-self.px0)/self.scaling).max()<=self.step*1e-12)
			):
				continue
			
			# Compute vector difference for i-th point
			dx=(x-self.px0)/self.scaling
			X[:,ii]=dx
			
			# Compute function differences at i-th point
			df=f-(
				0.5*np.dot(dx.reshape((1,self.ndim)), np.dot(self.pH[0], dx.reshape((self.ndim,1))))
				+np.dot(self.pg[0].reshape((1,self.ndim)), dx.reshape((self.ndim,1)))
				+self.pf0
			)
			dfc[ii,0]=df
			
			# Loop through constraints
			for jj in range(self.nc):
				dc=c[jj]-(
					0.5*np.dot(dx.reshape((1,self.ndim)), np.dot(self.pH[jj+1], dx.reshape((self.ndim,1))))
					+np.dot(self.pg[jj+1].reshape((1,self.ndim)), dx.reshape((self.ndim,1)))
					+self.pc0[jj]
				)
				dfc[ii,jj+1]=dc
			
			#flist.append(f)
			ii+=1
		
		# Truncate X, dfc
		X=X[:,:ii]
		dfc=dfc[:ii,:]
		npts=ii
		
		failed=True
		if npts>self.ndim+1:
			# Form matrix A
			A=0.5*np.dot(X.T,X)**2
			
			# Form matrix of the system
			W=np.zeros((self.ndim+npts,self.ndim+npts))
			W[:npts,:npts]=A 
			W[:npts,npts:]=X.T 
			W[npts:,:npts]=X 
			rhs=np.zeros((self.ndim+npts, self.nc+1))
			rhs[:npts,:]=dfc
			
			# Solve
			
			try:
				lamdg=np.linalg.solve(W, rhs)
				# Check if solution makes sense (no Inf or NaN)
				if np.isfinite(lamdg).all():
					failed=False
				else:
					failed=True
			except:
				pass
			
			# Rows are lambda and dg, columns are objective and constraints
			
			if not failed:
				# Update Hessians
				for jj in range(npts):
					# Compute (xj-x0) (xj-x0).T
					dx=X[:,jj].reshape((self.ndim,1))
					xxt=np.dot(dx, dx.T)
					for kk in range(self.nc+1):
						self.pH[kk]+=xxt*lamdg[jj,kk]
				
				# Update gradients
				for kk in range(self.nc+1):
					self.pg[kk]+=lamdg[npts:,kk]
		
		self.H=self.pH[0]
		
		# Verify H, g, and f0 vs known function (Hilbert)
		#Hexact=scipy.linalg.hilbert(self.ndim)
		#gexact=np.dot(Hexact,self.px0.reshape((self.ndim,1))).reshape(self.ndim)
		#fexact=0.5*np.dot(self.px0.reshape((1,self.ndim)), np.dot(Hexact,self.px0.reshape((self.ndim,1))))
		#print failed
		#print "Hdif", ((self.pH[0]-Hexact)**2).sum()**0.5
		#print "gdif", ((self.pg[0]-gexact)**2).sum()**0.5
		#print "fdif", np.abs(self.pf0-fexact).sum()
		
		# Verify points vs updated model
		#for kk in range(npts):
		#	dx=X[:,kk].reshape(self.ndim)
		#	print "diff", kk, (
		#		0.5*np.dot(dx.reshape((1,self.ndim)), np.dot(self.pH[0], dx.reshape((self.ndim,1))))+
		#		np.dot(self.pg[0].reshape((1,self.ndim)), dx.reshape((self.ndim,1)))+
		#		self.pf0
		#	) - flist[kk]
			
	
	def updateOrigin_powell(self, x0, f0, c0):
		# Do we have an origin
		# If no x0 available, store x0 - initialization
		if self.px0 is None:
			self.px0=x0
			self.pf0=f0
			self.pc0=c0
		elif (x0!=self.px0).any():
			# Compute x0 change
			dx0=(x0-self.px0)/self.scaling
			
			for kk in range(self.nc+1):
				# Update gradient
				self.pg[kk]+=np.dot(self.pH[kk], dx0.reshape((self.ndim,1))).reshape(self.ndim)
				
			# Update x0, f0, c0
			self.px0=x0
			self.pf0=f0
			self.pc0=c0
		
	def collect_powell(self):
		"""
		Used together with the Powell update. 
		
		Copies the gradient, the Hessian, and the Jabobian from the 
		internal structures of the update procedure. 
		
		Returns the gradient of objective and the Jacobian of 
		the constraints. 
		"""
		# Jacobian of the constraints
		J=np.zeros((self.nc,self.ndim))
		for kk in range(self.nc):
			J[kk,:]=self.pg[kk+1]
	
		# Gradient and Hessian of the objective
		g=self.pg[0]
		
		return g, J
	
	def explicit_gHJ(self, x0):
		"""
		Computes the explicitly given gradients and Hessian. 
		"""
		if self.cJ is not None:
			J=self.cJ(x0)*self.scaling[None,:]
		else:
			J=None
		if self.fgH is not None:
			g,H=self.fgH(x0)
			g=g*self.scaling
			H=H*self.scaling[None,:]*self.scaling[:,None]
		else:
			g=None
			H=None
		
		return g, H, J
		
	def lookupPoint(self, x0):
		for age, point in self.pointSet.iteritems():
			x,f,c=point
			# if np.abs((x-x0)/self.scaling).max()<self.step*1e-14:
			# if np.abs((x-x0)/self.scaling).max()<=self.mesh:
			# if np.abs((x-x0)/self.scaling).max()<=self.mesh/2:
			# if withinTol(x, x0, self.step*self.scaling*1e-14, 1e-14):
			# This is much faster than withinTol
			# if np.abs((x-x0)/self.scaling).max()<=self.step*1e-14:
			if (x==x0).all(): 
				return x, f, c
		
		return None, None, None
	
	def appendPoint(self, x, f, c):
		# Do this only if model is built
		if not self.model:
			return False
		
		# Skip infinite values of f and c
		if not (np.isfinite(f) and np.isfinite(c).all()):
			return False
		
		# Skip duplicates
		if self.lookupPoint(x)!=(None, None, None):
			return False
			
		
		self.pointSet[self.niter]=(x, f, c)
		
		# Purge excessive points, distance criterion
		# Lower bound below which we never decrease the region radius
		lowBound=0.0 # 1e-7 # QRC worse
		if self.step<lowBound:
			purgeLow=lowBound
			purgeHigh=lowBound*self.rho
		else:
			purgeLow=0.0
			purgeHigh=self.step*self.rho
		if 0 and len(self.pointSet)>self.maxPoints:
			# Remove all points with distance from current best point greater than rho*step 
			# Start with oldest points
			ages=self.pointSet.keys()
			ages.sort()
			todelete=[]
			for age in ages:
				entry=self.pointSet[age]
				x,f,c=entry
				# dist=np.abs((x-self.x)/self.scaling).max()
				dist=((x-self.x)**2).sum()**0.5
				if dist<purgeLow or dist>purgeHigh:
					todelete.append(age)
			
			for age in todelete:
				del self.pointSet[age]
				
		# Purge excessive points, age criterion
		if len(self.pointSet)>self.maxPoints:
			# Remove oldest points, order with oldest last so we pop them first
			ages=self.pointSet.keys()
			ages.sort()
			ages.reverse()
			todelete=[]
			while len(self.pointSet)>self.maxPoints and len(ages)>0:
				age=ages.pop()
				# Do not delete best point
				if age!=self.bestIter:
					del self.pointSet[age]
		
		return True
			
	def gradientRegression(self, xb, fb, cb, H):
		npts=len(self.pointSet)
		
		# Build system of equations, origin is given by xb, fb, cb
		nc=cb.size
		V=np.zeros((npts, self.ndim))
		
		# Compute the number of RHS columns, prepare RHS
		rhslen=0
		if self.fgH is None:
			rhslen+=1
		if self.cJ is None:
			rhslen+=self.nc
		FC=np.zeros((npts,rhslen))
		
		ii=0
		ndxlist=[]
		for age in self.pointSet.keys():
			# Skip best point by age
			#if age==self.bestIter:
			#	continue
			
			x,f,c=self.pointSet[age]
			
			# Skip best point
			# if dist<=self.mesh/2:
			# if withinTol(x, xb, self.step*self.scaling*1e-14, 1e-14):
			if (x==xb).all():
				continue
			
			# Direction from xbest
			d=(x-xb)/self.scaling
			# dist=np.abs(d).max()
			dist=(d**2).sum()**0.5
			
			# Skip points outside rho
			if dist>self.rho*self.step:
				continue
			
			
			ndxlist.append(age)
			
			# Add to LHS
			V[ii,:]=d[:]
			
			# Function difference from fbest, subtract second order term
			if self.fgH is None:
				# WARNING - a really nasty bug removed
				# this is wrong: (d*np.dot(self.H, d.reshape((self.ndim,1)))).sum()
				FC[ii,0]=f-fb-(
					0.5*np.dot(d.reshape((1,self.ndim)), np.dot(H, d.reshape((self.ndim,1))))
				)
				#FC[ii,0]=f-fb-(
				#	0.5*(d*np.dot(self.H, d.reshape((self.ndim,1)))).sum()
				#)
				#print
				#print (d.reshape((self.ndim,1))*np.dot(self.H, d.reshape((self.ndim,1)))).sum()
				#print np.dot(d.reshape((1,self.ndim)), np.dot(H, d.reshape((self.ndim,1))))
				#print
			
			# Constraint difference from cbest
			if self.cJ is None:
				if self.fgH is None:
					FC[ii,1:]=(c-cb)[:]
				else:
					FC[ii,0:]=(c-cb)[:]
			ii+=1
		
		if ii<self.minPoints:
			if self.debug:
				DbgMsgOut("MADSOPT", ("%d points insufficient for regression.") % (ii))
			return None, None
		
		# Truncate V and FC
		V=V[:ii,:]
		FC=FC[:ii,:]
		
		if self.debug:
			DbgMsgOut("MADSOPT", ("Computing gradients with %d points.") % (ii))
		
		# Solve linear least squares
		try:
			# G1, residues, rank, s=np.linalg.lstsq(V,FC)
			
			u,s,v=np.linalg.svd(V, full_matrices=False)
			eps=np.spacing(1)
			s=np.where(s<eps, eps, s)
			G=np.dot(v.T, np.dot(np.diag(s**-1), np.dot(u.T, FC)))
			
			# print np.abs(G-G1).max()
			
			# print rank, residues, self.ndim
			#if rank<self.ndim:
			#	# V does not have full rank, ignore the result
			#	# print G
			#	if self.debug:
			#		DbgMsgOut("MADSOPT", ("Insufficient rank in regression."))
			#	return None, None
		except:
			if self.debug:
				DbgMsgOut("MADSOPT", ("Regression failed."))
			return None, None
			
		# Extract gradient and Jacobian of constraints
		if self.fgH is None:
			g=G[:,0]
			g=g.copy()
		else:
			g=None
		if self.cJ is None:
			if self.fgH is None:
				J=G[:,1:].T
			else:
				J=G[:,0:].T
			J=J.copy()
		else:
			J=None
			
		# verify point distance
		#print "--"
		#for age in ndxlist:
			#x,f,c=self.pointSet[age]
			#d=x-xb
			#mf=(
				#0.5*np.dot(d.reshape((1,self.ndim)), np.dot(H, d.reshape((self.ndim,1))))+
				#(g.reshape(self.ndim)*d.reshape(self.ndim)).sum()
			#)
			#df=f-fb
			
			#rel=(mf-df)/np.abs(df)
			#print age, rel, np.abs(H).max(), (g**2).sum()**0.5
		
		# Hilbert
		#HH=scipy.linalg.hilbert(self.ndim)
		#hdif=((self.H-HH)**2).sum()**0.5
		# print "hdif", hdif, self.step
		#print "xn", (xb**2).sum()**0.5
		#g1=np.dot(HH, xb).reshape((self.ndim,1)).reshape((self.ndim))
		#g2=g+np.dot(self.H, xb.reshape((self.ndim,1))).reshape((self.ndim))
		#print g1
		#print g2
		#print self.step
		#print npts, ii, self.step, "gdif", ((g2-g1)**2).sum()**0.5, "hdif", hdif, residues # /(g1**2).sum()**0.5
		# print V
		
		return g, J
		
	def hessianModification(self, H):
		# Hessian modification with Cholesky factorization
		minHdiag=np.diag(H).min()
		if minHdiag>0:
			tau=0.0
		else:
			tau=-minHdiag+self.beta
		while True:
			try:	
				Hmod=H+tau*np.eye(self.ndim)
				L=np.linalg.cholesky(Hmod)
				if self.debug:
					DbgMsgOut("MADSOPT", ("Cholesky OK, tau=%e.") % (tau))
				break
			except:
				if 2*tau<self.beta:
					tau=self.beta
				else:
					tau=2*tau

				if self.debug:
					DbgMsgOut("MADSOPT", ("Trying Cholesky with tau=%e.") % (tau))
				
		return L, Hmod, tau
	
	def hessianModificationEig(self, H, g, betaMin=1e-9):
		# Estimate beta
		# beta^-1 * ||g|| = rho_neg
		gnorm=(g**2).sum()**0.5
		beta=self.beta*gnorm/self.rhoNeg
		if beta<betaMin:
			beta=betaMin
		e,V = np.linalg.eig(H)
		mine=e.min()
		if mine<beta:
			tau=np.abs(mine)
		else:
			tau=0.0
		e=np.where(e<beta, beta, e)
		Hmod=np.dot(np.dot(V, np.diag(e)), V.T)
		return Hmod, tau, mine
	
	def doModelSearchCVXOPT(self, x, f, c, g, H, J, tr=None, minTr=1e-5):
		# Hessian modification with Cholesky factorization
		L, Hmod, tau = self.hessianModification(H)
		# Hmod, tau, mine = self.hessianModificationEig(H, g)
		
		# Handle lower and upper bounds
		xlo=(self.xlo-x)/self.scaling
		xhi=(self.xhi-x)/self.scaling
		
		# Clip it to rho * step (comment out to disable trust region)
		# Turn on trust region for indefinite H
		if (tau>0 and tr is not None and np.isfinite(tr)):
			# print "NEG"
			trr=tr*self.step
			if trr<minTr:
				trr=minTr
			xlo=np.where(xlo<-trr, -trr, xlo)
			xhi=np.where(xhi>trr, trr, xhi)
			# print "NEG", tau 
		#else:
		#	trr=self.rho*self.step
		#	xhi=np.where(xhi>trr, trr, xhi)
		#	xlo=np.where(xlo<-trr, -trr, xlo)
			
		# Prepare CVXOPT constraints, cvxopt default is <=
		G=np.vstack((
			np.eye(self.ndim), 
			-np.eye(self.ndim)
		))
		h=np.hstack((xhi, -xlo))
		
		# Purge all bounds that are not finite
		nzi=np.where(np.isfinite(h))[0]
		G=G[nzi,:]
		h=h[nzi]
		
		# Get direction by solving the quadratic model
		d=None
		mr=None
		if self.nc==0:
			if g is not None and Hmod is not None:
				# Unconstrained problem
				
				# Solve min 0.5 x.T H x + g.T x
				try:
					# Use LP when problem is not positive definite
					if 0 and tau>0:
						sol=solvers.lp(
							matrix(g), 
							matrix(G), matrix(h)
						)
						mr=-sol['primal objective']
						d=np.array(sol['x']).reshape((self.ndim))
						# print "LP"
					else:
						sol=solvers.qp(
							# Function: 1/2 x.T Hmod x + g.T x 
							# matrix(np.dot(L, L.T)),
							matrix(Hmod), 
							matrix(g), 
							# Inequality: G x <= h
							matrix(G), matrix(h), 
							# Equality: A x = b
							None, None
						)
						mr=-sol['primal objective']
						d=np.array(sol['x']).reshape((self.ndim))
				except KeyboardInterrupt:
					raise
				except:
					pass
		else:
			# QP problem
			if g is not None and J is not None and Hmod is not None:
				# Problem is convex (Hmod is positive definite) 
				
				# Add inequality constraints
				# Normalize rows for >= constraints
				G1=J[self.ndxge,:]
				h1=(self.clo[self.ndxge]-c[self.ndxge])
				if self.ndxge.size>0:
					G1norm=(np.abs(G1)).max(axis=1)
					G1norm=np.where(G1norm==0.0, 1.0, G1norm)
					G1=G1/G1norm[:,None]
					h1=h1/G1norm
					G=np.vstack((
						G,
						-G1
					))
					h=np.hstack((h, -h1))
				# Normalize rows for <= constraints
				G2=J[self.ndxle,:]
				h2=(self.chi[self.ndxle]-c[self.ndxle])
				if self.ndxle.size>0:
					G2norm=(np.abs(G2)).max(axis=1)
					G2norm=np.where(G2norm==0.0, 1.0, G2norm)
					G2=G2/G2norm[:,None]
					h2=h2/G2norm
					G=np.vstack((
						G,
						G2
					))
					h=np.hstack((h, h2))
				
				# Equality constraints
				if self.ndxeq.size==0:
					# No equality constraints
					A=None
					b=None
				else:
					# At least one equality constraint
					# Normalize rows
					A1=J[self.ndxeq,:]
					A1norm=(np.abs(A1)).max(axis=1)
					A1norm=np.where(A1norm==0.0, 1.0, A1norm)
					A=matrix(J[self.ndxeq,:]/A1norm[:,None])
					b=matrix((self.clo[self.ndxeq]-c[self.ndxeq])/A1norm)
				
				try:
					if 0 and tau>0:
						sol=solvers.lp(
							matrix(g), 
							matrix(G), matrix(h)
						)
						mr=-sol['primal objective']
						d=np.array(sol['x']).reshape((self.ndim))
						# print "LP"
					else:
						sol=solvers.qp(
							# Function: 1/2 x.T Hmod x + p.T x 
							# matrix(np.dot(L, L.T)),
							matrix(Hmod), 
							matrix(g), 
							# Inequality: G x <= h
							matrix(G), matrix(h), 
							# Equality: A x = b
							A, b
						)
						mr=-sol['primal objective']
						d=np.array(sol['x']).reshape((self.ndim))
				except KeyboardInterrupt:
					raise
				except:
					pass
		
		return d, mr
	
	def doModelSearchSLSQP(self, x, f, c, g, H, J, tr=None):
		# Prepare bounds
		xlo=(self.xlo-x)/self.scaling
		xhi=(self.xhi-x)/self.scaling
		
		if tr is not None:
			trr=tr*self.step
			xlo=np.where(xlo<-trr, -trr, xlo)
			xhi=np.where(xhi>trr, trr, xhi)
			
		bounds=[]
		for ii in range(self.ndim):
			if np.isfinite(xlo[ii]):
				lo=xlo[ii]
			else:
				lo=None
			if np.isfinite(xhi[ii]):
				hi=xhi[ii]
			else:
				hi=None
			bounds.append((lo,hi))
		
		# Prepare constraints
		constr=[]
		for ii in self.ndxle:
			# Add upper bound on constraint
			constr.append(
				{
					'type': 'ineq', 
					'fun': lambda xv: -((J[ii,:].reshape(self.ndim)*xv.reshape(self.ndim)).sum()+c[ii]-self.chi[ii]), 
					'jac': lambda xv:  (J[ii,:]).reshape(self.ndim)
				}
			)
		for ii in self.ndxge:
			# Add lower bound on constraint
			constr.append(
				{
					'type': 'ineq', 
					'fun': lambda xv: -((J[ii,:].reshape(self.ndim)*xv.reshape(self.ndim)).sum()+c[ii]-self.clo[ii]), 
					'jac': lambda xv:  (J[ii,:]).reshape(self.ndim)
				}
			)
		
		# Prepare function
		function=lambda xv: 0.5*np.dot(
			xv.reshape((1, self.ndim)), np.dot(H, xv.reshape((self.ndim, 1)))
		)+(g.reshape(self.ndim)*xv.reshape(self.ndim)).sum()+f
		
		# Prepare gradient
		gradient=lambda xv: np.dot(H, xv.reshape((self.ndim,1))).reshape(self.ndim)+g.reshape(self.ndim)
		
		# Solve
		sol=scipy.optimize.minimize(
			function, np.zeros(self.ndim), jac=gradient, 
			bounds=bounds, constraints=constr, 
			method='SLSQP' #, options={'disp': True}
		)
		
		if sol.success and sol.status==0:
			return sol.x
		else:
			return None
		
	def doModelSearchIPOPT(self, x, f, c, g, H, J, tr=None):
		xlo=(self.xlo-x)/self.scaling
		xhi=(self.xhi-x)/self.scaling
		clo=self.clo.copy()
		chi=self.chi.copy()
		
		xlo=np.where(np.isfinite(xlo), xlo, -2e19)
		xhi=np.where(np.isfinite(xhi), xhi,  2e19)
		clo=np.where(np.isfinite(clo), clo, -2e19)
		chi=np.where(np.isfinite(chi), chi,  2e19)
		
		if tr is not None:
			trr=tr*self.step
			xlo=np.where(xlo<-trr, -trr, xlo)
			xhi=np.where(xhi>trr, trr, xhi)
		
		#print xlo
		#print xhi
		#print clo
		#print chi
		
		pw=IPOPTproblemWrapper(g, H, c, J)
		
		#print
		#print pw.objective(np.zeros(self.ndim))
		#print pw.gradient(np.zeros(self.ndim))
		#print pw.constraints(np.zeros(self.ndim))
		#print pw.jacobian(np.zeros(self.ndim))
		
		#1/0
		
		nlp = ipopt.problem(
			n=self.ndim,
			m=self.nc,
			problem_obj=pw,
			lb=xlo,
			ub=xhi,
			cl=clo,
			cu=chi
		)
		
		nlp.addOption('mu_strategy', 'adaptive')
		nlp.addOption('tol', 1e-7)
		nlp.addOption('hessian_approximation', 'limited-memory')
		nlp.addOption('print_level', 0)
		nlp.addOption('max_iter', 100)

		x, info = nlp.solve(np.zeros(self.ndim))
		
		# print "OK", info['status']
		if info['status']==0:
			return x
		else:
			return None
		
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("MADSOPT", "Starting a MADS run at i="+str(self.niter))
		
		# Reset stop flag
		self.stop=False
		
		# Check 
		self.check()
		
		# Evaluate initial point
		if self.f is None:
			self.f,self.c=self.funcon(self.x)
			h=self.aggregateConstraintViolation(self.c)
			
			if not np.isfinite(h):
				raise Exception, DbgMsg("MADSOPT", "Initial point constraint violation must be finite.")
			
			# If filtering is enabled
			if self.filt.hmax>0.0 and h>self.filt.hmax:
				# Stretch initial hmax when h>hmax
				self.filt.reset(h)
			
			# Try to accept initial point
			(dominates,dominated,accepted)=self.filt.accept(self.f, h, (self.x, self.c))
			if not accepted:
				raise Exception, DbgMsg("MADSOPT", "Initial point not accepted, increase hmax (enable filtering).")
			
		# Current point
		x=self.x.copy()
		f=self.f
		c=self.c.copy()
		it=self.niter
		
		if self.model:
			self.appendPoint(x, f, c)
		
		self.nqp=0
		self.nqpok=0
		
		# Rounding control
		mustRound=True
		
		# Last success - for aligning poll steps
		self.xbestold=self.x.copy()
		self.fbestold=self.f
		iterationSuccess=False
		
		tr=self.rho*self.step
		
		# Main loop
		outCnt=0	# Counter for poll steps that were extended beacuse some poll steps were outside bounds
		outPolls=0	# Number of poll sets for which poll step extension took place
		while not self.stop:
			itx1=self.niter
			# No success
			iterationSuccess=False
			ittypeCum=0 # iteration type
			
			# Step was not cut
			stepCut=False
			
			if self.step<self.stopStep:
				if self.debug:
					DbgMsgOut("MADSOPT", "Iteration i="+str(self.niter)+": step size small enough, stopping")
				break
			
			# Update origin of Powell update
			if self.powellUpdate:
				self.updateOrigin_powell(x, f, c)
								
			# Remember position so we can calculate speculative search direction
			xold=x.copy()
			fold=f
			# Reset speculative search flag
			trySpeculative=False
			
			# QP search requires a minimal number of points in the pointSet
			haveModel=False
			if (
				self.model and self.modelSearch and 
				len(self.pointSet)>=self.minPoints
			):
				if self.debug:
					DbgMsgOut("MADSOPT", "Trying QP step, point set size=%d" % (len(self.pointSet))) 
				
				# Evaluate explicit parts of the model
				g, H, J = self.explicit_gHJ(x)
				# Use explicit Hessian, if available
				if H is not None:
					self.H=H
					
				# Construct a model (compute gradients for linear and simplical update)
				if not self.powellUpdate or self.forceRegression:
					# self.modg, self.modJ=self.gradientRegression(x, f, c, self.H)
					if self.powellUpdate:
						self.H=self.pH[0]
					gr, Jr=self.gradientRegression(x, f, c, self.H)
				else:
					# Use Powell update gradients
					gr, Jr=self.collect_powell()
				
				# Use regression results if explicit information not available
				if g is None:
					g=gr
				if J is None:
					J=Jr
				# Store
				self.modg=g
				self.modJ=J
				
				# Store gradient origin and Hessian
				if self.modg is not None and self.modJ is not None:
					self.xgo=x.copy()
					self.fo=f
					self.co=c.copy()
					self.modH=self.H.copy()
					self.ito=it
								
				# If we have a model, use it for QP
				self.nqp+=1
				if self.modg is not None and self.modJ is not None:
					# At this point we have a model
					haveModel=True
					
					# Model search step, select solver by uncommenting
					d,mr = self.doModelSearchCVXOPT(x, f, c, self.modg, self.H, self.modJ, tr=self.rhoNeg, minTr=0e-5)
					# d = self.doModelSearchIPOPT(x, f, c, self.modg, self.H, self.modJ, tr=self.rho)
					# d = self.doModelSearchSLSQP(x, f, c, self.modg, self.H, self.modJ, tr=self.rho)
					
					# Do we have a step
					if d is not None:
						stepCount=0
						while True:
							if self.debug:
								DbgMsgOut("MADSOPT", "Trying QP step length %.1e" % ((d**2).sum()**0.5))
							
							# Default is to use the QP direction
							duse=d 
							
							# Snap to boundary
							if self.boundSnap:
								if self.qpFeasibilityRestoration:
									# xt,a,slide = self.restoreFeasibility(x, duse, self.H, self.modg, self.modJ, c, True, self.boundSlide, mustRound)
									# no slide
									xt,a,slide = self.restoreFeasibility(x, duse, self.H, self.modg, self.modJ, c, True, False, mustRound)
								else:
									# xt,a,slide = self.restoreFeasibility(x, duse, None, None, None, None, True, self.boundSlide, mustRound)
									# no slide
									xt,a,slide = self.restoreFeasibility(x, duse, None, None, None, None, True, False, mustRound)
								
							else:
								if mustRound:
									xt = x+self.gridRestrain(duse)*self.scaling
								else:
									xt = x+duse*self.scaling
								a = 1.0
								slide=False
								
							# If xt after rounding is too close to x QP search fails
							# if (np.abs((xt-x)/self.scaling)<=self.mesh/2).all(): 
							# if withinTol(xt, x, self.step*self.scaling*1e-14, 1e-14): 
							if (xt==x).all():
								if self.debug:
									DbgMsgOut("MADSOPT", "QP step rounded to zero.")
								break
							else:
								# Evaluate
								ft,ct=self.funcon(xt)
								ht=self.aggregateConstraintViolation(ct)
								itt=self.niter
								
								# Add to list
								if self.model and self.appendPoint(xt, ft, ct):
									# Added to list
									if self.powellUpdate:
										self.updateH_powell(x, f, c)
										
								# Model reduction vs function reduction
								#if mr is not None:
								#	fr=f-ft
								#	reduction=fr/mr
								#	if reduction>0.75:
								#		tr*=2
								#		if tr>1:
								#			tr=1.0
								#	elif reduction<0.25:
								#		tr/=4
								#		if tr<2*self.mesh:
								#			tr=2*self.mesh
								#		
								#	print "tr", tr, "step", self.step
								
								# New better point if QP succeeded, constraints are not violated, and f is lower
								oldPosition=self.filt.position(f, self.aggregateConstraintViolation(c))
								(dominates, dominated, accepted)=self.filt.accept(ft, ht, (xt, ct))
								position=self.filt.position(ft, ht)
								
								# Verify acceptance
								# if dominates or accepted and position<=oldPosition:
								if dominates:
									ittype=2
								elif accepted: # and position<=oldPosition:
									# itytype=1
									itytype=2
								else:
									ittype=0
								
								# Model reduction vs function reduction
								#if mr is not None:
								#	fr=f-ft
								#	reduction=fr/mr
								#	if reduction>0.75:
								#		tr*=2
								#		if tr>1:
								#			tr=1.0
								#	elif reduction<0.25:
								#		tr/=4
								#		if tr<2*self.mesh:
								#			tr=2*self.mesh
								#	
								#	if reduction>0.75 and fr>0 and ittype==2:
								#		print fr/np.abs(f), f, "step", self.step
								
								#if ittype>ittypeCum:
								#	ittypeCum=ittype
								ittypeCum=ittype
								
								if ittype==2 or ittype==1:
									# Dominating or improving
									# print (duse**2).sum()**0.5, f, f-ft, stepCount
									x=xt
									f=ft
									c=ct
									it=itt
									iterationSuccess=True
									stepCut = stepCut or (a<1.0) or slide
									self.nqpok+=1
									
									# Try speculative step if QP step was not shrinked and the point dominates
									if self.speculativeModelSearch and stepCount==0 and ittype==2:
										# Dominating
										trySpeculative=True
									
									if self.debug:
										DbgMsgOut("MADSOPT", "QP accepted, mesh=%.1e step=%.1e ittype=%d" % (self.mesh, self.step, ittype))
									
									break
								else:
									if self.debug:
										DbgMsgOut("MADSOPT", "QP rejected, mesh=%.1e step=%.1e ittype=%d" % (self.mesh, self.step, ittype))
							
							d=d/self.modelSearchShrinkFactor
							stepCount+=1
							if stepCount>=self.modelSearchStepLimit:
								break
							
							if self.debug:
								DbgMsgOut("MADSOPT", "QP step shrinked, retrying")
					else:
						if self.debug:
								DbgMsgOut("MADSOPT", "Failed to solve QP subproblem")
				else:
					if self.debug:
						DbgMsgOut("MADSOPT", "No gradient, QP step skipped")
			else:
				if self.debug and self.model and self.modelSearch:
					DbgMsgOut("MADSOPT", "Cannot compute model (insufficient points), QP step skipped")
						
			
			# Poll if search failed
			if not iterationSuccess:
				# Update origin of Powell update
				if self.powellUpdate:
					self.updateOrigin_powell(x, f, c)
				
				# Construct a model (again, because a failed QP step may have added a point)
				if self.model:
					# Evaluiate explicit parts of the model
					g, H, J = self.explicit_gHJ(x)
					# Use explicit Hessian, if available
					if H is not None:
						self.H=H
						
					# Construct a model (compute gradients for linear and simplical update)
					if not self.powellUpdate or self.forceRegression:
						# self.modg, self.modJ=self.gradientRegression(x, f, c, self.H)
						if self.powellUpdate:
							self.H=self.pH[0]
						gr, Jr=self.gradientRegression(x, f, c, self.H)
					else:
						# Use Powell update gradients
						gr, Jr=self.collect_powell()
					
					# Use regression results if explicit information not available
					if g is None:
						g=gr
					if J is None:
						J=Jr
					# Store
					self.modg=g
					self.modJ=J
					
					# Store gradient origin and Hessian
					if self.modg is not None and self.modJ is not None:
						self.xgo=x.copy()
						self.fo=f
						self.co=c.copy()
						self.modH=self.H.copy()
						self.ito=it
					
				else:
					self.modg=None
					self.modh=None
					
				if self.modg is not None and self.modJ is not None:
					# At this point we have a model
					haveModel=True
				else:
					haveModel=False
				
				# Generate poll steps
				if self.meshExp==self.minMeshExp:
					# Get steps generated with alternate generator
					steps=self.minMeshSteps
					
					# This resets the main generator so that its state becomes identical to
					# that of the alternate generator whenever mesh size decreases or becomes 
					# same as the smallest observed mesh size. 
					if self.sequenceReset:
						# Skip bmDim+(maxMeshExp-minMeshExp)+1 values
						# This makes the sequence generators behave like in OrthoMADS. 
						
						# Reset Halton
						if self.generator==0:
							self.halton=ghalton.Halton(self.bmDim)
							# self.halton.get(self.bmDim+(self.maxMeshExp-self.minMeshExp)+1)
							self.halton.get(self.bmDim+(0-self.minMeshExp)+1)
						
						# Reset random generator
						self.gen.set_state(self.genAlt.get_state())
						
						# Reset Sobol generator
						self.sobol.set_state(self.sobolAlt.get_state())
						
						# Reset big Sobol generator
						if self.unifmat==5:
							self.bsobol.set_state(self.bsobolAlt.get_state())
				else:
					steps=self.generatePollSteps()
				
				# Make a copy of steps
				steps=steps.copy()
				
				# Check bound violations, mirror poll steps
				if self.boundStepMirroring:
					mirrored=0
					if (self.protoset==0):
						# General case
						ndir=steps.shape[1]
						for ii in range(ndir):
							p=steps[:,ii]
							x1=x+p*self.scaling
							v_hi=x1>self.xhi
							v_lo=x1<self.xlo
							if v_hi.any() or v_lo.any():
								p=np.where((v_hi|v_lo), -p, p)
								steps[:,ii]=p
								mirrored+=1
					else:
						# 2n poll steps, +p, -p pairs
						for ii in range(self.ndim):
							# Check positive direction
							p=steps[:,(2*ii)]
							x1=x+p*self.scaling
							vp_hi=x1>self.xhi
							vp_lo=x1<self.xlo
							vp=vp_hi.any() or vp_lo.any()
							
							# Check negative direction
							n=steps[:,(2*ii+1)]
							x1=x+n*self.scaling
							vn_hi=x1>self.xhi
							vn_lo=x1<self.xlo
							vn=vn_hi.any() or vn_lo.any()
							
							# If p and n violate, reverse violating components of p, use the negative as n
							if vp and vn:
								p=np.where((vp_hi|vp_lo), -p, p)
								steps[:,(2*ii)]=p
								steps[:,(2*ii+1)]=-p
								mirrored+=1
						
				# Order poll steps
				if self.protoset==0:
					# UniMADS n+1
					# n+1 steps can be odered as is
					if self.model and self.modelOrdering and haveModel:
						# Model ordering, do not allow reversing of individual directions
						# because it affects the distribution of directions
						steps=self.modelOrderSteps(x, steps, self.H, self.modg, self.modJ, c, reverse=False)
					elif self.lastDirectionOrdering and self.pGood is not None:
						# Last successfull direction based ordering, do not allow reversing of individual directions
						# because it affects the distribution of directions
						steps=self.orderSteps(steps, self.pGood, reverse=False)
				else:
					# 2n must be first decomposed into a linear basis by taking, 
					# every other step, ordered, and then reassembled
					if self.linearUpdate:
						# 2n must be first decomposed into a linear basis by taking, 
						# every other step, ordered, and then reassembled
						# Orders points as +d1, -d1, +d2, -d2, ... 
						# so that linear update can be applied as often as possible
						steps1=None
						if self.model and self.modelOrdering and haveModel:
							# Model ordering
							steps1=self.modelOrderSteps(x, steps[:,::2], self.H, self.modg, self.modJ, c, reverse=True)
						elif self.lastDirectionOrdering and self.pGood is not None: 
							# Last successfull direction based ordering
							steps1=self.orderSteps(steps[:,::2], self.pGood, reverse=True)
							
						if steps1 is not None:
							steps=np.zeros((self.ndim, self.ndim*2))
							steps[:,0::2]=steps1
							steps[:,1::2]=-steps1
					else:
						# Simple ordering
						if self.model and self.modelOrdering and haveModel:
							# Model ordering
							steps=self.modelOrderSteps(steps, self.H, self.modg, self.modJ, c, reverse=False)
						elif self.pGood is not None: 
							# Last successfull direction based ordering
							steps=self.orderSteps(steps, self.pGood, reverse=False)
						

				
				# Verify poll steps outside bounds, prepare extension points of inward pointing directions
				extensionDir=None
				if (
					self.model and 
					self.linearUpdate and
					self.linearUpdateExtend
				):
					# Extension points for linear update enabled
					flag=False
					extCount=0
					if self.protoset==0:
						# General case - do nothing
						pass
					else:
						# 2n poll steps, +p, -p pairs
						extensionDir=np.zeros(2*self.ndim)
						for ii in range(self.ndim):
							ii1=2*ii
							ii2=ii1+1
							p1=steps[:,ii1]
							p2=steps[:,ii2]
							xt1=x+p1*self.scaling
							xt2=x+p2*self.scaling
							out1=False
							out2=False
							if (xt1<self.xlo).any() or (xt1>self.xhi).any():
								out1=True
							if (xt2<self.xlo).any() or (xt2>self.xhi).any():
								out2=True
							# Extend the direction not pointing outside
							if out1 and not out2:
								extensionDir[ii2]=1
								extCount+=1
								flag=True
							if not out1 and out2:
								extensionDir[ii1]=1
								extCount+=1
								flag=True
							
					if flag:
						outCnt+=extCount
						outPolls+=1
				
				# Initialize function and constraint value storage
				cutfactor=np.zeros(steps.shape[1])
				slidevec=np.zeros(steps.shape[1])
				fevaluated=np.zeros(steps.shape[1])
				cevaluated=np.zeros((self.nc, steps.shape[1]))
				
				# Poll
				pollSuccess=False
				for ii in range(steps.shape[1]):
					if self.powellUpdate:
						self.updateOrigin_powell(x, f, c)
						self.H=self.pH[0]
						
					if self.boundSnap:
						xt,a,slide = self.restoreFeasibility(x, steps[:,ii], None, None, None, None, True, self.boundSlide, mustRound)
					else:
						# Steps are already rounded
						xt=x+steps[:,ii]*self.scaling
						a=1.0
						slide=False
					
					#if (steps[:,ii]**2).sum()**0.5<self.step*1e-6:
					#	print steps[:,ii], self.step
					#	raise Exception, "zero"
					
					# Was step cut to the origin
					# if np.abs((xt-x)/self.scaling).max()<=self.mesh/2:
					# if withinTol(xt, x, self.step*self.scaling*1e-14, 1e-14):
					if (xt==x).all():
						# Yes, skip evaluation
						ft=f
						ct=c
						itt=it
						a=0.0
						slide=False
					else:
						# No, evaluate
						ft,ct=self.funcon(xt)
						itt=self.niter
					
					ht=self.aggregateConstraintViolation(ct)
					
					# Store f, x, and cut factor
					fevaluated[ii]=ft
					if self.nc>0 and len(ct)==self.nc:
						cevaluated[:,ii]=ct
					cutfactor[ii]=a
					if slide:
						slidevec[ii]=1
					if self.model:
						# Append to point set
						if self.appendPoint(xt, ft, ct):
							# Added to list
							if self.powellUpdate:
								self.updateH_powell(x, f, c)
						
					# Extension point for linear update
					xe=None
					if (
						self.model and 
						self.linearUpdate and
						extensionDir is not None and
						extensionDir[ii] and 
						np.isfinite(fevaluated[ii]) and
						cutfactor[ii]==1.0 and
						slidevec[ii]==0
					):
						xe=x+2.0*steps[:,ii]*self.scaling
						fe,ce=self.funcon(xe)
						ite=self.niter
						he=self.aggregateConstraintViolation(ce)
						# No bound snapping this time
						if np.isfinite(fe):
							# Apply linear update
							self.updateH_linear(x0=x/self.scaling, d=steps[:,ii], 
								f0=f, fp=fevaluated[ii], fn=fe, 
								ap=1.0, an=2.0
							)
							# Simplical update, requires
							#   - step results in finite f
							#   - step was not cut to the origin (a>0.0)
							if (
								self.simplicalUpdate and
								np.isfinite(fe)
							):
								self.updateH_simplical(x, f, c, xe, fe, ce)
								# self.updateH_simplical()
								
							if self.appendPoint(xe, fe, ce):
								# Added to list
								if self.powellUpdate:
									self.updateH_powell(x, f, c)
							
					
					# Hessian update
					if self.model:
						# Linear update, requires
						#   - maximal positive basis
						#   - on even steps (d and -d available)
						#   - last two steps result in finite f
						#   - last two steps were not cut (a=1.0)
						#   - no sliding in last two steps
						if (
							self.linearUpdate and
							not self.protoset==0 and 
							(ii%2==1) and 
							np.isfinite(fevaluated[ii-1]) and np.isfinite(fevaluated[ii]) and
							cutfactor[ii-1]==1.0 and cutfactor[ii]==1.0 and
							slidevec[ii-1]==0 and slidevec[ii]==0
						): 
							# 2n steps, even +, odd -
							self.updateH_linear(x0=x/self.scaling, d=steps[:,ii-1], 
								f0=f, fp=fevaluated[ii-1], fn=fevaluated[ii], 
								ap=1.0, an=-1.0
							)
					
						# Simplical update, requires
						#   - step results in finite f
						#   - step was not cut to the origin (a>0.0)
						if (
							self.simplicalUpdate and
							np.isfinite(ft) and
							a>0
						):
							self.updateH_simplical(x, f, c, xt, ft, ct)
							# self.updateH_simplical()
					
					# Replace xt with xe if xe dominates xt
					if xe is not None and self.filt.dominates(fe, he, ft, ht):
						xt=xe
						ft=fe 
						ct=ce 
						ht=he
					
					# Check for improvement
					oldPosition=self.filt.position(f, self.aggregateConstraintViolation(c))
					#if oldPosition==0:
					#	oldPosition=1
					(dominates, dominated, accepted)=self.filt.accept(ft, ht, (xt, ct))
					position=self.filt.position(ft, ht)
					
					# if dominates and (position<oldPosition or position==0):
					if (dominates or accepted) and (position<=oldPosition):
					# if (dominates or accepted) and (position<=oldPosition or oldPosition==0): 
						ittype=2
					#elif accepted and position<=oldPosition:
					#	ittype=2
					#elif accepted and position==oldPosition:
					#	ittype=1
					#elif accepted and position<=oldPosition:
					#	ittype=1
					else:
						ittype=0
					
					#if ittype>ittypeCum:
					#	ittypeCum=ittype
					ittypeCum=ittype
					
					if ittype==2 or ittype==1: 
						x=xt
						f=ft
						c=ct
						it=itt
						pollSuccess=True
						iterationSuccess=True
						stepCut = stepCut or (a<1.0) or slide
						# Try speculative step if the poll step dominates
						if 1 or ittype==2:
							# Dominating
							if self.speculativePollSearch:
								trySpeculative=True
						if self.debug:
							DbgMsgOut("MADSOPT", "Poll step accepted, i=%d mesh=%.1e step=%.1e ittype=%d" % (ii, self.mesh, self.step, ittype))
							#print (self.x[:-3]**2).sum()**0.5, (xt[:-3]**2).sum()**0.5
						
						if self.greedy: # break always
						# if self.greedy and ittype==2: # break only on dominating
							break
					else:
						if self.debug:
							DbgMsgOut("MADSOPT", "Poll step rejected, i=%d mesh=%.1e step=%.1e ittype=%d" % (ii, self.mesh, self.step, ittype))
							#print (self.x[:-3]**2).sum()**0.5, (xt[:-3]**2).sum()**0.5
			
					# Exit if stopping condition satisfied
					if self.stop:
						break
				
				# Successfull poll step
				if pollSuccess:
					if self.debug:
						DbgMsgOut("MADSOPT", "Poll successfull")
				else: 
					if self.debug:
						DbgMsgOut("MADSOPT", "Poll failed")
						
				# Exit if stopping condition satisfied
				if self.stop:
					if self.debug:
						DbgMsgOut("MADSOPT", "Iteration i="+str(self.niter)+": stopping condition satisfied.")
					break
				
			# Speculative search
			if trySpeculative and not stepCut:
				# Remember the best point at the beginning of speculative search
				x0=x.copy()
				f0=f
				
				# Speculative search in direction (x0-xold) from x0
				scale=1
				ii=0
				doptimist=(x0-xold)
				while not self.stop and ii<self.speculativeSteps:
					if self.powellUpdate:
						self.updateOrigin_powell(x, f, c)
						
					# Calculate step
					delta=scale*doptimist
					
					# Incumbent
					fspecinc=f 
					hspecinc=self.aggregateConstraintViolation(c)
					
					# Snap to boundary
					if self.boundSnap:
						xt,a,slide = self.restoreFeasibility(x0, delta/self.scaling, None, None, None, None, True, self.boundSlide, mustRound)
					else:
						# Step is already rounded (i.e. it was a QP search or poll step)
						xt=x0+delta
						a=1.0
						slide=False
					
					# Was step cut back so much we do not move from x0? 
					# if withinTol(xt, x0, self.step*self.scaling*1e-14, 1e-14):
					if (xt==x).all():
						# Skip evaluation, stop speculative search
						break
					
					# Evaluate
					ft,ct=self.funcon(xt)
					itt=self.niter
					ht=self.aggregateConstraintViolation(ct)
					
					# Check if ft is finite
					if not np.isfinite(ft):
						# Infinite f, stop immediately
						break
					
					# Hessian update
					if self.model:
						# Append to point set (may cause problems with Powell update)
						if self.appendPoint(xt, ft, ct): 
							# Added to list
							if self.powellUpdate:
								self.updateH_powell(x, f, c)
							
						# Linear update, requires
						#   - step was finite (a>0)
						#   - step was not cut (a=1.0)
						#   - no sliding
						
						# d = (x0-xold)
						#
						# n        0        p
						# x0-d     x0       x0+scale*d
						# fn=fold  f0       fp=ft
						# an=-1    0        ap=scale
						if (
							self.linearUpdate and
							a==1.0 and
							not slide
						): 
							self.updateH_linear(
								x0=x0/self.scaling, d=doptimist/self.scaling, 
								fn=fold, f0=f0, fp=ft, 
								an=-1.0, ap=scale
							)
							
						# Simplical update, requires
						#   - step results in finite f
						if (
							self.simplicalUpdate and
							np.isfinite(ft)
						):
							self.updateH_simplical(x, f, c, xt, ft, ct)
							# self.updateH_simplical()
					
					# Add to filter, ignore point acceptance
					oldPosition=self.filt.position(f, self.aggregateConstraintViolation(c))
					#if oldPosition==0:
					#	oldPosition=1
					(dominates, dominated, accepted)=self.filt.accept(ft, ht, (xt, ct))
					position=self.filt.position(ft, ht)
					
					# Improvement check
					# if Filter.dominates(ft, ht, f, h): # Dominating the origin of speculative search
					# if not Filter.dominates(f, h, ft, ht): # Not dominated by the origin
					# if (dominates or accepted) and position<=oldPosition:
					if (dominates or accepted) and (position<=oldPosition):
					# if (dominates or accepted) and (position<=oldPosition or oldPosition==0): 
						x=xt
						f=ft
						c=ct
						it=itt
						scale*=2
						iterationSuccess=True
						stepCut = stepCut or (a<1.0) or slide
						if self.debug:
							DbgMsgOut("MADSOPT", "Speculative step accepted")
					else:
						# Reduce ittypeCum to 1 if speculative step fails
						if self.speculativeStepAffectsUpdate:
							ittypeCum=1
							if self.debug:
								DbgMsgOut("MADSOPT", "Speculative step rejected, mesh will not be changed")
						else:
							if self.debug:
								DbgMsgOut("MADSOPT", "Speculative step rejected")
						# Stop on failure
						break
					
					# If the step was cut, stop speculative search
					if a<1.0 or slide:
						break

					# Count speculative steps
					ii+=1
			
			
			# Update last direction of success
			if iterationSuccess:
				self.pGood=(x-self.xbestold)/self.scaling
				self.xbestold=x.copy()
				self.fbestold=f
						
			# Update step size
			mustRound=True
			# print "ittypeCum", ittypeCum
			if ittypeCum==2:
				if self.stepCutAffectsUpdate and stepCut:
					if self.debug:
						DbgMsgOut("MADSOPT", "Mesh unchanged (last step was cut), mesh=%.1e step=%.1e" % (self.mesh, self.step))
				else:
					self.meshExp+=1
					self.newMeshStep()
					
					# If maximal step exceeded, go back to previous step
					if self.step>self.maxStep:
						self.meshExp-=1
						self.newMeshStep()
						if self.debug:
							DbgMsgOut("MADSOPT", "Mesh unchanged (maximal step reached), mesh=%.1e step=%.1e" % (self.mesh, self.step))
					else:
						if self.debug:
							DbgMsgOut("MADSOPT", "Mesh coarsened, mesh=%.1e step=%.1e" % (self.mesh, self.step))
					
			elif ittypeCum==0:
				# Adjust mesh and step size
				self.meshExp-=1
				self.newMeshStep()
				
				# Smallest mesh size update 
				if self.meshExp<self.minMeshExp:
					# New minimal mesh basis 
					self.minMeshExp=self.meshExp
					self.minMeshSteps=self.generatePollSteps(reduction=True)
					mustRound=self.roundOnFinestMeshEntry
					
					if self.debug:
						DbgMsgOut("MADSOPT", "Finest mesh yet, mesh=%.1e step=%.1e" % (self.mesh, self.step))
				else:
					if self.debug:
						DbgMsgOut("MADSOPT", "Mesh refined, mesh=%.1e step=%.1e" % (self.mesh, self.step))
			else:
				if self.debug:
					DbgMsgOut("MADSOPT", "Mesh unchanged, mesh=%.1e step=%.1e" % (self.mesh, self.step))
			
			# print self.niter-itx1, self.ndim
			
			# Update hmax
			#if ittypeCum==1:
				## Improving iteration
				#fh, hh, data = self.filt.mostInfeasible()
				#if hh is not None:
					#self.filt.updateHmax(hh)
			
			# Print progress of Hessian approximation
			#print "%.1e" % ((self.hfunc(x)-self.H)**2/self.ndim**2).sum()**0.5, self.step, self.ndim, self.niter
			#e1=np.linalg.eig(self.H)[0]
			#e2=np.linalg.eig(self.hfunc(x))[0]
			#e1.sort()
			#e2.sort()
			#print self.hfunc(x)
			#print self.H
			#print e2
			#print np.linalg.cond(self.H), np.linalg.cond(self.hfunc(x))
			#print self.H
			# print (self.x[:-3]**2).sum()**0.5
			
			# Print progress of Hilbert Hessian
			# print "%.1e" % ((scipy.linalg.hilbert(self.ndim)-self.H)**2).sum()**0.5, self.step, self.ndim, self.niter
			
		if self.debug:
			DbgMsgOut("MADSOPT", "Search finished.")
		
		# print "qp", self.nqpok, "/", self.nqp, 
		
		#if outPolls>0:
		#	print ":: out steps:", outCnt, "out polls:", outPolls, "::", 
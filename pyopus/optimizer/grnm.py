"""
.. inheritance-diagram:: pyopus.optimizer.grnm
    :parts: 1

**Unconstrained grid-restrained Nelder-Mead simplex optimizer 
(PyOPUS subsystem name: GRNMOPT)**

A provably convergent version of the Nelder-Mead simplex algorithm. The 
algorithm performs unconstrained optimization. Convergence is achieved by 
restraining the simplex points to a gradually refined grid and by keeping the 
simplex internal angles away from 0. 

The algorithm was published in 

Buermen A., Puhan J., Tuma T.: Grid Restrained Nelder-Mead Algorithm.
Computational Optimization and Applications, vol. 34, pp. 359-375, 2006. 

There is an error in Algorithm 2, step 5. The correct step 5 is: 
If $f^{pe}<\min(f^{pe}, f^1, f^2, ..., f^{n+1})$ replace $x^i$ with $x^{pe}$ 
where $x^{i}$ denotes the point for which $f(x^i)$ is the lowest of all points. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import Optimizer
from nm import NelderMead

from numpy import abs, argsort, where, round, sign, diag, sqrt, log, array, zeros, dot, ones
from numpy.linalg import qr, det

import matplotlib.pyplot as pl

__all__ = [ 'GRNelderMead' ]

class GRNelderMead(NelderMead): 
	"""
	Unconstrained grid-restrained Nelder-Mead optimizer class
	
	Default values of the expansion (1.2) and shrink (0.25) coefficients are 
	different from the original Nelder-Mead values. Different are also the 
	values of relative tolerance (1e-16), and absolute function (1e-16) and 
	side length size (1e-9) tolerance. 
	
	*lam* and *Lam* are the lower and upper bound on the simplex side length 
	with respect to the grid. The shape (side length determinant) is bounded 
	with respect to the grid density by *psi*. 
	
	The grid density has a continuity bound due to the finite precision of 
	floating point numbers. Therefore the grid begins to behave as continuous 
	when its density falls below the relative(*tau_r*) and absolute (*tau_a*) 
	bound with respect to the grid origin. 
	
	If *originalGrid* is ``True`` the initial grid has the same density in all 
	directions (as in the paper). If ``False`` the initial grid density adapts 
	to the bounding box shape. 
	
	If *gridRestrainInitial* is ``True`` the points of the initial simplex are 
	restrained to the grid. 
	
	See the :class:`~pyopus.optimizer.nm.NelderMead` class for more 
	information. 
	"""
	def __init__(self, function, debug=0, fstop=None, maxiter=None, 
					reflect=1.0, expand=1.2, outerContract=0.5, innerContract=-0.5, shrink=0.25, 
					reltol=1e-15, ftol=1e-15, xtol=1e-9, simplex=None, 
					lam=2.0, Lam=2.0**52, psi=1e-6, tau_r=2.0**(-52), tau_a=1e-100, 
					originalGrid=False, gridRestrainInitial=False):
		NelderMead.__init__(self, function, debug, fstop, maxiter, 
										reflect, expand, outerContract, innerContract, shrink, 
										reltol, ftol, xtol, simplex)
		
		# Simplex
		self.simplex=None
		
		# Grid origin and scaling
		self.z=None
		self.Delta=None
		
		# Side length bounds wrt. grid 
		self.lam=lam
		self.Lam=Lam
		
		# Simplex shape lower bound
		self.psi=1e-6
		
		# Grid continuity bound (relative and absolute)
		self.tau_r=tau_r
		self.tau_a=tau_a
		
		# Create initial grid with the procedure described in the paper
		self.originalGrid=originalGrid
		
		# Grid restrain initial simplex
		self.gridRestrainInitial=gridRestrainInitial
		
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		NelderMead.check(self)
		
		if self.lam<=0:
			raise Exception, DbgMsg("GRNMOPT", "lambda should be positive.")
			
		if self.Lam<=0:
			raise Exception, DbgMsg("GRNMOPT", "Lambda should be positive.")
			
		if self.lam>self.Lam:
			raise Exception, DbgMsg("GRNMOPT", "Lambda should be greater or equal lambda.")
		
		if self.psi<0:
			raise Exception, DbgMsg("GRNMOPT", "psi should be greater or equal zero.")
		
		if self.tau_r<0 or self.tau_a<0:
			raise Exception, DbgMsg("GRNMOPT", "Relative and absolute grid continuity bounds should be positive.")

	def buildGrid(self, density=10.0):
		"""
		Generates the intial grid density for the algorithm. The grid is 
		determined relative to the bounding box of initial simplex sides. 
		*density* specifies the number of points in every grid direction that 
		covers the corresponding side of the bounding box. 
		
		If any side of the bounding box has zero length, the mean of all side 
		lengths divided by *density* is used as grid density in the 
		corresponding direction. 
		
		Returns the 1-dimensional array of length *ndim* holding the grid 
		densities. 
		"""
		if self.debug:
			DbgMsgOut("GRNMOPT", "Building initial grid for initial simplex.")
		# Side vectors (to the first point)
		v=self.simplex[1:,:]-self.simplex[0,:]
		
		if not self.originalGrid:
			# Maximal absolute components (bounding box sides)
			vmax=abs(v).max(0)
			
			# Maximal bounding box side
			vmax_max=vmax.max()
			
			# If any component maximum is 0, set it to vmax value
			vmax=where(vmax==0.0, vmax_max, vmax)
		
			# Bounding box dimensions divided by density
			return vmax/density
		else:
			# Shortest side length
			lmin=sqrt((v*v).sum(1).min())
			
			# Shortest side length divided by density, uniform across all dimensions
			return ones(self.ndim)*lmin/density
	
	def gridRestrain(self, x):
		"""
		Returns the point on the grid that is closest to *x*. 
		"""
		xgr=round((x-self.z)/self.delta)*self.delta+self.z
		
		return xgr
		
	def sortedSideVectors(self):
		"""
		Returns a tuple (*vsorted*, *lsorted*) where *vsorted* is an array 
		holding the simplex side vectors sorted by their length with longest 
		side first. The first index of the 2-dimensional array is the side 
		vector index while the second one is the component index. *lsorted* 
		is a 1-dimensional array of corresponding simplex side lengths. 
		"""
		# Side vectors
		v=self.simplex[1:,:]-self.simplex[0,:]
		
		# Get length
		l2=(v*v).sum(1)
		
		# Order by length (longest first)
		i=argsort(l2, 0, 'mergesort')	# shortest first
		i=i[-1::-1]	# longest first
		
		vsorted=v[i,:]
		lsorted=sqrt(l2[i])
		
		return (vsorted, lsorted)
		
	def reshape(self, v=None, Q=None, R=None):
		"""
		Reshapes simpex side vectors given by rows of *v* into orthogonal sides 
		with their bounding box bounded in length by *lam* and *Lam* with 
		respect to the grid density. If *v* is ``None`` it assumes that it is a 
		product of matrices *Q* and *R*. 
		
		Returns a tuple (*vnew*, *l*) where *vnew* holds the reshaped simplex 
		sides and *l* is the 1-dimensional array of reshaped side lengths. 
		"""
		# Rows are side vectors
		
		# QR decomposition of a matrix with side vectors as columns
		if v is not None:
			(Q, R)=qr(v.T)
		
		# Get scaling factors and their signs 
		Rdiag=R.diagonal()
		Rsign=sign(Rdiag)
		Rsign=where(Rsign!=0, Rsign, 1.0)
		
		# Get side lengths
		l=abs(Rdiag)
		
		# Calculate side length bounds
		norm_delta=sqrt((self.delta**2).sum())
		lower=self.lam*sqrt(self.ndim)*norm_delta/2
		upper=self.Lam*sqrt(self.ndim)*norm_delta/2
				
		# Bound side length
		l=where(l<=upper, l, upper)
		l=where(l>=lower, l, lower)
		
		# Scale vectors
		# Vectors are in columns of Q. Therefore transpose Q. 
		vnew=dot(diag(l*Rsign), Q.T)
		
		return (vnew, l)
		
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array or list *x0*. The length of the array 
		becomes the dimension of the optimization problem 
		(:attr:`ndim` member). 
		
		The initial simplex is built around *x0* by calling the 
		:meth:`buildSimplex` method with default values for the *rel* and *abs* 
		arguments. 
		
		If *x0* is a 2-dimensional array or list of size 
		(*ndim*+1) times *ndim* it specifies the initial simplex. 
		
		A corresponding grid is created by calling the :meth:`buildGrid` method. 
		
		The initial value of the natural logarithm of the simplex side vectors 
		determinant is calculated and stored. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("GRNMOPT", "Resetting.")
		
		# Make it an array
		x0=array(x0)
		
		# Is x0 a point or a simplex?
		if x0.ndim==1:
			# Point
			# Set x now
			NelderMead.reset(self, x0)
			
			if self.debug:
				DbgMsgOut("GRNMOPT", "Generating initial simplex from initial point.")
				
			sim=self.buildSimplex(x0)
			self._setSimplex(sim)
			self.delta=self.buildGrid()
			self.z=x0
		else:
			# Simplex or error (handled in _setSimplex())
			self._setSimplex(x0)
			self.delta=self.buildGrid()
			self.z=x0[0,:]
			
			if self.debug:
				DbgMsgOut("GRNMOPT", "Using specified initial simplex.")
				
			# Set x to first point in simplex after it was checked in _setSimplex()
			Optimizer.reset(self, x0[0,:])
			
		# Reset point moves counter 
		self.simplexmoves=zeros(self.ndim+1)
		
		# Make x tolerance an array
		self.xtol=array(self.xtol)
	
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("GRNMOPT", "Starting a run at i="+str(self.niter))

		# Checks
		self.check()
		
		# Reset stop flag
		self.stop=False
		
		# Grid-restrain initial simplex
		if self.gridRestrainInitial:
			for i in range(0, self.ndim+1):
				self.simplex[i,:]=self.gridRestrain(self.simplex[i,:])
		
		# Evaluate if needed
		if self.simplexf is None:
			self.simplexf=zeros(self.npts)
			for i in range(0, self.ndim+1):
				self.simplexf[i]=self.fun(self.simplex[i,:])
				if self.debug:
					DbgMsgOut("GRNMOPT", "Initial simplex point i="+str(self.niter)+": f="+str(self.simplexf[i]))
		
		# Loop
		while not self.stop:
			# Order simplex (best point first)
			self.orderSimplex()
			
			# Centroid
			xc=self.simplex[:-1,:].sum(0)/self.ndim
			
			# Worst point
			xw=self.simplex[-1,:]
			fw=self.simplexf[-1]
			
			# Second worst point
			xsw=self.simplex[-2,:]
			fsw=self.simplexf[-2]
			
			# Best point
			xb=self.simplex[0,:]
			fb=self.simplexf[0]
			
			# No shrink
			shrink=False
			
			# Reflect
			xr=self.gridRestrain(xc+(xc-xw)*self.reflect)
			fr=self.fun(xr)
			if self.debug:
				DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": reflect   : f="+str(fr))
			
			if fr<fb:
				# Try expansion
				xe=self.gridRestrain(xc+(xc-xw)*self.expand)
				fe=self.fun(xe)
				if self.debug:
					DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": expand    : f="+str(fe))
				
				if fe<fr:
					# Accept expansion
					self.simplex[-1,:]=xe
					self.simplexf[-1]=fe
					self.simplexmoves[-1]+=1
					if self.debug:
						DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": accepted expansion")
				else:
					# Accept reflection
					self.simplex[-1,:]=xr
					self.simplexf[-1]=fr
					self.simplexmoves[-1]+=1
					if self.debug:
						DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": accepted reflection after expansion")
						
			elif fb<=fr and fr<fsw:
				# Accept reflection
				self.simplex[-1,:]=xr
				self.simplexf[-1]=fr
				self.simplexmoves[-1]+=1
				if self.debug:
					DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": accepted reflection")
					
			elif fsw<=fr and fr<fw:
				# Try outer contraction
				xo=self.gridRestrain(xc+(xc-xw)*self.outerContract)
				fo=self.fun(xo)
				if self.debug:
					DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": outer con : f="+str(fo))
				if fo<fsw:
					# Accept
					self.simplex[-1,:]=xo
					self.simplexf[-1]=fo
					self.simplexmoves[-1]+=1
					if self.debug:
						DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": accepted outer contraction")
				else:
					# Shrink
					shrink=True
					
			elif fw<=fr:
				# Try inner contraction
				xi=self.gridRestrain(xc+(xc-xw)*self.innerContract)
				fi=self.fun(xi)
				if self.debug:
					DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": inner con : f="+str(fi))
				if fi<fsw:
					# Accept
					self.simplex[-1,:]=xi
					self.simplexf[-1]=fi
					self.simplexmoves[-1]+=1
					if self.debug:
						DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": accepted inner contraction")
				else:
					# Shrink
					shrink=True
					
			# self._checkSimplex()
			
			# self._plotSimplex()
			
			# Reshape, pseudo-expand, and shrink loop
			if shrink:
				# Normal NM steps failed
				
				# No reshape happened yet
				reshaped=False
				
				# Create origin vector and function value
				x0=zeros(self.ndim)
				f0=0.0
				
				# Check simplex shape
				# Simplex is already sorted
				(v, l)=self.sortedSideVectors()
				
				# Rows of v are side vectors, need to QR decompose a matrix 
				# with columns holding side vectors
				(Q, R)=qr(v.T)
				
				# Diagonal of R
				Rdiag=R.diagonal()
				
				# Grid density norm
				norm_delta=sqrt((self.delta**2).sum())
		
				if abs(Rdiag).min()<self.psi*sqrt(self.ndim)*norm_delta/2:
					# Shape not good, reshape
					(v, l)=self.reshape(Q=Q, R=R)
					reshaped=True
					
					# Origin for building the new simplex
					x0[:]=self.simplex[0,:]
					f0=self.simplexf[0]
					
					# Build new simplex
					for i in range(self.ndim):
						self.simplex[i+1,:]=self.gridRestrain(v[i,:]+x0)
						f=self.fun(self.simplex[i+1,:])
						self.simplexf[i+1]=f
						if self.debug:
							DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": reshape   : f="+str(f))
					
					self.simplexmoves[:]=0
				
				# Do not order simplex here, even if reshape results in a point that improves over x0. 
				# The algorithm in the paper orders the simplex here. This is not in the sense of the
				# Price-Coope-Byatt paper, which introduced pseudo-expand. Therefore do not sort. 
				
				# Centroid of the n worst points (or if a reshape took place - n new points)
				xcw=self.simplex[1:,:].sum(0)/self.ndim
					
				# Pseudo-expand point
				xpe=self.gridRestrain(xb+(self.expand/self.reflect-1.0)*(xb-xcw))
				fpe=self.fun(xpe)
				if self.debug:
					DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": pseudo exp: f="+str(fpe))
					
				# Check if there is any improvement
				if fpe<fb:
					# Pseudo-expand point is better than old best point
					self.simplex[0,:]=xpe
					self.simplexf[0]=fpe
					self.simplexmoves[0]+=1
					if self.debug:
						DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": accepted pseudo exp")
				elif self.simplexf.min()<fb:
					# One of the points obtained by reshape is better than old best point
					if self.debug:
						DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": accepted reshape")
				else:
					# No improvement, enter shrink loop
					
					# Even though we had a reshape the reshape did not improve the best point, 
					# and neither did pseudo-expand. This means that the best point before 
					# reshape is still the best point.
					
					if not reshaped:
						# No reshape yet, reshape now
						(v, l)=self.reshape(Q=Q, R=R)
						reshaped=True
											
						# Origin for building the new simplex
						x0[:]=self.simplex[0,:]
						f0=self.simplexf[0]
						
						self.simplexmoves[:]=0
						
						if self.debug:
							DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": reshape")
						
						# This is the first shrink step
						shrink_step=0
					else:
						# This is the second shrink step 
						# (first one happened at reshape and failed to produce improvement)
						shrink_step=1
					
					# Shrink loop
					while self.simplexf.min()>=f0:
						# Reverse side vectors if this is not the first shrink step
						if shrink_step>0:
							v=-v
						
						# If not first even shrink step, shrink vectors and check grid
						if shrink_step>=2 and shrink_step % 2 == 0:
							# Shrink vectors
							v=v*self.shrink
							l=l*self.shrink
							if self.debug:
								DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": shrink vectors")
							
							# Find shortest side vector
							i=argsort(l, 0, 'mergesort')
							lmin=l[i[0]]
							vmin=v[i[0],:]
							
							# Do we need a new grid?
							if lmin < self.lam*sqrt(self.ndim)*sqrt((self.delta**2).sum())/2:
								# New grid origin
								self.z=x0
								
								# New (refined) grid density
								vmin_norm=sqrt((vmin**2).sum())/sqrt(self.ndim)
								abs_vmin=abs(vmin)
								deltaprime=1.0/(250*self.lam*self.ndim)*where(abs_vmin>vmin_norm, abs_vmin, vmin_norm)
								
								# Enforce continuity bound on density
								contbound_r=abs(self.z)*self.tau_r
								contbound=where(contbound_r>self.tau_a, contbound_r, self.tau_a)
								deltanew=where(deltaprime>contbound, deltaprime, contbound)
								
								# Update grid density
								self.delta=where(deltanew<self.delta, deltanew, self.delta)
								
								if self.debug:
									DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": refine grid")
								
						# Evaluate points
						self.simplex[1:,:]=x0+v
						for i in range(self.ndim):
							self.simplex[i+1,:]=self.gridRestrain(x0+v[i,:])
							f=self.fun(self.simplex[i+1,:])
							self.simplexf[i+1]=f
							if self.debug:
								DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+(": shrink %1d: f=" % (shrink_step % 2))+str(f))
						
						# self._checkSimplex()
						
						# self._plotSimplex()
						
						# if f0!=self.simplexf[0] or (x0!=self.simplex[0,:]).any():
						# 	raise Exception, "x0, f0 not matching."
						
						# Stopping condition
						if (self.checkFtol() and self.checkXtol()) or self.stop:
							break
						
						# Increase shrink step counter
						shrink_step+=1
				
			# Check stopping condition
			if self.checkFtol() and self.checkXtol(): 
				if self.debug:
					DbgMsgOut("GRNMOPT", "Iteration i="+str(self.niter)+": simplex x and f tolerance reached, stopping.")
				break
			
		# Debug message
		if self.debug:
			DbgMsgOut("GRNMOPT", "Finished.")
	
	#
	# Internal functions for debugging purposes
	#
	
	def _checkSimplex(self):
		"""
		Check if the approximate cost function values corresponding to simplex 
		points are correct. 
		"""
		for i in range(0, self.ndim+1):
			ff=self.simplexf[i]
			f=self.fun(self.simplex[i,:], False)
			
			if ff!=f and self.debug:
				DbgMsgOut("GRNMOPT", "Simplex consistency broken for member #"+str(i))
				raise Exception, ""
	
	def _checkLogDet(self):
		"""
		Check if the natural logarithm of the simplex side vectors is correct. 
		"""
		(v,l)=self.sortedSideVectors()
		vdet=abs(det(v))
		DbgMsgOut("GRNMOPT", " logDet="+str(exp(self.logDet))+"  vdet="+str(vdet))
		if (1.0-exp(self.logDet)/vdet)>1e-3:
			raise Exception, DbgMsG("GRNMOPT", "Simplex determinat consistency broken. Relative error: %e" % (1.0-exp(self.logDet)/vdet))
			
	def _plotSimplex(self):
		"""
		Plot the projection of simplex side vectors to the first two dimensions. 
		"""
		p1=self.simplex[0,:2]
		p2=self.simplex[1,:2]
		p3=self.simplex[2,:2]
		
		pl.clf()
		pl.hold(True)
		pl.plot([p1[0]], [p1[1]], 'ro')
		pl.plot([p2[0]], [p2[1]], 'go')
		pl.plot([p3[0]], [p3[1]], 'bo')
		pl.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b')
		pl.plot([p1[0], p3[0]], [p1[1], p3[1]], 'b')
		pl.axis('equal')
		pl.hold(False)
		pl.show()

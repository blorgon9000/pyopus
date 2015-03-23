"""
.. inheritance-diagram:: pyopus.optimizer.sanm
    :parts: 1

**Unconstrained successive approximation Nelder-Mead simplex optimizer 
(PyOPUS subsystem name: SANMOPT)**

A provably convergent version of the Nelder-Mead simplex algorithm. The 
algorithm performs unconstrained optimization. Convergence is achieved by 
performing optimization on gradually refined approximations of the cost 
function and keeping the simplex internal angles away from 0. Function 
approximations are constructed with the help of a regular grid of points. 

The algorithm was published in 

Buermen A., Tuma T.: Unconstrained derivative-free optimization by 
successive approximation. Journal of computational and applied mathematics, 
vol. 223, pp. 62-74, 2009. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import Optimizer
from nm import NelderMead

from numpy import abs, argsort, where, round, sign, diag, sqrt, log, array, zeros, dot
from numpy.linalg import qr, det

# import matplotlib.pyplot as pl

__all__ = [ 'SANelderMead' ]

class SANelderMead(NelderMead): 
	"""
	Unconstrained successive approximation Nelder-Mead optimizer class
	
	Default values of the expansion (1.2) and shrink (0.25) coefficients are 
	different from the original Nelder-Mead values. Different are also the 
	values of relative tolerance (1e-16), and absolute function (1e-16) and 
	side length size (1e-9) tolerance. 
	
	*lam* and *Lam* are the lower and upper bound on the simplex side length 
	with respect to the grid. The shape (side length determinant) is bounded 
	with respect to the product of simplex side lengths with *c*. 
	
	The grid density has a continuity bound due to the finite precision of 
	floating point numbers. Therefore the grid begins to behave as continuous 
	when its density falls below the relative(*tau_r*) and absolute (*tau_a*) 
	bound with respect to the grid origin. 
	
	See the :class:`~pyopus.optimizer.nm.NelderMead` class for more 
	information. 
	"""
	def __init__(self, function, debug=0, fstop=None, maxiter=None, 
					reflect=1.0, expand=1.2, outerContract=0.5, innerContract=-0.5, shrink=0.25, 
					reltol=1e-15, ftol=1e-15, xtol=1e-9, simplex=None, 
					lam=2.0, Lam=2.0**52, c=1e-6, tau_r=2.0**(-52), tau_a=1e-100):
		NelderMead.__init__(self, function, debug, fstop, maxiter, 
										reflect, expand, outerContract, innerContract, shrink, 
										reltol, ftol, xtol, simplex)
		
		# Simplex
		self.simplex=None
		
		# Side vector determinant
		self.logDet=None
		
		# Grid origin and scaling
		self.z=None
		self.Delta=None
		
		# Side length bounds wrt. grid 
		self.lam=lam
		self.Lam=Lam
		
		# Simplex shape lower bound
		self.c=c
		
		# Grid continuity bound (relative and absolute)
		self.tau_r=tau_r
		self.tau_a=tau_a
		
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		NelderMead.check(self)
		
		if self.lam<=0:
			raise Exception, DbgMsg("SANMOPT", "lambda should be positive.")
			
		if self.Lam<=0:
			raise Exception, DbgMsg("SANMOPT", "Lambda should be positive.")
			
		if self.lam>self.Lam:
			raise Exception, DbgMsg("SANMOPT", "Lambda should be greater or equal lambda.")
		
		if self.c<0:
			raise Exception, DbgMsg("SANMOPT", "c should be greater or equal zero.")
		
		if self.tau_r<0 or self.tau_a<0:
			raise Exception, DbgMsg("SANMOPT", "Relative and absolute grid continuity bounds should be positive.")

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
			DbgMsgOut("SANMOPT", "Building initial grid for initial simplex.")
		# Side vectors (to the first point)
		v=self.simplex[1:,:]-self.simplex[0,:]
		
		# Maximal absolute components
		vmax=abs(v).max(0)
		
		# Mean maximal absolute component
		vmax_mean=vmax.mean()
		
		# If any component maximum is 0, set it to the mean vmax value
		ndx=where(vmax==0.0)[0]
		vmax[ndx]=vmax_mean
		
		return vmax/density
	
	def continuityBound(self):
		"""
		Finds the components of the vector for which the corresponding grid 
		density is below the continuity bound. 
		
		Returns a tuple (*delta_prime*, *ndx_cont*). *delta_prime* is the 
		vector of grid densities where the components that are below the 
		continuity bound are replaced with the bound. *ndx_cont* is a vector 
		of grid component indices for which the grid is below the continuity 
		bound. 
		"""
		# Find continuous components (delta below grid continuity or 0.0)
		rtol=abs(self.z*self.tau_r)
		conttol=where(rtol>self.tau_a, rtol, self.tau_a)
		ndx_cont=where((self.delta<conttol) | (self.delta==0.0))[0]
		
		# If below tolerance, set delta to tolerance
		delta_prime=where(self.delta>conttol, self.delta, conttol)
		
		return (delta_prime, ndx_cont)
		
	def gridRestrain(self, x):
		"""
		Returns the point on the grid that is closest to *x*. The componnets 
		of *x* that correspond to the grid directions for whch the density is 
		below the continuity bound are left unchanged. The remaining components 
		are rounded to the nearest value on the grid. 
		"""
		# Get bound delta and indices where delta was bound
		(delta_work, ndx_cont)=self.continuityBound()
		
		# Grid-restrain
		xgr=round((x-self.z)/delta_work)*delta_work+self.z
		
		# Copy continuous components
		xgr[ndx_cont]=x[ndx_cont]
		
		return xgr
		
	def grfun(self, x, count=True):
		"""
		Returns the value of the cost function approximation at *x* 
		corresponding to the current grid. If *count* is ``False`` the cost 
		function evaluation happened for debugging purposes and is not counted 
		or registered in any way.  
		"""
		return self.fun(self.gridRestrain(x), count)
	
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
		
	def reshape(self, v):
		"""
		Reshapes simpex side vectors given by *v* into orthogonal sides with 
		their bounding box bounded in length by *lam* and *Lam* with respect 
		to the grid density. 
		
		Returns a tuple (*vnew*, *logDet*, *l*) where *vnew* holds the reshaped 
		simplex sides, *logDet* is the natural logarithm of the reshaped 
		simplex's determinant, and *l* is the 1-dimensional array of reshaped 
		side lengths. 
		
		*logDet* is in fact the natural logarithm of the side lengths product, 
		because the reshaped sides are orthogonal. 
		"""
		# Rows are side vectors
		
		# QR decomposition of a matrix with side vectors as columns
		(Q, R)=qr(v.T)
		
		# Get scaling factors and their signs 
		Rdiag=R.diagonal()
		Rsign=sign(Rdiag)
		Rsign=where(Rsign!=0, Rsign, 1.0)
		
		# Get side lengths
		l=abs(Rdiag)
		
		# Calculate side length bounds
		norm_delta=sqrt((self.delta**2).sum())
		lower=self.lam*sqrt(self.ndim)*norm_delta
		upper=self.Lam*sqrt(self.ndim)*norm_delta
				
		# Bound side length
		l=where(l<=upper, l, upper)
		l=where(l>=lower, l, lower)
		
		# Scale vectors
		# Vectors are in columns of Q. Therefore transpose Q. 
		vnew=dot(diag(l*Rsign), Q.T)
		
		# Calculate log of side vector determinat
		logDet=log(l).sum()
		
		return (vnew, logDet, l)
		
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
		determinant is calculated and stored. This value gets updated at every 
		simplex algorithm step. The only time it needs to be reevaluated is at 
		reshape. But that is also quite simple because the reshaped simplex 
		is orthogonal. The only place where a full determinant needs to be 
		calculated is here. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("SANMOPT", "Resetting.")
		
		# Make it an array
		x0=array(x0)
		
		# Is x0 a point or a simplex?
		if x0.ndim==1:
			# Point
			# Set x now
			NelderMead.reset(self, x0)
			
			if self.debug:
				DbgMsgOut("SANMOPT", "Generating initial simplex from initial point.")
				
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
				DbgMsgOut("SANMOPT", "Using specified initial simplex.")
				
			# Set x to first point in simplex after it was checked in _setSimplex()
			Optimizer.reset(self, x0[0,:])
		
		# Reset point moves counter 
		self.simplexmoves=zeros(self.ndim+1)
		
		# Calculate log of side vector determinat
		(v, l)=self.sortedSideVectors()
		self.logDet=log(abs(det(v)))
		
		# Make x tolerance an array
		self.xtol=array(self.xtol)
	
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("SANMOPT", "Starting a run at i="+str(self.niter))

		# Checks
		self.check()
		
		# Reset stop flag
		self.stop=False
		
		# Evaluate initial simplex if needed
		if self.simplexf is None:
			self.simplexf=zeros(self.npts)
			for i in range(0, self.ndim+1):
				self.simplexf[i]=self.grfun(self.simplex[i,:])
				
				if self.debug:
					DbgMsgOut("SANMOPT", "Initial simplex point i="+str(self.niter)+": f="+str(self.simplexf[i]))
		
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
			xr=xc+(xc-xw)*self.reflect
			fr=self.grfun(xr)
			if self.debug:
				DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": reflect   : f="+str(fr))
			
			if fr<fb:
				# Try expansion
				xe=xc+(xc-xw)*self.expand
				fe=self.grfun(xe)
				if self.debug:
					DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": expand    : f="+str(fe))
				
				if fe<fr:
					# Accept expansion
					self.simplex[-1,:]=xe
					self.simplexf[-1]=fe
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.expand))
					
					if self.debug:
						DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": accepted expansion")
				else:
					# Accept reflection
					self.simplex[-1,:]=xr
					self.simplexf[-1]=fr
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.reflect))
					
					if self.debug:
						DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": accepted reflection after expansion")
			elif fb<=fr and fr<fsw:
				# Accept reflection
				self.simplex[-1,:]=xr
				self.simplexf[-1]=fr
				self.simplexmoves[-1]+=1
				self.logDet+=log(abs(self.reflect))
				if self.debug:
					DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": accepted reflection")
			elif fsw<=fr and fr<fw:
				# Try outer contraction
				xo=xc+(xc-xw)*self.outerContract
				fo=self.grfun(xo)
				if self.debug:
					DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": outer con : f="+str(fo))
				if fo<fw:
					# Accept
					self.simplex[-1,:]=xo
					self.simplexf[-1]=fo
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.outerContract))
					if self.debug:
						DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": accepted outer contraction")
				else:
					# Shrink
					shrink=True
			elif fw<=fr:
				# Try inner contraction
				xi=xc+(xc-xw)*self.innerContract
				fi=self.grfun(xi)
				if self.debug:
					DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": inner con : f="+str(fi))
				if fi<fw:
					# Accept
					self.simplex[-1,:]=xi
					self.simplexf[-1]=fi
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.innerContract))
					
					if self.debug:
						DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": accepted inner contraction")
				else:
					# Shrink
					shrink=True
					
			# self._checkSimplex()
			# self._checkLogDet()
			
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
				if self.logDet-log(l).sum()<log(self.c)*self.ndim:
					# Shape not good, reshape
					(v, self.logDet, l)=self.reshape(v)
					reshaped=True
										
					# Origin for building the new simplex
					x0[:]=self.simplex[0,:]
					f0=self.simplexf[0]
					
					# Build new simplex
					self.simplex[1:,:]=v+x0
					
					# Evaluate new points
					for i in range(1, self.ndim+1):
						f=self.grfun(self.simplex[i,:])
						self.simplexf[i]=f
						if self.debug:
							DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": reshape   : f="+str(f))
					
					self.simplexmoves[:]=0
					
				# Do not order simplex here, even if reshape results in a point that improves over x0. 
				# The algorithm in the paper orders the simplex here. This is not in the sense of the
				# Price-Coope-Byatt paper, which introduced pseudo-expand. Therefore do not sort. 
				
				# Centroid of the n worst points
				xcw=self.simplex[1:,:].sum(0)/self.ndim
				
				# Pseudo-expand point
				xpe=xb+(self.expand/self.reflect-1.0)*(xb-xcw)
				fpe=self.grfun(xpe)
				if self.debug:
					DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": pseudo exp: f="+str(fpe))
				
				# Check if there is any improvement
				if fpe<fb: 
					# Pseudo-expand point is better than old best point
					self.simplex[0,:]=xpe
					self.simplexf[0]=fpe 
					self.simplexmoves[0]+=1
					
					# Error in the paper - should be gammae/gammar, not gammae/gammar-1
					self.logDet+=log(abs(self.expand/self.reflect))	
					
					if self.debug:
						DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": accepted pseudo exp")
				elif self.simplexf.min()<fb: 
					# One of the points obtained by reshape is better than old best point
					if self.debug:
						DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": accepted reshape")
				else:
					# No improvement, enter shrink loop
					
					# Even though we had a reshape and the simplex was reordered, 
					# if we end up here the ordering didn't change the first point 
					# of the simplex (it must still be the best point). If not, 
					# we would end up in the if branch. 
					
					if not reshaped:
						# No reshape yet, reshape now
						(v, l)=self.sortedSideVectors()
						(v, self.logDet, l)=self.reshape(v)
						reshaped=True
						
						# Origin for building the new simplex and for shrink steps
						x0[:]=self.simplex[0,:]
						f0=self.simplexf[0]
						
						self.simplexmoves[:]=0
						
						if self.debug:
							DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": reshape")
						
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
							self.logDet+=log(abs(self.shrink))*self.ndim
							if self.debug:
								DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": shrink vectors")
							
							# Find shortest side vector
							i=argsort(l, 0, 'mergesort')
							lmin=l[i[0]]
							vmin=v[i[0],:]
							
							# Do we need a new grid?
							if lmin < self.lam*sqrt(self.ndim)*sqrt((self.delta**2).sum()):
								# New grid origin
								self.z=self.gridRestrain(x0)
								
								# Move x0 to grid origin to resolve inconsistency between 
								# the old and the new grid-restrained value of f. 
								x0=self.z
								self.simplex[0,:]=x0
								
								# New (refined) grid
								vmin_norm=sqrt((vmin**2).sum())/sqrt(self.ndim)
								abs_vmin=abs(vmin)
								self.delta=1.0/(128.0*self.lam*self.ndim)*where(abs_vmin>vmin_norm, abs_vmin, vmin_norm)
								
								if self.debug:
									DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": refine grid")
								
						# Evaluate points
						self.simplex[1:,:]=x0+v
						for i in range(1, self.ndim+1):
							f=self.grfun(self.simplex[i,:])
							self.simplexf[i]=f
							if self.debug:
								DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+(": shrink %1d: f=" % (shrink_step % 2))+str(f))
						
						# self._checkSimplex()
						# self._checkLogDet()
						
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
					DbgMsgOut("SANMOPT", "Iteration i="+str(self.niter)+": simplex x and f tolerance reached, stopping.")
				break
			
		# Debug message
		if self.debug:
			DbgMsgOut("SANMOPT", "Finished.")
	
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
			f=self.grfun(self.simplex[i,:], False)
			
			if ff!=f and self.debug:
				DbgMsgOut("SANMOPT", "Simplex consistency broken for member #"+str(i))
	
	def _checkLogDet(self):
		"""
		Check if the natural logarithm of the simplex side vectors is correct. 
		"""
		(v,l)=self.sortedSideVectors()
		vdet=abs(det(v))
		DbgMsgOut("SANMOPT", " logDet="+str(exp(self.logDet))+"  vdet="+str(vdet))
		if (1.0-exp(self.logDet)/vdet)>1e-3:
			raise Exception, DbgMsG("SANMOPT", "Simplex determinat consistency broken. Relative error: %e" % (1.0-exp(self.logDet)/vdet))
			
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

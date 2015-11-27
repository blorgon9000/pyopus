"""
.. inheritance-diagram:: pyopus.optimizer.nm
    :parts: 1

**Unconstrained Nelder-Mead optimizer (PyOPUS subsystem name: NMOPT)**

A very popular unconstrained optimization algorithm first published in [nm]_,

Unfortunately no convergence theory is available. There is even a 
counterexample available showing how the algorithm can fail. See [mck]_.

.. [nm] Nelder, J. A.,Mead, R.: A simplex method for function minimization. 
        Computer Journal, vol. 7, pp. 308-313, 1965. 

.. [mck] McKinnon, K. I. M.: Convergence of the Nelder-Mead Simplex Method to a 
        Nonstationary Point. SIAM Journal on Optimization, vol. 9, pp. 148-158, 1998. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg

from base import Optimizer

from numpy import array, abs, lexsort, zeros, where

__all__ = [ 'NelderMead' ]


class NelderMead(Optimizer):
	"""
	Nelder-Mead optimizer class
	
	*reflect*, *expand*, *outerContract*, *innerContract*, and *shrink* are 
	step size factors for the reflection, expansion, outer contraction, inner 
	contraction, and shrink step, respectively. 
	
	*expansion* must be above 1. *reflection* must be greater than 0 and 
	smaller than *expansion*. *outerContraction* must be between 0 and 1, 
	while *innerContraction* must be between -1 and 0. *shrink* must be 
	between 0 and 1. 
	
	*reltol* is the relative stopping tolerance. *ftol* and *xtol* are the 
	absolute stopping tolerances for cost function values at simplex points 
	and simlex side lengths. See the :meth:`checkFtol` and :meth:`checkXtol` 
	methods. 
	
	*simplex* is the initial simplex given by a (*ndim*+1) times *ndim* array 
	where every row corresponds to one simplex point. If *simplex is not given 
	an initial simplex is constructed around the initial point *x0*. See the 
	:meth:`buildSimplex` method for details. 
	
	If *looseContraction* is ``True`` the acceptance condition for contraction 
	steps requres that the new point is not worse than the worst point. This is 
	the behavior of the original algorithm. If this parameter is ``False`` 
	(which is also the default) the new point is accepted if it is better than 
	the worst point. 
	
	See the :class:`~pyopus.optimizer.base.Optimizer` class for more 
	information. 
	"""
	# Note: shrink coefficient should be <0.5, because larger values may cause stagnation
	#       due to roundoff errors (succesive shrinks do not result in a zero-diameter
	#       simplex after infinite number of steps). If the value of the coefficient is 0.5 
	#       or larger, roundoff errors may keep the simplex size at floating point precision 
	#       limit (relative tolerance 2**-52 = 2.22e-16) and it never reaches 0. 
	def __init__(self, function, debug=0, fstop=None, maxiter=None, 
					reflect=1.0, expand=2.0, outerContract=0.5, innerContract=-0.5, shrink=0.5, 
					reltol=1e-15, ftol=1e-15, xtol=1e-9, simplex=None, looseContraction=False):
		Optimizer.__init__(self, function, debug, fstop, maxiter)
		
		# Coefficients
		self.reflect=reflect
		self.expand=expand
		self.outerContract=outerContract
		self.innerContract=innerContract
		self.shrink=shrink
		
		# Stopping condition
		self.reltol=reltol
		self.ftol=ftol
		self.xtol=xtol
		
		# Simplex
		self.simplex=simplex
		
		# Modifications
		self.looseContraction=looseContraction
		
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		Optimizer.check(self)
		
		if self.expand<=1.0:
			raise Exception, DbgMsg("NMOPT", "Expansion coefficient should be gerater than 1.")
		
		if self.reflect>self.expand:
			raise Exception, DbgMsg("NMOPT", "Reflection coefficient should be smaller than expansion coefficient.")
		
		if self.reflect<=0.0:
			raise Exception, DbgMsg("NMOPT", "Reflection coefficient should be greater than 0.")
		
		if (self.outerContract<=0.0) or (self.outerContract>=self.reflect):
			raise Exception, DbgMsg("NMOPT", "Outer contraction coefficient should be between 0 and reflection coefficient.")
			
		if (self.innerContract>=0.0) or (self.innerContract<=-1.0):
			raise Exception, DbgMsg("NMOPT", "Inner contraction coefficient must be from (-1,0).")

		if (self.shrink<=0.0) or (self.shrink>=1.0):
			raise Exception, DbgMsg("NMOPT", "Shrink coefficient must be from (0,1).")
			
		if self.reltol<0:
			raise Exception, DbgMsg("NMOPT", "Negative relative tolerance.")
		
		if self.ftol<0:
			raise Exception, DbgMsg("NMOPT", "Negative f tolerance.")
		
		if (self.xtol<0).any():
			raise Exception, DbgMsg("NMOPT", "Negative x tolerance.")

	def _setSimplex(self, sim):
		"""
		Sets the initial simplex to the array given by *sim* and checks it. 
		"""
		self.npts=sim.shape[0]
		if sim.ndim!=2:
			raise Exception, DbgMsg("NMOPT", "Simplex must be a 2-dimensional array.")
		
		if sim.shape[0]!=sim.shape[1]+1:
			raise Exception, DbgMsg("NMOPT", "Simplex must have dimension+1 points.")
		
		self.simplexf=None
		self.simplex=sim
		
	def buildSimplex(self, x0, rel=0.05, abs=0.00025):
		"""
		Builds an initial simplex around point given by a 1-dimensional array 
		*x0*. *rel* and *abs* are used for the relative and absolute simplex 
		size. 
		
		The initial simplex has its first point positioned at *x0*. The *i*-th 
		point among the remaining *ndim* points is obtained by movin along the 
		*i*-th coordinate direction by :math:`x_0^i \\cdot rel` or *abs* if 
		:math:`x_0^i` is zero. 
		"""
		ndim=x0.shape[0]
		sim=zeros([ndim+1, ndim])
		sim[0,:]=x0
		
		for i in range(1,ndim+1):
			x=x0.copy()
			c=x[i-1]
			if c==0.0:
				x[i-1]+=abs
			else:
				x[i-1]+=c*rel
			sim[i,:]=x
		
		return sim
		
	def orderSimplex(self):
		"""
		Reorders the points and the corresponding cost function values of the 
		simplex in such way that the point with the lowest cost function value 
		is the first point in the simplex. 
		"""
		# Order simplex
		i=lexsort((-self.simplexmoves, self.simplexf), 0)
		self.simplexf=self.simplexf[i]
		self.simplex=self.simplex[i,:]
		self.simplexmoves=self.simplexmoves[i]
	
	def checkFtol(self):
		"""
		Checks the function value tolerance and returns ``True`` if the 
		function values are within :math:`\\max(ftol, reltol \\cdot |f_{best}|)` 
		of the point with the lowest cost function value (:math:`f_{best}`). 
		"""
		tol=max(self.ftol, self.reltol*abs(self.simplexf[0]))
		if abs(self.simplexf[1:]-self.simplexf[0]).max()<tol:
			return True
		else:
			return False
	
	def checkXtol(self):
		"""
		Returns ``True`` if the components of points in the simplex are within 
		:math:`max(reltol \\cdot |x_{best}^i|, xtol)` of the corresponding 
		components of the point with the lowest cost function value 
		(:math:`x_{best}`). 
		"""
		tolr=self.xtol*abs(self.simplex[0,:])
		tol=where(tolr>=self.xtol, tolr, self.xtol)
		if (abs(self.simplex[1:,:]-self.simplex[0,:]).max(0)<tol).all():
			return True
		else:
			return False
			
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array *x0*. The length of the array becomes the 
		dimension of the optimization problem (:attr:`ndim` member). 
		
		The initial simplex is built around *x0* by calling the 
		:meth:`buildSimplex` method with default values for the *rel* and 
		*abs* arguments. 
		
		If *x0* is a 2-dimensional array of size (*ndim*+1) times *ndim* it 
		specifies the initial simplex. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("NM", "Resetting.")
			
		# Make it an array
		x0=array(x0)

		# Is x0 a point or a simplex?
		if x0.ndim==1:
			# Point
			# Set x now
			Optimizer.reset(self, x0)
			
			if self.debug:
				DbgMsgOut("NM", "Generating initial simplex from initial point.")
				
			sim=self.buildSimplex(x0)
			self._setSimplex(sim)
		else:
			# Simplex or error (handled in _setSimplex())
			self._setSimplex(x0)
			
			if self.debug:
				DbgMsgOut("NM", "Using specified initial simplex.")
				
			# Set x to first point in simplex after it was checked in _setSimplex()
			Optimizer.reset(self, x0[0,:])
		
		# Reset point moves counter 
		self.simplexmoves=zeros(self.ndim+1)
		
		# Make x tolerance an array
		self.xtol=array(self.xtol)
		
		# Reset counters
		self.nr=0
		self.ne=0
		self.noc=0
		self.nic=0
		self.ns=0

		self.nrok=0
		self.neok=0
		self.nocok=0
		self.nicok=0
		
		self.icconv=0
		self.occonv=0
	
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("NM", "Starting a run at i="+str(self.niter))
			
		# Checks
		self.check()
		
		# Reset stop flag
		self.stop=False
		
		# Evaluate initial simplex if needed
		if self.simplexf is None:
			self.simplexf=zeros(self.npts)
			for i in range(0, self.ndim+1):
				self.simplexf[i]=self.fun(self.simplex[i,:])
				
				if self.debug:
					DbgMsgOut("NM", "Initial simplex point i="+str(self.niter)+": f="+str(self.simplexf[i]))
		
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
			fr=self.fun(xr)
			self.nr+=1
			if self.debug:
				DbgMsgOut("NM", "Iteration i="+str(self.niter)+": reflect   : f="+str(fr))
			
			if fr<fb:
				# Try expansion
				xe=xc+(xc-xw)*self.expand
				fe=self.fun(xe)
				self.ne+=1
				if self.debug:
					DbgMsgOut("NM", "Iteration i="+str(self.niter)+": expand    : f="+str(fe))
				
				if fe<fr:
					# Accept expansion
					self.simplex[-1,:]=xe
					self.simplexf[-1]=fe
					self.simplexmoves[-1]+=1
					self.neok+=1
				else:
					# Accept reflection
					self.simplex[-1,:]=xr
					self.simplexf[-1]=fr
					self.simplexmoves[-1]+=1
					self.nrok+=1
			elif fb<=fr and fr<fsw:
				# Accept reflection
				self.simplex[-1,:]=xr
				self.simplexf[-1]=fr
				self.simplexmoves[-1]+=1
				self.nrok+=1
			elif fsw<=fr and fr<fw:
				# Try outer contraction
				xo=xc+(xc-xw)*self.outerContract
				fo=self.fun(xo)
				self.noc+=1
				if fo<((1+self.outerContract)*fw+(self.reflect-self.outerContract)*fr):
					self.occonv+=1
				if self.debug:
					DbgMsgOut("NM", "Iteration i="+str(self.niter)+": outer con : f="+str(fo))
				if fo<fw or (self.looseContraction and fo==fw):
					# Accept
					self.simplex[-1,:]=xo
					self.simplexf[-1]=fo
					self.simplexmoves[-1]+=1
					self.nocok+=1
				else:
					# Shrink
					shrink=True
			elif fw<=fr:
				# Try inner contraction
				xi=xc+(xc-xw)*self.innerContract
				fi=self.fun(xi)
				self.nic+=1
				if fi<((1+self.innerContract)*fw+(self.reflect-self.innerContract)*fr):
					self.icconv+=1
				if self.debug:
					DbgMsgOut("NM", "Iteration i="+str(self.niter)+": inner con : f="+str(fi))
				if fi<fw or (self.looseContraction and fi==fw):
					# Accept
					self.simplex[-1,:]=xi
					self.simplexf[-1]=fi
					self.simplexmoves[-1]+=1
					self.nicok+=1
				else:
					# Shrink
					shrink=True
			
			# Shrink
			if shrink:
				for i in range(1, self.ndim+1):
					xs=xb+(self.simplex[i,:]-xb)*self.shrink
					fs=self.fun(xs)
					self.ns+=1
					self.simplex[i,:]=xs
					self.simplexf[i]=fs
					self.simplexmoves[i]+=1
					if self.debug:
						DbgMsgOut("NM", "Iteration i="+str(self.niter)+": shrink    : f="+str(fs))
			
			# Check stopping condition
			if self.checkFtol() and self.checkXtol():
				if self.debug:
					DbgMsgOut("NM", "Iteration i="+str(self.niter)+": simplex x and f tolerance reached, stopping.")
				break
				
		# Debug message
		if self.debug:
			DbgMsgOut("NM", "Finished.")

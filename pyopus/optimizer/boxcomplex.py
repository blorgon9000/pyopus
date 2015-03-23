"""
.. inheritance-diagram:: pyopus.optimizer.boxcomplex
    :parts: 1

**Box's constrained simplex optimizer (PyOPUS subsystem name: BCOPT)**

A version of the Box's simplex algorithm that is capable of handling box 
constraints. First published (as a general constrained algorithm) in 

Box, M. J.: A new method of constrained optimization and a comparison with 
other methods. Computer Journal, vol. 8, pp. 42-52, 1965. 

Unfortunately no convergence theory is available. 

A simplex in the Box's version of the algorithm is referred to as the 
**complex**. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import BoxConstrainedOptimizer

from numpy import sqrt, isinf, Inf, abs, floor, array, where, zeros, ones, argsort
from numpy import random

__all__ = [ 'BoxComplex' ]

class BoxComplex(BoxConstrainedOptimizer):
	"""
	Box's constrained simplex optimizer class
	
	*population_factor* determines the number of points in the automatically 
	generated somplex (*population_factor* times *ndim*). The number of points 
	in the complex must be greater than *ndim*. 
	
	*initial_box* is the relative size (see normalization scale in 
	:class:`~pyopus.optimizer.base.BoxConstrainedOptimizer` class) of the box 
	from which the initial complex points are chosen. 
	
	*reflection* and *contraction* are the reflection and contraction step 
	coefficients. *reflection* must be positive and greater than 1. 
	*contraction* must be between 0 and 1. 
	
	*gamma* specifies the relative distance (with respect to the normalization 
	defined by the bounds) that is considered as close enough. Used in the 
	process of deciding whether one the contractions should be performed 
	towards the best point in the simplex. 
	
	*gamma_stop* is the relative simplex size at which the algorithm stops. 
	If it is not given the value provided as *gamma* is used. 
	
	See the :class:`~pyopus.optimizer.base.coordinate.BoxConstrainedOptimizer` 
	class for more information. 
	"""
	def __init__(self, function, xlo, xhi, debug=0, fstop=None, maxiter=None, 
					population_factor=2.0, initial_box=1.0, reflection=1.3, contraction=0.5, 
					gamma=1e-5, gamma_stop=None):
		BoxConstrainedOptimizer.__init__(self, function, xlo, xhi, debug)
		
		# The size of the population relative to the dimension of the problem
		self.population_factor=population_factor
		
		# Relative size of the box from which the initial complex is chosen
		self.initial_box=initial_box
		
		# Reflection  and contraction factors
		self.reflect=reflection
		self.contract=contraction
		
		# Relative distance that is considered as close enough
		self.gamma=gamma
		
		# Stopping condition (relative simplex size)
		if gamma_stop is not None:
			self.gamma_stop=gamma_stop
		else:
			self.gamma_stop=gamma
		
		# Array with the complex's points
		self.csimplex=None
		
		# Number of points in the complex
		self.npts=None
	
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		BoxConstrainedOptimizer.check(self)
		
		# We require box constraints
		if (self.xlo is None):
			raise Exception, DbgMsg("BCOPT", "Lower bound is not defined.")
		
		if (self.xhi is None):
			raise Exception, DbgMsg("BCOPT", "Upper bound is not defined.")
		
		if ((self.initial_box>1) or (self.initial_box<=0)):
			raise Exception, DbgMsg("BCOPT", "Initial box must be from (0,1].")
		
		if (self.population_factor<=1):
			raise Exception, DbgMsg("BCOPT", "Population factor must be greater than 1.")
		
		if (self.reflect<=1):
			raise Exception, DbgMsg("BCOPT", "Reflection factor must be greater than 1.")
		
		if ((self.contract<=0) or (self.contract>=1)):
			raise Exception, DbgMsg("BCOPT", "Contraction factor must be from (0,1).")
			
		if (self.gamma<=0):
			raise Exception, DbgMsg("BCOPT", "Minimal normalized distance for contraction must be positive.")
		
		if (self.gamma_stop<=0):
			raise Exception, DbgMsg("BCOPT", "Normalized simplex size for stopping must be positive.")

	def _setComplex(self, csim):
		"""
		Sets the initial complex to the array given by *csim* and checks it. 
		"""
		self.npts=csim.shape[0]
		if csim.ndim!=2:
			raise Exception, DbgMsg("BCOPT", "Complex must be a 2-dimensional array.")
		
		if csim.shape[1]!=self.ndim:
			raise Exception, DbgMsg("BCOPT", "Complex points must be of same dimension as bounds.")
		
		if csim.shape[0]<self.ndim+1:
			raise Exception, DbgMsg("BCOPT", "Complex must have at least dimension+1 points.")
		
		# Complex rows are points
		for p in csim:
			if (p<self.xlo).any():
				raise Exception, DbgMsg("BCOPT", "One of the complex points violates lower bound.")
			if (p>self.xhi).any():
				raise Exception, DbgMsg("BCOPT", "One of the complex points violates upper bound.")
		
		self.csimplexf=None
		self.csimplex=csim
		
	def buildComplex(self, x0):
		"""
		Builds an initial complex around point given by a 1-dimensional array 
		*x0*. The number of points (*npts*) is determined as the closest 
		integer not exceeding the product of *x0* and *population_factor*. 
		
		A box with size equal to the product of :math:`n_s` and *initial_box* 
		around *x0* is created and *npts*-1 points are chosen randomly from the 
		box. Together with *x0* they form the initial complex. 
		
		See the 
		:class:`~pyopus.optimizer.base.coordinate.BoxConstrainedOptimizer` 
		class for the definition of :math:`n_s`. 
		"""
		# Calculate number of points
		self.npts=int(floor(self.ndim*self.population_factor))
		if (self.npts<self.ndim+1):
			raise Exception, DbgMsg("BCOPT", "Population factor times dimension must be at least dimension+1.")
			
		# Set up box for choosing complex points
		boxw=self.normScale*self.initial_box
		boxlo=x0-boxw/2
		boxhi=x0+boxw/2
		
		# Adjust lower complex box bound
		i=where(boxlo<self.xlo)
		boxlo[i]=self.xlo[i]
		boxhi[i]=self.xlo[i]+boxw[i]
		
		# Adjust upper complex box bound
		i=where(boxhi>self.xhi)
		boxhi[i]=self.xhi[i]
		boxlo[i]=self.xhi[i]-boxw[i]
		
		# Create complex
		csim=zeros([self.npts, self.ndim])
		csim[0,:]=x0
		for i in range(1,self.npts):
			csim[i,:]=boxlo+random.rand(self.ndim)*(boxhi-boxlo)
		
		return csim
		
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array or list *x0*. The length of the array 
		becomes the dimension of the optimization problem (:attr:`ndim` 
		member). The shape of *x0* must match that of *xlo* and *xhi*. 
		The initial complex is built around *x0* by calling the 
		:meth:`buildComplex` method. 
		
		If *x0* is a 2-dimensional array or list of size *npts* times *ndim* it 
		specifies the initial complex. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("BCOPT", "Resetting complex optimizer.")
		
		# Make it an array
		x0=array(x0)
		
		# Is x0 a point or a complex?
		if x0.ndim==1:
			# Point
			# Set x now
			BoxConstrainedOptimizer.reset(self, x0)
			
			if self.debug:
				DbgMsgOut("BCOPT", "Generating initial complex from initial point and random samples.")
				
			csim=self.buildComplex(x0)
			self._setComplex(csim)
		else:
			# Set x to first point in complex after it was checked in _setComplex()
			BoxConstrainedOptimizer.reset(self, x0[0,:])

			# Optimizer's reset method is not called yet, but we need self.ndim. Set it now. 
			self.ndim=self.xlo.shape[0]
			
			# Complex or error (handled in _setComplex())
			self._setComplex(x0)
			
			if self.debug:
				DbgMsgOut("BCOPT", "Using specified initial complex.")
				
			
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("BCOPT", "Starting a complex run at i="+str(self.niter))
		
		# Reset stop flag
		self.stop=False
		
		# Evaluate initial complex if needed
		if self.csimplexf is None:
			self.csimplexf=zeros(self.npts)
			for i in range(0, self.npts):
				self.csimplexf[i]=self.fun(self.csimplex[i,:])
				
				if self.debug:
					DbgMsgOut("BCOPT", "Initial complex point i="+str(self.niter)+": f="+str(self.csimplexf[i]))
		
		# Checks
		self.check()

		# Loop
		while not self.stop:
			# Order complex
			i=argsort(self.csimplexf, 0, 'mergesort')
			self.csimplexf=self.csimplexf[i]
			self.csimplex=self.csimplex[i,:]
			
			# Centroid
			xc=self.csimplex[:-1,:].sum(0)/(self.npts-1)
			
			# Check simplex size
			ssize=0
			for i in range(0, self.npts):
				ssize+=self.normDist(xc, self.csimplex[i,:])
			ssize=ssize/self.npts
			if ssize<self.gamma_stop:
				if self.debug:
					DbgMsgOut("BCOPT", "Iteration i="+str(self.niter)+": complex small enough, stopping")
				break
			
			# Worst point
			xw=self.csimplex[-1,:]
			fw=self.csimplexf[-1]
			
			# Best point
			xb=self.csimplex[0,:]
			fb=self.csimplexf[0]
			
			# Reflect
			xr=xc-(xw-xc)*self.reflect
			
			# Force within bounds
			self.bound(xr)
			
			# Evaluate
			fr=self.fun(xr)
			if self.debug:
				DbgMsgOut("BCOPT", "Iteration i="+str(self.niter)+": reflect   : f="+str(fr))
			
			if fr<fw:
				# Accept
				self.csimplex[-1,:]=xr
				self.csimplexf[-1]=fr
			else:
				# Not accepted, try to contract xr to centroid
				xx=xr
				fx=fr
				while ((self.normDist(xx, xc) > self.gamma) and (fx>=fw)):
					xx=xc+(xx-xc)*self.contract
					fx=self.fun(xx)
					
					if self.debug:
						DbgMsgOut("BCOPT", "Iteration i="+str(self.niter)+": contract c: f="+str(fx))
				
				if fx<fw:
					# Accept
					self.csimplex[-1,:]=xx
					self.csimplexf[-1]=fx
				else:
					# Not accepted, try to contract xr to best point
					xx=xr
					fx=fr
					while ((self.normDist(xx, xb) > self.gamma) and (fx>=fw)):
						xx=xb+(xx-xb)*self.contract
						fx=self.fun(xx)
						
						if self.debug:
							DbgMsgOut("BCOPT", "Iteration i="+str(self.niter)+": contract b: f="+str(fx))
					
					if fx<fw:
						# Accept
						self.csimplex[-1,:]=xx
						self.csimplexf[-1]=fx
					else:
						# Not accepted, copy best
						self.csimplex[-1,:]=xb
						self.csimplexf[-1]=fb
						
						if self.debug:
							DbgMsgOut("BCOPT", "Iteration i="+str(self.niter)+": copy     b: f="+str(fb))
			
		# Debug message
		if self.debug:
			DbgMsgOut("BCOPT", "ComplexSearch finished.")

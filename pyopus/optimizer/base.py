"""
.. inheritance-diagram:: pyopus.optimizer.base
    :parts: 1

**Base classes for optimization algorithms and plugins 
(PyOPUS subsystem name: OPT)**

Every optimization algorthm should be used in the following way. 

1. Create the optimizer object. 
2. Call the :meth:`reset` method of the object to set the initial point. 
3. Optionally call the :meth:`check` method to check the consistency of 
   settings. 
4. Run the algorithm by calling the :meth:`run` method. 

The same object can be reused by calling the :meth:`reset` method followed by 
an optional call to :meth:`check` and a call to :meth:`run`. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg

from numpy import sqrt, isinf, Inf, where, zeros, array
from numpy import random
from time import sleep

import sys

__all__ = [ 'RandomDelay', 'Plugin', 'Reporter', 'Stopper', 
			'Optimizer', 'BoxConstrainedOptimizer' ]

class RandomDelay(object):
	"""
	A wrapper class for introducing a random delay into function evaluation
	
	Objects of this class are callable. On call they evaluate the callable 
	object given by *obj* with the given args. A random delay with uniform 
	distribution specified by the *delayInterval* list is generated and 
	applied before the return value from *obj* is returned. The two members of 
	*delayInterval* list specify the lower and the upper bound on the delay. 
	
	Example::
	
	  from pyopus.optimizer.mgh import RandomDelay
	  
	  def f(x):
		return 2*x
	  
	  # Delayed evaluator with a random delay between 1 and 5 seconds
	  fprime=RandomDelay(f, [1.0, 5.0])
	  
	  # Return f(10) without delay
	  print f(x)
	  
	  # Return f(10) with a random delay between 1 and 5 seconds
	  print fprime(10)
	"""
	def __init__(self, obj, delayInterval): 
		self.obj=obj
		self.delayInterval=delayInterval
		self.start=delayInterval[0]
		self.range=delayInterval[1]-delayInterval[0]
		
		# Seed our own random generator, remember its state and restore original state
		state=random.get_state()
		random.seed()
		self.state=random.get_state()
		random.set_state(state)
	
	def __call__(self, *args): 
		# Switch to our own random generator
		state=random.get_state()
		random.set_state(self.state)
		
		# Generate delay
		s=self.start+self.range*random.rand(1)[0]
		
		# Remember our state and switch back random generator
		self.state=random.get_state()
		random.set_state(state)
		
		# Sleep
		sleep(s)
		
		# Evaluate
		return self.obj(*args)


class Plugin(object):
	"""
	Base class for optimization algorithm plugins.
	
	A plugin is a callable object with the following calling convention
	
	``plugin_object(x, f, opt, annotation)``
	
	where x is an array representing a point in the search space and *f* is 
	its corresponding cost function value. *opt* is a reference to the 
	optimization algorithm where the plugin is installed. 
	
	The plugin's job is to produce output or update some internal structures 
	with the data collected from *x*, *f*, and *opt*. For transferring 
	auxiliary data that is not included in *x* or *f* from remote workers the 
	annotations mechanism can be used. 
	
	A plugin collects this auxiliary data from local structures returns it when 
	it is called with *annotation* set to ``None``. We say that the annotation 
	is produced. 
	
	If the *annotation* argument that is not ``None`` is passed to the plugin 
	at call time the plugin must use this data for updating the local 
	structures. We say that the annotation is consumed. 
	
	Usually the annotation is produced on remote workers when the cost function 
	is evaluated. It is the job of the optimization algorithm to send the value 
	of *f* along with the corresponding annotations from worker (where the 
	evaluation took place) to the master where the annotation is consumed. 
	This way the master can access all the auxiliary data which is normally 
	produced only on the machine where the evaluation of the cost function 
	takes place. 
	
	If *quiet* is ``True`` the plugin supresses its output. Useful on remote 
	workers where the output of a plugin is often uneccessary. 
	"""
	def __init__(self, quiet=False):
		# Stop flag
		self.stop=False
		
		# Should the reporter be silent
		self.quiet=quiet
		
		pass
	
	def reset(self):
		"""
		Resets the plugin to its initial state. 
		"""
		pass
	
	def setQuiet(self, quiet):
		"""
		If *quiet* is ``True`` all further output is supressed by the plugin. 
		To reenable it, call this method with *quiet* set to ``False``. 
		"""
		self.quiet=quiet
	
	def __call__(self, x, f, opt, annotation=None): 
		return None

class Reporter(Plugin):		
	"""
	A base class for plugins used for reporting the cost function value and its 
	details. 
	
	If *onImprovement* is ``True`` the cost function value is reported only 
	when it improves on the best-yet (lowest) cost function value. 
	
	*onIterStep* specifies the number of calls to the reporter after which the 
	cost function value is reported regardless of the *onImprovement* setting. 
	Setting *onIterStep* to 0 disables this feature. 
	
	The value of the cost function at the first call to the reporter (after the 
	last call to the :meth:`reset` method) is always reported. 
	
	This basic reporter class produces and consumes no annotations. Advanced 
	reporters which report the details of the cost function must implement the 
	annotations mechanism in order for them to work correctly in parallel 
	computing environments. 
	
	The iteration number is obtained from the :attr:`niter` member of the *opt* 
	object passed when  the reporter is called. The best-yet value of the cost 
	function is obtained from the :attr:`f` member of the *opt* object. This 
	member is ``None`` at the first iteration. Note that the :attr:`f` member 
	is updated after all plugins are called. 
	
	The :attr:`bestIter` member of *obj* lists the iteration in which the 
	best-yet function value was obtained. 
	"""
	
	def __init__(self, onImprovement=True, onIterStep=1):
		Plugin.__init__(self)
		
		# onIterStep=0 turns of regular reporting
		self.onImprovement=onImprovement
		self.onIterStep=onIterStep
		
	def __call__(self, x, f, opt, annotation=None):
		# This reporter uses no annotations
		
		# We have improvement at first iteration or whenever optimizer says there is improvement
		if opt.f is None or opt.niter==opt.bestIter:
			improved=True
		else:
			improved=False
		
		# Report
		if not self.quiet:
			if (self.onImprovement and improved) or (self.onIterStep!=0 and opt.niter%self.onIterStep==0): 
				print("iter="+str(opt.niter)+" f="+str(f)+" fbest="+str(opt.f))

		# Return annotation
		return None

class Stopper(Plugin):
	"""
	Stopper plugins are used for stopping the optimization algorithm when a 
	particular condition is satisfied. These plugins produce an annotation. 
	The annotation is usually a flag. If this flag is ``True`` the optimization 
	algorithm should be stopped. Annotations enable the stopper to stop the 
	optimization algorithm if the stopping condition is satisfied on some 
	remote worker. 
	
	The actuall stopping (consumption of the annotation) is achieved by 
	setting the :attr:`stop` member of the *opt* object to ``True``. 
	
	This base class merely propagates the stopping condition in the *opt* 
	object at the worker to the *opt* object of the master. 
	"""
	
	def __call__(self, x, f, opt, annotation=None):
		if annotation is not None: 
			# Use annotation for setting the stop condition
			opt.stop=opt.stop or annotation
		
		# Return annotation
		return opt.stop

class Optimizer(object):
	"""
	Base class for unconstrained optimization algorithms. 
	
	*function* is the cost function (Python function or any other callable 
	object) which should be minimzied. 
	
	If *debug* is greater than 0, debug messages are printed at standard output. 
	
	*fstop* specifies the cost function value at which the algorithm is 
	stopped. If it is ``None`` the algorithm is never stopped regardless of 
	how low the cost function's value gets. 
	
	*maxiter* specifies the number of cost function evaluations after which 
	the algorithm is stopped. If it is ``None`` the number of cost function 
	evaluations is unlimited. 
	
	The following members are available in every object of the 
	:class:`Optimizer` class
	
	* :attr:`ndim` - dimension of the problem. Updated when the :meth:`reset` 
	  method is called. 
	* :attr:`niter` - the consecutive number of cost function evaluation.
	* :attr:`x` - the argument to the cost function resulting in the best-yet 
	  (lowest) value.
	* :attr:`f` - the best-yet value of the cost function. 
	* :attr:`bestIter` - the iteration in which the best-yet value of the cost 
	  function was found. 
	* :attr:`bestAnnotations` - a list of annotations produced by the installed 
	  plugins for the best-yet value of the cost function.
	* :attr:`stop` - boolean flag indicating that the algorithm should stop. 
	* :attr:`annotations` - a list of annotations produced by the installed 
	  plugins for the last evaluated cost function value
	* :attr:`plugins` - a list of installed plugin objects
	
	Plugin objects are called at every cost function evaluation or whenever a 
	remotely evaluated cost function value is registered by the 
	:meth:`newResult` method. 
	
	Values of *x* and related members are arrays. 
	"""
	def __init__(self, function, debug=0, fstop=None, maxiter=None):
		# Function subject to optimization, must be picklable for parallel optimization methods. 
		self.function=function
		
		# Debug mode flag
		self.debug=debug
		
		# Problem dimension
		self.ndim=None
		
		# Stopping conditions
		self.fstop=fstop
		self.maxiter=maxiter
		
		# Iteration counter
		self.niter=0
		
		# Best-yet point
		self.x=None
		self.f=None
		self.bestIter=None
		self.bestAnnotations=None
		
		# Plugins
		self.plugins=[]
		
		# Stop flag
		self.stop=False
		
		# Annotations produced at last cost function evaluation
		self.annotations=None
		
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception 
		if something is wrong. 
		"""
		if (self.fun is None):
			raise Exception, DbgMsg("OPT", "Cost function not defined.")
		
		if self.maxiter is not None and self.maxiter<1:
			raise Exception, DbgMsg("OPT", "Maximum number of iterations must be at least 1.")

	def installPlugin(self, plugin, index=-1):
		"""
		Installs a plugin object at *index*-th position in the plugins list. 
		The old plugin at that position is discarded. 
		
		If *index* is negative, the plugin is installed at the end of the 
		plugins list. 
		
		Returns the index of the installed plugin in the plugins list. 
		"""
		if index<0:
			self.plugins.append(plugin)
			return len(self.plugins)-1
		else:
			self.plugins[index]=plugin
			return index
	
	def fun(self, x, count=True):
		"""
		Evaluates the cost function at *x* (array). If *count* is ``True`` the 
		:meth:`newResult` method is invoked with *x*, the obtained cost 
		function value, and *annotations* argument set to ``None``. This means 
		that the reuslt is registered (best-yet point information are updated) 
		and the plugins are called to produce annotations. 
		
		Use ``False`` for *count* if you need to evaluate the cost function 
		for degugging purposes. 
		
		Returns the value of the cost function at *x*. 
		"""
		# Evaluate
		f=self.function(x)
		
		# Do the things that need to be done with a new result
		# No annotation is provided telling the newResult() method that the 
		# function evaluation actually happened in this process. 
		if count:
			self.newResult(x, f)
			
		return f
	
	def newResult(self, x, f, annotations=None):
		"""
		Registers the cost function value *f* obtained at point *x* with 
		annotations list given by *annotations*. 
		
		Increases the :attr:`niter` member to reflect the iteration number of 
		the point being registered and updates the :attr:`f`, :attr:`x`, and 
		:attr:`bestIter` members. 
		
		If the *annotations* argument is not given, produces annotations by 
		calling the plugin objects from the :attr:`plugins` member. The 
		plugins produce annotations that get stored in the :attr:`annotations` 
		member. If *f* improves on the best-yet value, the obtained annotations 
		are also stored in the :attr:`bestAnnotations` member. 
		
		If the *annotations* argument is given, it must be a list with as many 
		members as there are plugin objects installed in the optimizer. The 
		annotations list is stored in the :attr:`annotations` member. If *f* 
		improves the best-yet value annotations are also stored in the 
		:attr:`bestAnnotations` member.  Plugins are invoked with *annotations* 
		as their last agrument thus applying the annotations to the objects in 
		the local Python interpreter. 
		
		Finally it is checked if the best-yet value of cost function is below 
		:attr:`fstop` or the number of iterations exceeded :attr:`maxiter`. If 
		any of these two conditions is satisfied, the algorithm is stopped by 
		setting the :attr:`stop` member to ``True``. 
		"""
		# Increase evaluation counter
		self.niter+=1
	
		# Update best-yet
		if (self.f is None) or (self.f>f):
			self.f=f
			self.x=x
			self.bestIter=self.niter
			updated=True
		else:
			updated=False
		
		# If no annotation is given, function evaluation happened in this process
		if annotations is None:
			# Produce a list of None annotations for plugins
			self.annotations=[]
			for plugin in self.plugins:
				self.annotations.append(None)
		else:
			# Put annotations in annotations list
			self.annotations=annotations
		
		# Update best-yet annotations
		if updated:
			self.bestAnnotations=self.annotations
		
		# Annotations are set up. Call plugins. 
		nplugins=len(self.plugins)
		for index in range(0,nplugins):
			plugin=self.plugins[index]
			if plugin is not None:
				stopBefore=self.stop
				
				self.annotations[index]=plugin(x, f, self, self.annotations[index])
				
				if self.debug and self.stop and not stopBefore: 
					DbgMsgOut("OPT", "Run stopped by plugin object.")
		
		# Force stop condition on f<=fstop
		if (self.fstop is not None) and (self.f<=self.fstop):
			self.stop=True
			
			if self.debug:
				DbgMsgOut("OPT", "Function fell below desired value. Stopping.")
		
		# Force stop condition on niter>maxiter
		if self.maxiter is not None and self.niter>=self.maxiter:
			self.stop=True
			
			if self.debug:
				DbgMsgOut("OPT", "Maximal number of iterations exceeded. Stopping.")
	
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array or list *x0*. The length of the array 
		becomes the dimension of the optimization problem 
		(:attr:`ndim` member). 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("OPT", "Resetting.")
			
		# Determine dimension of the problem from initial point
		x0=array(x0)
		self.ndim=x0.shape[0]
		
		if x0.ndim!=1:
			raise Exception, DbgMsg("OPT", "Initial point must be a vector.")
			
		# Store initial point
		self.x=x0.copy()
		self.f=None
		
		# Reset iteration counter
		self.niter=0
		
		# Reset plugins
		for plugin in self.plugins:
			if plugin is not None:
				plugin.reset()
			
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Does nothing, reimplement this in a derived class. 
		pass
 

class BoxConstrainedOptimizer(Optimizer):
	"""
	Box-constrained optimizer class 
	
	*xlo* and *xhi* are 1-dimensional arrays or lists holding the lower and 
	upper bounds on the components of *x*. Some algorithms allow the 
	components of *xlo* to be :math:`- \infty` and the components of *xhi* to 
	be :math:`+ \infty`. 
	
	See the :class:`optimizer` class for more information. 
	"""
	def __init__(self, function, xlo=None, xhi=None, debug=0, fstop=None, maxiter=None):
		Optimizer.__init__(self, function, debug, fstop, maxiter)
		
		# Constraints
		self.xlo=xlo
		self.xhi=xhi
	
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		if self.xlo is not None:
			if (self.xlo.ndim!=1 or self.xhi.ndim!=1): 
				raise Exception, DbgMsg("OPT", "Bounds must be one-dimensional vectors.")
		
		if self.xhi is not None:
			if (self.xlo.shape[0]!=self.xhi.shape[0]):
				raise Exception, DbgMsg("OPT", "Bounds must match in length.")
		
		if (self.xlo is not None) and (self.xhi is not None):
			if (self.xlo>=self.xhi).any():
				raise Exception, DbgMsg("OPT", "Lower bound must be below upper bound.")
				
		Optimizer.check(self)
		
		if self.x is not None:
			if self.xlo is not None:
				if (self.x<self.xlo).any():
					raise Exception, DbgMsg("OPT", "Current point violates lower bound.")
			if self.xhi is not None:
				if (self.x>self.xhi).any():
					raise Exception, DbgMsg("OPT", "Current point violates upper bound.")
		
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array or list *x0*. The length of the array 
		becomes the dimension of the optimization problem 
		(:attr:`ndim` member). 
		The shape of *x* must match that of *xlo* and *xhi*. 
		
		Normalization origin :math:`n_o` and scaling :math:`n_s` are calculated 
		from the values of *xlo*, *xhi*, and intial point *x0*:
		
		* If a lower bound :math:`x_{lo}^i` is :math:`- \infty` 
		  
		  .. math::
		  
		    n_s^i &= 2 (x_{hi}^i - x_0^i) \\\\
		    n_o^i &= x_0^i - n_s^i / 2
		  
		* If an upper bound :math:`x_{hi}^i` is :math:`+ \infty` 
		
		  .. math::

		    n_s^i &= 2 (x_0^i - x_{lo}^i) \\\\
		    n_o^i &= x_0^i - n_s^i / 2
			
		* If both lower and upper bound are infinite
		  
		  .. math::
		  
		    n_s^i &= 2 \\\\
		    n_o^i &= x_0^i 
			
		* If bouth bounds are finite
		
		  .. math::
		    n_s^i &= x_{hi}^i - x_{lo}^i \\\\
		    n_o^i &= x_{lo}
		"""
		x0=array(x0)
		Optimizer.reset(self, x0)
		
		if self.debug:
			DbgMsgOut("BCOPT", "Resetting.")
		
		# The dimension is now known
		# Set default bounds to Inf, -Inf
		if self.xlo is None:
			self.xlo=zeros(self.ndim)
			self.xlo.fill(-Inf)
		else:
			self.xlo=array(self.xlo)
		
		if self.xhi is None:
			self.xhi=zeros(self.ndim)
			self.xhi.fill(Inf)
		else:
			self.xhi=array(self.xhi)
		
		if (x0<self.xlo).any():
			raise Exception, DbgMsg("OPT", "Initial point violates lower bound.")
		
		if (x0>self.xhi).any():
			raise Exception, DbgMsg("OPT", "Initial point violates upper bound.")
		
		# Normalization (defaults to low-high range)
		self.normScale=self.xhi-self.xlo
		self.normOrigin=self.xlo.copy()
				
		# Update normalization origin and scaling for unbound coordinates based on initial point. 
		ndx=where(isinf(self.xlo) & ~isinf(self.xhi))
		if len(ndx[0])>0:
			self.normScale[ndx]=2*(self.xhi[ndx]-x0[ndx])
			self.normOrigin[ndx]=x0[ndx]-self.normScale[ndx]/2.0
		
		ndx=where(~isinf(self.xlo) & isinf(self.xhi))
		if len(ndx[0])>0:
			self.normScale[ndx]=2*(x0[ndx]-self.xlo[ndx])
			self.normOrigin[ndx]=xlo[ndx]
			
		ndx=where(isinf(self.xlo) & isinf(self.xhi))
		if len(ndx[0])>0:
			self.normScale[ndx]=x0[ndx]*0.0+2.0
			self.normOrigin[ndx]=x0[ndx]
		
	def bound(self, x):
		"""
		Fixes components of *x* so that teh bounds are enforced. If a component 
		is below lower bound it is set to the lower bound. If a component is 
		above upper bound it is set to the upper bound. 
		"""
		pos=where(x<self.xlo)[0]
		x[pos]=self.xlo[pos]
		
		pos=where(x>self.xhi)[0]
		x[pos]=self.xhi[pos]
	
	def normDist(self, x, y):
		"""
		Calculates normalized distance between *x* and *y*. 
		
		Normalized distance is calculated as 
		
		.. math:: \\sqrt{\\sum_{i=1}^{n} (\\frac{x^i - y^i}{n_s^i})^2}
		"""
		d=(x-y)
		return sqrt(sum((d/self.normScale)**2))
	
	def normalize(self, x):
		"""
		Returnes a normalized point *y* corresponding to *x*. 
		Components of *y* are
		
		.. math:: y^i = \\frac{x^i - n_o^i}{n_s^i}
		
		If both bounds are finite, the result is within the :math:`[0,1]` 
		interval. 
		"""
		return (x-self.normOrigin)/self.normScale
	
	def denormalize(self, y):
		"""
		Returns a denormalized point *x* corresponding to *y*. 
		Components of *x* are
		
		.. math:: x^i = y^i n_s^i + n_o^i
		"""
		return y*self.normScale+self.normOrigin
	
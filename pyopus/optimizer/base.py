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
from cache import Cache

import numpy as np
from numpy import sqrt, isinf, Inf, where, zeros, array, concatenate
from numpy import random
from time import sleep

import sys

__all__ = [ 'RandomDelay', 'Plugin', 'Reporter', 'Stopper', 'Annotator', 'AnnotatorGroup', 
		'Optimizer', 'BoxConstrainedOptimizer', 'ConstrainedOptimizer',
		'UCEvaluator', 'BCEvaluator', 'CEvaluator', 
		'normalizer', 'denormalizer', 'CostCollector',  ]

class RandomDelay(object):
	"""
	A wrapper class for introducing a random delay into function evaluation
	
	Objects of this class are callable. On call they evaluate the callable 
	object given by *obj* with the given args. A random delay with uniform 
	distribution specified by the *delayInterval* list is generated and 
	applied before the return value from *obj* is returned. The two members of 
	*delayInterval* list specify the lower and the upper bound on the delay. f
	
	Example::
	
	  from pyopus.optimizer.base import RandomDelay
	  
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
	
	``plugin_object(x, ft, opt)``
	
	where *x* is an array representing a point in the search space and *ft* is 
	the corresponding cost function value. If *ft* is a tuple the first member 
	is the cost function value, the second member is an array of nonlinear 
	constraint function values, and the third member is the vector of constraint 
	violations. A violation of 0 means that none of the nonlinear constraints 
	is violated. Positive/negative violations indicate a upper/lower bound on 
	the constraint function is violated. *opt* is a reference to the optimization 
	algorithm where the plugin is installed. 
	
	The plugin's job is to produce output or update some internal structures 
	with the data collected from *x*, *ft*, and *opt*. 
	
	If *quiet* is ``True`` the plugin supresses its output. This is useful on 
	remote workers where the output of a plugin is often uneccessary. 
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
	
	def __call__(self, x, ft, opt): 
		pass

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
	last call to the :meth:`reset` method) is always reported. The mechanism 
	for detecting an improvement is simple. It compares the :attr:`niter` and 
	the :attr:`bestIter` attributes of *opt*. The latter becomes equal to the 
	former whenever a new better point is found. See the :meth:`updateBest` 
	method of the optimization algorithm for more details. 
	
	*ft* is a tuple for nonlinearly constrained optimizers. More details can be 
	found in the documantation of the :class:`Plugin` class. 
	
	The reporter prints the function value and the cumulative constraint 
	violation (h). 
	
	The iteration number is obtained from the :attr:`niter` member of the *opt* 
	object passed when  the reporter is called. The best-yet value of the cost 
	function is obtained from the :attr:`f` member of the *opt* object. This 
	member is ``None`` at the first iteration. 
	"""
	
	def __init__(self, onImprovement=True, onIterStep=1):
		Plugin.__init__(self)
		
		# onIterStep=0 turns of regular reporting
		self.onImprovement=onImprovement
		self.onIterStep=onIterStep
		
	def __call__(self, x, ft, opt):
		# We have improvement at first iteration or whenever optimizer says there is improvement
		if opt.f is None or opt.niter==opt.bestIter:
			improved=True
		else:
			improved=False
		
		# Report
		if not self.quiet:
			if (self.onImprovement and improved) or (self.onIterStep!=0 and opt.niter%self.onIterStep==0): 
				# Is ft a tuple
				if type(ft) is tuple:
					# Nonlinear constraint violations given, print cumulative squared violation (h)
					print("iter="+str(opt.niter)+" f="+str(ft[0])+" h="+str((ft[2]**2).sum())+" fbest="+str(opt.f))
				else:
					# No nonlinear constraints, just f
					print("iter="+str(opt.niter)+" f="+str(ft)+" fbest="+str(opt.f))

class Stopper(Plugin):
	"""
	Stopper plugins are used for stopping the optimization algorithm when a 
	particular condition is satisfied. 
	
	The actuall stopping is achieved by setting the :attr:`stop` member of 
	the *opt* object to ``True``. 
	"""
	
	def __call__(self, x, ft, opt):
		pass

class Annotator(object):
	"""
	Annotators produce annotations of the function value. 
	These annotations can be used for restoring the state of 
	local objects from remotely computed results. 
	
	Usually the annotation is produced on remote workers when the cost function 
	is evaluated. It is the job of the optimization algorithm to send the value 
	of *ft* along with the corresponding annotations from worker (where the 
	evaluation took place) to the master where the annotation is consumed (by
	a call to the :meth:`newResult` method of the optimization algorithm). 
	This way the master can access all the auxiliary data which is normally 
	produced only on the machine where the evaluation of the cost function 
	took place. 
	"""
	def produce(self):
		"""
		Produce an annotation.
		"""
		return None
	
	def consume(self, annotation):
		"""
		Consume an annotation. 
		"""
		return

class AnnotatorGroup(object):
	"""
	This object is a container holding annotators. 
	"""
	def __init__(self):
		self.annotators=[]
	
	def add(self, annotator, index=-1):
		"""
		Adds an annotator at position *index*. If no *index* 
		is given the annotator is appended at the end. The index 
		of the annotator is returned. 
		"""
		if index<0:
			self.annotators.append(annotator)
			return len(self.annotators)-1
		else:
			self.annotators[index]=annotator
			return index
		
	def produce(self):
		"""
		Produces a list of annotations corresponding to annotators. 
		"""
		annot=[]
		for a in self.annotators:
			annot.append(a.produce())
		return annot
	
	def consume(self, annotations):
		"""
		Consumes a list of annotations. 
		"""
		for ii in range(len(self.annotators)):
			self.annotators[ii].consume(annotations[ii])

def UCEvaluator(x, f, annGrp, nanBarrier):
	"""
	Evaluator for unconstrained optimization. 
	
	Returns the function value and the annotations. 
	
	This function should be used on a remote computing node 
	for evaluating the function. Its return value should be sent 
	to the master that invoked the evaluation. 
	
	The arguments for this function are generated by the optimizer. 
	See the :meth:`getEvaluator` method. 
	
	The evaluator and its arguments are the minimum of things 
	needed for a remote evaluation. 
	
	By using this function one can avoid pickling and sending the 
	whole optimizer object with all of its auxiliary data. Instead 
	one can just pickle and send the bare minimum needed for a remote 
	evaluation. 
	"""
	fval=f(x)
	if nanBarrier and np.isnan(fval):
		fval=np.Inf
	
	return np.array(fval), annGrp.produce()

def BCEvaluator(x, xlo, xhi, f, annGrp, extremeBarrierBox, nanBarrier):
	"""
	Evaluator for box-constrained optimization. 
	
	Returns the function value and the annotations. 
	
	The arguments are generated by the optimizer. 
	See the :meth:`getEvaluator` method. 
	
	The evaluator and its arguments are the minimum of things 
	needed for a remote evaluation. 
	
	See the :func:`UCEvaluator` function for more information. 
	"""
	# Check box constraint violations
	violatedLo=(x<xlo).any()
	violatedHi=(x>xhi).any()
		
	# Enforce box constraints if extreme barrier approach is used
	if extremeBarrierBox:
		if violatedLo or violatedHi:
			return Inf
	else:
		if violatedLo:
			raise Exception, DbgMsg("BCOPT", "Point violates lower bound.")
		
		if violatedHi:
			raise Exception, DbgMsg("BCOPT", "Point violates upper bound.")
		
	fval=f(x)
		
	if nanBarrier and np.isnan(fval):
		fval=np.Inf
	
	return np.array(fval), annGrp.produce()

def CEvaluator(x, xlo, xhi, f, fc, c, annGrp, extremeBarrierBox, nanBarrier):
	"""
	Evaluator for constrained optimization. 
	
	Returns the function value, the constraints values, and the annotations. 
	
	The arguments are generated by the optimizer. 
	See the :meth:`getEvaluator` method. 
	
	The evaluator and its arguments are the minimum of things 
	needed for remote evaluation. 
	
	See the :func:`UCEvaluator` function for more information. 
	"""
	# Check box constraint violations
	violatedLo=(x<xlo).any()
	violatedHi=(x>xhi).any()
	
	# No function value
	# Enforce box constraints with extreme barrier approach 
	if extremeBarrierBox:
		if violatedLo or violatedHi:
			fval=Inf
			cval=np.array([])
			return fval, cval, None
	else:
		if violatedLo:
			raise Exception, DbgMsg("COPT", "Point violates lower bound.")
		
		if violatedHi:
			raise Exception, DbgMsg("COPT", "Point violates upper bound.")
		
		
	# Do we have fc
	if fc is not None: 
		# Yes
		# Evaluate f and c
		(fval,cval)=fc(x)
	else:
		# No
		# Evaluate constraints 
		if c is not None:
			cval=c(x)
		else:
			cval=np.array([])
			
		# Evaluate function (always)
		fval=f(x)
				
		# Nan barrier
		if nanBarrier and np.isnan(fval):
			fval=np.Inf
	
	return np.array(fval), np.array(cval), annGrp.produce()
	
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
	
	*nanBarrier* specifies if NaN function values should be treated as 
	infinite thus resulting in an extreme barrier. 
	
	*cache* turns on local point caching. Currently works only for 
	algorithms that do not use remote evaluations. See the 
	:mod:`~pyopus.optimizer.cache` module for more information.
	
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
	  annotators for the best-yet value of the cost function.
	* :attr:`stop` - boolean flag indicating that the algorithm should stop. 
	* :attr:`annotations` - a list of annotations produced by the installed 
	  annotators for the last evaluated cost function value
	* :attr:`annGrp` - :class:`AnnotatorGroup` object that holds the installed 
	  annotators
	* :attr:`plugins` - a list of installed plugin objects
	
	Plugin objects are called at every cost function evaluation or whenever a 
	remotely evaluated cost function value is registered by the 
	:meth:`newResult` method. 
	
	Values of *x* and related members are arrays. 
	"""
	def __init__(self, function, debug=0, fstop=None, maxiter=None, nanBarrier=False, cache=False):
		# Function subject to optimization, must be picklable for parallel optimization methods. 
		self.function=function
		
		# Debug mode flag
		self.debug=debug
		
		# Problem dimension
		self.ndim=None
		
		# Stopping conditions
		self.fstop=fstop
		self.maxiter=maxiter
		
		# NaN barrier
		self.nanBarrier=nanBarrier
		
		# Cache
		if cache:
			self.cache=Cache()
		else:
			self.cache=None
		
		# Iteration counter
		self.niter=0
		
		# Best-yet point
		self.x=None
		self.f=None
		self.bestIter=None
		self.bestAnnotations=None
		
		# Plugins
		self.plugins=[]
		
		# Annotator group
		self.annGrp=AnnotatorGroup()
		
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

	def installPlugin(self, plugin):
		"""
		Installs a plugin object or an annotator in the plugins list 
		and/or annotators list
		
		Returns two indices. The first is the index of the installed 
		annotator while the second is the index of the instaleld 
		plugin. 
		
		If teh object is not an annotator, the first index is 
		``None``. Similarly, if the object is not a plugin the second 
		index is ``None``. 
		"""
		i1=None
		if issubclass(type(plugin), Annotator):
			i1=self.annGrp.add(plugin)
		i2=None
		if issubclass(type(plugin), Plugin):
			self.plugins.append(plugin)
			i2=len(self.plugins)-1
		
		return (i1, i2)
		
	def getEvaluator(self, x):
		"""
		Returns a tuple holding the evaluator function and its positional 
		arguments that evaluate the problem at *x*. This tuple can be 
		sent to a remote computing node and evaluation is invoked using:: 
		
		  # Tuple t holds the function and its arguments
		  func,args=t
		  retval=func(*args)
		  # Send retval back to the master
		
		The first positional argument is the point to be evaluated. 
		
		The evaluator returns a tuple of the form (f,annotations).
		"""
		return UCEvaluator, [x, self.function, self.annGrp, self.nanBarrier]
	
	def fun(self, x, count=True):
		"""
		Evaluates the cost function at *x* (array). If *count* is ``True`` the 
		:meth:`newResult` method is invoked with *x*, the obtained cost 
		function value, and *annotations* argument set to ``None``. This means 
		that the result is registered (best-yet point information are updated), 
		the plugins are calls and the annotators are invoked to produce 
		annotations. 
		
		Use ``False`` for *count* if you need to evaluate the cost function 
		for debugging purposes. 
		
		Returns the value of the cost function at *x*. 
		"""
		data=None
		if self.cache is not None:
			data=self.cache.lookup(x)
			
		if data is not None:
			f,annot,it=data
		else:
			# Evaluate
			evf, args = self.getEvaluator(x)
			f,annot=evf(*args)
		
		# Do the things that need to be done with a new result
		# No annotation is provided telling the newResult() method that the 
		# function evaluation actually happened in this process. 
		if count:
			self.newResult(x, f, annot)
			
		return np.array(f)
	
	def updateBest(self, x, f):
		"""
		Updates best yet function value. 
		Returns ``True`` if an update takes place. 
		"""
		if (self.f is None) or (self.f>f):
			self.f=f
			self.x=x
			self.bestIter=self.niter
			return True
		
		return False
		
	def newResult(self, x, f, annotations=None):
		"""
		Registers the cost function value *f* obtained at point *x* with 
		annotations list given by *annotations*. 
		
		Increases the :attr:`niter` member to reflect the iteration number of 
		the point being registered and updates the :attr:`f`, :attr:`x`, and 
		:attr:`bestIter` members. 
		
		If the *annotations* argument is given, it must be a list with as many 
		members as there are annotator objects installed in the optimizer. The 
		annotations list is stored in the :attr:`annotations` member. If *f* 
		improves the best-yet value annotations are also stored in the 
		:attr:`bestAnnotations` member. The *annotations* are consumed by calling 
		the :meth:`consume` method of the annotators. 
		
		Finally it is checked if the best-yet value of cost function is below 
		:attr:`fstop` or the number of iterations exceeded :attr:`maxiter`. If 
		any of these two conditions is satisfied, the algorithm is stopped by 
		setting the :attr:`stop` member to ``True``. 
		"""
		# Increase evaluation counter
		self.niter+=1
		
		# Store in cache
		if self.cache and self.cache.lookup(x) is None:
			self.cache.insert(x, (f, annotations, self.niter))
		
		# Update best-yet
		updated=self.updateBest(x, f)
		
		# If no annotation are given, function evaluation happened in this process
		if annotations is not None:
			# Put annotations in annotations list
			self.annotations=annotations
			# Consume annotations
			self.annGrp.consume(annotations)
		
		# Update best-yet annotations
		if updated:
			self.bestAnnotations=self.annotations
		
		# Annotations are set up. Call plugins. 
		nplugins=len(self.plugins)
		for index in range(0,nplugins):
			plugin=self.plugins[index]
			if plugin is not None:
				stopBefore=self.stop
				plugin(x, f, self)
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

def normalizer(x, origin, scale):
	"""
	Normalizer for bound constrained optimization.	
	"""
	return (x-origin)/scale

def denormalizer(y, origin, scale):
	"""
	Denormalizer for bound constrained optimization. 
	"""
	return y*scale+origin
	
class BoxConstrainedOptimizer(Optimizer):
	"""
	Box-constrained optimizer class 
	
	*xlo* and *xhi* are 1-dimensional arrays or lists holding the lower and 
	the upper bounds on the components of *x*. Some algorithms allow the 
	components of *xlo* to be :math:`- \infty` and the components of *xhi* to 
	be :math:`+ \infty`. 
	
	If *extremeBarrierBox* is set to ``True`` the :meth:`fun` method returns 
	:math:`\infty` if the supplied point violates the box constraints. 
	
	See the :class:`Optimizer` class for more information. 
	"""
	def __init__(self, function, xlo=None, xhi=None, debug=0, fstop=None, maxiter=None,
		nanBarrier=False, extremeBarrierBox=False, cache=False):
		Optimizer.__init__(self, function, debug, fstop, maxiter, nanBarrier, cache)
		
		# Constraints
		self.xlo=xlo
		self.xhi=xhi
		
		self.extremeBarrierBox=False
	
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		# Check bounds
		if self.xlo is not None:
			if (self.xlo.ndim!=1 or self.xhi.ndim!=1): 
				raise Exception, DbgMsg("OPT", "Bounds must be one-dimensional vectors.")
		
		if self.xhi is not None:
			if (self.xlo.shape[0]!=self.xhi.shape[0]):
				raise Exception, DbgMsg("OPT", "Bounds must match in length.")
		
		if (self.xlo is not None) and (self.xhi is not None):
			if (self.xlo>=self.xhi).any():
				raise Exception, DbgMsg("OPT", "Lower bound must be below upper bound.")
		
		# Unconstraint checks
		Optimizer.check(self)
		
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
		
		# Check initial point against bounds
		if (x0<self.xlo).any():
			raise Exception, DbgMsg("BCOPT", "Initial point violates lower bound.")
		
		if (x0>self.xhi).any():
			raise Exception, DbgMsg("BCOPT", "Initial point violates upper bound.")
		
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
			self.normOrigin[ndx]=self.xlo[ndx]
			
		ndx=where(isinf(self.xlo) & isinf(self.xhi))
		if len(ndx[0])>0:
			self.normScale[ndx]=x0[ndx]*0.0+2.0
			self.normOrigin[ndx]=x0[ndx]
		
	def bound(self, x):
		"""
		Fixes components of *x* so that the bounds are enforced. If a component 
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
		return normalizer(x, self.normOrigin, self.normScale)
		
	def denormalize(self, y):
		"""
		Returns a denormalized point *x* corresponding to *y*. 
		Components of *x* are
		
		.. math:: x^i = y^i n_s^i + n_o^i
		"""
		return denormalizer(y, self.normOrigin, self.normScale)
		
	def getEvaluator(self, x):
		"""
		Returns the evaluator function and its positional arguments 
		that evaluate the problem at *x*. 
		
		The first positional argument is the point to be evaluated. 
		
		The evaluator returns a tuple of the form (f,annotations).
		"""
		return BCEvaluator, [x, self.xlo, self.xhi, self.function, self.annGrp, self.extremeBarrierBox, self.nanBarrier]

	def fun(self, x, count=True):
		"""
		Evaluates the cost function at *x* (array). If *count* is ``True`` the 
		:meth:`newResult` method is invoked with *x*, the obtained cost 
		function value, and *annotations* argument set to ``None``. This means 
		that the result is registered (best-yet point information are updated) 
		and the plugins are called to produce annotations. 
		
		Use ``False`` for *count* if you need to evaluate the cost function 
		for debugging purposes. 
		
		Returns the value of the cost function at *x*. 
		"""
		data=None
		if self.cache is not None:
			data=self.cache.lookup(x)
			
		if data is not None:
			f,annot,it=data
		else:
			# Evaluate
			evf, args = self.getEvaluator(x)
			f,annot=evf(*args)
		
		# Do the things that need to be done with a new result
		# No annotation is provided telling the newResult() method that the 
		# function evaluation actually happened in this process. 
		if count:
			self.newResult(x, f)
			
		return np.array(f)
	
class ConstrainedOptimizer(BoxConstrainedOptimizer):
	"""
	Constrained optimizer class 
	
	*xlo* and *xhi* are 1-dimensional arrays or lists holding the lower and 
	upper bounds on the components of *x*. Some algorithms allow the 
	components of *xlo* to be :math:`- \infty` and the components of *xhi* to 
	be :math:`+ \infty`. 
	
	*constraints* is a function that returns an array holding the values of 
	the general nonlinear constraints. 
	
	*clo* and *chi* are vectors of lower and upper bounds on the constraint 
	functions in *constraints*. Nonlinear constraints are of the form 
	:math:`\leq c_{lo} \leq f(x) \leq c_{hi}`. 
	
	*fc* is a function that simultaneously evaluates the function and the 
	constraints. When it is given, *function* and *constraints* must be 
	``None``. 
	
	See the :class:`BoxConstrainedOptimizer` class for more information. 
	"""
	def __init__(self, function, xlo=None, xhi=None, constraints=None, 
		clo=None, chi=None, fc=None, debug=0, fstop=None, maxiter=None, nanBarrier=False, 
		extremeBarrierBox=False, cache=False):
		BoxConstrainedOptimizer.__init__(self, function, xlo, xhi, debug, fstop, maxiter, nanBarrier, extremeBarrierBox, cache)
		# Constraints
		self.constraints=constraints
		
		# Function and constraints
		self.fc=fc
		
		if fc is not None and (function is not None or constraints is not None): 
			raise Exception, DbgMsg("COPT", "When fc is given, function and constraints must be None.")
		
		# Constraint bounds
		self.clo=clo
		self.chi=chi
		
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		# Unconstrained and box constrained checks
		BoxConstrainedOptimizer.check(self)
		
		# Nonlinear constraint checks
		if self.constraints is not None:
			if (self.clo.ndim!=1 or self.chi.ndim!=1): 
				raise Exception, DbgMsg("COPT", "Constraint bounds must be one-dimensional vectors.")
			
			if (self.clo.shape[0]!=self.chi.shape[0]):
				raise Exception, DbgMsg("COPT", "Constraint bounds must match in length.")
			
			if (self.clo>self.chi).any():
				raise Exception, DbgMsg("COPT", "Lower constraint bound must not be greater than upper constraint bound.")
		
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array or list *x0*. The length of the array 
		becomes the dimension of the optimization problem 
		(:attr:`ndim` member). 
		The shape of *x* must match that of *xlo* and *xhi* and *x* must 
		be within the bounds specified by *xlo* and *xhi*. 
		
		See the :meth:`reset` method of :class:`BoxConstrainedOptimizer` class 
		for more information. 
		"""
		x0=array(x0)
		BoxConstrainedOptimizer.reset(self, x0)
		
		if self.debug:
			DbgMsgOut("COPT", "Resetting.")
			
		# Default constraint bounds - don't known number of constraints, error 
		if (
			(self.constraints is not None or self.fc is not None) and 
			self.clo is None and self.chi is None
		):
			raise Exception, DbgMsg("COPT", "Cannot deduce the number of constraints")
		
		if (self.constraints is not None or self.fc is not None):
			if self.clo is None:
				self.clo=zeros(self.chi.shape[0])
				self.clo.fill(-Inf)
			else:
				self.clo=array(self.clo)
			
			if self.chi is None:
				self.chi=zeros(self.clo.shape[0])
				self.chi.fill(Inf)
			else:
				self.chi=array(self.chi)
		
			# Count nonlinear constraints
			self.nc=self.clo.shape[0]
		else:
			self.nc=0
		
		# Reset nonlinear constraint values at best point
		self.c=None
	
	def fun(self, x, count=True):
		"""
		The use of this function is not allowed in constrained optimization.
		"""
		raise DbgMsg("COPT", "fun() is not allowed. Use funcon().")
	
	def getEvaluator(self, x):
		"""
		Returns the evaluator function and its positional arguments 
		that evaluate the problem at *x*. 
		
		The first positional argument is the point to be evaluated. 
		
		The evaluator returns a tuple of the form (f,c,annotations).
		"""
		return CEvaluator, [x, self.xlo, self.xhi, self.function, self.fc, self.constraints, self.annGrp, self.extremeBarrierBox, self.nanBarrier]
	
	def funcon(self, x, count=True):
		"""
		Evaluates the cost function and the nonlinear constraints at *x*. 
		
		If *count* is ``True`` the :meth:`newResult` method is invoked with 
		*x*, the obtained cost function value, the nonlinear constraint 
		values, and the *annotations* argument set to ``None``. 
		This means that the result is registered (best-yet point information 
		are updated) and the plugins are called to produce annotations. 
		
		Use ``False`` for *count* if you need to evaluate the cost function 
		and the constraints for debugging purposes. 
		
		Returns the value of the cost function at *x* adjusted for extreme 
		barrier (i.e. ``Inf`` when box constraints are violated or the function 
		value is ``Nan``) and a vector of constraint function values. 
		"""
		data=None
		if self.cache is not None:
			data=self.cache.lookup(x)
			
		if data is not None:
			f,c,annot,it=data
		else:
			evf, args = self.getEvaluator(x)
			f,c,annot=evf(*args)
		
		# Register result
		if count:
			self.newResult(x, f, c, annot)

		return np.array(f), np.array(c)
	
	def constraintViolation(self, c):
		"""
		Returns constraint violation vector. Negative values correspond to 
		lower bound violation while positive values correspond to upper bound 
		violation. 
		
		Returns 0 if *c* is zero-length. 
		"""
		if c.size>0:
			vlo=c-self.clo
			vlo=np.where(vlo>=0.0, 0.0, vlo)
			
			vhi=c-self.chi
			vhi=np.where(vhi<=0.0, 0.0, vhi) 
			
			return vlo+vhi
		else:
			return np.array([])
	
	def aggregateConstraintViolation(self, c, useL2squared=False):
		"""
		Computes the aggregate constraint violation. If no nonlinear 
		constraints are violated this value is 0. Otherwise it is greater 
		than zero. 
		
		*c* is the vector of constraint violations returned by the 
		:meth:`constraintViolation` method. 
		
		if *useL2squared* is ``True`` the L2 norm is used for computing 
		the aggregate violation. Otherwise L1 norm is used. 
		
		"""
		h=self.constraintViolation(c)
		if h.size==0:
			return 0.0
		else:
			if useL2squared:
				return (h**2).sum() # L2
			else:
				return np.abs(h).sum() # L1
		
	def updateBest(self, x, f, c):
		"""
		Updates best yet function and constraint values. 
		
		Calculates cumulative squared constraint violation (h). It is 0 if 
		there are no constraints. 
		
		If h<hbest, replace best point.
		If h==hbest and f<fbest, replace best point.
		
		Returns ``True`` if an update takes place. 
		"""
		if self.f is None:
			self.f=f
			self.x=x
			self.c=c
			self.bestIter=self.niter
			return True
		else:
			h=self.aggregateConstraintViolation(c)
			hbest=self.aggregateConstraintViolation(self.c)
			
			if (h<hbest) or (h==hbest and f<self.f):
				self.f=f
				self.x=x
				self.c=c
				self.bestIter=self.niter
				return True
		
		return False
		
	def newResult(self, x, f, c, annotations=None):
		"""
		Registers the cost function value *f* and constraints values *c* 
		obtained at point *x* with annotations list given by *annotations*. 
		
		Increases the :attr:`niter` member to reflect the iteration number of 
		the point being registered and updates the :attr:`f`, :attr:`x`, 
		:attr:`c`, and :attr:`bestIter` members. 
		
		See the :meth:`newResult` method of the :class:`Optimizer` class for 
		more information. 
		"""
		# Increase evaluation counter
		self.niter+=1
		
		# Store in cache
		if self.cache and self.cache.lookup(x) is None:
			self.cache.insert(x, (f,c,annotations,self.niter))
	
		# Update best-yet
		updated=self.updateBest(x, f, c)
		
		# If no annotation are given, function evaluation happened in this process
		if annotations is not None:
			# Put annotations in annotations list
			self.annotations=annotations
			# Consume annotations
			self.annGrp.consume(annotations)
		
		# Update best-yet annotations
		if updated:
			self.bestAnnotations=self.annotations
		
		# Annotations are set up. Call plugins. 
		nplugins=len(self.plugins)
		for index in range(0,nplugins):
			plugin=self.plugins[index]
			if plugin is not None:
				stopBefore=self.stop
				plugin(x, (f,c,self.constraintViolation(c)), self)
				
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


# Cost function and input parameter collector
class CostCollector(Plugin):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Plugin` iterative 
	algorithm plugin class. This is a callable object invoked at every 
	iteration of the algorithm. It collects the input parameter vector 
	(n components) and the aggregate function value. 
	
	Let niter denote the number of stored iterations. The input parameter 
	values are stored in the :attr:`xval` member which is an array of shape 
	(niter, n) while the aggregate function values are stored in the 
	:attr:`fval` member (array with shape (niter)). If the algorithm 
	supplies constraint violations they are stored in the :attr:`hval` 
	member. Otherwise this member is set to ``None``. 
	
	Some iterative algorithms do not evaluate iterations sequentially. Such 
	algorithms denote the iteration number with the :attr:`index` member. If 
	the :attr:`index` member is not present in the iterative algorithm object 
	the internal iteration counter of the :class:`CostCollector` is used. 
	
	The first index in the *xval*, *fval*, and *hval* arrays is the iteration 
	index. If iterations are not performed sequentially these two arrays may 
	contain gaps where no valid input parameter or aggregate function value is 
	found. The gaps are denoted by the *valid* array (of shape (niter)) where 
	zeros denote a gap. 
	
	*xval*, *fval*, *hval*, and *valid* arrays are resized in chunks of size 
	*chunkSize*. 
	"""
	def __init__(self, chunkSize=100): 
		Plugin.__init__(self)
		
		self.xval=None
		self.fval=None
		self.hval=None
		self.valid=None
		self.n = chunkSize
		self.memLen = chunkSize
		
	def __call__(self, x, ft, opt):
		if self.xval is None:
			#allocate space in memory
			self.xval = zeros([self.n,len(x)])
			self.fval = zeros([self.n])
			if type(ft) is tuple:
				self.hval = zeros([self.n])
			self.valid = zeros([self.n])
			self.localindex = 0 
		
		if 'index' in opt.__dict__:
			index = opt.index 
		else:
			index = self.localindex
			self.localindex += 1
							
		#check if the index is inside the already allocated space -> if not allocate new space in memory
		while index >= self.memLen: 
			self.xval = concatenate((self.xval, zeros([self.n,len(x)])), axis=0)
			self.fval = concatenate((self.fval, zeros([self.n])), axis=0)
			if type(ft) is tuple:
				self.hval = concatenate((self.hval, zeros([self.n])), axis=0)
			self.valid = concatenate((self.valid, zeros([self.n])), axis=0)
			self.memLen += self.n
				
		#write data
		self.xval[index] = x
		
		if type(ft) is tuple:
			self.fval[index] = ft[0]
			self.hval[index] = ft[2].sum()
		else:
			self.fval[index] = ft
			
		self.valid[index] += 1 
					
		return None

	def finalize(self): 
		"""
		Removes the space beyond the recorded iteration with highest iteration 
		number. This space was reserved for the last chunk, but the highest 
		recorded iteration may not be the one recorded at the end of the chunk. 
		
		Must be called after the iterative algorithm is stopped. 
		"""
		if self.valid is None:
			return
		
		nonZeroIndex = self.valid.nonzero()
		lastIndex=nonZeroIndex[0].max()+1
		self.xval = self.xval[0:lastIndex][:]
		self.fval = self.fval[0:lastIndex]
		if self.hval is not None:
			self.hval = self.hval[0:lastIndex]
		self.valid = self.valid[0:lastIndex]

	def reset(self):
		"""
		Clears the :attr:`xval`, :attr:`fval`, :attr:`hval`, and :attr:`valid` 
		members.
		"""
		
		self.xval = None
		self.fval = None
		self.hval = None
		self.valid = None
		self.memLen = self.n

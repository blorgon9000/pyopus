"""
.. inheritance-diagram:: pyopus.optimizer.coordinate
    :parts: 1

**Box constrained coordinate search optimizer (PyOPUS subsystem name: CSOPT)**

The algorithm (asymptotically) converges to a point where the projection of the 
cost function gradient is normal to the active constraint (i.e. stationary 
point in the sense of constrained optimization) if the following holds

* the cost function is continuously differentiable, 
* the step size is a rational multiple of :math:`n_s` 
  (see :class:`~pyopus.optimizer.base.BoxConstrainedOptimizer`), 
* the step scaling factors *stepup* and *stepdn* are rational. 

If the above conditions hold the algorithm can be classified as a pattern 
search algorithm. For a proof see

Lewis R. M., Torczon V.: Pattern search algorithms for bound constrained 
minimization. SIAM Journal on Optimization, vol. 9, pp. 1082-1099, 1999. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import BoxConstrainedOptimizer
from numpy import max, abs, array

__all__ = [ 'CoordinateSearch' ] 


class CoordinateSearch(BoxConstrainedOptimizer):
	"""
	Coordinate search optimizer class
	
	*stepup* is the factor by which the step size is increased after a 
	successfull step. 
	
	*stepdn* is the factor by which the step size is decreased after steps in 
	all directions fail. 
	
	*step0* is the initial vector of *ndim* step sizes corresponding to 
	coordinate directions. It can also be a scalar in which case it is applied 
	to all directions. 
	
	*minstep* is a vector of step sizes used in the stopping condition. 
	After all step sizes fall below the corresponding values in *minstep* the 
	algorithm is stopped. *minstep* can also be a scalar in which case it 
	applies to step sizes in all directions. 
	
	If *step0* and *minstep* are not given they are set to :math:`n_s / 10` 
	and :math:`n_s / 1000` when the :meth:`reset` method is called. 
	
	Box constraints are handled by setting the components of a vector that 
	violate a bound to the bound. 
	
	Infinite bounds are allowed. This means that the algorithm behaves as 
	an unconstrained algorithm if lower bounds are :math:`-\\infty` and 
	upper bounds are :math:`+\\infty`. 
	
	See the :class:`~pyopus.optimizer.base.BoxConstrainedOptimizer` class for 
	more information. 
	"""
	def __init__(self, function, xlo=None, xhi=None, debug=0, fstop=None, maxiter=None, 
					stepup=1.0, stepdn=0.5, step0=None, minstep=None):
		BoxConstrainedOptimizer.__init__(self, function, xlo, xhi, debug, fstop, maxiter)
		
		# Initial step
		self.step0=step0
		if self.step0 is not None:
			self.step0=array(self.step0)
			
		# Stopping condition (step size)
		self.minstep=minstep
		if self.minstep is not None:
			self.minstep=array(self.minstep)
		
		# Step size control
		self.stepup=array(stepup)
		self.stepdn=array(stepdn)
	
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		BoxConstrainedOptimizer.check(self)
		
		if self.step0 is not None and (self.step0<=0).any():
			raise Exception, DbgMsg("CSOPT", "Initial step must be greater than 0.")
		
		if (self.stepup<1):
			raise Exception, DbgMsg("CSOPT", "Step increase must be greater or equal 1.")
			
		if (self.stepdn>=1) or (self.stepdn<=0):
			raise Exception, DbgMsg("CSOPT", "Step decrease must be between 0 and 1.")
			
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array *x0*. The length of the array becomes the 
		dimension of the optimization problem (:attr:`ndim` member). The shape 
		of *x* must match that of *xlo* and *xhi*. 
		"""
		
		BoxConstrainedOptimizer.reset(self, x0)
		
		# Debug message
		if self.debug:
			DbgMsgOut("CSOPT", "Resetting coordinate search.")
		
		if self.step0 is None:
			self.step0=self.normScale/10.0
		
		if self.minstep is None:
			self.minstep=self.normScale/1000.0
			
		self.step=self.step0
	
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("CSOPT", "Starting a coordinate search run at i="+str(self.niter))
		
		# Reset stop flag
		self.stop=False
		
		# Check 
		self.check()
		
		# Evaluate initial point (if needed)
		if self.f is None:
			self.f=self.fun(self.x)
		
		# Retrieve best-yet point
		x=self.x.copy()
		f=self.f.copy()
		
		# Retrieve step
		step=self.step
		
		# Main loop
		while not self.stop:
			if (step<self.minstep).all():
				if self.debug:
					DbgMsgOut("CSOPT", "Iteration i="+str(self.niter)+": step small enough, stopping")
				break;
				
			# Trial steps
			i=0
			while i<self.ndim:
				if step.size==1:
					delta=step
				else:
					delta=step[i]
				xnew=x.copy()
				xnew[i]=x[i]+delta
				self.bound(xnew)
				fnew=self.fun(xnew)
				
				# Debug message
				if self.debug:
					DbgMsgOut("CSOPT", "Iteration i="+str(self.niter)+": di="+str(i+1)+" f="+str(f)+" step="+str(max(abs(step))))
				if fnew<f:
					break
									
				xnew[i]=x[i]-delta
				self.bound(xnew)
				fnew=self.fun(xnew)
				
				# Debug message
				if self.debug:
					DbgMsgOut("CSOPT", "Iteration i="+str(self.niter)+": di="+str(-i-1)+" f="+str(f)+" step="+str(max(abs(step))))
				if fnew<f:
					break
				
				i+=1
			
			if fnew<f:
				x=xnew
				f=fnew
				if step.size==1:
					step=step*self.stepup
				else:
					step[i]=step[i]*self.stepup
			else:
				step=step*self.stepdn
		
		if self.debug:
			DbgMsgOut("CSOPT", "Coordinate search finished.")
		
		self.step=step
		
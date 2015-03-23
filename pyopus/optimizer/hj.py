"""
.. inheritance-diagram:: pyopus.optimizer.hj
    :parts: 1

**Box constrained Hooke-Jeeves optimizer (PyOPUS subsystem name: HJOPT)**

An extended version of coordinate search where so-called **speculative steps** 
are taken from time to time. These steps are hoped to speed up the search. 

The convergence theory applied to coordinate search can also be applied to 
Hooke-Jeeves algorithm if the speculative step length factor is an integer. 

The algorithm (unconstrained version) was first published in 

Hooke R., Jeeves T. A., Direct Search Solution of Numerical and Statistical 
Problems. Journal of the ACM (JACM), vol. 8, pp. 212-229, 1961.
"""

from ..misc.debug import DbgMsgOut, DbgMsg

from coordinate import CoordinateSearch

from numpy import where, isinf, where, abs, max, array

__all__ = [ 'HookeJeeves' ]


class HookeJeeves(CoordinateSearch):
	"""
	Hooke-Jeeves optimizer class
	
	*speculative* is the step length factor for the speculative step. 
	
	See the :class:`~pyopus.optimizer.base.coordinate.CoordinateSearch` 
	class for more information. 
	"""
	def __init__(self, function, xlo=None, xhi=None, debug=0, fstop=None, maxiter=None, 
					stepup=1.0, stepdn=0.5, step0=None, minstep=None, speculative=2.0):
		# It has all members of a coordinate search
		CoordinateSearch.__init__(self, function, xlo, xhi, debug, fstop, maxiter, 
				stepup=stepup, stepdn=stepdn, step0=step0, minstep=minstep)
		
		# Speculative step length (factor)
		self.speculative=speculative
		
		# Speculative step flag
		self.isSpeculative=None
		
		# Trial step origin
		self.xt=None
		self.ft=None
		
		# Fallback origin
		self.xo=None
		self.fo=None
	
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		CoordinateSearch.check(self)
		
		if (self.step0<=0).any():
			raise Exception, DbgMsg("HJOPT", "Initial step must be greater than 0.")
		
		if (self.stepup<1):
			raise Exception, DbgMsg("HJOPT", "Step increase must be greater or equal 1.")
			
		if (self.stepdn>=1):
			raise Exception, DbgMsg("HJOPT", "Step decrease must be less than 1.")
		
	def reset(self, x0):
		"""
		Puts the optimizer in its initial state and sets the initial point to 
		be the 1-dimensional array or list *x0*. The length of the array 
		becomes the dimension of the optimization problem (:attr:`ndim` 
		member). The shape of *x0* must match that of *xlo* and *xhi*. 
		"""
		CoordinateSearch.reset(self, x0)
		
		# Debug message
		if self.debug:
			DbgMsgOut("HJOPT", "Resetting Hooke-Jeeves.")
		
		self.step=self.step0
		self.isSpeculative=False
		self.xt=array(x0)
		self.ft=None
		self.xo=array(x0)
		self.fo=None

		# Debug message
		if self.debug:
			DbgMsgOut("HJOPT", "Done.")
	
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("HJOPT", "Starting a Hooke-Jeeves run at i="+str(self.niter))
		
		# Reset stop flag
		self.stop=False
		
		# Check and evaluate initial point
		self.check()
				
		# Evaluate initial point (if needed)
		if self.f is None:
			self.f=self.fun(self.x)
			
		if self.ft is None:
			self.ft=self.f
		if self.fo is None:
			self.fo=self.f
		
		# Prepare
		x=self.x.copy()
		f=self.f.copy()
		xt=self.xt
		ft=self.ft
		xo=self.xo
		fo=self.fo
		step=self.step
		isSpeculative=self.isSpeculative
		
		# Main loop
		while not self.stop:
			# Speculative step
			if isSpeculative:
				xspec=(xt-xo)*self.speculative
				self.bound(xspec)
				xo=xt
				fo=ft
				xt=xspec
				if self.debug:
					DbgMsgOut("HJOPT", "Iteration i="+str(self.niter)+": speculative step")
			
			# Trial steps
			i=0
			while i<self.ndim:
				# Origin of trial steps in one dimension
				xtmp=xt.copy()
				ftmp=ft.copy()
				
				if step.size==1:
					delta=step
				else:
					delta=step[i]
				
				# +step
				xnew=xtmp.copy()
				xnew[i]=xtmp[i]+delta
				self.bound(xnew)
				fnew=self.fun(xnew)
				# Debug message
				if self.debug:
					DbgMsgOut("HJOPT", "Iteration i="+str(self.niter)+": f="+str(ft)+" step="+str(max(abs(step))))
				if fnew<ft:
					xt=xnew
					ft=fnew
				if self.stop:
					break;
				
				# -step	
				xnew=xtmp.copy()
				xnew[i]=xtmp[i]-delta
				self.bound(xnew)
				fnew=self.fun(xnew)
				# Debug message
				if self.debug:
					DbgMsgOut("HJOPT", "Iteration i="+str(self.niter)+": f="+str(ft)+" step="+str(max(abs(step))))
				if fnew<ft:
					xt=xnew
					ft=fnew
				if self.stop:
					break;
				
				#  Speculative step failed
				if isSpeculative and not (ft<fo):
					break
				
				# Next dimension
				i+=1
			
			if self.stop:
				break;
				
			# Check if we need to change the step
			if isSpeculative:
				# Speculative step
				if ft>=fo:
					# Failed
					# Return to previous origin, next one is ordinary
					isSpeculative=False
					xt=xo
					ft=fo
					if self.debug:
						DbgMsgOut("HJOPT", "Iteration i="+str(self.niter)+": speculative step FAILED")
				else:
					# OK, increase step
					step*=self.stepup
					None
			else:
				# Ordinary step
				if ft>=fo:
					# Failed, reduce step
					step*=self.stepdn
				else:
					# OK, increase step, next one is speculative
					isSpeculative=True
					step*=self.stepup
			
			# Check step size
			if (step<self.minstep).all():
				if self.debug:
					DbgMsgOut("HJOPT", "Iteration i="+str(self.niter)+": step small enough, stopping")
				break;
	
		if self.debug:
			DbgMsgOut("HJOPT", "Hooke-Jeeves run finished.")
		
		self.step=step		


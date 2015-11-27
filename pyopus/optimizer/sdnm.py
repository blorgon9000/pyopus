"""
.. inheritance-diagram:: pyopus.optimizer.sdnm
    :parts: 1

**Sufficient descent Nelder-Mead simplex optimizer (Price-Coope-Byatt)
(PyOPUS subsystem name: SDNMOPT)**

A provably convergent version of the Nelder-Mead simplex algorithm. The 
algorithm performs unconstrained optimization. Convergence is achieved by 
imposing sufficient descent on simplex steps and by keeping the simplex 
internal angles away from 0. 

The algorithm was published in 

Price C.J., Coope I.D., Byatt D.: A Convergent Variant of the Nelder-Mead 
Algorithm. Journal of Optimization Theory and Applications, vol. 113, 
pp. 5-19, 2002. 

Byatt D.: Convergent Variants of the Nelder-Mead Algorithm, MSc thesis, 
University of Canterbury, 2000. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import Optimizer
from nm import NelderMead

from numpy import abs, argsort, lexsort, where, round, sign, diag, sqrt, log, array, zeros, dot, ones, pi, e
from numpy.linalg import qr, det
from scipy.misc import factorial

import matplotlib.pyplot as pl

__all__ = [ 'SDNelderMead' ]

class SDNelderMead(NelderMead): 
	"""
	Unconstrained sufficient-descent Nelder-Mead optimizer class 
	(Price-Coope-Byatt algorithm)
	
	*kappa* is the frame shrink factor. 
	
	*K0* is the maximal length of a vector in basis. 
	
	*N0* defines initial sufficient descent, which is *N0* times smaller than 
	average function difference between the best point and the remaining $n$ 
	points of the initial simplex
	
	*nu* is the exponential (>1) for calculating new sufficient descent. 
	
	*tau* is the bound on basis determinant. 
	
	Initial value of h is not given in the paper. The MSc thesis, however, 
	specifies that it is 1. 
	
	See the :class:`~pyopus.optimizer.nm.NelderMead` class for more 
	information. 
	"""
	def __init__(self, function, debug=0, fstop=None, maxiter=None, 
					reflect=1.0, expand=2.0, outerContract=0.5, innerContract=-0.5, shrink=0.5, 
					reltol=1e-15, ftol=1e-15, xtol=1e-9, simplex=None, 
					kappa=4.0, K0=1e3, N0=100.0, nu=4.5, tau=1e-18):
		NelderMead.__init__(self, function, debug, fstop, maxiter, 
										reflect, expand, outerContract, innerContract, shrink, 
										reltol, ftol, xtol, simplex)
		
		# Simplex
		self.simplex=None
		self.simplexf=None
		self.simpelxmoves=None
		
		# log(n! det([v])) where [v] are the n side vectors
		# arranged as columns of a matrix
		self.logDet=None
				
		# Algorithm parameters 
		self.kappa=kappa
		self.K0=K0
		self.N0=N0
		self.nu=nu
		self.tau=tau
		
		# Dependent parameters
		self.N=None
		self.h=None
		self.epsilon=None
		
	def check(self):
		"""
		Checks the optimization algorithm's settings and raises an exception if 
		something is wrong. 
		"""
		NelderMead.check(self)
		
		if self.K0<=0:
			raise Exception, DbgMsg("SDNMOPT", "K0 should be positive.")
			
		if self.N0<=0:
			raise Exception, DbgMsg("SDNMOPT", "N0 should be positive.")
			
		if self.nu<=1:
			raise Exception, DbgMsg("SDNMOPT", "nu should be greater than 1.")
		
		if self.tau<=0:
			raise Exception, DbgMsg("SDNMOPT", "tau should be positive.")
	
	def logFactorial(self, n):
		"""
		Calculates log(n!) where log() is the natiral logarithm. 
		Uses Stirling's approximation for n>50. 
		"""
		
		if n<=50:
			return log(factorial(n))
		else:
			return 0.5*log(2*pi*n)+n*log(n/e)
		
	def orderSimplex(self):
		"""
		Overrides default sorting in Nelder-Mead simplex. 
		
		Reorders the points and the corresponding cost function values of the 
		simplex in such way that the point with the lowest cost function value 
		is the first point in the simplex. Secondary sort key is the number of 
		moves of the point. It increses by 1 every time a point moves and is 
		reset to 0 at simplex reshape. Of two points with same f the one with 
		higher number of moves is first. 
		"""
		# Order simplex
		i=lexsort((-self.simplexmoves, self.simplexf), 0)
		self.simplexf=self.simplexf[i]
		self.simplex=self.simplex[i,:]
		self.simplexmoves=self.simplexmoves[i]
	
	
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
		Reshapes basis given by rows of *v* into an orthogonal linear base. 
		
		Returns a tuple (*bnew*, *logDet*) where *bnew* holds the reshaped 
		basis and *logDet* is log(n! det([v])). 
		"""
		# Rows are old basis
		# Transpose and QR decompose them. 
		(Q, R)=qr(v.T)
		
		# Get scaling factors and their signs 
		Rdiag=R.diagonal()
		Rsign=sign(Rdiag)
		Rsign=where(Rsign!=0, Rsign, 1.0)
		
		# Get side lengths and mean side length
		l=abs(Rdiag)
		lmean=l.mean()
		
		# Calculate basis length bounds
		lower=lmean/10.0
		upper=self.K0
				
		# Bound basis vector lengths 
		l=where(l<=upper, l, upper)
		l=where(l>=lower, l, lower)
		
		# Scale vectors to form a new base. 
		# Vectors are in columns of Q. Therefore transpose Q. 
		bnew=dot(diag(l*Rsign), Q.T)
		
		# Calculate logDet of basis (side vectors divided by h)
		logDet=log(l).sum()
		
		return (bnew, logDet)
		
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
		
		The initial value of the natural logarithm of the simplex side vectors 
		determinant is calculated and stored. This value gets updated at every 
		simplex algorithm step. The only time it needs to be reevaluated is at 
		reshape. But that is also quite simple because the reshaped simplex 
		is orthogonal. The only place where a full determinant needs to be 
		calculated is here. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("SDNMOPT", "Resetting.")
		
		# Make it an array
		x0=array(x0)
		
		# Is x0 a point or a simplex?
		if x0.ndim==1:
			# Point
			# Set x now
			NelderMead.reset(self, x0)
			
			if self.debug:
				DbgMsgOut("SDNMOPT", "Generating initial simplex from initial point.")
				
			sim=self.buildSimplex(x0)
			self._setSimplex(sim)
		else:
			# Simplex or error (handled in _setSimplex())
			self._setSimplex(x0)
			
			if self.debug:
				DbgMsgOut("SDNMOPT", "Using specified initial simplex.")
				
			# Set x to first point in simplex after it was checked in _setSimplex()
			Optimizer.reset(self, x0[0,:])
		
		# Reset point moves counter 
		self.simplexmoves=zeros(self.ndim+1)
		
		# Calculate log(n! det([v])) where [v] are the n side vectors
		# arranged as columns of a matrix
		(v, l)=self.sortedSideVectors()
		self.logDet=log(abs(det(v)))
		
		# Initial h 
		self.h=1.0
		
		# Make x tolerance an array
		self.xtol=array(self.xtol)
	
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Debug message
		if self.debug:
			DbgMsgOut("SDNMOPT", "Starting a run at i="+str(self.niter))

		# Checks
		self.check()
		
		# Reset stop flag
		self.stop=False
		
		# Evaluate if needed
		if self.simplexf is None:
			self.simplexf=zeros(self.npts)
			for i in range(0, self.ndim+1):
				self.simplexf[i]=self.fun(self.simplex[i,:])
				if self.debug:
					DbgMsgOut("SDNMOPT", "Initial simplex point i="+str(self.niter)+": f="+str(self.simplexf[i]))
			
			# Initial epsilon and N
			self.orderSimplex()
			self.epsilon=(self.simplexf[1:]-self.simplexf[0]).mean()/self.N0
			self.N=self.epsilon/self.h**self.nu
		
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
			if self.debug:
				DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": reflect   : f="+str(fr))
			
			if fr<fb:
				# Try expansion
				xe=xc+(xc-xw)*self.expand
				fe=self.fun(xe)
				if self.debug:
					DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": expand    : f="+str(fe))
				
				if fe<fr:
					# Accept expansion
					self.simplex[-1,:]=xe
					self.simplexf[-1]=fe
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.expand))
					if self.debug:
						DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": accepted expansion")
				else:
					# Accept reflection
					self.simplex[-1,:]=xr
					self.simplexf[-1]=fr
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.reflect))
					if self.debug:
						DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": accepted reflection after expansion")
						
			elif fb<=fr and fr<fsw:
				# Accept reflection
				self.simplex[-1,:]=xr
				self.simplexf[-1]=fr
				self.simplexmoves[-1]+=1
				self.logDet+=log(abs(self.reflect))
				if self.debug:
					DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": accepted reflection")
					
			elif fsw<=fr and fr<fw:
				# Try outer contraction
				xo=xc+(xc-xw)*self.outerContract
				fo=self.fun(xo)
				if self.debug:
					DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": outer con : f="+str(fo))
				if fo<=fw:
					# Accept
					self.simplex[-1,:]=xo
					self.simplexf[-1]=fo
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.outerContract))
					if self.debug:
						DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": accepted outer contraction")
				else:
					shrink=True
					
			elif fw<=fr:
				# Try inner contraction
				xi=xc+(xc-xw)*self.innerContract
				fi=self.fun(xi)
				if self.debug:
					DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": inner con : f="+str(fi))
				if fi<=fw:
					# Accept
					self.simplex[-1,:]=xi
					self.simplexf[-1]=fi
					self.simplexmoves[-1]+=1
					self.logDet+=log(abs(self.innerContract))
					if self.debug:
						DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": accepted inner contraction")
				else:
					shrink=True
			
			# self._checkSimplex()
			
			# self._plotSimplex()
			
			# Stopping condition fo inner loop not satisfied
			stoppingSatisfied=False
			
			# Check sufficient descent (look at worst point)
			if fw-self.simplexf.max()<=self.epsilon:
				# Normal NM steps failed to produce sufficient descent 
				
				# Check simplex shape
				# Simplex is already sorted
				(v, l)=self.sortedSideVectors()
				
				# Basis vectors
				vbasis=v/self.h
				lbasis=l/self.h
				
				# No reshape yet
				reshaped=False
				
				# Create origin vector and function value
				x0=zeros(self.ndim)
				f0=0.0
				
				# Prepare frame points and function values
				frame=zeros((self.ndim+1, self.ndim))
				frameb=zeros((self.ndim+1, self.ndim))
				framef=zeros(self.ndim+1)
				
				# Calculate log determinant of basis (side vectors divided by h)
				logDetBase=self.logDet-self.ndim*log(self.h)
				
				# Check shape
				if logDetBase<=log(self.tau) or lbasis.max()>self.K0: 
					# Shape not good, reshape
					
					# Origin for building the new simplex
					x0[:]=self.simplex[0,:]
					f0=self.simplexf[0]
					
					# Reshape and calculate logDet of simplex sides
					(basis, logDet)=self.reshape(vbasis)
					frameb[:-1,:]=basis[:,:]
					reshaped=True
					
					# Calculate pseudo-expand direction 
					frameb[-1,:]=-basis.sum(0)/self.ndim*(self.expand/self.reflect-1.0)
					
					# Evaluate new simplex
					self.simplexmoves[:]=0
					for i in range(self.ndim):
						self.simplex[i+1,:]=x0+self.h*frameb[i,:]
						f=self.fun(self.simplex[i+1,:])
						self.simplexf[i+1]=f
						if self.debug:
							DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": reshape 1 : f="+str(f))
							
				# Centroid of the n worst points
				xcw=self.simplex[1:,:].sum(0)/self.ndim
				
				# Pseudo-expand point
				xpe=xb+(self.expand/self.reflect-1.0)*(xb-xcw)
				fpe=self.fun(xpe)
				if self.debug:
					DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": pseudo ex : f="+str(fpe))
				
				# Do we have sufficient descent wrt best (not a quasiminimal frame)
				if fpe<fb-self.epsilon: 
					# Pseudo-expand is better than old best (x0)
					self.simplex[0,:]=xpe
					self.simplexf[0]=fpe
					self.simplexmoves[0]+=1
					
					# Calculate new logDet (logDet * h**n * gammae/gammar) 
					if reshaped:
						self.logDet=logDet+self.ndim*log(self.h)+log(abs(self.expand/self.reflect))
					
					if self.debug:
						DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": accepted pseudo exp")
				elif self.simplexf.min()<fb-self.epsilon:
					# One of the points obtained by reshape is better than old best point
					# Calculate new logDet (logDet * h**n) 
					if reshaped:
						self.logDet=logDet+self.ndim*log(self.h)
					
					if self.debug:
						DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": accepted reshape")
				else:
					# Reset point moves counter
					self.simplexmoves[:]=0
					
					# Frame-based loop
					exitFrameLoop=False
					while not exitFrameLoop: 
						# No sufficient descent, first point in simplex is still best
						if not reshaped: 
							# No reshape yet, reshape now
							
							# Origin for building the new simplex
							x0[:]=self.simplex[0,:]
							f0=self.simplexf[0]
							
							# Reshape 
							(basis, logDet)=self.reshape(vbasis)
							frameb[:-1,:]=basis[:,:]
							reshaped=True
							
							# Calculate pseudo-expand direction 
							frameb[-1,:]=-basis.sum(0)/self.ndim*(self.expand/self.reflect-1.0)
							
							if self.debug:
								DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": reshape 2")
						else:
							# Reshaped already, reverse basis and shrink
							frameb=-frameb
							self.h/=self.kappa
							self.epsilon=self.N*self.h**self.nu
						
						# Evaluate frame 
						for i in range(self.ndim+1):
							frame[i,:]=x0+self.h*frameb[i,:]
							framef[i]=self.fun(frame[i,:])
							if self.debug:
								DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": frame     : f="+str(framef[i]))
						
						# Build would-be simplex for the case we exit loop now
						self.simplex[:,:]=frame[:,:]
						self.simplexf[:]=framef[:]
						self.logDet=logDet+self.ndim*log(self.h)
						
						# Is pseudo-expand worse than x0
						if framef[-1]>=f0:
							# Replace pseudo-expand with x0
							self.simplex[-1,:]=x0[:]
							self.simplexf[-1]=f0
							self.logDet+=log(abs(self.expand/self.reflect))
							
						# Check for sufficient descent
						if framef.min()<f0-self.epsilon: 
							# Exit loop 
							exitFrameLoop=True
						
						# Stopping condition
						if (self.checkFtol() and self.checkXtol()) or self.stop:
							exitFrameLoop=True
							stoppingSatisfied=True
						
			# Check stopping condition
			if stoppingSatisfied or (self.checkFtol() and self.checkXtol()): 
				if self.debug:
					DbgMsgOut("SDNMOPT", "Iteration i="+str(self.niter)+": simplex x and f tolerance reached, stopping.")
				break
			
		# Debug message
		if self.debug:
			DbgMsgOut("SDNMOPT", "Finished.")
	
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
				DbgMsgOut("SDNMOPT", "Simplex consistency broken for member #"+str(i))
				raise Exception, ""
	
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

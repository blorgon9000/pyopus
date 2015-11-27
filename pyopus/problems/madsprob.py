# -*- coding: UTF-8 -*-
"""
.. inheritance-diagram:: pyopus.problems.madsprob
    :parts: 1
	
**Optimization test functions from papers on MADS. 
(PyOPUS subsystem name: MADSPROB)**

All test functions in this module are maps from :math:`R^n` to :math:`R`. 

The *nc* nonlinear constraints are of the form

.. math::
  cl \\leq c(x) \\leq ch
  
Beside linear constraints a problem can also have bounds of the form

.. math::
  xl \\leq x \\leq xh
  
This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. It depends only on the :mod:`cpi` 
and the :mod:`mgh` module. 

The STYRENE and the MDO problem depend on the :mod:`_mads` module. 
HS114 and MAD6 depend on the :mod:`_lvns` module. 

References:

.. [mads]  Audet C., Dennis J.E. Jr., Mesh Adaptive Direct Search Algorithms 
           for Constrained Optimization. SIAM Journal on Optimization, 
           vol. 17, pp. 188-217, 2006. 

.. [madspsd] Audet C., Dennis Jr J.E., Le Digabel S., Parallel Space 
           Decomposition of the Mesh Adaptive Direct Search Algorithm. 
           SIAM Journal on Optimization, vol. 19, pp. 1150-1170, 2008. 

.. [madsort] Abramson M.A., Audet C., Dennis J.E. Jr., Le Digabel S.,
           ORTHO-MADS: A Deterministic MADS Instance with Orthogonal 
           Directions. SIAM Journal on Optimization, vol. 20, pp. 948-966, 
           2009.

.. [madspb] Audet C., Dennis J.E. Jr., A Progressive Barrier for 
           Derivative-Free Nonlinear Programming. SIAM Journal on 
           Optimization, vol. 20, pp. 445-472, 2009. 

.. [madsqm] Conn A.R., Le Digabel S., Use of Quadratic Models with Mesh 
           Adaptive Direct Search for Constrained Black Box Optimization. 
           Optimization Methods and Software, vol. 28, pp. 139-158, 2013. 

.. [madsvns] Audet C., Bechard V., Le Digabel S., Nonsmooth Optimization Through 
	   Mesh Adaptive Direct Search and Variable Neighbourhood Search. Journal 
	   of Global Optimization, vol 41, pp. 299-318, 2008. 
		
.. [ufo]   Lukšan L, et. al., UFO 2011 interactive system for univeral functional 
	   optimization. TR 1151 ICS, AS CR, 2011. 
"""

from cpi import CPI, MemberWrapper
import numpy as np
import mgh

try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y

__all__ = [ 
	'MADSPROBsuite', 
	'SNAKE', 
	'DISK', 
	'CRESCENT', 
	'G2', 
	'B250', 
	'B500', 
	'DIFF2', 
	'UFO7_26', 
	'UFO7_29',
	'MADSPROB'
]

class MADSPROB(CPI):
	"""
	Base class for the test functions 
	
	The fesible initial point can be obtained from the :attr:`initial` member. 
	The infeasible initial point is in the :attr:`initialinf` member. If any of 
	these two points is not given, it is set to ``None``. 
	
	The full name of the problem is in the :attr:`name` member. The :attr:`n` 
	member holds the dimension of the problem. :attr:`nc` is the number of 
	nonlinear constraints. 
	
	The best known minimum value of the function is in the :attr:`fmin` member. 
	If this value is unknown it is set to ``None``. 
	
	Objects of this class are callable. The calling convention is 
	
	``object(x, gradients)``
	
	where *x* is the input values vector and *gradients* is a boolean flag 
	specifying whether the gradients should be evaluated. The values of the 
	auxiliary functions and their gradients are stored in the :attr:`fi` and 
	:attr:`J` members. The values of the cobnstraints and the corresponding 
	Jacobian are stored in the :attr:`ci` and :attr:`J` members. 
	
	This creates an instance of the CRESCENT function and evaluates the 
	function and the constraints along with the gradient and the constraint 
	Jacobian at the feasible initial point:: 
	  
	  from pyopus.optimizer.madsprob import CRESCENT
	  cr=CRESCENT(n=10)
	  (f, g, c, Jc)=cr.fgcjc(cr.initial)
	"""
	name=None
	"The name of the test function"
	
	def __init__(self, n, m=1, nc=0):
		self.n=n
		self.nc=nc
		self.initial=None
		self.initialinf=None
		self.fmin=None
		
		# No bounds by default
		self.xl=np.ones(n)
		self.xl.fill(-np.Inf)
		self.xh=np.ones(n)
		self.xh.fill(np.Inf)
		
		# Equality constraint of the form =0 by default
		self.cl=np.zeros(nc)
		self.ch=np.zeros(nc)
		
		self.fi=np.zeros(m)
		self.J=np.zeros([m,n])
		self.ci=np.zeros(nc)
		self.Jc=np.zeros([nc,n])
		
	def __call__(self, x, gradients=False):
		self.fi=None
		self.ci=None
		if gradients:
			self.J=None
			self.Jc=None
	
	def freductor(self):
		# Return the first auxiliary function by default
		return self.fi[0]*1.0
			
	def greductor(self):
		# Return the gradient of the first auxiliary function by default
		return self.J[0,:]*1.0
		
	def f(self, x):
		"""
		Returns the value of the test function at *x*. 
		"""
		self(x)
		return self.freductor()
		
	def g(self, x):
		"""
		Returns the value of the gradient at *x*. 
		"""
		self(x, True)
		return self.greductor()
	
	def c(self, x):
		"""
		Returns the value of the constraints at *x*. 
		"""
		self(x)
		return self.ci*1.0
	
	def fc(self, x):
		"""
		Returns the value of teh function and constraints at *x*. 
		"""
		self(x)
		return self.freductor(), self.ci*1.0
		
	def cjc(self, x):
		"""
		Returns the constraint Jacobian at *x*. 
		"""
		self(x, True)
		return self.Jc*1.0
		
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		xmin is not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=self.nc)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		itf['c']=MemberWrapper(self, 'c')
		
		itf['g']=MemberWrapper(self, 'g')
		itf['cg']=MemberWrapper(self, 'cjc')
		
		if 'fmin' in self.__dict__:
			itf['fmin']=self.fmin
		
		itf['xlo']=self.xl
		itf['xhi']=self.xh
		itf['clo']=self.cl
		itf['chi']=self.ch
		
		return self.fixBounds(itf)

class SNAKE(MADSPROB):
	"""
	SNAKE test function (n=2, nc=2). 
	
	Published as the first problem in section 4.1 of [madspb]_. 
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="SNAKE"
	
	def __init__(self):
		MADSPROB.__init__(self, n=2, nc=2)
		self.initial=np.array([0.0, 0])
		
		self.initialinf=np.array([0.0, -10])
		
		self.fmin=0.08098
		
		self.cl[0]=-np.Inf
		self.ch[0]=0
		
		self.cl[1]=0
		self.ch[1 ]=np.Inf
		
	def __call__(self, x, gradient=False):
		self.fi[0]=((x[0]-20)**2+(x[1]-1)**2)**0.5
		
		self.ci[0]=np.sin(x[0])-0.1-x[1]
		self.ci[1]=np.sin(x[0])-x[1]
		
		if gradient:
			self.J[0,0]=(x[0]-20)/self.fi[0]
			self.J[0,1]=(x[1]-1)/self.fi[0]
			
			self.Jc[0,0]=np.cos(x[0])
			self.Jc[0,1]=-1.0
			
			self.Jc[1,0]=np.cos(x[0])
			self.Jc[1,1]=-1.0
	
class DISK(MADSPROB):
	"""
	DISK test function (n=arbitrary, nc=1). 
	Also referred to a the Hypersphere problem. 
	
	Published as the first problem in section 4.2 of [madspb]_
	or as the first problem in section 5.3 of [mads]_.
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="DISK"
	
	def __init__(self, n=10):
		MADSPROB.__init__(self, n=n, nc=1)
		self.initial=np.zeros(n)
		
		self.initialinf=np.ones(n)*2
		
		self.fmin=-3**0.5*n
		
		self.cl[0]=-np.Inf
		self.ch[0]=0.0
		
	def __call__(self, x, gradient=False):
		self.fi[0]=x.sum()
		
		self.ci[0]=(x**2).sum()-3*self.n
		
		if gradient:
			self.J[0,:]=1.0
			
			self.Jc[0,:]=2*x

class CRESCENT(MADSPROB):
	"""
	CRESCENT test function (n=arbitrary, nc=2). 
	
	Published as the first problem in section 4.3 of [madspb]_.
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="CRESCENT"
	
	def __init__(self, n=10):
		MADSPROB.__init__(self, n=n, nc=2)
		self.initial=np.zeros(n)
		self.initial[0]=n+0.1
		
		self.initialinf=np.zeros(n)
		self.initialinf[0]=n
		self.initialinf[-1]=-n
		
		self.fmin=1.0-n
		
		self.cl[0]=-np.Inf
		self.ch[0]=0.0
		
		self.cl[1]=0.0
		self.ch[1]=np.Inf
		
	def __call__(self, x, gradient=False):
		self.fi[0]=x[-1]
		
		self.ci[0]=((x-1.0)**2).sum()-self.n**2
		self.ci[1]=((x+1.0)**2).sum()-self.n**2
		
		# One row contains derivatives wrt n dimensions (has n columns)
		# There is one row per component (m or nc in total)
		if gradient:
			self.J[:,:]=0.0
			self.J[0,-1]=1.0
			
			self.Jc[0,:]=2*(x-1.0)
			self.Jc[1,:]=2*(x+1.0)

class G2(MADSPROB):
	"""
	G2 test function (n=arbitrary, nc=1). 
	
	Published as problem A in section 5.2 of [madspsd]_.
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="G2"
	
	def __init__(self, n=10):
		MADSPROB.__init__(self, n=n, nc=2)
		self.initial=np.ones(n)*5
		
		self.initialinf=None
		
		if n==10:
			self.fmin=-0.803619
		
		self.xl[:]=0.0
		self.xh[:]=10.0
		
		self.cl[0]=-np.Inf
		self.ch[0]=0.0
		
		self.cl[1]=-np.Inf
		self.ch[1]=0.0
		
	def __call__(self, x, gradient=False):
		ii=np.arange(self.n)
		cv=np.cos(x)
		den2=((ii+1)*x**2).sum()
		
		self.fi[0]=((cv**4).sum()-2*(cv**2).prod())/den2**0.5
		
		self.ci[0]=-x.prod()+0.75
		self.ci[1]=x.sum()-7.5*self.n
		
		if gradient:
			for i in ii:
				self.J[0,i]=(
					(
						4*np.cos(x[i])**3*(-np.sin(x[i]))
						-2*cv.prod()*cv[:i].prod()*cv[i+1:].prod()*(-np.sin(x[i]))
					)/den2**0.5
					-(
						((cv**4).sum()-2*(cv**2).prod())/den2*
						0.5/den2**0.5*(i+1)*x[i]*2
					)
				)
					
				self.Jc[0,i]=-x[:i].prod()*x[i+1:].prod()
			
			self.Jc[1,:]=1.0
			
	def freductor(self):
		if self.fi[0]<0:
			return self.fi[0]
		else:
			return -self.fi[0]
			
	def greductor(self):
		if self.fi[0]<0:
			return self.J[0,:]*1.0
		else:
			return -self.J[0,:]*1.0

class B250(MADSPROB):
	"""
	B250 test function (n=60, nc=1). 
	
	Mentioned as problem B in section 5.2 of [madspsd]_.
	
	Rewritten from C++ source obtained from the NOMAD test problems collection. 
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="B250"
	
	def __init__(self):
		MADSPROB.__init__(self, n=60, nc=1)
		self.initial=np.array([
			0.25,
			0.85,
			0.25,
			0.85,
			0.25,
			0.85,
			0.25,
			-0.35,
			0.85,
			0.25,
			0.85,
			0.85,
			-0.35,
			-0.35,
			0.25,
			0.825,
			0.125,
			0.825,
			0.825,
			0.125,
			0.825,
			0.825,
			0.825,
			-0.575, 
			-0.575, 
			0.825,
			0.125,
			0.825,
			0.125,
			0.125,
			0.125,
			0.125,
			0.125,
			0.125,
			-0.575,
			0.125,
			0.825,
			0.125,
			-0.575,
			0.825,
			-0.575,
			0.825,
			0.825,
			0.125,
			0.825,
			0.9,
			1.78,
			0.9,
			0.02,
			0.9,
			0.9,
			0.9,
			0.02,
			0.9,
			0.9,
			0.9,
			0.9,
			0.02,
			0.9,
			1.78
		])
		
		self.initialinf=None
		
		self.xl[:15]=-0.5
		self.xh[:15]=1.0
		
		self.xl[15:45]=-0.75
		self.xh[15:45]=1.0
		
		self.xl[45:]=-0.2
		self.xh[45:]=2.0
		
		self.cl[0]=250.0
		self.ch[0]=np.inf
		
		self.func1=mgh.PenaltyII(n=15)
		self.func2=mgh.Trigonometric(n=30)
		self.func3=mgh.BrownAlmostLinear(n=15)
		
		self.func4=mgh.BroydenBanded(n=15)
		self.func5=mgh.BroydenTridiagonal(n=30)
		self.func6=mgh.DiscreteBoundaryValue(n=15)
		
	def __call__(self, x, gradient=False):
		self.func1(x[:15], gradient)
		self.func2(x[15:45], gradient)
		self.func3(x[45:], gradient)
		
		self.func4(x[:15], gradient)
		self.func5(x[15:45], gradient)
		self.func6(x[45:], gradient)
		
		self.fi[0]=(
			(self.func1.fi**2).sum()
			+(self.func2.fi**2).sum()
			+(self.func3.fi**2).sum()
		)
		
		self.ci[0]=(
			(self.func4.fi**2).sum()
			+(self.func5.fi**2).sum()
			+(self.func6.fi**2).sum()
		)
		
		if gradient:
			self.J[0,:15]=2*np.dot(self.func1.J.T, self.func1.fi)
			self.J[0,15:45]=2*np.dot(self.func2.J.T, self.func2.fi)
			self.J[0,45:]=2*np.dot(self.func3.J.T, self.func3.fi)
			
			self.Jc[0,:15]=2*np.dot(self.func4.J.T, self.func4.fi)
			self.Jc[0,15:45]=2*np.dot(self.func5.J.T, self.func5.fi)
			self.Jc[0,45:]=2*np.dot(self.func6.J.T, self.func6.fi)
			
class B500(B250):
	"""
	B500 test function (n=60, nc=1). 
	
	Mentioned as problem B in section 5.2 of [madspsd]_.
	
	Rewritten from C++ source obtained from the NOMAD test problems collection. 
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="B500"
	
	def __init__(self):
		B250.__init__(self)
		self.initial=np.array([
			0.2500000000000000000000000, 0.2500000000000000000000000,
			-0.0500000000000000000108420, -0.0500000000000000000108420,
			0.8500000000000000000216840, -0.3500000000000000000216840,
			0.8500000000000000000216840, 0.5500000000000000000108420,
			1.0000000000000000000000000, -0.3500000000000000000216840,
			1.0000000000000000000000000, -0.0500000000000000000108420,
			0.8500000000000000000216840, -0.3500000000000000000216840,
			-0.0500000000000000000108420, -0.5749999999999999999891580,
			-0.2249999999999999999945790, -0.7500000000000000000000000,
			0.8249999999999999999891580, -0.5749999999999999999891580,
			0.4749999999999999999945790, -0.7500000000000000000000000,
			1.0000000000000000000000000, 0.8249999999999999999891580,
			-0.2249999999999999999945790, -0.7500000000000000000000000,
			0.8249999999999999999891580, -0.2249999999999999999945790,
			0.4749999999999999999945790, -0.2249999999999999999945790,
			-0.7500000000000000000000000, 0.4749999999999999999945790,
			-0.2249999999999999999945790, 0.8249999999999999999891580,
			-0.2249999999999999999945790, -0.2249999999999999999945790,
			-0.2249999999999999999945790, 0.4749999999999999999945790,
			0.4749999999999999999945790, 0.1250000000000000000000000,
			-0.2249999999999999999945790, 0.4749999999999999999945790,
			0.8249999999999999999891580, 0.4749999999999999999945790,
			1.0000000000000000000000000, 0.8999999999999999999783160,
			-0.2000000000000000000433681, 1.7799999999999999999739791,
			0.8999999999999999999783160, -0.2000000000000000000433681,
			1.7799999999999999999739791, 1.3400000000000000000303577,
			0.0199999999999999999284427, 1.7799999999999999999739791,
			-0.2000000000000000000433681, 1.7799999999999999999739791,
			0.4599999999999999999533793, 0.8999999999999999999783160,
			1.3400000000000000000303577, 0.4599999999999999999533793
		])
		self.cl[0]=500.0
		self.ch[0]=np.Inf
	
class DIFF2(MADSPROB):
	"""
	DIFF2 test function (n=2, nc=0). 
	
	Published as the problem (4.1) in [madsqm]_.
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="DIFF2"
	
	def __init__(self):
		MADSPROB.__init__(self, n=2, nc=0)
		
		# Intial point is not given, so we use (-100, -100)
		self.initial=np.array([-90.0, -90.0])
		
		self.fmin=-2e-4
		
		self.xl[0]=-100.0
		self.xh[0]=100.0
		
		self.xl[1]=-100.0
		self.xh[1]=100.0
		
	def __call__(self, x, gradient=False):
		self.fi[0]=np.abs(x[0]-x[1])-1e-6*(x[0]+x[1])
		
		if gradient:
			if x[0]>x[1]:
				self.J[0,0]=1.0-1e-6
				self.J[0,1]=-1.0-1e-6
			else:
				self.J[0,0]=-1.0-1e-6
				self.J[0,1]=1.0-1e-6

class UFO7_26(MADSPROB):
	"""
	Problem from UFO manual (n, nc=int(3*(n-2)/2)). 
	
	Published as the first problem in section 7.26 of [ufo]_.
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="UFO7_26"
	
	def __init__(self, n=10):
		nc=int(3*(n-2)/2)
		MADSPROB.__init__(self, n=n,nc=nc)
		
		self.initial=np.zeros(n)
		
		self.xl=np.zeros(n)
		self.xh=np.zeros(n)
		self.xl.fill(-np.Inf)
		self.xh.fill(np.Inf)
		
		self.cl=np.zeros(nc)
		self.ch=np.zeros(nc)
		self.cl.fill(-np.Inf)
		
	def __call__(self, x, gradient=False):
		self.fi[0]=(
			x[0:-3]**2+x[1:-2]**2+2*x[2:-1]**2+x[3:]**2
			-5*x[0:-3]-5*x[1:-2]-21*x[2:-1]+7*x[3:]
		).sum()
		
		for k in range(self.nc):
			j=2*(k/3+1)
			j1=j-1
			if k%3==0:
				self.ci[k]=x[j1-1]**2+x[j1]**2+x[j1+1]**2+x[j1+2]**2+x[j1-1]-x[j1]+x[j1+1]-x[j1+2]-8.0
			elif k%3==1:
				self.ci[k]=x[j1-1]**2+2*x[j1]**2+x[j1+1]**2+2*x[j1+2]**2-x[j1-1]-x[j1+2]-10.0
			else:
				self.ci[k]=2*x[j1-1]**2+x[j1]**2+x[j1+1]**2+2*x[j1-1]-x[j1]-x[j1+2]-5.0
		
		if gradient:
			raise Exception, DbgMsg("MADSPROB", "Gradient not implemented.")
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		xmin is not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=MADSPROB.cpi(self)
		itf['g']=None
		itf['cg']=None
		
		return itf

class UFO7_29(MADSPROB):
	"""
	Problem from UFO manual (n, nc=n-2). 
	
	Published as the first problem in section 7.29 of [ufo]_.
	
	See the :class:`MADSPROB` class for more details. 
	"""
	name="UFO7_29"
	
	def __init__(self, n=10):
		nc=n-2
		MADSPROB.__init__(self, n=n,nc=nc)
		
		self.initial=np.zeros(n)
		self.initial[0::2]=-1.2
		self.initial[1::2]=1.0
		
		self.xl=np.zeros(n)
		self.xh=np.zeros(n)
		self.xl.fill(-np.Inf)
		self.xh.fill(np.Inf)
		
		self.cl=np.zeros(nc)
		self.ch=np.zeros(nc)
		self.ch.fill(np.Inf)
		
	def __call__(self, x, gradient=False):
		self.fi[0]=(
			10*np.abs(x[0:-1]**2-x[1:])+np.abs(x[0:-1]-1.0)
		).sum()
		
		for ii in range(self.nc):
			self.ci[ii]=(
				3*x[ii+1]**3+2*x[ii+2]-5.0
				+np.sin(x[ii+1]-x[ii+2])*np.sin(x[ii+1]+x[ii+2])
				+4*x[ii+1]-x[ii]*np.exp(x[ii]-x[ii+1])
				-3.0-2.0
			)
			
		if gradient:
			raise Exception, DbgMsg("MADSPROB", "Gradient not implemented.")
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		xmin is not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=MADSPROB.cpi(self)
		itf['g']=None
		itf['cg']=None
		
		return itf
		
		
MADSPROBsuite=[
	# Nonlinearly constrained problems
	SNAKE, 
	DISK, 
	CRESCENT, 
	G2,	# nonsmooth objective 
	B250,  
	B500, 
	DIFF2, 
	UFO7_26, 
	UFO7_29, 
]
"""
A list holding references to all function classes in this module. 
"""

try:
	import _mads
	
	__all__.append("MDO")
	__all__.append("STYRENE")
	
	class MDO(CPI):
		"""
		A multidisciplinary design optimization problem - maximization of 
		aircraft range. 
		
		Warning - possible memory leaks and crashes the c++ code is not 
		cleaned up. 
		
		Published in [madsvns]_.
		"""
		def __init__(self, eps=1e-12, maxiter=100):
			self.name="MDO"
			self.n=10
			self.nc=10
			self.eps=eps
			self.maxiter=maxiter
			
			self.cl=np.zeros(10)
			self.ch=np.zeros(10)
			self.cl.fill(-np.Inf)
			self.ch.fill(1e-14)
			
			self.xl=np.array([0.10, 0.75, 0.75, 0.1, 0.01, 30000, 1.4, 2.5, 40, 500])
			self.xh=np.array([0.40, 1.25, 1.25, 1.0, 0.09, 60000, 1.8, 8.5, 70, 1500])
			
			self.initial=np.array([0.4, 1, 0.872, 0.4433, 0.05, 45000, 1.728, 3.196, 62.68, 1000])
			self.initialinf=np.array([0.4, 0.75, 0.75, 0.189296875, 0.09, 57000, 1.4, 2.5, 70, 1500])
			
			self.fmin=-3964.0
		
		def fc(self, x):
			"""
			Returns the value of the function and the constraints at *x*. 
			"""
			return _mads.mdo_wrap(x, self.eps, self.maxiter)
		
		def cpi(self):
			"""
			Returns the common problem interface. 
			
			xmin, f, and c are not available. Only fc is available. 
			
			See the :class:`CPI` class for more information. 
			"""
			itf=self.prepareCPI(self.n, m=self.nc)
			itf['name']=self.name
			itf['x0']=self.initial
			itf['fc']=MemberWrapper(self, 'fc')
			
			itf['xlo']=self.xl
			itf['xhi']=self.xh
			itf['clo']=self.cl
			itf['chi']=self.ch
			
			return self.fixBounds(itf)
			
	class STYRENE(CPI):
		"""
		An engineering optimization problem - styrene process optimization
		
		Warning - possible memory leaks and crashes the c++ code is not 
		cleaned up. 
		
		Published in [madsvns]_.
		"""
		def __init__(self):
			self.name="STYRENE"
			self.n=8
			self.nc=11
			
			self.cl=np.zeros(11)
			self.ch=np.zeros(11)
			self.cl.fill(-np.Inf)
			self.ch.fill(1e-14)
			
			self.xl=np.zeros(8)
			self.xh=np.ones(8)*100.0
			
			self.initial=np.array([54.0, 66, 86, 8, 29, 51, 32, 15])
			self.initialinf=None
			
			self.fmin=-3.35
		
		def fc(self, x):
			"""
			Returns the value of the function and the constraints at *x*. 
			"""
			return _mads.sty_wrap(x)
		
		def cpi(self):
			"""
			Returns the common problem interface. 
			
			xmin, f, and c are not available. Only fc is available. 
			
			See the :class:`CPI` class for more information. 
			"""
			itf=self.prepareCPI(self.n, m=self.nc)
			itf['name']=self.name
			itf['x0']=self.initial
			itf['fc']=MemberWrapper(self, 'fc')
			
			itf['xlo']=self.xl
			itf['xhi']=self.xh
			itf['clo']=self.cl
			itf['chi']=self.ch
			
			return self.fixBounds(itf)
		
	MADSPROBsuite.append(MDO)
	MADSPROBsuite.append(STYRENE)
except:
	pass
	
try:
	import _lvns
	
	__all__.append("MAD6_mod")
	__all__.append("HS114_mod")
	
	# Eliminate x7 and x6 from MAD6 to get a n=5 dimensional problem
	class MAD6_mod(CPI):
		"""
		Modification of the MAD6 problem from the :mod:`lvns` module. 
		
		The MAD6 problem has 7 variables. One of them has an equality 
		constraint imposed (x7). There is also a linear equality constraint 
		involving x4 and x6. This modification eliminates x6 and x7 resulting 
		in a problem with 5 variables and no equality constraint. 
		
		Published in [madsort]_.
		"""
		
		def setup(self):
			"""
			Initializes the binary implementation of the function. 
			After this function is called no other function from the same test 
			set may be created or initialized because that will change the 
			internal variables and break the function. 
			Returns an info structure. 
			"""
			return _lvns.eild22(5)
		
		def __init__(self):
			self.name="MAD6_mod"
			
			info=self.setup()
		
			self.n=info['n']-2
			self.m=info['m']
			self.nc=info['nc']-2
			self.initial=np.array([0.5, 1.0, 1.5, 2.0, 2.5])
			self.fmin=0.10183089
			
			# Initial point
			#   -x4+x6=1  ... x6=1+x4 ... x6=1+2=3
			#    x7=3.5
			# Constraints (equality)
			#    -x4+x6=1.0 ... x6=1.0+x4
			#    x7=3.5
			# Elimination (constraints)
			#    -x5+x6>=0.4 ... -x5+(1+x4)>=0.4 ... x4-x5>=-0.6
			#    -x6+x7>=0.4 ... -(1.0+x4)+3.5>=0.4 ... -x4>=-2.1 ... x4<=2.1
			
			# Bounds
			self.xl=np.array([0.4,   -np.Inf, -np.Inf, -np.Inf, -np.Inf])
			self.xh=np.array([np.Inf, np.Inf,  np.Inf,  2.1, np.Inf])
			
			# Constraints
			self.cl=np.array([0.4, 0.4, 0.4, 0.4, -0.6])
			self.ch=np.array([np.Inf, np.Inf, np.Inf, np.Inf, np.Inf])
			
			# Jacobian
			self.Jc=np.array([
				[-1.0, 1.0, 0.0, 0.0, 0.0], 
				[ 0.0,-1.0, 1.0, 0.0, 0.0], 
				[ 0.0, 0.0,-1.0, 1.0, 0.0], 
				[ 0.0, 0.0, 0.0,-1.0, 1.0], 
				[ 0.0, 0.0, 0.0, 1.0,-1.0] 
			])
			
		def fc(self, x):
			"""
			Returns the value of the function and the constraints at *x*. 
			"""
			xx=np.hstack((x,1.0+x[3],3.5))
			f=_lvns.tafu22(5, xx).max()
			c=np.dot(self.Jc, x.reshape((self.n,1))).reshape((self.nc))

			return (f,c)
		
		def cpi(self):
			"""
			Returns the common problem interface. 
			
			xmin, f, and c are not available. Only fc is available. 
			
			See the :class:`CPI` class for more information. 
			"""
			itf=self.prepareCPI(self.n, m=self.nc)
			itf['name']=self.name
			itf['x0']=self.initial
			itf['fc']=MemberWrapper(self, 'fc')
			itf['setup']=MemberWrapper(self, 'setup')
			
			itf['xlo']=self.xl
			itf['xhi']=self.xh
			itf['clo']=self.cl
			itf['chi']=self.ch
			
			return self.fixBounds(itf)
			
			
	# Eliminate x5 from HS114 to get a n=9 dimensional problem
	class HS114_mod(CPI):
		"""
		Modification of the HS114 problem from the :mod:`lvns` module. 
		
		The HS114 problem has 10 variables. One linear equality constraint 
		links x1, x4, and x5. This modification eliminates x5 resulting 
		in a problem with 9 variables and no equality constraint. 
		
		Published in [madsort]_.
		"""
		
		def setup(self):
			"""
			Initializes the binary implementation of the function. 
			After this function is called no other function from the same test 
			set may be created or initialized because that will change the 
			internal variables and break the function. 
			Returns an info structure. 
			"""
			return _lvns.eild22(11)
		
		def __init__(self):
			self.name="HS114_mod"
			
			info=self.setup()
		
			self.n=info['n']-1
			self.m=info['m']
			self.nc=info['nc']-1
			self.initial=np.hstack((info['x0'][:4], info['x0'][5:]))
			self.fmin=-1768.8070
			
			# Constraints (equality)
			#   1.22 x4 - x1 - x5 = 0 ... x5 = 1.22 x4 - x1
			# Initial point
			#   x5 = 1.22 x4 - x1 = 1973.56
			
			# Bounds
			bt=info['bt']
			if bt.shape[0]==0:
				# No bounds vector given, infinite bounds
				self.xl=np.zeros(self.n)
				self.xl.fill(-np.Inf);
				self.xh=np.zeros(self.n)
				self.xh.fill(np.Inf);
			else:
				self.xl=info['xl']
				self.xh=info['xh']
				# No bound
				self.xl=np.where(bt==0, -np.Inf, self.xl)
				self.xh=np.where(bt==0, np.Inf, self.xh)
				# Lower bound only
				self.xh=np.where(bt==1, np.Inf, self.xh)
				# Upper bound only
				self.xl=np.where(bt==2, -np.Inf, self.xl)
			
			self.xl=np.hstack((self.xl[:4], self.xl[5:]))
			self.xh=np.hstack((self.xh[:4], self.xh[5:]))
			
			# Constraints
			ct=info['ct']
			self.cl=info['cl']
			self.ch=info['ch']
			# No constraint
			self.cl=np.where(ct==0, -np.Inf, self.cl)
			self.ch=np.where(ct==0, np.Inf, self.ch)
			# Lower bound only
			self.ch=np.where(ct==1, np.Inf, self.ch)
			# Upper bound only
			self.cl=np.where(ct==2, -np.Inf, self.cl)
			
			self.cl=self.cl[:-1]
			self.ch=self.ch[:-1]
			
			self.Jc=info['Jc']
			
		def fc(self, x):
			"""
			Returns the value of the function and the constraints at *x*. 
			"""
			xx=np.hstack((x[:4], 1.22*x[3]-x[0], x[4:]))
			f=_lvns.tafu22(11, xx).max()
			c=np.dot(self.Jc, xx.reshape((self.n+1,1))).reshape((self.nc+1))[:-1]

			return (f,c)
		
		def cpi(self):
			"""
			Returns the common problem interface. 
			
			xmin, f, and c are not available. Only fc is available. 
			
			See the :class:`CPI` class for more information. 
			"""
			itf=self.prepareCPI(self.n, m=self.nc)
			itf['name']=self.name
			itf['x0']=self.initial
			itf['fc']=MemberWrapper(self, 'fc')
			itf['setup']=MemberWrapper(self, 'setup')
			
			itf['xlo']=self.xl
			itf['xhi']=self.xh
			itf['clo']=self.cl
			itf['chi']=self.ch
			
			return self.fixBounds(itf)
	
	MADSPROBsuite.append(MAD6_mod)
	MADSPROBsuite.append(HS114_mod)
	
except:
	pass



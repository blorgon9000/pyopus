"""
.. inheritance-diagram:: pyopus.problems.cpi
    :parts: 1

**Common problem interface (PyOPUS subsystem name: CPI)**

Common problem interface (CPI) is a picklable dictionary containing all the 
information that is needed to describe an optimization problem. 

The dictionary contains the following members:
	
	* ``setup`` -- Setup function. If not ``None`` this function must be called 
	  just before the problem is evaluated for the first time. 
	* ``name``  -- problem name
	* ``n``     -- number of variables
	* ``m``     -- number of nonlinear constraints
	* ``x0``    -- the initial point
	* ``f``     -- returns the cost function value
	* ``g``     -- returns the cost function gradient
	* ``c``     -- returns the vector of nonlinear constraints
	* ``cg``    -- returns the Jacobian of the constraint functions, 
	  one row corresponds to one constraint, columns represent variables
	* ``fc``    -- returns the cost function value and the constraint function values 
	* ``xlo``   -- a vector of lower bounds on variables
	* ``xhi``   -- a vector of upper bounds on variables
	* ``clo``   -- a vector of lower bounds on nonlinear constraints
	* ``chi``   -- a vector of upper bounds on nonlinear constraints
	* ``fmin``  -- best known minimum function value
	* ``xmin``  -- best known minimum 
	* ``info``  -- additional problem information dictionary (suite dependent)
	
Lower and upper bounds are always defined. If some bound is not defined, it is 
set to ``+Inf`` or ``-Inf``. If some memeber is not available it is set to ``None``. 

If the ``setup`` member is not ``None`` the problem belongs to a suite where 
multiple problems share internal variables. For such problems the setup 
function should be called just before the first problem evaluation. If later 
the setup function of a different problem from the same suite is called the 
values of the shared internal variables change and all subsequent evaluations 
of the previous problem result in incorrect values. 

This class is inherited by optimization problems that support CPI.
The :meth:`cpi` method returns the CPI dictionary of the problem. 

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. 

Classes that inherit this class must reimplement the :meth:`cpi` method. 
"""

try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y
import numpy as np
from copy import deepcopy
from pprint import pprint

__all__ = [ 'CPI', 'MemberWrapper', 'rotateProblem' ]

			
class CPI(object):
	def __init__(self):
		pass
	
	def prepareCPI(self, n, m):
		"""
		Prepares a blank CPI dictionary for a problem with *n* variables and 
		*m* nonlinear constraints. 
		
		Sets missing members to ``None`` or 0. 
		"""
		return {
			"setup": None, 
			"name": "unknown", 
			"n": n, 
			"m": m, 
			"x0": None, 
			"f": None, 
			"g": None, 
			"c": None, 
			"cg": None, 
			"fc": None, 
			"xlo": None, 
			"xhi": None, 
			"clo": None, 
			"chi": None, 
			"fmin": None, 
			"xmin": None, 
			"info": {}, 
		}
		
	def fixBounds(self, itf):
		"""
		Set default bounds and change <-1e20 (>1e20) to -Inf(+Inf).
		Assumes n and m are already set. 
		"""
		if itf['xlo'] is None:
			itf['xlo']=np.zeros(itf['n'])
			itf['xlo'].fill(-np.Inf)
			
		if itf['xhi'] is None:
			itf['xhi']=np.zeros(itf['n'])
			itf['xhi'].fill(np.Inf)
			
		if itf['m']>0 and itf['clo'] is None:
			itf['clo']=np.zeros(itf['m'])
			itf['clo'].fill(-np.Inf)
			
		if itf['m']>0 and itf['chi'] is None:
			itf['chi']=np.zeros(itf['m'])
			itf['chi'].fill(np.Inf)
		
		itf['xlo']=np.where(itf['xlo']<=-1e20, -np.Inf, itf['xlo'])
		itf['xhi']=np.where(itf['xhi']>=1e20, np.Inf, itf['xhi'])
		
		if itf['m']>0:
			itf['clo']=np.where(itf['clo']<=-1e20, -np.Inf, itf['clo'])
			itf['chi']=np.where(itf['chi']>=1e20, np.Inf, itf['chi'])
			
		return itf
		
	def cpi(self):
		raise Exception, DbgMsg("CPI", "Common problem interface is not supported by this problem.")
	
class MemberWrapper(object):
	"""
	Wraps a member function in a callable object. 
	
	*obj* is the object and *memberName* is the name of the member to wrap. 
	"""
	
	def __init__(self, obj, memberName):
		self.obj=obj
		self.n=memberName
		self.compile()
	
	def compile(self):
		self.f=getattr(self.obj, self.n)
		
	def __call__(self, *args, **kwargs):
		return self.f(*args, **kwargs)
	
	# For pickling - copy object's dictionary and remove members 
	# with references to member functions so that the object can be pickled. 
	def __getstate__(self):
		state=self.__dict__.copy()
		del state['f']
		
		return state
	
	# For unpickling - update object's dictionary and rebuild members with references
	# to member functions. 
	def __setstate__(self, state):
		self.__dict__.update(state)
		self.compile()

class RotateF(object):
	"""
	Rotated function and gradient evaluator. 
	"""
	def __init__(self, f, Q, xlo=None, xhi=None):
		self.f=f
		self.Q=Q
		self.xlo=xlo
		self.xhi=xhi
	
	def __call__(self, x):
		xr=np.dot(self.Q, x.reshape((self.Q.shape[1],1))).reshape(self.Q.shape[1])
		
		# Extreme barrier
		if (
			(self.xlo is not None and (xr<self.xlo).any()) or
			(self.xhi is not None and (xr>self.xhi).any())
		):
			return np.Inf
		
		return self.f(xr)

class RotateC(object):
	"""
	Rotated constraints evaluator. 
	"""
	def __init__(self, c, Q, cndx):
		self.c=c
		self.Q=Q
		self.cndx=cndx
		
	def __call__(self, x):
		xr=np.dot(self.Q, x.reshape((self.Q.shape[1],1))).reshape(self.Q.shape[1])
		
		if self.c is None:
			return xr[self.cndx]
		else:
			return np.concatenate((
				self.c(xr), 
				xr[self.cndx]
			), axis=0)

class RotateFC(object):
	"""
	Rotated function and constraints evaluator. 
	"""
	def __init__(self, fc, Q, cndx, xlo=None, xhi=None):
		self.fc=fc
		self.Q=Q
		self.cndx=cndx
		self.extremeBarrierBounds=extremeBarrierBounds
		self.xlo=xlo
		self.xhi=xhi
	
	def __call__(self, x):
		xr=np.dot(self.Q, x.reshape((self.Q.shape[1],1))).reshape(self.Q.shape[1])
		
		# Extreme barrier
		if (
			(self.xlo is not None and (xr<self.xlo).any()) or
			(self.xhi is not None and (xr>self.xhi).any())
		):
			return np.Inf, []
		
		f,c=self.fc(xr)
		return f,np.concatenate((
			c, 
			xr[self.cndx]
		), axis=0)

class RotateCG(object):
	"""
	Rotated constraints Jacobian evaluator. 
	"""
	def __init__(self, fc, Q, cndx):
		self.cg=cg
		self.Q=Q
		self.cndx=cndx
	
	def __call__(self, x):
		xr=np.dot(self.Q, x.reshape((self.Q.shape[1],1))).reshape(self.Q.shape[1])
		J=self.cg(xr)
		return np.concatenate((
			J, 
			Q[self.cndx,:]
		), axis=0)

def rotateProblem(itf, Q, extremeBarrierBounds=False):
	"""
	Rotate a problem described by *itf* using orthogonal matrix *Q*. 
	
	Feeds all functions with Qx instead of x. 
	
	Converts bounds to constraints. 
	
	Treats rotated bounds with extreme barrier approach if
	*extremeBarrierBounds* is ``True``. 
	"""
	rotItf=deepcopy(itf)
	
	# Count finite bounds
	cndx=np.where(np.isfinite(itf["xlo"])|np.isfinite(itf["xhi"]))[0]
	
	# Convert bounds to constraints
	m=itf["m"]
	rotItf["m"]=m+cndx.size
	rotItf["xlo"].fill(-np.Inf)
	rotItf["xhi"].fill(np.Inf)
	
	clo=np.zeros(rotItf["m"])
	chi=np.zeros(rotItf["m"])
	clo[:m]=itf["clo"]
	chi[:m]=itf["chi"]
	clo[m:]=itf["xlo"][cndx]
	chi[m:]=itf["xhi"][cndx]
	
	rotItf["clo"]=clo
	rotItf["chi"]=chi
	
	# Update xmin
	if itf["xmin"] is not None:
		rotItf["xmin"]=np.dot(Q.T, itf["xmin"].reshape((Q.shape[0],1))).reshape(Q.shape[0])
		
	# Update x0
	if itf["x0"] is not None:
		rotItf["x0"]=np.dot(Q.T, itf["x0"].reshape((Q.shape[0],1))).reshape(Q.shape[0])
	
	# Note that the info structure is not rotated and applies to unrotated problem
	
	# Replace functions with rotated versions
	# Take into account additional constraints
	if itf["f"] is not None:
		if extremeBarrierBounds and cndx.size>0:
			rotItf["f"]=RotateF(itf["f"], Q, itf["xlo"], itf["xhi"])
		else:
			rotItf["f"]=RotateF(itf["f"], Q)
	if itf["g"] is not None:
		rotItf["g"]=RotateF(itf["g"], Q)
	if itf["c"] is not None:
		rotItf["c"]=RotateC(itf["c"], Q, cndx)
	if itf["fc"] is not None:
		if extremeBarrierBounds and cndx.size>0:
			rotItf["fc"]=RotateFC(itf["fc"], Q, cndx, itf["xlo"], itf["xhi"])
		else:
			rotItf["fc"]=RotateFC(itf["fc"], Q, cndx)
	if itf["cg"] is not None:
		rotItf["cg"]=RotateCG(itf["cg"], Q, cndx)
	
	# Add constraint function if original problem was unconstrained
	if m==0 and cndx.size>0 and itf["c"] is None and itf["fc"] is None:
		rotItf["c"]=RotateC(None, Q, cndx)
	
	return rotItf
			
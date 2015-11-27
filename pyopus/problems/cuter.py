"""
.. inheritance-diagram:: pyopus.problems.cuter
    :parts: 1

**Wrapper for CUTEr problems (PyOPUS subsystem name: CUTER)**

This is a CPI interface to CUTEr. You nedd to have CUTEr installed. 
See the :mod:`cutermgr` module for more information. 

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. It depends only on the cpi, 
the cutermgr, and the cuteritf modules. 
"""

try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y
		
try:
	import cutermgr as pcm
	available=True
except:
	available=False
	raise Exception, DbgMsgOut("CUTER", "CUTEr is not available. Install CUTEr.")

from cpi import CPI, MemberWrapper

__all__ = [ 'CUTEr' ]

class CUTEr(CPI):
	"""
	A wrapper class for accessing CUTEr problems.
	
	* *cuterName*    - CUTEr problem name (described by cuterName.sif)
	* *cachedName*   - compiled problem name, defaults to cuterName
	* *sifParams*    - SIF parameters (e.g. for specifying problem size)
	* *sifOptions*   - addittional SIF decoder command line options
	* *efirst*       - put equality constraints first (boolean)
	* *lfirst*       - put linear constraints first (boolean)
	* *nvfirst*      - put nonlinear variables first (boolean)
	* *forceRebuild* - force rebuilding even if the problem is already in cache
	* *quiet*        - supress output messages from compilers
	
	*cachedName* must not contain dots because it is a part of a Python module 
	name. 
	
	The actual compiled problem module is in the :attr:`mod` attribute. See 
	the :mod:`cutermgr` module on how to compile and import a CUTEr problem 
	without using this wrapper. 
	"""
	def __init__(self, cuterName, cachedName=None, 
				sifParams=None, sifOptions=None, 
				efirst=False, lfirst=False, nvfirst=False, 
				forceRebuild=True, quiet=True): 
		
		self.name=cuterName
		if cachedName is not None:
			self.cachedName=cachedName
		else:
			self.cachedName=cuterName
		
		# Build if needed
		if not pcm.isCached(self.cachedName) or forceRebuild:
			self.mod=pcm.prepareProblem(
				cuterName, 
				destination=self.cachedName, 
				sifParams=sifParams, 
				sifOptions=sifOptions, 
				efirst=efirst, lfirst=lfirst, nvfirst=nvfirst, 
				quiet=quiet
			)
		else:
			self.mod=pcm.importProblem(self.cachedName)
		
	def cpi(self): 
		"""
		Returns the common problem interface. 
		
		Best known minimum information is not available. 
		
		The info member of the returned dictionary is itself a dictionary with 
		the following members:
		
		* ``sifParams``  - SIF parameters
		* ``sifOptions`` - additional SIF decoder command line options
		
		An additional member is available named ``module``. It is a reference
		to the imported CUTEr problem module. 
		
		See the :class:`CPI` class for more information. 
		"""
		# Get problem information
		info=self.mod.getinfo()
		
		itf=self.prepareCPI(info['n'], info['m'])
		itf['name']=info['name']
		itf['xlo']=info['bl']
		itf['xhi']=info['bu']
		itf['x0']=info['x']
		itf['f']=self.mod.obj
		itf['g']=CUTErWrapper(self.mod.obj, args=[True], selector=1)
		
		# Problems with nonlinear constraints
		if info['m']>0:
			itf['clo']=info['cl']
			itf['chi']=info['cu']
			itf['c']=self.mod.cons
			itf['cg']=CUTErWrapper(self.mod.cons, args=[True], selector=1)
		
		itf['module']=self.mod
		
		itf['info']={
			'sifParams': info['sifparams'], 
			'sifOptions': info['sifoptions'], 
		}
		
		return self.fixBounds(itf)

class CUTErWrapper(object):
	"""
	Wraps a function *fun* in a callable object. On __call__ invokes it with 
	addittional positional arguments *args* and keyword arguments *kwargs*. 
	Extracts and returns the *selector* -th component of the returned 
	tuple/list/dictionary. If *selector* is ``None``, *fun*'s return value is 
	returned untouched. 
	
	*fun*, *args*, *kwargs*, and *selector* must be picklable. 
	"""
	
	def __init__(self, fun, args=[], kwargs={}, selector=None):
		self.fun=fun
		self.args=args
		self.kwargs=kwargs
		self.selector=selector
	
	def __call__(self, x):
		if self.selector is None:
			return self.fun(x, *self.args, **self.kwargs)
		else:
			return self.fun(x, *self.args, **self.kwargs)[self.selector]
	
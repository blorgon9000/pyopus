# -*- coding: UTF-8 -*-
"""
.. inheritance-diagram:: pyopus.problems.karmitsa
    :parts: 1
	
**Large scale nonsmooth test functions (Karmitsa set)
(PyOPUS subsystem name: KARMITSA)**

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. It depends only on the :mod:`cpi` 
and the :mod:`_karmitsa` modules. 

.. [karmitsa] Karmitsa N.: Test Problems for Large-Scale Nonsmooth 
              Minimization, Technical report B.4/2007,
              University of Jyvaskyla, Jyvaskyla, 2007. 
"""
import _karmitsa, os
import numpy as np
from cpi import CPI, MemberWrapper
try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y

__all__ = [ 'LSNSU', 'LSNSB', 'LSNSI' ]
		
		
class LSNSU(CPI):
	"""
	Unconstrained problems from the Karmitsa test suite.  Problems 10 and 11
	were added by Á. Bűrmen. 
	
	* *name*   - problem name
	* *number* - problem number (0-11)
	* *n*      - problem dimension
	
	Attributes:
	
	* :attr:`name`    - problem name
	* :attr:`n`       - number of variables
	* :attr:`initial` - initial values of variables
	* :attr:`fmin`    - best known minimum
	
	This module is independent of PyOPUS, meaning that it can be taken as is 
	and used as a module in some other package. It depends only on the cpi and 
	the _karmitsa modules. 
	"""
	
	fminTab=[
		0.0, 
		0.0, 
		np.NaN, 
		np.NaN, 
		np.NaN, 
		0.0, 
		0.0, 
		np.NaN, 
		0.0, 
		0.0, 
		0.0, 
		np.NaN
	]
	names=[
		"GeneralizedMAXQ", 
		"GeneralizedMXHILB", 
		"ChainedLQ", 
		"ChainedCB3I", 
		"ChainedCB3II", 
		"ActiveFaces", 
		"GeneralizedBrown2", 
		"ChainedMifflin2", 
		"ChainedCrescentI", 
		"ChainedCrescentII", 
		"GeneralizedL1HILB", 
		"GeneralizedWatson"
	]
	"List of all function names"
	
	functionNumber=dict(zip(names, range(len(names))))
	
	def __init__(self, name=None, number=None, n=2):
		if number is None and name is None:
			raise Exception, DbgMsg("KARMITSA", "Must specify name or number.")
			
		if number is not None and name is not None:
			raise Exception, DbgMsg("KARMITSA", "Name and number cannot be specified at the same time.")
			
		if number is not None:
			self.number=number
			if number<0 or number>11:
				raise Exception, DbgMsg("KARMITSA", "Bad problem number.")
			self.name=self.names[number]
			
		if name is not None:
			if name not in self.functionNumber:
				raise Exception, DbgMsg("KARMITSA", "Function not found.")
			self.number=self.functionNumber[name]
			self.name=name
		
		self.n=n
		
		self.initial=_karmitsa.startxu(self.number, self.n)
		if self.initial is None:
			raise Exception, DbgMsg("KARMITSA", "Problem selection error.")
		
		if self.number==2:
			self.fmin=-(self.n-1)*2.0**0.5
		elif self.number==3:
			self.fmin=2.0*(self.n-1)
		elif self.number==4:
			self.fmin=2.0*(self.n-1)
		elif self.number==7:
			if self.n==50:
				self.fmin=-34.795
			elif self.n==200:
				self.fmin=-140.86
			elif self.n==1000:
				self.fmin=-706.55
			else:
				self.fmin=None
		else:
			self.fmin=self.fminTab[self.number]
		
	def f(self, x):
		"""
		Returns the value of the function at *x*. 
		"""
		if x.size!=self.n:
			raise Exception, DbgMsg("KARMITSA", "Bad function argument.")
			
		dat=_karmitsa.funcu(self.number, x)
		if dat is None:
			raise Exception, DbgMsg("KARMITSA", "Evaluation failed.")
		(f,g)=dat
		
		return f
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Subgradient is not supported. 
		
		xmin is also not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=0)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		
		# Gradient is not supported
		# itf['g']=MemberWrapper(self, 'g')
		
		if 'fmin' in self.__dict__:
			itf['fmin']=self.fmin
		
		return self.fixBounds(itf)


class LSNSB(CPI):
	"""
	Bound constrained problems from the Karmitsa test suite.  
	
	* *name*     - problem name
	* *number*   - problem number (0-9)
	* *n*        - problem dimension
	* *feasible* - 0=initial point unchanged, 1=feasible, 2=strictly feasible, default=1
	
	Attributes:
	
	* :attr:`name`    - problem name
	* :attr:`n`       - number of variables
	* :attr:`xl`      - lower bounds on variables
	* :attr:`xh`      - upper bounds on variables
	* :attr:`initial` - initial values of variables
	* :attr:`fmin`    - best known minimum
	
	This module is independent of PyOPUS, meaning that it can be taken as is 
	and used as a module in some other package. It depends only on the cpi and 
	the _karmitsa modules. 
	"""
	
	fminTab=[
		0.0, 
		0.0, 
		np.NaN, 
		np.NaN, 
		np.NaN, 
		0.0, 
		0.0, 
		np.NaN, 
		0.0, 
		0.0
	]
	
	names=[
		"GeneralizedMAXQ", 
		"GeneralizedMXHILB", 
		"ChainedLQ", 
		"ChainedCB3I", 
		"ChainedCB3II", 
		"ActiveFaces", 
		"GeneralizedBrown2", 
		"ChainedMifflin2", 
		"ChainedCrescentI", 
		"ChainedCrescentII", 
	]
	"List of all function names"
	
	functionNumber=dict(zip(names, range(len(names))))
	
	def __init__(self, name=None, number=None, n=2, feasible=1):
		if number is None and name is None:
			raise Exception, DbgMsg("KARMITSA", "Must specify name or number.")
			
		if number is not None and name is not None:
			raise Exception, DbgMsg("KARMITSA", "Name and number cannot be specified at the same time.")
			
		if number is not None:
			self.number=number
			if number<0 or number>9:
				raise Exception, DbgMsg("KARMITSA", "Bad problem number.")
			self.name=self.names[number]
			
		if name is not None:
			if name not in self.functionNumber:
				raise Exception, DbgMsg("KARMITSA", "Function not found.")
			self.number=self.functionNumber[name]
			self.name=name
		
		self.n=n
		
		x0=_karmitsa.startxb(self.number, self.n)
		if x0 is None:
			raise Exception, DbgMsg("KARMITSA", "Problem selection error.")
		
		(bt, xl, xh)=_karmitsa.bounds(self.number, self.n)
		self.xl=np.where(bt==3, -np.Inf, xl)
		self.xh=np.where(bt==1, np.Inf, xh)
		
		# Project to feasible region
		if feasible==1:
			x0=np.where(x0<self.xl, xl, x0)
			x0=np.where(x0>self.xh, xh, x0)
		elif feasible==2:
			feas=1e-4
			x0=np.where(x0<=self.xl, xl+feas, x0)
			x0=np.where(x0>=self.xh, xh-feas, x0)
		
		self.initial=x0
		
		if self.number==2:
			self.fmin=-(self.n-1)*2.0**0.5
		elif self.number==3:
			self.fmin=2.0*(self.n-1)
		elif self.number==4:
			self.fmin=2.0*(self.n-1)
		elif self.number==7:
			if self.n==50:
				self.fmin=-34.795
			elif self.n==200:
				self.fmin=-140.86
			elif self.n==1000:
				self.fmin=-706.55
			else:
				self.fmin=None
		else:
			self.fmin=self.fminTab[self.number]
		
	def f(self, x):
		"""
		Returns the value of the function at *x*. 
		"""
		if x.size!=self.n:
			raise Exception, DbgMsg("KARMITSA", "Bad function argument.")
			
		dat=_karmitsa.funcb(self.number, x)
		(f,g)=dat
		if dat is None:
			raise Exception, DbgMsg("KARMITSA", "Evaluation failed.")
		return f
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Subgradient is not supported. 
		
		xmin is also not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=0)
		itf['name']=self.name
		itf['xlo']=self.xl
		itf['xhi']=self.xh
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		
		# Gradient is not supported
		# itf['g']=MemberWrapper(self, 'g')
		
		if 'fmin' in self.__dict__:
			itf['fmin']=self.fmin
		
		return self.fixBounds(itf)


class LSNSI(CPI):
	"""
	Inequality constrained problems from the Karmitsa test suite.  
	
	* *name*     - function name
	* *number*   - function number (0-9)
	* *cname*    - constraint name
	* *cnumber*  - constraint number (0-7)
	* *n*        - problem dimension
	
	Attributes:
	
	* :attr:`name`    - problem name
	* :attr:`fname`   - function name
	* :attr:`cname`   - constraint name
	* :attr:`n`       - number of variables
	* :attr:`m`       - number of constraints
	* :attr:`cl`      - lower bounds on constraint functions
	* :attr:`ch`      - upper bounds on constraint functions
	* :attr:`initial` - initial values of variables
	
	This module is independent of PyOPUS, meaning that it can be taken as is 
	and used as a module in some other package. It depends only on the cpi and 
	the _karmitsa modules. 
	"""
	
	names=[
		"GeneralizedMAXQ", 
		"GeneralizedMXHILB", 
		"ChainedLQ", 
		"ChainedCB3I", 
		"ChainedCB3II", 
		"ActiveFaces", 
		"GeneralizedBrown2", 
		"ChainedMifflin2", 
		"ChainedCrescentI", 
		"ChainedCrescentII", 
	]
	"List of all function names"
	
	cnames=[
		"TridiagonalI", 
		"TridiagonalII", 
		"MAD1I", 
		"MAD1II", 
		"ModifiedMAD1I", 
		"ModifiedMAD1II", 
		"P20I", 
		"P20II", 
	]
	"List of all constraint names"
	
	functionNumber=dict(zip(names, range(len(names))))
	constraintNumber=dict(zip(cnames, range(len(cnames))))
	
	def __init__(self, name=None, number=None, cname=None, cnumber=None, n=10):
		if number is None and name is None:
			raise Exception, DbgMsg("KARMITSA", "Must specify function name or number.")
			
		if number is not None and name is not None:
			raise Exception, DbgMsg("KARMITSA", "Function name and number cannot be specified at the same time.")
		
		if cnumber is None and cname is None:
			raise Exception, DbgMsg("KARMITSA", "Must specify constraint name or number.")
			
		if cnumber is not None and cname is not None:
			raise Exception, DbgMsg("KARMITSA", "Constraint name and number cannot be specified at the same time.")
				
		if number is not None:
			self.fnumber=number
			if number<0 or number>9:
				raise Exception, DbgMsg("KARMITSA", "Bad function number.")
			self.fname=self.names[number]
			
		if name is not None:
			if name not in self.functionNumber:
				raise Exception, DbgMsg("KARMITSA", "Function not found.")
			self.fnumber=self.functionNumber[name]
			self.fname=name
		
		if cnumber is not None:
			self.cnumber=cnumber
			if cnumber<0 or cnumber>7:
				raise Exception, DbgMsg("KARMITSA", "Bad constraint number.")
			self.cname=self.cnames[cnumber]
			
		if cname is not None:
			if cname not in self.constraintNumber:
				raise Exception, DbgMsg("KARMITSA", "Constraint not found.")
			self.cnumber=self.constraintNumber[cname]
			self.cname=cname
		
		self.name=self.fname+"_"+self.cname
		self.n=n
		
		dat=_karmitsa.xinit3(self.fnumber, self.cnumber, self.n)
		if dat is None:
			raise Exception, DbgMsg("KARMITSA", "Problem selection error.")
		(x0,m)=dat
		
		self.initial=x0
		self.m=m
		
		self.cl=np.zeros(m)
		self.cl.fill(-np.Inf)
		self.ch=np.zeros(m)
		
	def f(self, x):
		"""
		Returns the value of the function at *x*. 
		"""
		if x.size!=self.n:
			raise Exception, DbgMsg("KARMITSA", "Bad function argument.")
			
		dat=_karmitsa.funci(self.fnumber, x)
		if dat is None:
			raise Exception, DbgMsg("KARMITSA", "Function evaluation failed.")
		(f,g)=dat
		
		return f
	
	def c(self, x):
		"""
		Returns the value of the constraints at *x*. 
		"""
		if x.size!=self.n:
			raise Exception, DbgMsg("KARMITSA", "Bad function argument.")
			
		dat=_karmitsa.cineq(self.fnumber, self.cnumber, x)
		if dat is None:
			raise Exception, DbgMsg("KARMITSA", "Constraint evaluation failed.")
		(c,J)=dat
		
		return c
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Subgradient is not supported. 
		
		xmin is also not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, self.m)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		itf['c']=MemberWrapper(self, 'c')
		
		# Gradient is not supported
		# itf['g']=MemberWrapper(self, 'g')
		
		if 'fmin' in self.__dict__:
			itf['fmin']=self.fmin
		
		itf['clo']=self.cl
		itf['chi']=self.ch
		
		return self.fixBounds(itf)
		
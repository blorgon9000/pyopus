# -*- coding: UTF-8 -*-
"""
.. inheritance-diagram:: pyopus.problems.lvns
    :parts: 1
	
**Nonsmooth test functions by Lukšan and Vlček 
(PyOPUS subsystem name: LVNS)**

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. It depends only on the :mod:`cpi` 
and the :mod:`_lvns` modules. 

The code in the binary module shares variables. Therefore the function should 
be created and then used immediately. Creating another function may change 
the previously created one. This sucks, but what can you do? I know! Rewrite the 
FORTRAN code :)

.. [lvns] Lukšan L., Vlček J.: Test Problems for Nonsmooth Unconstrained and 
          Linearly Constrained Optimization, Technical report V-798, 
          ICS AS CR, Prague, 2000. 
"""
import _lvns, os
import numpy as np
from cpi import CPI, MemberWrapper
try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y

__all__ = [ 'UMM', 'UNS', 'LCMM' ]
		
		
class UMM(CPI):
	"""
	Unconstrained minimax problems from the Lukšan-Vlček test suite.  
	
	* *name*   - problem name
	* *number* - problem number (0-24)
	
	Attributes:
	
	* :attr:`name`    - problem name
	* :attr:`n`       - number of variables
	* :attr:`m`       - number of partial functions
	* :attr:`initial` - initial values of variables
	* :attr:`fmin`    - best known minimum
	
	All test functions in this class are maps from :math:`R^n` to :math:`R`. 
	Every function is comprised of *m* partial functions 

	.. math::
	  f_1(x) ... f_m(x)
  
	where *x* is a *n*-dimensional vector. 

	The actual test function is then constructed as 

	.. math::
	  f(x) = \\max_{i=1}^m  f_i(x) 
  
	or 

	.. math::
	  f(x) = \\max_{i=1}^m \\left| f_i(x) \\right|
	"""
	
	fminTab=[
		1.9522245,
		0.0,
		0.0,
		3.5997193,
		-44.0,
		-44.0,
		0.42021427e-2,
		0.50816327e-1,
		0.80843684e-2,
		115.70644,
		0.26359735e-2,
		0.20160753e-2,
		0.12041887e-6,
		0.12237125e-3,
		0.22340496e-1,
		0.34904926e-1,
		0.19729063,
		0.61852848e-2,
		680.63006,
		24.306209,
		133.72828,
		54.598150,
		261.08258,
		0.14743027e-7,
		0.48027401e-1,
	]
	
	names=[
		"CB2", 
		"WF", 
		"SPIRAL", 
		"EVD52", 
		"RosenSuzuki", 
		"Polak6", 
		"PBC3", 
		"Bard", 
		"KowalikOsborne", 
		"Davidon2", 
		"OET5", 
		"OET6", 
		"GAMMA", 
		"EXP", 
		"PBC1", 
		"EVD61", 
		"Transformer", 
		"Filter", 
		"Wong1", 
		"Wong2", 
		"Wong3", 
		"Polak2", 
		"Polak3", 
		"Watson", 
		"Osborne2", 
	]
	"List of all function names"
	
	functionNumber=dict(zip(names, range(len(names))))
	
	def setup(self):
		"""
		Initializes the binary implementation of the function. 
		After this function is called no other function from the same test set 
		may be created or initialized because that will change the internal 
		variables and break the function. Returns an info structure. 
		"""
		return _lvns.tiud06(self.number)
	
	def __init__(self, name=None, number=None):
		if number is None and name is None:
			raise Exception, DbgMsg("LVNS", "Must specify name or number.")
			
		if number is not None and name is not None:
			raise Exception, DbgMsg("LVNS", "Name and number cannot be specified at the same time.")
		
		if number is not None:
			self.number=number
			if number<0 or number>24:
				raise Exception, DbgMsg("LVNS", "Bad problem number.")
			self.name=self.names[number]
		
		if name is not None:
			if name not in self.functionNumber:
				raise Exception, DbgMsg("LVNS", "Function not found.")
			self.number=self.functionNumber[name]
			self.name=name
			
		info=self.setup()
		
		self.n=info['n']
		self.m=info['m']
		self.initial=info['x0']
		self.maxAbs=info['type']>=0
		self.fmin=self.fminTab[self.number]
		
	def fi(self, x, i=None):
		"""
		Returns the value of the *i*-th partial function at *x*. 
		If *i* is not given a vector of all partial functions is returned. 
		"""
		if i is not None:
			return _lvns.tafu06(self.number, x, i)
		else:
			return _lvns.tafu06(self.number, x)
	
	def f(self, x):
		"""
		Returns the value of the function at *x*. 
		"""
		if self.maxAbs:
			return np.abs(self.fi(x)).max()
		else:
			return self.fi(x).max()
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Gradient is not supported. 
		Anyway it is valid only where the functions are smooth. 
		
		xmin is also not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=0)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		itf['setup']=MemberWrapper(self, 'setup')
		
		# Gradient is not supported
		# itf['g']=MemberWrapper(self, 'g')
		
		if 'fmin' in self.__dict__:
			itf['fmin']=self.fmin
		
		return self.fixBounds(itf)
			

class UNS(CPI):
	"""
	Unconstrained nonsmooth problems from the Lukšan-Vlček test suite.  
	
	* *name*   - problem name
	* *number* - problem number (0-24)
	
	Attributes:
	
	* :attr:`name`    - problem name
	* :attr:`n`       - number of variables
	* :attr:`initial` - initial values of variables
	* :attr:`fmin`    - best known minimum
	
	All test functions in this module are maps from :math:`R^n` to :math:`R`. 
	Gradients are valid only where the functions are smooth (continuously 
	differentiable). 
	"""
	
	fminTab=[
		0.0, 
		0.0, 
		1.9522245, 
		2.0, 
		-3.0, 
		7.20, 
		-1.4142136, 
		-1.0, 
		-1.0, 
		-8.0, 
		-44.0, 
		22.600162, 
		-32.348679, 
		-2.9197004, 
		0.5598131, 
		-0.8414083, 
		9.7857721, 
		16.703838, 
		0.0, 
		0.0, 
		-638565.0, 
		0.0, 
		0.0, 
		0.0, 
		32.348679
	]
	
	names=[
		"Rosenbrock", 
		"Crescent", 
		"CB2", 
		"CB3", 
		"DEM", 
		"QL", 
		"LQ", 
		"Mifflin1", 
		"Mifflin2", 
		"Wolfe", 
		"RosenSuzuki", 
		"Shor", 
		"Colville1", 
		"HS78", 
		"ElAttar", 
		"Maxquad", 
		"Gill", 
		"Steiner2", 
		"Maxq", 
		"Maxl", 
		"TR48", 
		"Goffin", 
		"MXHILB", 
		"L1HILB", 
		"ShellDual", 
	]
	"List of all function names"
	
	functionNumber=dict(zip(names, range(len(names))))
	
	def setup(self):
		"""
		Initializes the binary implementation of the function. 
		After this function is called no other function from the same test set 
		may be created or initialized because that will change the internal 
		variables and break the function. Returns an info structure. 
		"""
		# Problem #20 requires an external file named TEST19.DAT. 
		# We must temporarily change the working directory so that the file is  
		# found by the underlying FORTRAN code. 
		# Note that the file was changed. The f2c compiled code does not handle 
		# input separated by newlines. Therefore we removed them. 
		if self.number==20:
			# Store working directory
			wd=os.getcwd()
			# Set the working directory to the path of the _lvns binary module
			os.chdir(os.path.dirname(_lvns.__file__))
			
		retval=_lvns.tiud19(self.number)
		
		if self.number==20:
			# Go back to original working directory
			os.chdir(wd)
		
		return retval
		
	def __init__(self, name=None, number=None):
		if number is None and name is None:
			raise Exception, DbgMsg("LVNS", "Must specify name or number.")
			
		if number is not None and name is not None:
			raise Exception, DbgMsg("LVNS", "Name and number cannot be specified at the same time.")
		
		if number is not None:
			self.number=number
			if number<0 or number>24:
				raise Exception, DbgMsg("LVNS", "Bad problem number.")
			self.name=self.names[number]
		
		if name is not None:
			if name not in self.functionNumber:
				raise Exception, DbgMsg("LVNS", "Function not found.")
			self.number=self.functionNumber[name]
			self.name=name
		
		info=self.setup()
		
		self.n=info['n']
		self.initial=info['x0']
		self.fmin=self.fminTab[self.number]
	
	def f(self, x):
		"""
		Returns the value of the function at *x*. 
		"""
		return _lvns.tffu19(self.number, x)

	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Gradient is not supported because it is untested at this time. 
		Anyway it is valid only where the functions are smooth. 
		
		xmin is also not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=0)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		itf['setup']=MemberWrapper(self, 'setup')
		
		# Gradient is not supported
		# itf['g']=MemberWrapper(self, 'g')
		
		if 'fmin' in self.__dict__:
			itf['fmin']=self.fmin
		
		return self.fixBounds(itf)

class LCMM(CPI):
	"""
	Linearly constrained minimax problems from the Lukšan-Vlček test suite.  
	
	* *name*   - problem name
	* *number* - problem number (0-14)
	
	Attributes:
	
	* :attr:`name`    - problem name
	* :attr:`n`       - number of variables
	* :attr:`m`       - number of partial functions
	* :attr:`nc`      - number of linear constraints
	* :attr:`xl`      - lower bounds on variables
	* :attr:`xh`      - upper bounds on variables
	* :attr:`cl`      - lower bounds on constraint functions
	* :attr:`ch`      - upper bounds on constraint functions
	* :attr:`Jc`      - Jacobian of constraints, one row per constraint
	* :attr:`initial` - initial values of variables
	* :attr:`fmin`    - best known minimum
	
	All test functions in this module are maps from :math:`R^n` to :math:`R`. 
	Every function is comprised of *m* partial functions 

	.. math::
	  f_1(x) ... f_m(x)
	  
	where *x* is a *n*-dimensional vector. 

	The actual test function is then constructed as 

	.. math::
	  f(x) = \\max_{i=1}^m  f_i(x) 
	  
	or 

	.. math::
	  f(x) = \\max_{i=1}^m \\left| f_i(x) \\right|
	  
	The *nc* constraints are linear and are of the form

	.. math::
	  cl \\leq c(x) \\leq ch
	  
	Beside linear constraints a problem can also have bounds of the form

	.. math::
	  xl \\leq x \\leq xh
	"""
	
	fminTab=[
		-0.38965952, 
		-0.33035714, 
		-0.44891079, 
		-0.42928061, 
		-1.85961870, 
		0.10183089, 
		0.0, 
		24.306209, 
		133.72828, 
		0.50694799, 
		0.27607734e-3, 
		-1768.8070, 
		1227.2260, 
		7049.2480, 
		174.78699
	]
	
	names=[
		"MAD1", 
		"MAD2", 
		"MAD4", 
		"MAD5", 
		"PENTAGON", 
		"MAD6", 
		"EQUIL", 
		"Wong2", 
		"Wong3", 
		"MAD8", 
		"BPfilter",
		"HS114", 
		"Dembo3", 
		"Dembo5", 
		"Dembo7"
	]
	"List of all function names"
	
	functionNumber=dict(zip(names, range(len(names))))
	
	def setup(self):
		"""
		Initializes the binary implementation of the function. 
		After this function is called no other function from the same test set 
		may be created or initialized because that will change the internal 
		variables and break the function. Returns an info structure. 
		"""
		return _lvns.eild22(self.number)
		
	def __init__(self, name=None, number=None):
		if number is None and name is None:
			raise Exception, DbgMsg("LVNS", "Must specify name or number.")
			
		if number is not None and name is not None:
			raise Exception, DbgMsg("LVNS", "Name and number cannot be specified at the same time.")
		
		if number is not None:
			self.number=number
			if number<0 or number>14:
				raise Exception, DbgMsg("LVNS", "Bad problem number.")
			self.name=self.names[number]
		
		if name is not None:
			if name not in self.functionNumber:
				raise Exception, DbgMsg("LVNS", "Function not found.")
			self.number=self.functionNumber[name]
			self.name=name
		
		info=self.setup()
		
		self.n=info['n']
		self.m=info['m']
		self.nc=info['nc']
		
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

		# Fix a bug in original lvlcmm.f (eild22)
		#   Lower and upper bound for problem 6 (MAD6), variable 7 should be set to 3.5
		if self.number==5:
			self.xl[6]=3.5
			self.xh[6]=3.5
		
		# MAD6 description in the paper has different contraints
		# Implementation ommits c1, and c9 (c9 is actually an equality bound on variable 7). 
		# There are 7 constraints left. 
		
		# HS114 - in paper the bounds on x10 are 145 and 162
		#         in implementation the bounds are 140 and 160
		
		# PENTAGON - constraint function coefficients with sin() and cos() are exchanged in the implementations
		# initial point violates constraints
		
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
		
		self.Jc=info['Jc']
		
		self.initial=info['x0']
		self.maxAbs=info['type']>=0
		self.fmin=self.fminTab[self.number]
		
		# Fix the initial point of the PENTAGON problem so that it satisfies the constraints
		if self.number==4:
			self.initial[5]=0.0
		
	def fi(self, x, i=None):
		"""
		Returns the value of the *i*-th partial function at *x*. 
		If *i* is not given a vector of all partial functions is returned. 
		"""
		if i is not None:
			return _lvns.tafu22(self.number, x, i)
		else:
			return _lvns.tafu22(self.number, x)
	
	def f(self, x):
		"""
		Returns the value of the function at *x*. 
		"""
		if self.maxAbs:
			return np.abs(self.fi(x)).max()
		else:
			return self.fi(x).max()
			
	def c(self, x):
		"""
		Returns the values of the constraint functions at *x*. 
		"""
		return np.dot(self.Jc, x.reshape((self.n,1))).reshape((self.nc))
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Gradient is not supported. 
		Anyway it is valid only where the functions are smooth. 
		
		xmin is also not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=self.nc)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		itf['c']=MemberWrapper(self, 'c')
		itf['setup']=MemberWrapper(self, 'setup')
		
		# Gradient is not supported
		# itf['g']=MemberWrapper(self, 'g')
		# itf['cg']=MemberWrapper(self, 'cg')
		
		if 'fmin' in self.__dict__:
			itf['fmin']=self.fmin
		
		itf['xlo']=self.xl
		itf['xhi']=self.xh
		itf['clo']=self.cl
		itf['chi']=self.ch
		
		return self.fixBounds(itf)

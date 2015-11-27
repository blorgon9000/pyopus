# -*- coding: UTF-8 -*-
"""
.. inheritance-diagram:: pyopus.problems.lvu
    :parts: 1
	
**Unconstrained test functions by Lukšan and Vlček 
(PyOPUS subsystem name: LVU)**

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. It depends only on the :mod:`cpi` 
and the :mod:`_lvu` modules. 

The code in the binary module shares variables. Therefore the function should 
be created and then used immediately. Creating another function may change 
the previously created one. This sucks, but what can you do? I know! Rewrite 
the FORTRAN code :)

.. [lvu] Lukšan L., Vlček J.: Test Problems for Unconstrained Optimization, 
	 Technical report V-897, ICS AS CR, Prague, 2003. 
"""
import _lvu, os
import numpy as np
from cpi import CPI, MemberWrapper
try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y

__all__ = [ 'LVU' ]
		
		
class LVU(CPI):
	"""
	Unconstrained problems from the Lukšan-Vlček test suite.  
	
	* *name*   - problem name
	* *number* - problem number (0-91)
	
	Attributes:
	
	* :attr:`name`    - problem name
	* :attr:`n`       - number of variables
	* :attr:`initial` - initial values of variables
	
	All test functions in this class are maps from :math:`R^n` to :math:`R`. 
	"""
	
	names=[
		'ChainedRosenbrock', 
		'ChainedWood', 
		'ChainedPowelSingular', 
		'ChainedCraggAndLevy', 
		'GeneralizedBroydenTridiagonal1', 
		'GeneralizedBroydenBanded1', 
		'SevenDiagonalBroyden', 
		'ModifiedNazarethTrigonometric', 
		'AnotherTrigonometric', 
		'TointTrigonometric', 
		'AugmentedLagrangian', 
		'GeneralizedBrown1', 
		'GeneralizedBrown2', 
		'DiscreteBoundaryValue1', 
		'DiscreteVariational', 
		'BandedTrigonometric', 
		'Variational1', 
		'Variational2', 
		'Variational3', 
		'Variational4', 
		'Variational5', 
		'VariationalCalvar2', 
		'Penalty2', 
		'Penalty3', 
		'ExtendedRosenbrock', 
		'ExtendedPowellSingular', 
		'Penalty1', 
		'VariablyDimensioned', 
		'BrownAlmostLinear', 
		'DiscreteBoundaryValue2', 
		'BroydenTridiagonal1', 
		'GeneralizedBroydenTridiagonal2', 
		'GeneralizedBroydenBanded2', 
		'ChainedFreudensteinAndRoth', 
		'WrightAndHoltZeroResidual', 
		'TointQuadraticMerging', 
		'ChainedExponential', 
		'ChainedSerpentine', 
		'ChainedModifiedHS47', 
		'ChainedModifiedHS48_1', 
		'SparseSignomial', 
		'SparseExponential', 
		'SparseTrigonometric', 
		'CountercurrentReactors', 
		'TridiagonalSystem', 
		'StructuredJacobian', 
		'ModifiedDiscreteBoundaryValue', 
		'ChainedModifiedHS48_2', 
		'AttractingRepelling', 
		'TointExponentialTrigonometricMerging', 
		'CountercurrentReactors2', 
		'TrigonometricExponential1', 
		'TrigonometricExponential2', 
		'SingularBroyden', 
		'FiveDiagonal', 
		'SevenDiagonal', 
		'ExtendedFreudensteinAndRoth', 
		'ExtendedCraggAndLevy', 
		'BroydenTridiagonal2', 
		'ExtendedPowellBadlyScaled', 
		'ExtendedWood', 
		'TridiagonalExponential', 
		'Brent', 
		'Troesch', 
		'FlowInChannel', 
		'SwirlingFlow', 
		'Bratu', 
		'Poisson1', 
		'Poisson2', 
		'PorousMedium', 
		'ConvectionDifussion', 
		'NonlinearBiharmonic', 
		'DrivenCavity', 
		'Problem74', 
		'Problem201_27', 
		'Problem202_27', 
		'Problem205_27', 
		'Problem206_27', 
		'Problem207_27', 
		'Problem208_27', 
		'Problem209_27', 
		'Problem212_27', 
		'Problem213_27', 
		'Problem214_27', 
		'GheriAndMancino', 
		'OrtegaAndRheinboldt', 
		'AscherAndRussel1', 
		'AscherAndRussel2', 
		'AllgowerAndGeorg', 
		'PotraAndRheinboldt', 
		'Problem91', 
		'Problem92', 
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
		return _lvu.tiud28(self.number, self.n)
	
	def __init__(self, name=None, number=None, n=10):
		if number is None and name is None:
			raise Exception, DbgMsg("LVU", "Must specify name or number.")
			
		if number is not None and name is not None:
			raise Exception, DbgMsg("LVU", "Name and number cannot be specified at the same time.")
		
		if number is not None:
			self.number=number
			if number<0 or number>91:
				raise Exception, DbgMsg("LVU", "Bad problem number.")
			self.name=self.names[number]
		
		if name is not None:
			if name not in self.functionNumber:
				raise Exception, DbgMsg("LVU", "Function not found.")
			self.number=self.functionNumber[name]
			self.name=name
		
		self.n=n		
		
		info=self.setup()
		
		self.initial=info['x0']
		
	def f(self, x):
		"""
		Returns the value of the function at *x*. 
		"""
		if len(x.shape)>1 or x.shape[0]!=self.n:
			raise Exception, DbgMsg("LVU", "Bad x.")
		
		return _lvu.tffu28(self.number, x)
		
	
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Gradient is not supported. 
		
		xmin and fmin are not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=0)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		itf['setup']=MemberWrapper(self, 'setup')
		
		# Gradient is not supported
		# itf['g']=MemberWrapper(self, 'g')
		
		return self.fixBounds(itf)
	
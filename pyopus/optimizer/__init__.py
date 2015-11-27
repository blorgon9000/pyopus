"""
**Optimization algorithms and test function suites module**

This module provides unconstrained and bound constrained optimization 
algorithms. 

Nothing from the submodules of this module is imported into the main 
:mod:`optimizer` module. The :mod:`optimizer` module provides only the 
:func:`optimizerClass` function for on-demand loading of optimizer classes. 

Optimization algorithms search for the argument which results in the lowest 
possible value of the **cost function**. The search can be **constrained** 
meaning that only certain values of the argument are allowed. 

**Iteration of the optimization algorithm** is another name for the 
consecutive number of cost function evaluation. 
"""

__all__=[ 'optimizerClass' ]

optimizers={
	'CoordinateSearch': 'coordinate', 
	'HookeJeeves': 'hj', 
	'NelderMead': 'nm', 
	'BoxComplex': 'boxcomplex', 
	'GRNelderMead': 'grnm', 
	'SANelderMead': 'sanm', 
	'DifferentialEvolution': 'de', 
	'ParallelSADE': 'psade', 
	'QPMADS': 'qpmads'
}
"""
A dictionary with optimizer class name for key holding the module names 
corresponding to optimizer classes. 
"""

def optimizerClass(className): 
	"""
	Returns the class object of the optimizer named *className*. 
	Raises an exception if the optimizer class object is not found. 
	
	This function provides on-demand loading of optimizer classes. 
	
	To create an optimizer object of the class HookeJeeves that minimizes 
	function ``f`` and put it in ``opt`` use::
	
		from pyopus.optimizer import optimizerClass
		OptClass=optimizerClass('HookeJeeves')
		opt=OptClass(f)
	"""
	return __import__("pyopus.optimizer."+optimizers[className], globals(), locals(), [className]).__dict__[className]

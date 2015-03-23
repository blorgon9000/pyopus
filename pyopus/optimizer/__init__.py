"""
**Optimization algorithms and test function suites module**

This module provides unconstrained and bound constrained optimization 
algorithms. 

Suites of test functions for local and global optimization algorithms are also 
available. 

Nothing from the submodules of this module is imported into the main 
:mod:`optimizer` module. The :mod:`optimizer` module provides only the 
:func:`optimizerClass` function for on-demand loading of optimizer classes. 

The :mod:`mgh` and the :mod:`glbctf` submodule are self-contained and can be 
used separately from PyOPUS by simply copying ``mgh.py`` or the ``glbctf.py`` 
file to your own project. 

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
	'SANelderMead': 'sanm', 
	'DifferentialEvolution': 'de', 
	'ParallelSADE': 'psade', 
	'ParallelPointEvaluator': 'ppe'
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

"""
**Simulator support module**

Nothing from the submodules of this module is imported into the main 
:mod:`simulator` module. The :mod:`simulator` module provides only the 
:func:`simulatorClass` function for on-demand loading of simulator classes. 
"""

__all__=[ 'simulatorClass' ]

simulators={
	'SpiceOpus': 'spiceopus', 
	'HSpice': 'hspice', 
	'Spectre': 'spectre'
}
"""
A dictionary with simulator class name for key holding the 
module names corresponding to simulator classes. 
"""

def simulatorClass(className): 
	"""
	Returns the class object of the simulator named *className*. 
	Raises an exception if the simulator class object is not found. 
	
	This function provides on-demand loading of simulator classes. 
	
	To create a simulator object of the class SpiceOpus and put it in ``sim`` 
	use::
	
		from pyopus.simulator import simulatorClass
		SimClass=simulatorClass('SpiceOpus')
		sim=SimClass()
	"""
	return __import__("pyopus.simulator."+simulators[className], globals(), locals(), [className]).__dict__[className]


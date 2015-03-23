"""
**Visualization of simulation results**

This module provides the functionality for visulaizing the results stored in 
a :class:`~pyopus.evaluator.performance.PerformanceEvaluator`. 

Only portable parts of this module's submodules are imported into the main 
:mod:`visual` module. The memebrs of the mod:`wxmplplotter` module are 
therefore not imported (because it depends on wxPython and Matplotlib. 

So you can do::

	from pyopus.visual import IterationPlotter
	
instead of::

	from pyopus.visual.plotter import IterationPlotter
	
But you cannot import :class:`WxMplPlotter` with::

	from pyopus.visual import WxMplPlotter
	
Instead you must specify the full path to :class:`WxMplPlotter`:: 

	from pyopus.visual.wxmplplotter import WxMplPlotter
"""

# Import only portable stuff.
from plotter import *

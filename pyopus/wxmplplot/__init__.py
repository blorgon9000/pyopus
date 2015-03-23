"""
**Threaded plotting support based on Matplotlib and wxPython**

This module is based on wxMpl by Ken McIvor

http://agni.phys.iit.edu/~kmcivor/wxmpl/

It provides the basic plot window managemet that is performed in a separate 
thread so that MATLAB style plotting is possible in Python. The rendering is 
performed by Matplotlib on a wxPython canvas. 

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. 

Because this module depends on Matplotlib and wxPython its members are not 
imported into the main PyOPUS module. 

All members of the :mod:`~pyopus.wxmplplot.plotitf` module are imported 
into the :mod:`~pyopus.wxmplplot` module. This way you can import the plotting 
interface as::

	from pyopus import wxmplplot as pyopl
	
instead of more complicated::
	
	from pyopus.wxmplplot import plotitf as pyopl
"""
from plotitf import *

"""
**Performance evaluation module**

Provides functions and classes for extracting performance measures (like gain, 
bandwidth, rise time, etc.) from simulated responses, management of multiple 
simulators for obtaining a set of performance measure values, and the 
construction of a cost function.

A **performance measure** is a scalar or a vector giving an aspect of the 
performance of some simulated system. 

A **corner** is a set of operating conditions under which the system is 
simulated. Typically a system is simulated across several corners.

Everything from the :mod:`performance` and :mod:`cost` submodules of this 
module is imported to the main :mod:`evaluator` module. 
"""

from performance import *
from cost import *


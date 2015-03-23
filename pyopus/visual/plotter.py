"""
.. inheritance-diagram:: pyopus.visual.plotter
    :parts: 1
	
**An optimization algorithm plugin for visualization of simulation results**
"""

from ..optimizer.base import Reporter

import re
import sys

__all__ = [ 'IterationPlotter' ] 

		
class IterationPlotter(Reporter):
	"""
	A reporter plugin that uses a *plotter* to visualize the simulation results 
	during optimization. The plotting is initiated every time the optimization 
	algorithm finds a point with a lower value of the cost function. 
	
	This plugin produces an annotation at the computer where the evaluation of 
	the cost function takes place. The annotation that is produced is the 
	:attr:`results` member of the 
	:class:`~pyopus.evaluator.performance.PerformanceEvaluator` object 
	obtained from the *plotter*. On the master side this annotation is used 
	for updating the 
	:class:`~pyopus.evaluator.performance.PerformanceEvaluator` object of the 
	*plotter*. 
	
	See the :class:`pyopus.optimizer.base.Reporter` class for more details. 
	"""
	def __init__(self, plotter):
		Reporter.__init__(self)
		self.plotter=plotter
				
	def __call__(self, x, f, opt, annotation=None):
		# Update best no matter what the annotation is 
		if opt.f is None or opt.niter==opt.bestIter: 
			
			# Write annotation
			if annotation is not None:
				self.plotter.performanceEvaluator().results=annotation
				
			# Report (plot)
			if not self.quiet:
				self.plotter('iter='+str(opt.niter), 'f='+str(f))
		
		# Return annotation
		return self.plotter.performanceEvaluator().results
			

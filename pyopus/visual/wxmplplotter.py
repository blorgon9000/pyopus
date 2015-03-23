"""
.. inheritance-diagram:: pyopus.visual.wxmplplotter
    :parts: 1
	
**A wxPython and Matplotlib based simulation results plotter
(PyOPUS subsystem name: WxMplPL)**

This plotter takes the simulation results from a 
:class:`~pyopus.evaluator.performance.PerformanceEvaluator` object. 

Because this plotter depends on wxPython and Matplotlib it is not imported 
into the main PyOPUS module. 

The plot windows and their contents are described by a dictionary with the 
following keys:

* ``graphs`` - lists the plot windows and their axes
* ``styles`` - lists the style masks for traces
* ``traces`` - lists the traces that will be displayed on the axes of 
  plot windows
  
The value of ``graphs`` is a dictionary with graph name for key. The value 
for every key is a dictionary describing a graph with the following members:

* ``title`` - the title of the canvas displaying the graph
* ``shape`` - a dictionary specifying the size of the plot window.
  It has the following members:
  
  * ``figsize`` - a tuple giving the horizontal and vertical size of the plot 
    window in inches
  * ``figpx`` - a tuple giving the horizontal and vertical size of the plot 
    window in pixels
  * ``dpi`` - a number specifying the dots-per-inch value for the plot window
    This value is used for converting ``figsize`` to pixels when the plot 
    window is displayed on the screen. 
  
  If both ``figsize`` and ``figpx`` are given, ``figpx`` takes precedence over 
  ``figsize``. 
  
  If ``figpx`` is specified ``dpi`` is used for calculating the size of the 
  figure in inches (used when a plot window is saved to a Postscript file). 
  
  If ``figpx`` is not specified, ``dpi`` is used for convertion ``figsize`` 
  to pixels. The obtained values specify the size of the plot window's canvas. 
  
  If ``dpi`` is not specified the default Matplotlib value is used. 
  
  The contents of the ``shape`` dictionary are passed to the constructor of 
  the :class:`~pyopus.wxmplplot.wxmplitf.PlotFrame` object that corresponds to 
  the plot window. 

* ``axes`` - a dictionary with axes name for key decribing the axes that 
  reside in the plot window

  Every value in the ``axes`` dictionary describes one axes of the plot window. 
  It is itself a dictionary with the following members:

  * ``subplot`` - the subplot specification (tuple of 3 integers) passed to the 
    :func:`pyopus.wxmplplot.plotitf.subplot` function at axes creation
  * ``rectangle`` - the rectangle specification (tuple of 4 integers) passed to 
    the :func:`pyopus.wxmplplot.plotitf.rectangle` function at axes creation
  * ``options`` - a dictionary passed to the axes creation function 
    (:func:`pyopus.wxmplplot.plotitf.subplot` or 
    :func:`pyopus.wxmplplot.plotitf.rectangle`) as keyword arguments at axes 
    creation time 
  * ``gridtype`` - type of grid for the axes. 
    ``rect`` for rectilinear (default) or ``polar`` for polar grid. 
  * ``xscale`` - a dictionary describing the type of the x-axis scale
  * ``yscale`` - a dictionary describing the type of the y-axis scale
  * ``xlimits`` - a tuple with two values specifying the lower and the upper 
    limit for the x-axis scale
  * ``ylimits`` - a tuple with two values specifying the lower and the upper 
    limit for the y-axis scale
  * ``xlabel`` - the label for the x-axis
  * ``ylabel`` - the label for the y-axis
  * ``title`` - the title for the axes
  * ``legend`` - a boolean flag indicating if legend should be displayed. 
    ``False`` by default. 
  * ``grid`` - a boolean flag indicating if gridlines should be displayed. 
    ``False`` by default. `
    
  If both ``rectangle`` and ``subplot`` are specified, ``rectangle`` is used. 
  
  Axis scale type (``xscale`` and ``yscale``) is a dictionary with the 
  following members:
  
  * ``type`` - type of scale (``linear``, ``log``, or ``symlog``). 
    Default is ``linear``. 
  * ``base`` - the base for the log scale (default is 10). 
  * ``subticks`` - an array of log scale subticks 
    (for base 10 this is ``range(10)``)
  * ``linthresh`` - linearity threshold for the ``symlog`` type of scale

  See the :meth:`Axes.set_xscale` and :meth:`Axes.set_yscale` methods in 
  Matplotlib for more information. 

The value of ``styles`` is a list of dictionaries. Every dictionary specifyes 
a style with the following dictionary members:

* ``pattern`` - a tuple of 4 regular expressions for matching a trace to a 
  style. The first member of the tuple matches the graph name, the second one 
  matches the axes name, the third matches the corner name and the fourth 
  matches the trace name. See the :mod:`re` standard Python module for the 
  explanation of regular expressions. 
* ``style`` - a dictionary specifying style directives. Members of this 
  dictionary are keyword arguments to the 
  :func:`~pyopus.wxmplplot.plotitf.plot` function. 
  
A trace style is determined by starting with no style directives. The 
``styles`` list is traveversed and a style specified by the ``style`` 
member is applied if the ``pattern`` matches the graph, axes, corner, and 
trace name. 

Style directives that appear later in the list override those that appear 
earlier. The final trace style is obtained by applying the ``style`` specified 
in the trace definition. This style overrides the style directives obtained 
with matching ``patterns`` from the ``styles`` list. 

The ``traces`` member is a dictionary with trace name for key. Values are 
dictionaries with the following members: 

* ``graph`` - the name of the plot window in which the trace will appear
* ``axes`` - the name of the axes in the plot window where trace will appear
* ``xresult`` - the name of the performance measure in the 
  :class:`~pyopus.evaluator.performance.PerformanceEvaluator` object that is 
  used for x-axis values
* ``yresult`` - the name of the performance measure in the 
  :class:`~pyopus.evaluator.performance.PerformanceEvaluator` object that is 
  used for y-axis values
* ``corners`` - a list of corners for which the trace will be plotted. 
  If ommitted the trace is plotted for all corners in which the ``xresult`` 
  and ``yresult`` are evaluated. 
* ``style`` - a dictionary specifying the style directives 
  (keyword arguments to the :func:`~pyopus.wxmplplot.plotitf.plot` function)
"""

from ..misc.debug import *
import pyopus.wxmplplot as pyopl

import re 
import sys 

__all__ = [ 'WxMplPlotter' ] 

class WxMplPlotter(object):
	"""
	A class that plots the peformance measures from the 
	*perforamnceEvaluator* object. 
	
	The way these measures are plotted is specified by *setup*. 
	
	If *debug* is greater than 0 debug messages are printed at standard output. 
	
	Objects of this class are callable with the following calling convention: 
	
	``object(prefixText='', postfixText='', createPlots=True)``
	
	The title of every plot windows is the plot window name. 
	
	The title of every canvas is composed of *prefixText*, plot window title, 
	and *postfixText*. 
	
	The *createPlots* is ``True`` the plot windows are created if they do not 
	exist. If a plot window does not exist and it is not created then the 
	traces that are supposed to be displayed by that plot window are not 
	plotted. 
	
	If *initializePlotting* is ``True`` the plotting system is initialized 
	(graphical thread is started) automatically when needed. 
	"""
	log2minors=[2**(x/12.0)-1 for x in range(12)]
	log10minors=range(10)
	
	def __init__(self, setup, performanceEvaluator, debug=0):
		self.setup=setup
		self.pe=performanceEvaluator
		self.debug=debug
		
		self.figure={}
		self.titleArtist={}
		self.plotAxes={}
		
		self.compiledStyles=[]
		self.tracesOnAxes={}
		self.compiledCornerNames={}
		
		if self.debug:
			DbgMsgOut("WxMplPL", "Compiling.")
		self._compile()
	
	def performanceEvaluator(self):
		"""
		Returns the performance evaluator object specified when the plotter's 
		constructor was called. 
		"""
		return self.pe
		
	def _compile(self):
		"""
		Prepares internal structures for faster processing. 
		This function should never be called by the user. 
		"""
		# Prepare a list of traces for every (plot, axes) pair
		self.tracesOnAxes={}
		for (graphName, graph) in self.setup['graphs'].iteritems():
			for (axesName, axes) in graph['axes'].iteritems():
				key=(graphName, axesName)
				self.tracesOnAxes[key]=[]
		for (traceName, trace) in self.setup['traces'].iteritems(): 
			graph=trace['graph']
			axes=trace['axes']
			key=(graph, axes)
			if key in self.tracesOnAxes:
				self.tracesOnAxes[key].append(traceName)
			
		# Prepare regexps
		self.compiledStyles=[]
		regexp=[]
		for style in self.setup['styles']:
			# Compile regexps in a pattern
			compiled=[]
			for regexp in style['pattern']:
				compiled.append(re.compile(regexp))
			
			# Replace pattern with compiled pattern
			self.compiledStyles.append({'pattern': compiled, 'style':style['style']})
		
		# Build a list of result indices corresponding to desired corners for every trace
		# Build a list of corner names
		
		# Get the complete list of corners for a particular trace. 
		# Use x-axis vector for extracting the corners list. 
		self.compiledCornerNames={}
		for (traceName, trace) in self.setup['traces'].iteritems():
			xresult=trace['xresult']
			
			# Get complete set of corner
			if xresult in self.pe.measures:
				allCornerNames=self.pe.measures[xresult]['corners']
			else:
				allCornerNames=[]
			
			# Get initial set of corner names
			if 'corners' in trace and len(trace['corners'])>0:
				desiredCornerNames=set(trace['corners'])
			else:
				desiredCornerNames=set(allCornerNames)
			
			# Build a list of corner names and indices
			cornerNames=[]
			for cornerName in allCornerNames: 
				if cornerName in desiredCornerNames:
					cornerNames.append(cornerName)
			
			self.compiledCornerNames[traceName]=cornerNames

	def _traceStyle(self, traceName, trace, cornerName): 
		"""
		Matches the plot window, axes, corner, and trace names to the patterns 
		is the ``styles`` dictionary and constructs the dictionary of style 
		directives for the trace. 
		"""
		graphName=trace['graph']
		axesName=trace['axes']
		
		finalStyle={}
		# Go through all styles
		for style in self.compiledStyles:
			# Compiled patterns
			pattern=style['pattern']
			# Match to trace
			if (pattern[0].match(graphName) and
				pattern[1].match(axesName) and
				pattern[2].match(cornerName) and
				pattern[3].match(traceName)):
				finalStyle.update(style['style'])
		
		# Finally update with trace style
		if 'style' in trace:
			finalStyle.update(trace['style'])
		
		return finalStyle
		
	# For pickling
	def __getstate__(self):
		state=self.__dict__.copy()
		del state['compiledStyles']
		del state['figure']
		del state['titleArtist']
		del state['plotAxes']
		
		return state
	
	# For unpickling
	def __setstate__(self, state):
		self.__dict__.update(state)
		
		self._compile()
		self.figure={}
		self.titleArtist={}
		self.plotAxes={}
	
	def __call__(self, prefixText='', postfixText='', createPlots=True, initalizePlotting=True):
		try:
			# Initialize plotting system
			if initalizePlotting:
				pyopl.init()

			# Lock GUI
			pyopl.lock()
				
			# Are we creating plots?
			if createPlots: 
				# Check if figures were created and are alive, add missing figures to a list
				graphsToCreate=[]
				for (graphName, graph) in self.setup['graphs'].iteritems():
					if graphName not in self.figure:
						if self.debug:
							DbgMsgOut("WxMplPL", "Added missing graph (not in figure list) '"+graphName+"'.")
						graphsToCreate.append(graphName)
					elif not pyopl.alive(self.figure[graphName]): 
						if self.debug:
							DbgMsgOut("WxMplPL", "Added missing graph (not in on screen) '"+graphName+"'.")
						graphsToCreate.append(graphName)
				
				# Unlock GUI
				pyopl.lock(False)
				
				# OK, now create figures and store the tags
				for graphName in graphsToCreate:
					if self.debug:
						DbgMsgOut("WxMplPL", "  Creating figure for '"+graphName+"'")
					graph=self.setup['graphs'][graphName]
					
					fig=pyopl.figure(**(graph['shape']))
					self.figure[graphName]=fig
					pyopl.title(fig, graphName+" : "+graph['title'])
				
				# Lock GUI
				pyopl.lock(True)
				
				# Add axes to created graphs
				for graphName in graphsToCreate:
					if self.debug:
						DbgMsgOut("WxMplPL", "  Creating axes for '"+graphName+"'")
					
					# Get graph data
					graph=self.setup['graphs'][graphName]
					
					# Get figure
					fig=self.figure[graphName]
					
					# Check if it is alive
					if not pyopl.alive(fig):
						if self.debug:
							DbgMsgOut("WxMplPL", "    Figure not alive, skipped.")
						continue
					
					# Create axes
					axesDict={}
					for (axName, ax) in graph['axes'].iteritems():
						if self.debug:
							DbgMsgOut("WxMplPL", "    '"+axName+"'")
						opt=ax.get('options', {})
						# Handle polar axes
						if ax.get('gridtype', None)=='polar':
							opt.update(projection='polar')
						
						# Create axes
						if 'rectangle' in ax:
							axesDict[axName]=fig.add_axes(ax['rectangle'], **opt)
						elif 'subplot' in ax:
							axesDict[axName]=fig.add_subplot(*(ax['subplot']), **opt)
						else:
							axesDict[axName]=fig.add_axes((0.12, 0.12, 0.76, 0.76), **opt)
						
					# Put axes dict in self.plotAxes
					self.plotAxes[graphName]=axesDict

			# Go through all graphs
			for (graphName, graph) in self.setup['graphs'].iteritems():
				if self.debug:
					DbgMsgOut("WxMplPL", "Refreshing graph '"+graphName+"'")
				
				# Get figure
				fig=self.figure[graphName]
					
				# Check if it is alive
				if not pyopl.alive(fig):
					if self.debug:
						DbgMsgOut("WxMplPL", "  Figure not alive, skipped.")
					continue
				
				# Go through axes and add data.
				for (axName, axobj) in self.plotAxes[graphName].iteritems(): 
					if self.debug:
						DbgMsgOut("WxMplPL", "  Refreshing axes '"+axName+"'")
					
					# Get axes data 
					ax=graph['axes'][axName]
					
					# Clear axes
					axobj.clear()
					
					# Go through all traces on these axes
					for traceName in self.tracesOnAxes[(graphName, axName)]: 
						if self.debug:
							DbgMsgOut("WxMplPL", "    Refreshing trace '"+traceName+"'")
							
						trace=self.setup['traces'][traceName]
						xresult=trace['xresult']
						yresult=trace['yresult']
						
						# Go through all corners
						for cornerName in self.compiledCornerNames[traceName]: 
							if self.debug:
								DbgMsgOut("WxMplPL", "      in corner '"+cornerName+"'")
								
							# Get xresult and yresult
							if xresult in self.pe.results and cornerName in self.pe.results[xresult]: 
								x=self.pe.results[xresult][cornerName]
							else:
								x=None
								
							if yresult in self.pe.results and cornerName in self.pe.results[yresult]: 
								y=self.pe.results[yresult][cornerName]
							else:
								y=None
							
							# Calculate style
							style=self._traceStyle(traceName, trace, cornerName)
							
							# Set name
							style['label']=cornerName+'.'+traceName
							
							# Plot (TODO: handle polar plots correctly, need r, phi from x, y)
							if x is not None and y is not None: 
								axobj.plot(x, y, **style)
					
					if self.debug:
						DbgMsgOut("WxMplPL", "  Finalizing axes settings for '"+axName+"'")
						
					# Handle log scale
					if ax.get('gridtype', None)=='polar':
						pass
					else:
						# Rectilinear grid, handle log scale
						
						# x-axis
						xscale=ax.get('xscale', None) 
						kwargs={}
						if xscale is None:
							type='linear'
						else:
							if xscale.get('type', None)=='log':
								type='log'
								
								if 'linthresh' in xscale:
									type='symlog'
									kwargs['linthreshx']=xscale['linthresh']
								
								if 'base' in xscale:
									kwargs['basex']=xscale['base']
								else:
									kwargs['basex']=10
								
								if 'subticks' in xscale:
									kwargs['subsx']=xscale['subticks']
								elif kwargs['basex']==10:
									kwargs['subsx']=self.log10minors
								elif kwargs['basex']==2:
									kwargs['subsx']=self.log2minors
							else:
								type='linear'
						axobj.set_xscale(type, **kwargs)
						
						# y-axis
						yscale=ax.get('yscale', None) 
						kwargs={}
						if yscale is None:
							type='linear'
						else:
							if yscale.get('type', None)=='log':
								if 'linthresh' in yscale:
									type='symlog'
									kwargs['linthreshy']=yscale['linthresh']
								
								if 'base' in yscale:
									kwargs['basey']=yscale['base']
								else:
									kwargs['basey']=10
								
								if 'subticks' in yscale:
									kwargs['subsy']=yscale['subticks']
								elif kwargs['basey']==10:
									kwargs['subsy']=self.log10minors
								elif kwargs['basey']==2:
									kwargs['subsy']=self.log2minors
							else:
								type='linear'
						axobj.set_yscale(type, **kwargs)
						
					# Labels, title, legend, grid
					ax=graph['axes'][axName]
					if 'xlabel' in ax:
						axobj.set_xlabel(ax['xlabel'])
					if 'ylabel' in ax:
						axobj.set_ylabel(ax['ylabel'])
					if 'title' in ax:
						axobj.set_title(ax['title'])
					if 'legend' in ax and ax['legend']:
						axobj.legend()
					if 'grid' in ax:
						axobj.grid(ax['grid'])
						
					# Set axis limits
					if 'xlimits' in ax:
						axobj.set_xlim(ax['xlimits'])
					if 'ylimits' in ax:
						axobj.set_ylim(ax['ylimits'])
						
				# TODO: xlimits and ylimits on polar axes
				
				if self.debug:
					DbgMsgOut("WxMplPL", "Finalizing graph '"+graphName+"'.")
					
				# Set plot and window title
				if len(prefixText)>0:
					prefix=prefixText+' : '
				else:
					prefix=''
				if len(postfixText)>0:
					postfix=' : '+postfixText
				else:
					postfix=''
				if 'title' in graph:
					gt=graph['title']
				else:
					gt=''
				
				if not graphName in self.titleArtist:
					self.titleArtist[graphName]=fig.suptitle(prefix+gt+postfix)
				else:
					self.titleArtist[graphName].set_text(prefix+gt+postfix)
				
				# Draw the figure
				pyopl.draw(fig)
			
			# Unlock GUI
			pyopl.lock(False)
		
		except (KeyboardInterrupt, SystemExit):
			pyopl.lock(False)
			raise
		
		
		
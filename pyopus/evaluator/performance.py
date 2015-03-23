"""

.. inheritance-diagram:: pyopus.evaluator.performance
    :parts: 1
	
**System performance evaluation module (PyOPUS subsystem name: PE)**

A **system description module** is a fragment of simulated system description. 
Usually it corresponds to a file or a section of a library file. 

**Performance measure ordering** is a list of performance measure names that 
defines the order of performance measures. 

The **heads** data structure provides the list of simulators with available 
system description modules. The **analyses** data structure specifies the 
analyses that will be performed by the listed simulators. The **corners** data 
structure specifies the corners across which the systems will be evaluated. 
The **measures** data structure describes the performance measures which are 
extracted from simulation results. 


The **heads** data structure is a dictionary with head name for key. The values 
are also dictionaries describing a simulator and the description of the system 
to be simulated with the following keys:

* ``simulator`` - the name of the simulator to use 
  (see :data:`pyopus.simulator.simulators` member for the list of available 
  simulators)
* ``settings`` - a dictionary specifying the keyword arguments passed to the 
  simulator object's constructor 
* ``moddefs`` - definition of system description modules 
* ``options`` - simulator options valid for all analyses performed in this 
  simulator. This is a dictionary with option name for key. 
* ``params`` - system parameters valid for all analyses performed in this 
  simulator. This is a dictionary with parameter name for key. 

The definition of system description modules in the ``moddefs`` dictionary 
member are themselves dictionaries with system description module name for key. 
Values are dictionaries using the following keys for describing a system 
description module 

* ``file`` - file name in which the system description module is described
* ``sections`` - file section name where the system description module 
  description can be bound

Specifying only the ``file`` member translates into an ``.include`` simulator 
input directive (or its equivalent). If additionally the ``section`` member is 
also specified the result is a ``.lib`` directive (or its equivalent). 


The **analyses** data structure is a dictionary with analysis name for key. 
The values are also dictionaries describing an analysis using the following 
dictionary keys:

* ``head`` - the name of the head (simulator+) that will be used for this 
  analysis
* ``modules`` - the list of system description module names that apply only to 
  this analysis 
* ``options`` - simulator options that apply only to this analyses. This is a 
  dictionary with option name for key. 
* ``params`` - system parameters that apply only to this analysis. This is a 
  dictionary with parameter name for key. 
* ``saves`` - a list of strings which evaluate to save directives specifying 
  what simulated quantities should be included in simulator's output. See 
  individual simulator classes in the :mod:`pyopus.simulator` module for the 
  available save directive generator functions. 
* ``command`` - a string which evaluates to the analysis directive for the 
  simulator. See individual simulator classes in the :mod:`pyopus.simulator` 
  module for the available analysis directive generator functions. 
  
The environment in which the strings in the ``saves`` member and the string in 
the ``command`` member are evaluated is simulator-dependent. See individual 
simulator classes in the :mod:`pyopus.simulator` module for details. 

The environment in which the ``command`` string is evaluated has a member 
named ``param``. It is a dictionary containing all system parameters used in 
the analysis. 


The **corners** data structure is a dictionary with corner name for key. The 
values are also dictionaries describing individual corners using the following 
dictionary keys: 

* ``modules`` - the list of system description module names that apply only to 
  this corner
* ``params`` - a dictionary with the system parameters that apply only to 
  this corner


The **measures** data structure is a dictionary with performance measure name 
for key. The values are also dictionaries describing individual performance 
measures using the following dictionary keys

* ``analysis`` - the name of the analysis that produces the results from which 
  the performance measure's value is extracted
* ``corners`` - the list of corner names across which the performance measure 
  is evaluated. Corner indices obtained from the 
  :meth:`~pyopus.evaluator.cost.MNbase.worstCornerIndex` method of 
  normalization objects (defined in the :mod:`pyopus.evaluator.cost` module) 
  can be converted to corner names by looking up the corresponding members of 
  this list. 
* ``expression`` - a string specifying a Python expression that evaluates to 
  the performance measure's value
* ``script`` - a string specyfying a Python script that stores the performance 
  measure's value in a variable named ``__result``
* ``vector`` - a boolean flag which specifies that a performance measure's 
  value may be a vector. If it is ``False`` and the obtained performance 
  measure value is not a scalar (or scalar-like) the evaluation is considered 
  as failed. Defaults to ``False``. 

If both ``expression`` and ``script`` are given the ``script`` is ignored. 

If the ``analysis`` member is ``None`` the performance measure is a dependent 
performance measure and is evaluated after all other (independent) performance 
measure have been evaluated. Dependent performance measures can access the 
values of independent performance measures through the ``result`` data 
structure. 


``expression`` and ``script`` are evaluated in an environment with the 
following members: 

* ``m`` - a reference to the :mod:`pyopus.evaluator.measure` module providing 
  a set of functions for extracting common performance measures from simulated 
  response
* ``np`` - a reference to the NumPy module
* ``param`` - a dictionary with the values of system parameters that apply to 
  the particular analysis and corner used for obtaining the simulated response 
  form which the performance measure is being extracted
* ``result`` - a dictionary of dictionaries available to dependent performance 
  measures only. The first key is the performance measure name and the second 
  key is the corner name. The values represent performance measure values. 
  If a value is ``None`` the evaluation of the independent performance measure 
  failed in that corner. 
* ``thisCorner`` - a string that reflects the name of the corner in which the 
  dependent performance measure is currently under evaluation. Not available 
  for independent performance measures. 
  
Beside these members every simulator object provides additional members for 
accessing simulation results. See individual simulator classes in the 
:mod:`pyopus.simulator` module and the :meth:`getGenerators` method of 
simulator obejcts for details. 
"""

from ..optimizer.base import Plugin
from numpy import array, ndarray, iscomplex, dtype
from sys import exc_info
from traceback import format_exception, format_exception_only
from ..simulator import simulatorClass
from ..misc.debug import DbgMsgOut, DbgMsg

# Measurements and NumPy
import measure as m
import numpy as np

import sys

__all__ = [ 'PerformanceEvaluator', 'PerformanceAnnotator', 'PerformanceCollector' ] 

import pdb

class PerformanceEvaluator:
	"""
	Performance evaluator class. Obejcts of this class are callable. The 
	calling convention is ``object(paramDictionary)`` where *paramDictionary* 
	is a dictionary of input parameter values. The return value is a dictionary 
	of dictionaries where the first key represents the performance measure name 
	and the second key represents the corner name. The dictionary holds the 
	values of performance measure values across corners. If some value is 
	``None`` the performance measure evaluation failed for that corner. The 
	return value is also stored in the *results* member of the 
	:class:`PermormanceEvaluator` object. 
	
	*heads*, *analyses*, *corners*, and *measures* specify the heads, the 
	analyses, the corners, and the performance measures. 
	
	If *debug* is set to a nonzero value debug messages are generated at the 
	standard output. Two debug levels are available (1 and 2). A higher *debug* 
	value results in greater verbosity of the debug messages. 
	
	Objects of this class construct a list of simulator objects based on the 
	*heads* data structure. Every simulator object performs the analyses which 
	list the corresponding head under ``head`` in the analysis description. 
	
	Every analysis is performed across the set of corners obtained as the union 
	of ``corners`` found in the descriptions of performance measures that list 
	the corresponding analysis as their ``analysis``. 
	
	The system description for an analysis in a corner is constructed from 
	system description modules specified in the corresponding entries in 
	*corners*, and *analyses* data structures. The definitions of the system 
	description modules are taken from the *heads* data structure entry 
	corresponding to the ``head`` specified in the description of the analysis 
	(*analysis* data structure). 
	
	System parameters for an analysis in a particular corner are obtained as 
	the union of

	* the input parameters dictionary specified when an object of the 
	  :class:`PerformanceEvaluator` class is called 
	* the ``params`` dictionary of the *heads* data structure entry 
	  corresponding to the analysis
	* the ``params`` dictionary of the *corners* data structure entry 
	  corresponding to the corner
	* the ``params`` dictionary of the *analyses* data structure entry 
	  corresponding to the analysis
	
	If a parameter appears across multiple dictionaries the entries in the 
	input parameter dictionary have the lowest priority and the entries in the 
	``params`` dictionary of the *analyses* have the highest priority. 
	
	A similar priority is applied to simulator options specified in the 
	``options`` dictionaries (the values from *heads* have the lowest priority 
	and the values from *analyses* have the highest priority). 
	
	Independent performance measures (the ones with ``analysis`` not equal to 
	``None``) are evaluated before dependent performance measures (the ones 
	with ``analysis`` set to ``None``). 
	
	The evaluation results are stored in a dictionary of dictionaries with 
	performance measure name as the first key and corner name as the second 
	key. ``None`` indicates that the performance measure evaluation failed in 
	the corresponding corner. 
	"""
	# Constructor
	def __init__(self, heads, analyses, corners, measures, debug=0):
		# Debug mode flag
		self.debug=debug
		
		# Store problem
		self.heads=heads
		self.analyses=analyses;
		self.corners=corners;
		self.measures=measures
		
		# Results of the performance evaluation
		self.results=None
		
		# Compile
		if self.debug:
			DbgMsgOut("PE", "Compiling.")
			
		self._compile()
		
		if self.debug:
			DbgMsgOut("PE", "  Simulator objects (%d): " % len(self.jobListForHead))
			for (headName, jobList) in self.jobListForHead.iteritems(): 
				DbgMsgOut("PE", "    %s: %d analyses" % (headName, len(jobList)))
				if self.debug>1:
					for job in jobList:
						DbgMsgOut("PE", "      %s" % job['name'])
	
	def _compile(self):
		"""
		Prepares internal structures for faster processing. 
		This function should never be called by the user. 
		"""
		for (name, head) in self.heads.iteritems():
			if 'simulator' not in head or len(head['simulator'])<1:
				raise Exception, DbgMsg("PE", "No simulator specified for head '%s'." % name)
			if 'settings' not in head:
				head['settings']={}
			if 'moddefs' not in head or len(head['moddefs'])<1:
				raise Exception, DbgMsg("PE", "No definitions specified for head '%s'." % name)
			if 'options' not in head:
				head['options']={}
			if 'params' not in head:
				head['params']={}
		
		# Build a dictionary with head name as key containing lists of analyses that use a particular head
		head2an={}
		for (name, an) in self.analyses.iteritems():
			headName=an['head']
			# Sanity check
			if headName not in self.heads:
				raise Exception, DbgMsg("PE", "Head '%s' used by analysis '%s' is not defined." % (headName, name))
			
			if headName not in head2an:
				head2an[headName]=[name]
			else:
				head2an[headName].append(name)

		# Store head2an
		self.head2an=head2an
		
		# Build a dictionary with analysis name as key containig lists of corners corresponding to individual analyses. 
		# Build a dictionary with (corner,analysis) as key containing corresponding measure name lists. 
		an2corners={}
		key2measures={}
		for (measureName, measure) in self.measures.iteritems():
			anName=measure['analysis']
			# Sanity check
			if anName is not None and anName not in self.analyses:
				raise Exception, DbgMsg("PE", "Analysis '%s' used by measure '%s' is not defined." % (anName, measureName))
			
			# Get corner name list.  
			cornerNames=measure['corners']
			# Make corner names unique
			cornerNames=list(set(cornerNames))
			# Add corners to an2corners. 
			if anName not in an2corners:
				an2corners[anName]=cornerNames
			else:
				an2corners[anName]+=cornerNames
			# Remove duplicates
			an2corners[anName]=list(set(an2corners[anName]))
				
			# Add measure to key2measures for all corners
			for cornerName in cornerNames:
				key=(cornerName, anName)
				if key not in key2measures:
					key2measures[key]=[measureName]
				else:
					key2measures[key].append(measureName)
		
		# Store an2corners and key2measures. 
		self.an2corners=an2corners
		self.key2measures=key2measures
		
		# Build joblists for all heads, remember key=(corner,analysis) for every job. 
		jobListForHead={}
		keyListForHead={}
		# Go through all heads
		for (headName, anList) in head2an.iteritems(): 
			head=self.heads[headName]
			# For every head go through all analyses
			jobList=[]
			keyList=[]
			for anName in anList:
				analysis=self.analyses[anName]
				# For every analysis go through all corners
				if anName in an2corners:
					for cornerName in an2corners[anName]:
						corner=self.corners[cornerName]
						# Create job for analysis in corner
						job={}
						key=(cornerName, anName) 
						
						job['name']="C"+cornerName+"A"+anName
						job['command']=analysis['command']
						if 'saves' in analysis:
							job['saves']=analysis['saves']
						else:
							job['saves']=[]
											
						# Create a list of modules by joining analysis and corner lists. 
						modules=[]
						if 'modules' in corner:
							modules.extend(corner['modules'])
						if 'modules' in analysis:
							modules.extend(analysis['modules'])
						# Search for duplicates. 
						if len(modules)!=len(set(modules)):
							raise Exception, DbgMsg("PE", "Duplicate modules in corner '%s', analysis '%s'." % (cornerName, anName))

						# Translate to actual module definitions using information in the head. 
						job['definitions']=[]
						for module in modules: 
							# Sanity check
							if module not in head['moddefs']:
								raise Exception, DbgMsg("PE", "Module '%s' used by '%s/%s' is not defined." % (module,anName, cornerName))
								
							job['definitions'].append(head['moddefs'][module])
						
						# Merge params from head, corner, and analysis. 
						params={}
						if 'params' in head:
							params.update(head['params'])
						if 'params' in corner:
							params.update(corner['params'])
						if 'params' in analysis:
							params.update(analysis['params'])
						job['params']=params
						
						# Merge options from head, corner, and analysis. 
						options={}
						if 'options' in head:
							options.update(head['options'])
						if 'options' in corner:
							options.update(corner['options'])
						if 'options' in analysis:
							options.update(analysis['options'])
						job['options']=options
						# Append to job list for this head. 
						jobList.append(job)
						keyList.append(key)
			
			# Store in jobListforHead. 
			jobListForHead[headName]=jobList
			keyListForHead[headName]=keyList
		
		# Store jobListForHead and keyListForHead.  
		self.jobListForHead=jobListForHead
		self.keyListForHead=keyListForHead
		
		# Build simulator objects, one for every head. 
		# Build Local variable dictionaries for measurement evaluation. 
		self.simulatorForHead={}
		self.measureLocals={}
		for (headName, head) in self.heads.iteritems():
			# Get simulator class
			try:
				SimulatorClass=simulatorClass(head['simulator'])
			except:
				raise Exception, DbgMsg("PE", "Simulator '"+head['simulator']+"' not found.")
			
			# Create simulator
			simulator=SimulatorClass(**(head['settings']))
			simulator.setJobList(jobListForHead[headName]) 
			
			# Local variable dictionaries for measurement evaluation, one per simulator. 
			self.measureLocals[headName]={}
			self.measureLocals[headName].update(simulator.getGenerators())
			self.measureLocals[headName]['param']={}
			
			self.simulatorForHead[headName]=simulator
			
		# Local variable dictionary for dependent measurement evaluation. 
		self.dependentMeasureLocals={
			'param': {}, 
			'result': None, 
			'thisCorner': None
		}
	
	# For pickling
	def __getstate__(self):
		state=self.__dict__.copy()
		del state['measureLocals']
		del state['dependentMeasureLocals']
		del state['head2an']
		del state['an2corners']
		del state['key2measures']
		del state['jobListForHead']
		del state['keyListForHead']
		del state['simulatorForHead']
		
		return state
	
	# For unpickling
	def __setstate__(self, state):
		self.__dict__.update(state)
		
		self._compile()
		
	# Return simulators dictionary. 
	def simulators(self):
		"""
		Returns the dictionary with head name for key holding the corresponding 
		simulator objects. 
		"""
		return simulatorForHead
	
	# Cleanup simulator intermediate files and stop interactive simulators. 
	def finalize(self):
		"""
		Removes all intermediate simulator files and stops all interactive 
		simulators. 
		"""
		for (headName, simulator) in self.simulatorForHead.iteritems():
			simulator.cleanup()
			simulator.stopSimulator()
	
	def _postprocessMeasure(self, measureValue, isVector):
		"""
		Postprocesses *measureValue* obtained by evaluating the ``script`` or
		the ``expression`` string describing the measurement. 
		
		1. Converts the result to an array. 
		2. Signals an error if the array type is complex. 
		3. Converts the array of values to a double floating point array. 
		4. If the array is empty (size==0) signals an error. 
		5. Signals an error if *isVector* is ``False`` and the array has size>1. 
		6. Scalarizes array (makes it a 0D array) if *isVector* is ``False``. 
		"""
		# None indicates a failure, nothing further to do. 
		if measureValue is not None:
			# Pack it in an array
			if type(measureValue) is not ndarray: 
				# This will convert lists and tuples to arrays
				try:
					measureValue=array(measureValue)
				except KeyboardInterrupt:
					print "PE : keyboard interrupt"
					raise
				except:
					if self.debug:
						print "PE:         Result can't be converted to an array."
					measureValue=None
			
			# Was conversion successfull? 
			if measureValue is not None: 
				# It is an array
				# Check if it is complex
				if iscomplex(measureValue).any():
					# Bad. Complex value not allowed.
					if self.debug:
						print "PE:         Measurement produced a complex array."
					measureValue=None
				elif measureValue.dtype is not dtype('float64'):
					# Not complex. Convert it to float64. 
					# On conversion failure we get an exception and a failed measurement. 
					try:
						measureValue=measureValue.astype(dtype('float64'))
					except KeyboardInterrupt:
						print "PE : keyboard interrupt"
						raise
					except:
						if self.debug:
							print "PE:         Failed to convert result into a real array."
						measureValue=None
				
				# Check if it is empty
				if measureValue.size==0:
					# It is empty, this is bad
					if self.debug:
						print "PE:         Measurement produced an empty array."
					measureValue=None
				
				# Scalarize if measurement is scalar
				if (measureValue is not None) and (not isVector):
					# We are expecting a scalar.
					if measureValue.size==1:
						# Scalarize it
						measureValue=measureValue.ravel()[0]
					else:
						# But we have a vector, bad
						if self.debug:
							print "PE:         Scalar measurement produced a vector."
						measureValue=None
		
		return measureValue
		
	def __call__(self, inputParams={}):
		if self.debug: 
			DbgMsgOut("PE", "Evaluation started.") 
			
		# Go through all simulators
		for (headName, simulator) in self.simulatorForHead.iteritems():
			if self.debug: 
				DbgMsgOut("PE", "  Simulator/head %s" % headName)
			
			# Clear results.
			self.results={}
			
			# Get head.
			head=self.heads[headName]
			
			# Get job list. 
			jobList=self.jobListForHead[headName]
			
			# Get key list. 
			keyList=self.keyListForHead[headName]
			
			# Set input parameters. 
			simulator.setInputParameters(inputParams)
			
			# Count job groups. 
			ngroups=simulator.jobGroupCount()
			
			# Go through all job groups, write file, run it and collect results
			for i in range(ngroups):
				# Delete old loaded results (free memory). 
				simulator.resetResults()
				
				# Run jobs in job group and collect results. 
				(jobIndices, status)=simulator.runJobGroup(i)
				
				# Go through all job indices in i-th job group. 
				for j in jobIndices: 
					# Get (corner, analysis) key for j-th job. 
					key=keyList[j]
					(cornerName, anName)=key
					job=jobList[j]
					
					# Prepare dictionary of local variables for measurement evaluation
					locals=self.measureLocals[headName]
					locals['param'].clear()
					
					# Case: input parameters get overriden by job parameters - default
					# Update with input params
					locals['param'].update(inputParams)
					# Update with job params
					locals['param'].update(job['params'])
					
					# Case: job parameters get overriden by input parameters - unimplemented
					
					# Load result (one job at a time to save memory). 
					simulator.collectResults([j], status)
					
					# Do we have a result? 
					if simulator.activateResult(j) is None: 
						# No. 
						# Assume all measurements that depend on this analysis have failed
						if self.debug: 
							DbgMsgOut("PE", "    Corner: %s, analysis %s ... FAILED" % (cornerName, anName))
							
						# Set corresponding measurements to None
						for measureName in self.key2measures[key]:
							# Store result
							if measureName not in self.results:
								self.results[measureName]={}
							self.results[measureName][cornerName]=None
							
							if self.debug: 
								DbgMsgOut("PE", "      %s : FAILED" % measureName)
						
						# Continue with next analysis
						continue
					else:
						# Yes, we have a result. 
						if self.debug: 
							DbgMsgOut("PE", "    Corner: %s, analysis: %s ... OK" % (cornerName, anName))
						
						# Go through all measurements for this key. 
						for measureName in self.key2measures[key]:
							# Get measure. 
							measure=self.measures[measureName]
							
							# Are we expecting a vector?
							if 'vector' in measure:
								isVector=bool(measure['vector'])
							else:
								isVector=False
					
							# TODO: maybe compile it and run it in a function so it has its own namespace
							# Evaluate measure, catch exception that occurs during evaluation. 
							try: 
								if 'expression' in measure:
									measureValue=eval(measure['expression'], globals(), locals)	
								elif 'script' in measure:
									tmpLocals={}
									tmpLocals.update(locals)
									exec measure['script'] in globals(), locals
									measureValue=locals['__result']
								else:
									raise Exception, DbgMsg("PE", "No expression or script.")
								if self.debug: 
									DbgMsgOut("PE", "      %s OK" % measureName)
							except KeyboardInterrupt:
								DbgMsgOut("PE", "Keyboard interrupt.")
								raise
							except:
								measureValue=None
								if self.debug:
									DbgMsgOut("PE", "      %s FAILED" % measureName)
									ei=exc_info()
									if self.debug>1:
										for line in format_exception(ei[0], ei[1], ei[2]):
											DbgMsgOut("PE", "        "+line) 
									else:
										for line in format_exception_only(ei[0], ei[1]):
											DbgMsgOut("PE", "        "+line)
							
							# Store result
							if measureName not in self.results:
								self.results[measureName]={}
							self.results[measureName][cornerName]=self._postprocessMeasure(measureValue, isVector)

		# Handle measures that depend on other measures (analysis is None). 
		# Prepare a copy of results structure containing results for all measures
		# that are associated with an analysis. 
		self.dependentMeasureLocals['result']=self.results.copy()
		# Are there any? 
		if None in self.an2corners:
			# Go through all corners of anlysis None
			for cornerName in self.an2corners[None]:
				# Get corner. 
				corner=self.corners[cornerName]
				
				if self.debug: 
					DbgMsgOut("PE", "    Corner: %s, analysis: None" % cornerName)
							
				# Set thisCorner variable in local variable dictionary. 
				self.dependentMeasureLocals['thisCorner']=cornerName
				
				# Go through all measures for key (cornerName, None). 
				for measureName in self.key2measures[(cornerName, None)]:
					# Get measure. 
					measure=self.measures[measureName]
					
					# Are we expecting a vector?
					if 'vector' in measure:
						isVector=bool(measure['vector'])
					else:
						isVector=False
						
					# Prepare dictionary of local variables for measurement evaluation
					locals=self.dependentMeasureLocals 
					locals['param'].clear()
					
					# Case: input parameters get overriden by job parameters - default
					# Update with input params
					locals['param'].update(inputParams)
					# Update with corner params (job params not available since this 
					# measure is not associated with any analysis). 
					locals['param'].update(corner['params'])
					
					# Case: job parameters get overriden by input parameters - unimplemented

					# TODO: maybe compile it and run it in a function so it has its own namespace
					# Evaluate measure, catch exception that occurs during evaluation. 
					try: 
						if 'expression' in measure:
							measureValue=eval(measure['expression'], globals(), locals)	
						elif 'script' in measure:
							tmpLocals={}
							tmpLocals.update(locals)
							exec measure['script'] in globals(), locals
							measureValue=locals['__result']
						else:
							raise Exception, DbgMsg("PE", "No expression or script.")
						if self.debug: 
							DbgMsgOut("PE", "      %s OK" % measureName)
					except KeyboardInterrupt:
						DbgMsgOut("PE", "Keyboard interrupt.")
						raise
					except:
						measureValue=None
						if self.debug:
							DbgMsgOut("PE", "      %s FAILED" % measureName)
							ei=exc_info()
							if self.debug>1:
								for line in format_exception(ei[0], ei[1], ei[2]):
									DbgMsgOut("PE", "        "+line) 
							else:
								for line in format_exception_only(ei[0], ei[1]):
									DbgMsgOut("PE", "        "+line)
					
					# Store result
					if measureName not in self.results:
						self.results[measureName]={}
					self.results[measureName][cornerName]=self._postprocessMeasure(measureValue, isVector)
				
		return self.results
		
	def formatResults(self, outputOrder=None, nMeasureName=10, nCornerName=6, nResult=12, nPrec=3):
		"""
		Formats a string representing the results obtained with the last call 
		to this object. Generates one line for every performance measure 
		evaluation in a corner. 
		
		*outputOrder* (if given) specifies the order in which the performance 
		measures are listed. 
		
		*nMeasureName* specifies the formatting width for the performance 
		measure name. 
		
		*nCornerName* specifies the formatting width for the corner name. 
		
		*nResult* and *nPrec* specify the formatting width and the number of 
		significant digits for the performance measure value. 
		"""
		# List of measurement names
		if outputOrder is None:
			# Default is sorted by name
			nameList=[]
			for (measureName, measure) in self.measures.iteritems():
				nameList.append(measureName)
			nameList.sort()
		else:
			nameList=outputOrder

		# Format output
		outStr=""
		for measureName in nameList: 
			measure=self.measures[measureName]
			# Format measurement values
			first=True
			for cornerName in measure['corners']: 
				# First header
				if first:
					header="%*s | " % (nMeasureName, measureName)

				# Result in one corner
				if self.results[measureName][cornerName] is None:
					textVal='%*s: %-*s' % (nCornerName, cornerName, nResult, 'FAILED')
				else:
					if self.results[measureName][cornerName].size==1:
						textVal="%*s: %*.*e" % (nCornerName, cornerName, nResult, nPrec, self.results[measureName][cornerName])
					else:
						textVal="%*s: " % (nCornerName, cornerName) + str(self.results[measureName][cornerName])
				
				# Append
				outStr+=header+textVal+"\n"

				# Remaining headers
				if first:
					first=False
					header="%*s | " % (nMeasureName, '')
				
		return outStr
		
	# Return annotator plugin. 
	def getAnnotator(self):
		"""
		Returns an object of the :class:`PerformanceAnnotator` class which can 
		be used as a plugin for iterative algorithms. The plugin takes care of 
		cost function details (:attr:`results` member) propagation from the 
		machine where the evaluation of the cost function takes place to the 
		machine where the evaluation was requested (usually the master). 
		"""
		return PerformanceAnnotator(self)
	
	# Return collector plugin. 
	def getCollector(self):
		"""
		Returns an object of the :class:`PerformanceCollector` class which can 
		be used as a plugin for iterative algorithms. The plugin gathers 
		performance information from the :attr:`results` member of the 
		:class:`PerformanceEvaluator` object across iterations of the algorithm. 
		"""
		return PerformanceCollector(self)
		
	
# Default annotator for performance evaluator
class PerformanceAnnotator(Plugin):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Plugin` iterative 
	algorithm plugin class. This is a callable object whose job is to
	
	* produce an annotation (details of the evaluated performance) stored in the 
	  *performanceEvaluator* object (when invoked with ``None`` for 
	  *annotation*)
	* update the *performanceEvaluator* object with the given annotation 
	  (when invoked with an *annotation* that is not ``None``)
	
	Annotation is a copy of the :attr:`results` member of 
	*performanceEvaluator*. 
	
	Annotators are used for propagating the details of the cost function from 
	the machine where the evaluation takes place to the machine where the 
	evaluation was requested (usually the master). 
	"""
	def __init__(self, performanceEvaluator):
		self.pe=performanceEvaluator
	
	def __call__(self, x, f, opt, annotation=None):
		if annotation is None:
			return pe.results.copy()
		else:
			pe.results=annotation
					
# Performance record collector
class PerformanceCollector(Plugin):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Plugin` iterative 
	algorithm plugin class. This is a callable object invoked at every 
	iteration of the algorithm. It collects the summary of the evaluated 
	performance measures from the :attr:`results` member of the 
	*performanceEvaluator* object (member of the :class:`PerformanceEvaluator` 
	class). 
	
	Let niter denote the number of stored iterations. The *results* structures 
	are stored in a list where the index of an entry represents the iteration 
	number. The list can be obtained from the :attr:`performance` member of the 
	:class:`PerformanceCollector` object. 
	
	Some iterative algorithms do not evaluate iterations sequentially. Such 
	algorithms denote the iteration number with the :attr:`index` member. If 
	the :attr:`index` is not present in the iterative algorithm object the 
	internal iteration counter of the :class:`PerformanceCollector` is used. 
	
	If iterations are not performed sequentially the *performance* list may 
	contain gaps where no valid *results* structure is found. Such gaps are 
	denoted by ``None``.
	"""
	
	def __init__(self, performanceEvaluator):
		OptTrigger.__init__(self)
		
		# Performance evaluator object
		self.performanceEvaluator=performanceEvaluator
		
		# Colletion of performance records
		self.performance=[]
		
		# Local index - used when opt does not impose iteration ordering with an index member
		self.localIndex=0
				
	def __call__(self, x, f, opt, annotation=None):
		if 'index' in opt.__dict__:
			# Iteration ordering imposed by opt
			index = opt.index 
		else:
			# No iteration ordering
			index = self.localIndex
			self.localIndex += 1
							
		# Check if the index is inside the already allocated space -> if not allocate new space in memory
		while index >= len(self.performance): 
			newelems=len(self.performance)-index+1
			self.performance.extend([None]*newelems)
				
		# write data
		self.performance[index]=self.performanceEvaluator.results
					
		return index
		
	def reset(self):
		"""
		Clears the :attr:`performance` member. 
		"""
		
		self.performance=[]
		

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
  (see the :func:`pyopus.simulator.simulatorClass` function`for details on 
  how this name is resolved to a simulator class)
* ``settings`` - a dictionary specifying the keyword arguments passed to the 
  simulator object's constructor 
* ``moddefs`` - definition of system description modules 
* ``options`` - simulator options valid for all analyses performed by this 
  simulator. This is a dictionary with option name for key. 
* ``params`` - system parameters valid for all analyses performed in this 
  simulator. This is a dictionary with parameter name for key. 
* ``variables`` - Python variables available during the evaluation of all 
  save directives, analysis commands, and measurements associated with this 
  simulator. 
  
The definition of system description modules in the ``moddefs`` dictionary 
member are themselves dictionaries with system description module name for key. 
Values are dictionaries using the following keys for describing a system 
description module 

* ``file`` - file name in which the system description module is described
* ``section`` - file section name where the system description module 
  description can be bound

Specifying only the ``file`` member translates into an ``.include`` simulator 
input directive (or its equivalent). If additionally the ``section`` member is 
also specified the result is a ``.lib`` directive (or its equivalent). 


The **analyses** data structure is a dictionary with analysis name for key. 
The values are also dictionaries describing an analysis using the following 
dictionary keys:

* ``head`` - the name of the head describing the simulator that will be used 
  for this analysis
* ``modules`` - the list of system description module names that form the 
  system description for this analysis 
* ``options`` - simulator options that apply only to this analysis. This is a 
  dictionary with option name for key. 
* ``params`` - system parameters that apply only to this analysis. This is a 
  dictionary with parameter name for key. 
* ``variables`` - Python variables available during the evaluation of all 
  save directives, analysis commands, and measurements associated with this 
  analysis. 
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
named ``param``. It is a dictionary containing all system parameters defined  
for the analysis. It also has a member names ``var`` which is a dictionary 
containing all variables used associated with the analysis. The ``var`` 
dictionary is also available during save directive evaluation. 


The **measures** data structure is a dictionary with performance measure name 
for key. The values are also dictionaries describing individual performance 
measures using the following dictionary keys

* ``analysis`` - the name of the analysis that produces the results from which 
  the performance measure's value is extracted. Set it to ``None`` for 
  dependent measures (measures whose value is computed from the values of other 
  measures). 
* ``corners`` - the list of corner names across which the performance measure 
  is evaluated. Corner indices obtained from the 
  :meth:`~pyopus.evaluator.cost.MNbase.worstCornerIndex` method of 
  normalization objects (defined in the :mod:`pyopus.evaluator.cost` module) 
  can be converted to corner names by looking up the corresponding members of 
  this list. 
  If this list is omitted the measure is evaluated in all corners defined in 
  the **corners** structure. 
* ``expression`` - a string specifying a Python expression that evaluates to 
  the performance measure's value
* ``script`` - a string specifying a Python script that stores the performance 
  measure's value in a variable named ``__result``
* ``vector`` - a boolean flag which specifies that a performance measure's 
  value may be a vector. If it is ``False`` and the obtained performance 
  measure value is not a scalar (or scalar-like) the evaluation is considered 
  as failed. Defaults to ``False``. 
* ``depends`` - an optional name list of measures required for evaluation of 
  this performance measure. Specified for dependent performance measures. 

If both ``expression`` and ``script`` are given ``script`` is ignored. 

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
  from which the performance measure is being extracted. 
* ``var`` - a dictionary with the values of Python variables that apply to 
  the particular analysis and corner used for obtaining the simulated response 
  from which the performance measure is being extracted. 
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


The **corners** data structure is a dictionary with corner name for key. The 
values are also dictionaries describing individual corners using the following 
dictionary keys: 

* ``modules`` - the list of system description module names that form the system 
  description in this corner
* ``params`` - a dictionary with the system parameters that apply only to 
  this corner
* ``variables`` - Python variables available during the evaluation of all 
  save directives, analysis commands, and measurements for this corner. 
  
This data structure can be omitted by passing ``None`` to the :class:`PerformanceEvaluator`
class. In that case a corner named 'default' with no modules and no parameters 
is created. 
"""

from ..optimizer.base import Plugin, Annotator
from numpy import array, ndarray, iscomplex, dtype
from sys import exc_info
from traceback import format_exception, format_exception_only
from ..simulator import simulatorClass
from ..misc.debug import DbgMsgOut, DbgMsg
from ..parallel.cooperative import cOS

from cPickle import dumps

from pprint import pprint

# Measurements and NumPy
import measure as m
import numpy as np

import sys

__all__ = [ 'PerformanceEvaluator', 'updateAnalysisCount', 'PerformanceAnnotator', 'PerformanceCollector' ] 

def updateAnalysisCount(count, delta, times=1):
	"""
	Updates the analysis counts in dictionary *count* by adding the 
	values from dictionary *delta*. If *count* is not given the 
	current count of every analysis is assumed to be zero. 
	
	Returns the updated dictionary *count*. 
	"""
	if count is None:
		count={}
	
	for name,value in delta.iteritems():
		if name not in count:
			count[name]=0
		count[name]+=value*times
	
	return count	

class PerformanceEvaluator:
	"""
	Performance evaluator class. Objects of this class are callable. The 
	calling convention is ``object(paramDictionary)`` 
	where *paramDictionary* is a dictionary of input parameter values.
	The argument can also be a list of dictionaries containing parameter 
	values. The argument can be omitted (empty dictionary is passed). 
	
	*heads*, *analyses*, *measures*, and *corners* specify the heads, the 
	analyses, the corners, and the performance measures. If *corners* are 
	not specified, a default corner named ``default`` is created. 
	
	*activeMeasures* is a list of measure names that are evaluated by the 
	evaluator. If it is not specified all measures are active. Active measures 
	can be changed by calling the :meth:`setActiveMeasures` method. 
	
	*params* is a dictionary of parameters that have the same value 
	every time the object is called. They should not be passed in the 
	*paramDictionary* argument. This argument can also be a list of 
	dictionaries (dictionaries are joined to obtain one dictionary). 
	
	*variables* is a dictionary holding variables that have the same value 
	in every corner and every analysis. These variables are also available 
	during every performance measure evaluation in the ``var`` dictionary. 
	This can also be a list of dictionaries (dictionaries are joined). 
	
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
	* the ``params`` dictionary specified at evaluator construction. 
	* the ``params`` dictionary of the *heads* data structure entry 
	  corresponding to the analysis
	* the ``params`` dictionary of the *corners* data structure entry 
	  corresponding to the corner
	* the ``params`` dictionary of the *analyses* data structure entry 
	  corresponding to the analysis
	
	If a parameter appears across multiple dictionaries the entries in the 
	input parameter dictionary have the lowest priority and the entries in the 
	``params`` dictionary of the *analyses* have the highest priority. 
	
	A similar priority order is applied to simulator options and Python 
	variables specified in the ``options`` dictionaries (the values from 
	*heads* have the lowest priority and the values from *analyses* have the 
	highest priority). The only difference is that here we have no options 
	separately specified at evaluator construction because simulator options 
	are always associated with a particular simulator (i.e. head). 
	
	Variables specified at evaluator construction have the lowest priority, 
	followed by the values from *heads*, *corners*, and *analyses* dictionaries. 
	
	Independent performance measures (the ones with ``analysis`` not equal to 
	``None``) are evaluated before dependent performance measures (the ones 
	with ``analysis`` set to ``None``). 
	
	The evaluation results are stored in a dictionary of dictionaries with 
	performance measure name as the first key and corner name as the second 
	key. ``None`` indicates that the performance measure evaluation failed in 
	the corresponding corner. 
	
	Objects of this type store the number of analyses performed in the 
	*analysisCount* member. The couter is reset at every call to the 
	evaluator object. 
	
	A call to an object of this class returns a tuple holding the results 
	and the *analysisCount* dictionary. The results dictionary is a 
	dictionary of dictionaries where the first key represents the 
	performance measure name and the second key represents the corner name. 
	The dictionary holds the values of performance measure 
	values across corners. If some value is ``None`` the performance 
	measure evaluation failed for that corner. The return value is also stored 
	in the *results* member of the :class:`PermormanceEvaluator` object. 
	
	If *spawnerLevel* is not greater than 1, evaluations are distributed across 
	available computing nodes (that is unless task distribution takes place at 
	a higher level). Every computing node evaluates one job group. See 
	the :mod:`~pyopus.parallel.cooperative` module for details on parallel 
	processing. More information on job groups can be found in the 
	:mod:`~pyopus.simulator` module. 
	"""
	# Constructor
	def __init__(
		self, heads, analyses, measures, corners=None, params={}, variables={}, activeMeasures=None, 
		debug=0, 
		spawnerLevel=1
	):
		# Debug mode flag
		self.debug=debug
		
		# Store problem
		self.heads=heads
		self.analyses=analyses
		self.measures=measures
		
		self.spawnerLevel=spawnerLevel
		
		if corners is not None:
			self.corners=corners
		else:
			# Construct default corner with no modules and no params
			self.corners={
				'default': {
					'modules': [], 
					'params': {}
				}
			}
		
		# Set fixed parameters
		self.setParameters(params)
		
		# Set fixed variables
		self.skipCompile=True
		self.setVariables(variables)
		self.skipCompile=False
		
		# Set active measures and compile
		self.setActiveMeasures(activeMeasures)
		
		# Results of the performance evaluation
		self.results=None
		
		self.resetCounters()
	
	def _compile(self):
		"""
		Prepares internal structures for faster processing. 
		This function should never be called by the user. 
		"""
		# Sanity check
		for measureName in self.activeMeasures:
			if measureName not in self.measures:
				raise Exception, DbgMsg("PE", "Measure '%s' is not defined." % measureName)
		
		# Construct list of computed measures
		computedMeasures=set(self.activeMeasures)
		# Repeat this until no more measures are added
		while True:
			# Go through all measures and add all dependancies
			candidates=[]
			for measureName in computedMeasures:
				if 'depends' in self.measures[measureName]:
					# Check corner uniqueness
					if (
						'corners' in self.measures[measureName] and
						len(self.measures[measureName]['corners'])!=len(set(self.measures[measureName]['corners']))
					):
						raise Exception, DbgMsg("PE", "Measure '%s' has duplicate corners." % (measureName))
					
					deps=self.measures[measureName]['depends']
					for dep in deps:
						# Sanity check
						if dep not in self.measures:
							raise Exception, DbgMsg("PE", "Dependency '%s' of '%s' is not defined." % (dep, measureName))
						# Add to set 
						candidates.append(dep)
			# Form union
			oldLen=len(computedMeasures)
			computedMeasures=computedMeasures.union(candidates)
			if len(computedMeasures)==oldLen:
				break
			
		# Store
		self.computedMeasures=computedMeasures
		
		# Defaults in heads structure
		for (name, head) in self.heads.iteritems():
			if 'simulator' not in head or len(head['simulator'])<1:
				raise Exception, DbgMsg("PE", "No simulator specified for head '%s'." % name)
			if 'moddefs' not in head or len(head['moddefs'])<1:
				raise Exception, DbgMsg("PE", "No definitions specified for head '%s'." % name)
			
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
		# for (measureName, measure) in self.measures.iteritems():
		for measureName in self.computedMeasures:
			measure=self.measures[measureName]
			anName=measure['analysis']
			# Sanity check
			if anName is not None and anName not in self.analyses:
				raise Exception, DbgMsg("PE", "Analysis '%s' used by measure '%s' is not defined." % (anName, measureName))
			
			# Get corner name list. If a measure has no corners, use all defined corners. 
			if 'corners' in measure:
				cornerNames=measure['corners']
			else:
				cornerNames=self.corners.keys()
				
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
							# print "head", head['params']
						if 'params' in corner:
							params.update(corner['params'])
							# print "corner", corner['params']
						if 'params' in analysis:
							params.update(analysis['params'])
							# print "analysis", analysis['params']
						job['params']=params
						
						# Merge fixed variables with variables from head, corner, and analysis
						variables={}
						variables.update(self.fixedVariables)
						if 'variables' in head:
							variables.update(head['variables'])
						if 'variables' in corner:
							variables.update(corner['variables'])
						if 'variables' in analysis:
							variables.update(analysis['variables'])
						job['variables']=variables
						
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
						
						# print cornerName, corner
						# print "comp job -- ", job
			
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
			simulator=SimulatorClass(**(head['settings'] if 'settings' in head else {}))
			simulator.setJobList(jobListForHead[headName]) 
			
			# Local variable dictionaries for measurement evaluation, one per simulator. 
			self.measureLocals[headName]={}
			self.measureLocals[headName]['param']={}
			self.measureLocals[headName]['var']={}
			
			self.simulatorForHead[headName]=simulator
			
		# Local variable dictionary for dependent measurement evaluation. 
		self.dependentMeasureLocals={
			'param': {}, 
			'var': {}, 
			'result': None, 
			'thisCorner': None
		}
		
		if self.debug:
			DbgMsgOut("PE", "  Simulator objects (%d): " % len(self.jobListForHead))
			for (headName, jobList) in self.jobListForHead.iteritems(): 
				DbgMsgOut("PE", "    %s: %d analyses" % (headName, len(jobList)))
				if self.debug>1:
					for job in jobList:
						DbgMsgOut("PE", "      %s" % job['name'])
		
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
	
	# Reconfigure fixed parameters
	def setParameters(self, params):
		"""
		Sets the parameters dictionary. 
		Can handle a list of dictionaries. 
		"""
		if type(params) is list:
			inputList=params
		else:
			inputList=[params]
		self.fixedParameters={}
		for inputDict in inputList:
			self.fixedParameters.update(inputDict)
	
	# Reset analysis counters
	def resetCounters(self):
		"""
		Resets analysis counters to 0. 
		"""
		self.analysisCount={}
		for name in self.analyses.keys():
			self.analysisCount[name]=0
		
	# Set the variables dictionary. 
	def setVariables(self, variables):
		"""
		Sets the variables dictionary. 
		Can handle a list of dictionaries. 
		"""
		if type(variables) is list:
			inputList=variables
		else:
			inputList=[variables]
		self.fixedVariables={}
		for inputDict in inputList:
			self.fixedVariables.update(inputDict)
		# Need to recompile because the jobs have changed
		if not self.skipCompile:
			self._compile()
	
	# Reconfigure measures
	def setActiveMeasures(self, activeMeasures=None):
		"""
		Sets the list of measures that are going to be evaluated. 
		Specifying ``None`` as *activeMeasures* activates all 
		measures. 
		"""
		# Evaluate all measures by default
		if activeMeasures is not None:
			self.activeMeasures=activeMeasures
		else:
			self.activeMeasures=self.measures.keys()
		
		# Compile
		if self.debug:
			DbgMsgOut("PE", "Compiling.")
			
		self._compile()
		
	# Get active measures
	def getActiveMeasures(self):
		"""
		Returns the names of the active measures. 
		"""
		return self.activeMeasures
	
	def getComputedMeasures(self):
		"""
		Returns the names of all measures that are computed 
		by the evaluator. 
		"""
		return self.computedMeasures
	
	def getParameters(self):
		"""
		Returns the parameters dictionary. 
		"""
		return self.fixedParameters
	
	def getVariables(self):
		"""
		Returns the variables dictionary. 
		"""
		return self.fixedVariables
	
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
	
	@classmethod
	def _postprocessMeasure(cls, measureValue, isVector, debug):
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
					if debug:
						print "PE:         Result can't be converted to an array."
					measureValue=None
			
			# Was conversion successfull? 
			if measureValue is not None: 
				# It is an array
				# Check if it is complex
				if iscomplex(measureValue).any():
					# Bad. Complex value not allowed.
					if debug:
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
						if debug:
							print "PE:         Failed to convert result into a real array."
						measureValue=None
				
				# Check if it is empty
				if measureValue.size==0:
					# It is empty, this is bad
					if debug:
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
						if debug:
							print "PE:         Scalar measurement produced a vector."
						measureValue=None
		
		return measureValue
	
	def generateJobs(self, inputParams):
		# Go through all simulators
		for (headName, simulator) in self.simulatorForHead.iteritems():
			if self.debug: 
				DbgMsgOut("PE", "  Simulator/head %s" % headName)
			
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
			
			# Locals master dictionary
			localsMaster=self.measureLocals[headName]
			
			# Go through all job groups, prepare job
			for i in range(ngroups):
				yield (
					self.processJob, [
						headName, simulator, jobList, keyList, localsMaster, i, 
						inputParams, self.key2measures, self.measures, self.debug
					]
				)
	
	@classmethod
	def processJob(cls, headName, simulator, jobList, keyList, localsMaster, i, inputParams, key2measures, measures, debug):
		# Insert simulator's result access functions
		localsDict={}
		localsDict.update(localsMaster)
		localsDict.update(simulator.getGenerators())
		
		# Delete old loaded results (free memory). 
		simulator.resetResults()
		
		# Run jobs in job group and collect results. 
		(jobIndices, status)=simulator.runJobGroup(i)
		
		results={}
		analysisCount={}
		
		# Go through all job indices in i-th job group. 
		for j in jobIndices: 
			# Get (corner, analysis) key for j-th job. 
			key=keyList[j]
			(cornerName, anName)=key
			job=jobList[j]
			
			# Prepare dictionary of local variables for measurement evaluation
			localsDict['param'].clear()
			localsDict['var'].clear()
			
			# Case: input parameters get overriden by job parameters - default
			# Update with input params
			localsDict['param'].update(inputParams)
			# Update with job params
			localsDict['param'].update(job['params'])
			
			# Update with job variables
			localsDict['var'].update(job['variables'])
			
			# Case: job parameters get overriden by input parameters - unimplemented
			
			# Load result (one job at a time to save memory). 
			simulator.collectResults([j], status)
			
			# Do we have a result? 
			if simulator.activateResult(j) is None: 
				# No. 
				# Assume all measurements that depend on this analysis have failed
				if debug: 
					DbgMsgOut("PE", "    Corner: %s, analysis %s ... FAILED" % (cornerName, anName))
					
				# Set corresponding measurements to None
				for measureName in key2measures[key]:
					# Store result
					if measureName not in results:
						results[measureName]={}
					results[measureName][cornerName]=None
					
					if debug: 
						DbgMsgOut("PE", "      %s : FAILED" % measureName)
				
				# Continue with next analysis
				continue
			else:
				# Yes, we have a result. 
				if debug: 
					DbgMsgOut("PE", "    Corner: %s, analysis: %s ... OK" % (cornerName, anName))
				
				# Update analysis counter
				if key[1] not in analysisCount:
					analysisCount[key[1]]=1
				else:
					analysisCount[key[1]]+=1
				
				# Go through all measurements for this key. 
				for measureName in key2measures[key]:
					# Get measure. 
					measure=measures[measureName]
					
					# Are we expecting a vector?
					if 'vector' in measure:
						isVector=bool(measure['vector'])
					else:
						isVector=False
			
					# TODO: maybe compile it and run it in a function so it has its own namespace
					# Evaluate measure, catch exception that occurs during evaluation. 
					try: 
						if 'expression' in measure:
							measureValue=eval(measure['expression'], globals(), localsDict)	
						elif 'script' in measure:
							tmpLocals={}
							tmpLocals.update(localsDict)
							exec measure['script'] in globals(), tmpLocals
							measureValue=tmpLocals['__result']
						else:
							raise Exception, DbgMsg("PE", "No expression or script.")
						if debug>1:
							DbgMsgOut("PE", "      %s : %s" % (measureName, str(measureValue)))
						elif debug>0: 
							DbgMsgOut("PE", "      %s OK" % (measureName))
					except KeyboardInterrupt:
						DbgMsgOut("PE", "Keyboard interrupt.")
						raise
					except:
						measureValue=None
						if debug:
							DbgMsgOut("PE", "      %s FAILED" % measureName)
							ei=exc_info()
							if debug>1:
								for line in format_exception(ei[0], ei[1], ei[2]):
									DbgMsgOut("PE", "        "+line) 
							else:
								for line in format_exception_only(ei[0], ei[1]):
									DbgMsgOut("PE", "        "+line)
					
					# Store result
					if measureName not in results:
						results[measureName]={}
					results[measureName][cornerName]=cls._postprocessMeasure(measureValue, isVector, debug)
		
		return results, analysisCount
	
	def collectResults(self, analysisCount, results):
		while True:
			index, job, (res1, anCount) = (yield)
			updateAnalysisCount(analysisCount, anCount)
			for measName, cornerResults in res1.iteritems():
				if measName not in results:
					results[measName]={}
				for cornerName, value in cornerResults.iteritems():
					results[measName][cornerName]=value
		
	def __call__(self, parameters={}):
		if self.debug: 
			DbgMsgOut("PE", "Evaluation started.") 
		
		# Reset counters
		self.resetCounters()
		
		# Clear results.
		self.results={}
		
		# Collect parameters when they are given as a list of dictionaries
		if type(parameters) is tuple or type(parameters) is list:
			srcList=parameters
		else:
			srcList=[parameters]
		
		inputParams1={}
		for subList in srcList:
			# Update parameter dictionary
			inputParams1.update(subList)
		
		# Merge with fixed parameters
		inputParams={}
		inputParams.update(self.fixedParameters)
		inputParams.update(inputParams1)
		
		# Check for conflicts
		if len(inputParams)<len(inputParams1)+len(self.fixedParameters):
			# Find conflicts
			conflict=set(inputParams1.keys()).intersection(self.fixedParameters)
			raise Exception, DbgMsg("PE", "Input parameters "+str(list(conflict))+" conflict with parameters specified at construction.")
		
		# Reset temporary results storage
		results={}
		analysisCount={}
		
		# Dispatch tasks
		cOS.dispatch(
			jobList=self.generateJobs(inputParams), 
			collector=self.collectResults(analysisCount, results), 
			remote=self.spawnerLevel<=1
		)
		
		# Store results
		self.results=results
		self.analysisCount=analysisCount
		
		# Handle measures that depend on other measures (analysis is None). 
		# Prepare a copy of results structure containing results for all measures
		# that are associated with an analysis. 
		self.dependentMeasureLocals['result']=self.results.copy()
		# Are there any? 
		if None in self.an2corners:
			# Go through all corners of analysis None
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
					localsMaster=self.dependentMeasureLocals 
					localsMaster['param'].clear()
					localsMaster['var'].clear()
					
					# Case: input parameters get overriden by job parameters - default
					# Update with input params
					localsMaster['param'].update(inputParams)
					# Update with corner params (job params not available since this 
					# measure is not associated with any analysis). 
					if 'params' in corner:
						localsMaster['param'].update(corner['params'])
					# Update with corner variables 
					if 'variables' in corner:
						localsMaster['var'].update(corner['variables'])
					
					# Case: job parameters get overriden by input parameters - unimplemented

					# TODO: maybe compile it and run it in a function so it has its own namespace
					# Evaluate measure, catch exception that occurs during evaluation. 
					try: 
						if 'expression' in measure:
							measureValue=eval(measure['expression'], globals(), localsMaster)	
						elif 'script' in measure:
							tmpLocals={}
							tmpLocals.update(localsMaster)
							exec measure['script'] in globals(), tmpLocals
							measureValue=tmpLocals['__result']
						else:
							raise Exception, DbgMsg("PE", "No expression or script.")
						if self.debug>1:
							DbgMsgOut("PE", "      %s : %s" % (measureName, str(measureValue)))
						elif self.debug>0: 
							DbgMsgOut("PE", "      %s OK" % (measureName))
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
					self.results[measureName][cornerName]=self._postprocessMeasure(measureValue, isVector, self.debug)
		
		return self.results, self.analysisCount
		
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
			# for (measureName, measure) in self.measures.iteritems():
			for measureName in self.computedMeasures:
				measure=self.measures[measureName]
				nameList.append(measureName)
			nameList.sort()
		else:
			nameList=outputOrder

		# Format output
		outStr=""
		for measureName in nameList: 
			if measureName not in self.measures:
				raise Exception, DbgMsg("PE", "Measure '%s' is not defined." % measureName)
			measure=self.measures[measureName]
			# Format measurement values
			first=True
			if 'corners' in measure:
				cornerNames=measure['corners']
			else:
				cornerNames=self.corners.keys()
			for cornerName in cornerNames: 
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
class PerformanceAnnotator(Annotator):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Annotator` iterative 
	algorithm plugin class. This is a callable object whose job is to
	
	* produce an annotation (details of the evaluated performance) stored in the 
	  *performanceEvaluator* object 
	* update the *performanceEvaluator* object with the given annotation 
	
	Annotation is a copy of the :attr:`results` member of 
	*performanceEvaluator*. 
	
	Annotators are used for propagating the details of the cost function from 
	the machine where the evaluation takes place to the machine where the 
	evaluation was requested (usually the master). 
	"""
	def __init__(self, performanceEvaluator):
		self.pe=performanceEvaluator
	
	def produce(self):
		return pe.results.copy(), pe.analysisCount.copy()
	
	def consume(self, annotation):
		pe.results=annotation[0]
		pe.analysisCount=annotation[1]

					
# Performance record collector
class PerformanceCollector(Plugin, PerformanceAnnotator):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Plugin` iterative 
	algorithm plugin class. This is a callable object invoked at every 
	iteration of the algorithm. It collects the summary of the evaluated 
	performance measures from the :attr:`results` member of the 
	*performanceEvaluator* object (member of the :class:`PerformanceEvaluator` 
	class). 
	
	This class is also an annotator that collects the results at remote 
	evaluation and copies them to the host where the remote evaluation was 
	requested. 
	
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
		Plugin.__init__(self)
		PerformanceAnnotator.__init__(self, performanceEvaluator)
		
		# Performance evaluator object
		self.performanceEvaluator=performanceEvaluator
		
		# Colletion of performance records
		self.performance=[]
		
		# Local index - used when opt does not impose iteration ordering with an index member
		self.localIndex=0
				
	def __call__(self, x, ft, opt):
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
		
	def reset(self):
		"""
		Clears the :attr:`performance` member. 
		"""
		
		self.performance=[]
		

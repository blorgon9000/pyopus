"""
.. inheritance-diagram:: pyopus.simulator.spectre
    :parts: 1

**SPECTRE batch mode interface (PyOPUS subsystem name: SPSI)**

SPECTRE is simulator from Cadence. It is capable of modifying circuit, 
subcircuit and simulator parameters, but cannot change the topology 
without restarting and loading a new file. 

SPECTRE is not capable of changing the circuit's topology (system 
definition) without restarting the simulator and loading a new input 
file. It also cannot have a different set of save directives for 
every analysis. 

The ``temperature`` parameter represents the circuit's temperature in 
degrees centigrade (``options temp=...`` simulator directive). Consequently 
the ``temp`` simulator option is not allowed to appear in the simulator 
options list. 

Save statements are global in SPECTRE. Therefore they apply to all analyses 
in a file. If two analyses have different sets of save directives two separate 
input files are generated when *saveSplit* is set to ``True``. Otherwise all 
save directives from all jobs in a file are merged together. 

Nutmeg rawfile (binary) files are used for collecting the reults. 

A job sequence in SPECTRE is a list of lists containing the indices of jobs 
belonging to individual job groups. 

One result group always contains only one plot. See 
:mod:`pyopus.simulator.rawfile` module for the details on the result files. 
"""

import subprocess
from base import Simulator
from rawfile import raw_read
import os
import platform
from ..misc.env import environ
from ..misc.debug import DbgMsgOut, DbgMsg

__all__ = [ 'ipath', 'save_all', 'save_voltage', 'save_current', 'save_property', 
			'an_op', 'an_dc', 'an_ac', 'an_tran', 'an_noise', 'Spectre' ] 

#
# Hierarchical path handling 
#

def ipath(input, outerHierarchy=None, innerHierarchy=None, objectType='inst'):
	"""
	Constructs a hierarchical path for the instance with name given by *input*. 
	The object is located within *outerHierarchy* (a list of instances with 
	innermost instance listed first). *innerHierarchy* a list of names 
	specifying the instance hierarchy inner to the *input* instance. The 
	innermost instance name is listed first. If *outerHierarchy* is not given 
	*input* is assumed to be the outermost element in the hierarchy. Similarly 
	if *innerHierarchy* is not given *input* is assumed to be the innermost 
	element in the hierarchy. 
	
	Returns a string representing a hierarchical path. 
	
	If *input* is a list the return value is also a list representing 
	hierarchical paths corresponding to elements in *input*. 
	
	*innerHierarchy* and *outerHierarchy* can also be ordinary strings 
	(equivalent to a list with only one string as a member). 
	
	The *objectType* argument is for compatibility with other simulators. 
	Because SPICE OPUS treats the hierarchical paths of all objects in the 
	same way, the return value does not depend on *objectType*. The available 
	values of *objectType* are ``'inst'``, ``'mod'``, and ``'node'``. 
	
	SPICE OPUS hierarchical paths begin with the innermost instance followed 
	by its enclosing instances. Colon (``:``) is used as the separator between 
	instances in the hierarchy. So ``m1:x1:x2`` is an instance named ``m1`` 
	that is a part of ``x1`` (inside ``x1``) which in turn is a part of ``x2`` 
	(inside ``x2``). 
	
	Some examples:
	
	* ``ipath('m1', ['x1', 'x2'])`` - instance named ``m1`` inside ``x1`` 
	  inside ``x2``. Returns ``'x2.x1.m1'``. 
	* ``ipath('x1', innerHierarchy=['m0', 'x0'])`` - instance ``m0`` inside 
	  ``x0`` inside ``x1``. Returns ``'x1.x0.m0'``. 
	* ``ipath(['m1', 'm2'], ['x1', 'x2']) - instances ``m1`` and ``m2`` inside 
	  ``x1`` inside ``x2``. Returns ``['x2.x1.m1', 'x2.x1.m2']``. 
	* ``ipath(['xm1', 'xm2'], ['x1', 'x2'], 'm0')`` - instances named ``m0`` 
	  inside paths ``x2.x1.xm1``  and ``x2.x1.xm2``. Returns 
	  ``['x2.x1.xm1.m0', 'x2.x1.xm2.m0']``. 
	"""
	# Create outer and inner path
	
	# Outer hierarchy is represented by a prefix
	if outerHierarchy is None:
		prefStr=''
	else:
		if type(outerHierarchy ) is str:
			prefStr=outerHierarchy+'.'
		else:
			# Make a copy and reverse it
			tmpList=list(outerHierarchy)
			tmpList.reverse()
			prefStr=('.'.join(tmpList))+'.'
	
	# Inner hierarchy is represented by a suffix
	if innerHierarchy is None:
		suffStr=''
	else:
		if type(innerHierarchy) is str:
			suffStr='.'+innerHierarchy
		else:
			# Make a copy and reverse it
			tmpList=list(innerHierarchy)
			tmpList.reverse()
			suffStr='.'+('.'.join(tmpList))
	
	# Build results	
	if type(input) is not list:
		return prefStr+input+suffStr
	else:
		result=[]
		for inst in input:
			result.append(prefStr+inst+suffStr)
	
		return result

#
# Save directive generators
#

# no saves or --all-- given -> options save=all
# have saves, but no --all-- -> options save=selected

def save_all():
	"""
	Returns a save directive that saves all results the simulator normally 
	saves in its output (in SPICE OPUS these are all node voltages and all 
	currents flowing through voltage sources and inductances). 
	"""
	return [ '--all--' ]
	
def save_voltage(what):
	"""
	If *what* is a string it returns a save directive that instructs the 
	simulator to save the voltage of node named *what* in simulator output. 
	If *what* is a list of strings a multiple save directives are returned 
	instructing the simulator to save the voltages of nodes with names given 
	by the *what* list. 
	
	Equivalent of SPICE OPUS ``save v(what)`` simulator command. 
	"""
	compiledList=[]
	
	if type(what) is list:
		input=what
	else:
		input=[what]
		
	for name in input:
		compiledList.append(name)
	
	return compiledList
	
def save_current(what, terminal=1):
	"""
	If *what si a string it returns a save directive that instructs the 
	simulator to save the current flowing through instance names *what* in 
	simulator output. If *what* is a list of strings multiple save diretives 
	are returned instructing the simulator to save the currents flowing 
	through instances with names given by the *what* list. 
	*terminal* specifies the number of the terminal it which the current 
	is saved. 
	"""
	compiledList=[]
	
	if type(what) is list:
		input=what
	else:
		input=[what]
	
	for name in input:
		compiledList.append(name+':'+str(terminal))
	
	return compiledList

def save_property(devices, params, indices=None):
	"""
	Saves the properties given by the list of property names (*params*) of 
	instances given by the *devices* list. 
	
	If *params* and the *devices* have n and m members, n*m save directives 
	are returned describing all combinations of device name and property name. 
	"""
	compiledList=[]
	
	if type(devices) is list:
		inputDevices=devices
	else:
		inputDevices=[devices]
	
	if type(params) is list:
		inputParams=params
	else:
		inputParams=[params]
	
	if indices is None:
		for name in inputDevices:
			for param in inputParams:
				compiledList.append(name+':'+param)
	else:
		raise Exception, DbgMsg("SPSI", "Device properties do not have indices in SPECTRE.")
	
	return compiledList


#
# Analysis command generators
#

def an_parameterString(params):
	if len(params)==0:
		return ""
	
	anStr=" "
	for (key,value) in params.iteritems():
		anStr+=" "+str(key)+"="+str(value)
	return anStr

def an_op(**kwargs):
	"""
	Generates the SPECTRE simulator command that invokes an operating 
	point analysis. 
	
	Passes any additional arguments to the dc simulator directive. 
	"""
	# Actually sweeps idummy__ with one point (dc=0.0)
	return 'dc dev=idummy__ param=dc values=[0]'+an_parameterString(kwargs)

def an_dc(start=None, stop=None, sweep=None, points=None, name=None, parameter=None, 
	  index=None, **kwargs):
	"""
	Generates the SPECTRE simulator command that invokes an operating point 
	sweep (DC) analysis. See the SPECTRE manual for details on arguments. 
	
	Generates the SPECTRE simulator directive that invokes the operating point 
	sweep (DC) analysis. *start* and *stop* give the intial and the final 
	value of the swept parameter. 
	
	*sweep* can be one of 
	
	* ``'lin'`` - linear sweep with the number of points given by *points* 
	* ``'dec'`` - logarithmic sweep with points per decade 
	  (scale range of 1..10) given by *points*
	
	*name* gives the name of the instance whose *parameter* is swept. 
	Because SPECTRE knows no such thing as vector parameters, *index* 
	should never be used. 
	
	If *name* is not given a sweep of a circuit parameter (defined with 
	``.param``) is performed. The name of the parameter can be specified with 
	the *parameter* argument. If *parameter* is ``temperature`` a sweep of the 
	circuit's temperature is performed. 
	
	If *star*, *stop*, *sweep*, *points*, and *parameter* are given, a 
	common PyOPUS dc sweep is performed. Otherwise the parameters are passed 
	to the dc simulator directive. 
	
	Passes any additional arguments to the dc simulator directive. 
	"""
	if index is not None:
		raise Exception, DbgMsg("SPSI", "Device properties do not have indices in SPECTRE.")
	
	if (
		parameter is not None and 
		start is not None and stop is not None and 
		sweep is not None and points is not None and
		len(kwargs)==0
	):
		# Ordinary PyOPUS sweep
		if sweep=='lin':
			sweepName='lin'
		elif sweep=='dec':
			sweepName='dec'
		else:
			raise Exception, DbgMsg("SPSI", "SPECTRE supports only 'lin' and 'dec' sweep in a PyOPUS common sweep.")

		# Check if we are sweeping a parameter
		if name is None:
			# Temperature and netlist parameter sweep
			if param=='temperature':
				usePar='temp'
			else:
				usePar=parameter

			anStr=(
				'dc start='+str(start)+' stop='+str(stop)+
				' '+sweepName+'='+str(points)+' param='+str(usePar)
			)
		else:
			# Device parameter sweep
			anStr=(
				'dc start='+str(start)+' stop='+str(stop)+
				' '+sweepName+'='+str(points)+
				' dev='+str(name)+' param='+str(parameter)
			)
	else:
		# SPECTRE dc sweep, put start and stop in kwargs
		kwargs["start"]=start
		kwargs["stop"]=stop
		
		anStr='dc'
	
	# TODO: format vector parameters as string
	
	return anStr+an_parameterString(kwargs)

def an_ac(start=None, stop=None, sweep=None, points=None, **kwargs):
	"""
	Generats the SPECTRE simulator command that invokes a small signal 
	(AC) analysis. See the SPECTRE manual for details on arguments. 
	
	*sweep* can be one of 
	
	* ``'lin'`` - linear sweep with the number of points given by *points* 
	* ``'dec'`` - logarithmic sweep with points per decade 
	  (scale range of 1..10) given by *points*
	
	if *start*, *stop*, *sweep*, and *points* are given a common PyOPUS 
	ac sweep is performed. 
	
	Passes any additional arguments to the ac simulator directive. 
	"""
	if (
		start is not None and stop is not None and 
		sweep is not None and points is not None and 
		len(kwargs)==0
	):
		# Ordinary PyOPUS sweep
		if sweep=='lin':
			sweepName='lin'
		elif sweep=='dec':
			sweepName='dec'
		else:
			raise Exception, DbgMsg("SPSI", "SPECTRE supports only 'lin' and 'dec' in a PyOPUS common sweep")
		
		anStr=(
			'ac start='+str(start)+' stop='+str(stop)+
			' '+sweepName+'='+str(points)
		)
	else:
		# SPECTRE ac sweep, put start and stop in kwargs
		kwargs["start"]=start
		kwargs["stop"]=stop
		
		anStr='ac'
	
	# TODO: format vector parameters as string
	
	return anStr+an_parameterString(kwargs)
		
def an_tran(step=None, stop=None, start=0.0, maxStep=None, uic=False, **kwargs):
	"""
	Generats the SPECTRE simulator command that invokes a transient 
	analysis. See the SPECTRE manual for details on arguments.
	
	If *step* and *stop* (optionally *start*, *maxStep*, and *uic*)
	are given a common PyOPUS transient analysis is performed. To force 
	a spectre style analysis with only *stop* (and *start*), specify at 
	least one SPECTRE-only transient analysis parameter default value 
	(e.g. skipdc=no). 
	
	Passes any additional arguments to the tran simulator directive. 
	"""
	# step and stop
	if step is not None and stop is not None and len(kwargs)==0:
		# Ordinary PyOPUS transient, ignore step
		anStr='tran stop='+str(stop)+' outputstart='+str(start)
		if uic:
			anStr+=' ic=all skipdc=yes'
	elif (
		step is not None and stop is not None and 
		maxStep is not None and len(kwargs)==0
	):
		# Ordinary PyOPUS transient, ignore step
		anStr=(
			'tran stop='+str(stop)+' outputstart='+str(start)+
			' maxstep='+str(maxStep)
		)
		if uic:
			anStr+=' ic=all skipdc=yes'
	else:
		# SPECTRE transient, move start and stop to kwargs
		kwargs["start"]=start
		kwargs["stop"]=stop
		
		anStr='tran'
	
	# TODO: format vector parameters as string
	
	return anStr+an_parameterString(kwargs)

def an_noise(start=None, stop=None, sweep=None, points=None, input=None, outp=None, outn=None, ptsSum=1, **kwargs):
	"""
	Generats the SPECTRE simulator command that invokes a small signal 
	noise analysis. See the SPECTRE manual for details on arguments.
	
	If *start*, *stop*, *sweep*, and *points* are given a common PyOPUS 
	noise analysis is performed. 
	
	The *ptsSum* argument is ignored. 
	
	Passes any additional arguments to the noise simulator directive. 
	"""
	if (
		start is not None and stop is not None and 
		sweep is not None and points is not None and 
		len(kwargs)==0
	):
		# Ordinary PyOPUS sweep
		if sweep=='lin':
			sweepName='lin'
		elif sweep=='dec':
			sweepName='dec'
		else:
			raise Exception, DbgMsg("SPSI", "SPECTRE supports only 'lin' and 'dec' in a PyOPUS common sweep")
		
		if outp is None:
			raise Exception, DbgMsg("SPSI", "Need at least outp for noise analysis")
		
		if outn is None:
			anStr=str(outp)+" noise"
		else:
			anStr=str(outp)+" "+str(outn)+" noise"
		
		anStr+=(
			' start='+str(start)+' stop='+str(stop)+
			' '+sweepName+'='+str(points)+
			' iprobe='+str(input)
		)
	else:
		# SPECTRE noise, move start and stop to kwargs
		kwargs["start"]=start
		kwargs["stop"]=stop
		
		if outp is None:
			anStr='noise'
		elif outn is None:
			anStr=str(outp)+" noise"
		else:
			anStr=str(outp)+" "+str(outn)+" noise"
		
	return anStr+an_parameterString(kwargs)

class Spectre(Simulator):
	"""
	A class for interfacing with the SPECTRE simulator. 
	
	*binary* is the path to the SPECTRE simulator binary. If it is not given 
	the ``SPECTRE_BINARY`` environmental variable is used as the path to the 
	binary. If ``SPECTRE_BINARY`` is not set the binary is assumed to be 
	named 'spectre' and located in the system PATH. 
	
	*args* apecifies a list of additional arguments passed to the simulator 
	binary at startup. 
	
	If *debug* is greater than 0 debug messages are printed at the standard 
	output. If it is above 1 a part of the simulator output is also printed. 
	If *debug* is above 2 full simulator output is printed. 
	
	The save directives from the simulator job description are evaluated in an 
	environment where the following objects are available: 
	
	* ``all`` - a reference to the :func:`save_all` function
	* ``v`` - a reference to the :func:`save_voltage` function
	* ``i`` - a reference to the :func:`save_current` function
	* ``p`` - a reference to the :func:`save_property` function
	* ``ipath`` - a reference to the :func:`ipath` function
	
	Similarly the environment for evaluating the analysis command given in the 
	job description consists of the following objects: 
	
	* ``op`` - a reference to the :func:`an_op` function
	* ``dc`` - a reference to the :func:`an_dc` function
	* ``ac`` - a reference to the :func:`an_ac` function
	* ``tran`` - a reference to the :func:`an_tran` function
	* ``noise`` - a reference to the :func:`an_noise` function
	* ``ipath`` - a reference to the :func:`ipath` function
	* ``param`` - a dictionary containing the members of the ``params`` entry 
	  in the simulator job description together with the parameters from the 
	  dictionary passed at the last call to the :meth:`setInputParameters` 
	  method. The parameters values given in the job description take 
	  precedence over the values passed to the :meth:`setInputParameters` 
	  method. 
	  
	Seting *saveSplit* to ``True`` splits a job group in multiple 
	job groups with differing setws of save directives. Setting it to 
	``False`` (default) joins the save directives from all jobs in a job 
	group. 
	"""
	def __init__(self, binary=None, args=[], debug=0, saveSplit=False):
		Simulator.__init__(self, binary, args, debug)
		
		self.saveSplit=saveSplit
		self._compile()
	
	def _compile(self):
		"""
		Prepares internal structures. 
		
		* dictionaries of functions for evaluating save directives and analysis 
		  commands
		* constructs the binary name for invoking the simulator
		"""
		# Local namespace for save directive evaluation
		self.saveLocals={
			'all': save_all,
			'v': save_voltage,
			'i': save_current, 
			'p': save_property, 
			'ipath': ipath, 
			'var': {}
		}
		
		# Local namespace for analysis evaluation
		self.analysisLocals={
			'op': an_op, 
			'dc': an_dc, 
			'ac': an_ac, 
			'tran': an_tran, 
			'noise': an_noise, 
			'ipath': ipath, 
			'param': {}, 
			'var': {}
		}
		
		# Default binary based on SPECTRE_BINARY and platform
		if self.binary is None:
			if 'SPECTRE_BINARY' in environ:
				spectrebinary=environ['SPECTRE_BINARY']
			else:
				spectrebinary='spectre'
			
			self.binary=spectrebinary
		
	# For pickling - copy object's dictionary and remove members 
	# with references to member functions so that the object can be pickled. 
	def __getstate__(self):
		state=self.__dict__.copy()
		del state['saveLocals']
		del state['analysisLocals']
		
		return state
	
	# For unpickling - update object's dictionary and rebuild members with references
	# to member functions. Also rebuild simulator binary name. 
	def __setstate__(self, state):
		self.__dict__.update(state)
		
		self._compile()
	
	
	def _createSaves(self, saveDirectives, variables):
		"""
		Creates a list of save directives by evaluating the members of the 
		*saveDirectives* list. *variables* is a dictionary of extra 
		variables that are available during directive evaluation. 
		"""
		# Prepare evaluation environment
		evalEnv={}
		evalEnv.update(self.saveLocals)
		evalEnv['var']=variables
		
		compiledList=[]
		
		for saveDirective in saveDirectives:
			# A directive must be a string that evaluates to a list of strings
			saveList=eval(saveDirective, globals(), evalEnv)

			if type(saveList) is not list: 
				raise Exception, DbgMsg("SPSI", "Save directives must evaluate to a list of strings.")
				
			for save in saveList:
				if type(save) is not str:
					raise Exception, DbgMsg("SPSI", "Save directives must evaluate to a list of strings.")
			
			compiledList+=saveList
			
		# Make list memebers unique, sort
		ordered=list(set(compiledList))
		ordered.sort()
		return ordered

	#
	# Batch simulation
	#
	
	def writeFile(self, i):
		"""
		Prepares the simulator input file for running the *i*-th job group. 
		
		The file is named ``simulatorID_group_i.scs`` where *i* is the index 
		of the job group. 
		
		All output files with simulation results are .raw files in binary 
		format. 
		
		System description modules are converted to ``include`` simulator 
		directives. 
		
		Simulator options are set with the ``set`` simulator directive. 
		Integer, real, and string simulator options are converted with the 
		:meth:`__str__` method before they are written to the file. Boolean 
		options are converted to ``0`` or ``1t`` depending on whether they 
		are ``True`` or ``False``. 
		
		The parameters set with the last call to :meth:`setInputParameters` 
		method are joined with the parameters in the job description. The 
		values from the job description take precedence over the values 
		specified with the :meth:`setInputParameters` method. All parameters 
		are written to the input file in form of ``alter`` simulator directives. 
		
		The ``temperature`` parameter is treated differently. It is written to 
		the input file in form if a ``set`` simulator directive preceding its 
		corresponding analysis directive. 
		
		Save directives are written as a series of ``save`` simulator directives. 
		
		Every analysis command is evaluated in its corresponding environment 
		taking into account the parameter values passed to the 
		:meth:`setInputParameters` method. 
		
		All analyses write the results to a single binary .raw file. 
		
		The function returns the name of the simulator input file it generated. 
		"""
		# Build file name
		fileName=self.simulatorID+"_group"+str(i)+'.scs'
		
		if self.debug>0:
			DbgMsgOut("SPSI", "Writing job group '"+str(i)+"' to file '"+fileName+"'")
			
		f=open(fileName, 'w')
		
		# First line
		f.write('// Simulator input file for job group '+str(i)+'\n\n')
		f.write('simulator lang=spectre\n')
		
		# Dummy devices to avoid a bug in rawfile output where the last
		# operating poijnt analysis has 0 points if it is run as a
		# dc analysis without any parameters and there are save diretives
		# present. Operating point is performed as a 1 point sweep of 
		# this dummy device. 
		f.write('idummy__ (dummy_node__ 0) isource dc=0\n')
		f.write('rdummy__ (dummy_node__ 0) resistor r=1k\n\n')
		
		# Job group
		jobGroup=self.jobGroup(i)
		
		# Representative job
		repJob=self.jobList[jobGroup[0]]
		
		# Directive index counter
		dirNdx=0

		# Include definitions
		for definition in repJob['definitions']:
			if 'section' in definition:
				f.write('include \"'+definition['file']+'\" section='+definition['section']+'\n')
			else:
				f.write('include \"'+definition['file']+'\"\n')
		
		# Write representative options (as .option directives)
		if 'options' in repJob:
			for (option, value) in repJob['options'].iteritems():
				if value is True:
					strValue='1'
				elif value is False:
					strValue='0'
				else:
					strValue=str(value)
				f.write('dir'+str(dirNdx)+' options '+option+'='+strValue+'\n')
				dirNdx+=1
				
		# Prepare representative parameters dictionary. 
		# Case: input parameters get overriden by job parameters - default
		params={}
		params.update(self.inputParameters)
		if 'params' in repJob:
			params.update(repJob['params'])
			
		# Case: job parameters get overriden by input parameters - unimplemented

		# Write representative parameters, handle temperature as simulator option. 
		for (param, value) in params.iteritems():
			if value is True:
				strValue='1'
			elif value is False:
				strValue='0'
			else:
				strValue=str(value)
				
			if param!="temperature":
				f.write('parameters '+param+'='+strValue+'\n')
			else:
				f.write('dir'+str(dirNdx)+' options temp='+strValue+'\n')
				dirNdx+=1
			
		# Analyses
		f.write('\n');
		f.write('// output settings\n')
		f.write('setFmt options rawfmt=nutbin\n')
		f.write('\n');
		
		# Merge saves from all jobs
		allSaves=[]
		for j in jobGroup:
			job=self.jobList[j]
			allSaves.extend(self._createSaves(job['saves'], job['variables']))
			# Keep unique saves
			allSaves=list(set(allSaves))
		
		# Dump saves
		f.write('// save directives\n')
		count=0
		haveAll=False
		for save in allSaves:
			if save=='--all--':
				haveAll=True
				continue
			if count == 0: 
				f.write('save ')
			f.write(save+' ')
			count+=1
			if count == 10:
				count=0
				f.write('\n')
		f.write('\n')
		if not haveAll and len(allSaves)>0:
			f.write('setSave options save=selected\n')
		else:
			f.write('setSave options save=all\n')
		f.write('\n')
		
		f.write('// analyses begin here\n')
		f.write('\n');
		
		# Handle analyses
		for j in jobGroup:
			# Get job
			job=self.jobList[j]
			
			# Get job name
			if self.debug>0:
				DbgMsgOut("SPSI", "  job '"+job['name']+"'")
			
			# Prepare evaluation environment for analysis command
			evalEnv={}
			evalEnv.update(self.analysisLocals)
			evalEnv['var']=job['variables']
			
			# Prepare analysis params - used for evauating analysis expression. 
			# Case: input parameters get overriden by job parameters - default
			analysisParams={}
			analysisParams.update(self.inputParameters)
			if 'params' in job:
				analysisParams.update(job['params'])
				
			# Case: job parameters get overriden by input parameters - unimplemented
			
			# Analysis commands start here
			f.write('// '+job['name']+'\n')
			
			# Write options for analysis
			if 'options' in job:
				for (option, value) in job['options'].iteritems():
					if value is True:
						strValue='1'
					elif value is False:
						strValue='0'
					else:
						strValue=str(value)
					
					f.write('dir'+str(dirNdx)+' set '+option+'='+strValue+'\n')
					dirNdx+=1
			
			# Write parameter values
			for (param, value) in analysisParams.iteritems():
				if value is True:
					strValue='1'
				elif value is False:
					strValue='0'
				else:
					strValue=str(value)
					
				if param!="temperature":
					f.write('dir'+str(dirNdx)+' alter param='+param+' value='+strValue+'\n')
				else:
					f.write('dir'+str(dirNdx)+' alter param=temp value='+strValue+'\n')
				dirNdx+=1
					
			# Prepare parameters dictionary for local namespace
			self.analysisLocals['param'].clear()
			self.analysisLocals['param'].update(analysisParams)
			
			# Write analysis
			anStr=eval(job['command'], globals(), evalEnv)
			f.write('job'+str(j)+' '+anStr+'\n')
			
		f.close()
		
		return fileName
		
	def cleanupResults(self, i): 
		"""
		Removes all result files that were produced during the simulation of 
		the *i*-th job group. Simulator input files are left untouched. 
		""" 
		if self.debug>0:
			DbgMsgOut("SPSI", "Cleaning up result for job group "+str(i))
			
		jobGroup=self.jobGroup(i)
		
		# Remove old .raw file
		try:
			os.remove(self.simulatorID+"_group"+str(i)+'.raw')
		except KeyboardInterrupt:
			DbgMsgOut("SPSI", "Keyboard interrupt")
			raise
		except:
			None
		
	def runFile(self, fileName):
		"""
		Runs the simulator on the input file given by *fileName*. 
		
		Returns ``True`` if the simulation finished successfully. 
		This does not mean that any results were produced. 
		It only means that the return code from the simuator was 0 (OK). 
		"""
		if self.debug>0:
			DbgMsgOut("SPSI", "Running file '"+fileName+"'")
				
		# Run the file
		spawnOK=True
		p=None
		try:
			# Start simulator
			p=subprocess.Popen(
					[self.binary]+self.cmdline+[fileName], 
					universal_newlines=True, 
					stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE
				)
			
			# Collect output
			self.messages=p.stdout.read()
			
			if self.debug>2:
				DbgMsgOut("SPSI", self.messages)
			elif self.debug>1:
				DbgMsgOut("SPSI", self.messages[-400:])
			
			# Now wait for the process to finish. If we don't wait p might get garbage-collected before the
			# actual process finishes which can result in a crash of the interpreter. 
			retcode=p.wait()
			
			# Check return code. Nonzero return code means that something has gone bad. 
			# At least the simulator says so. 
			if retcode!=0:
				spawnOK=False
		except KeyboardInterrupt:
			DbgMsgOut("SPSI", "Keyboard interrupt")
			
			# Will raise an exception if process exits before kill() is called.
			try:
				p.kill()
			except:
				pass
			
			raise KeyboardInterrupt
		except:
			spawnOK=False
		
		if not spawnOK and self.debug>0:
			DbgMsgOut("SPSI", "  run FAILED")
			
		return spawnOK
		
	def runJobGroup(self, i):
		"""
		Runs the *i*-th job group. 
		
		First calls the :meth:`writeFile` method followed by the 
		:meth:`cleanupResults` method that removes any old results produced by 
		previous runs of the jobs in *i*-th job group. Finally the 
		:meth:`runFile` method is invoked. Its return value is stored in the 
		:attr:`lastRunStatus` member. 
		
		The function returns a tuple (*jobIndices*, *status*) where 
		*jobIndices* is a list of job indices corresponding to the *i*-th job 
		group. *status* is the status returned by the :meth:`runFile` method. 
		"""
		# Write file for job group. 
		filename=self.writeFile(i)
		
		# Delete old results. 
		self.cleanupResults(i)
		
		# Run file
		self.lastRunStatus=self.runFile(filename)
		
		# Get job indices for jobs in this job group. 
		jobIndices=self.jobGroup(i)
		
		return (jobIndices, self.lastRunStatus)
	
	def collectResults(self, indices, runOK=None):
		"""
		Collects the results produces by running jobs with indices given by 
		the *indices* list. *runOK* specifies the status returned by the 
		:meth:`runJobGroup` method which produced the results. 
		
		If *runOK* is ``False`` the result groups of the jobs with indices 
		given by *indices* are set to ``None``. 
		
		A results group corresponding to some job is set to ``None`` if the 
		.raw file is not successfully loaded. 
		"""
		if runOK is None:
			runOK=self.lastRunStatus
			
		if runOK:
			# Collect results
			count=0
			countOK=0
			for jobIndex in indices:
				# Because there are results of every job in a job group are in one file, 
				# we convert jobIndex to groupIndex
				groupIndex=self.jobGroupIndex[jobIndex]
				
				# The result may already be loaded because someone asked for the results
				# of a different job in the same group before
				if jobIndex in self.results:
					continue
				
				# Load the file, reverse bytes in double values (those Cadence folks and endianness don't mix)
				reslist=raw_read(self.simulatorID+"_group"+str(groupIndex)+'.raw', reverse=1)
				
				# Go through reslist and distribute results
				jobsInGroup=self.jobSequence[groupIndex]
				for ii in range(len(jobsInGroup)):
					jobNdx=jobsInGroup[ii]
					if ii<len(reslist):
						self.results[jobNdx]=[reslist[ii]]
					else:
						self.results[jobNdx]=None
			
			# Count
			for jobIndex in indices:
				if self.results[jobIndex] is not None:
					countOK+=1
				count+=1
			
			if self.debug>0:
				DbgMsgOut("SPSI", "  "+str(countOK)+"/"+str(count)+" jobs OK")
		else:
			# Simulator failed, no results
			for jobIndex in indices: 
				self.results[jobIndex]=None
	
	def jobGroupCount(self):
		"""
		Returns the number of job groups. 
		"""
		return len(self.jobSequence)
	
	def jobGroup(self, i):
		"""
		Returns a list of job indices corresponding to the jobs in *i*-th job 
		group. 
		"""
		return self.jobSequence[i]

	
	#
	# Job optimization
	#
	
	def unoptimizedJobSequence(self):
		"""
		Returns the unoptimized job sequence. If there are n jobs in the job list 
		the following list of lists is returned: ``[[0], [1], ..., [n-1]]``. 
		This means we have n job groups with one job per job group. 
		"""
		seq=[[0]]*len(self.jobList)
		for i in range(len(self.jobList)):
			seq[i]=[i];
		return seq
		
	def optimizedJobSequence(self):
		"""
		Returns the optimized job sequence. 
		
		Jobs in a job group have:
		
		* identical circuit definition, 
		* identical save directives
		"""
		# Count jobs
		jobCount=len(self.jobList)
		
		# Construct a list of job indices
		candidates=set(range(jobCount))
		
		# Evaluate save directives, sort them
		saves=[]
		for jobIndex in range(jobCount):
			job=self.jobList[jobIndex]
			if 'saves' in job:
				job["processedSaves"]=self._createSaves(job['saves'], job['variables'])
				# Make unique, sort
				
			else:
				job["processedSaves"]=[]
			
		# Empty job sequence
		seq=[]
		
		# Repeat while we have a nonempty indices list. 
		while len(candidates)>0:
			# Take one job
			i1=candidates.pop()
			
			# Start a new job group
			jobGroup=[i1]
			
			# Compare i1-th job with all other jobs
			peerCandidates=list(candidates)
			for i2 in peerCandidates:
				# Check if i1 and i2 can be joined together
				# Compare jobs, join them if all of the following holds  
				# - definitions are identical
				# - the list of processedSaves is identical
				if (
					self.jobList[i1]['definitions']==self.jobList[i2]['definitions'] and
					(
						self.saveSplit is False or
						self.jobList[i1]['processedSaves']==self.jobList[i2]['processedSaves']
					)
				):
					# Job i2 can be joined with job i1, add it to jobGroup
					jobGroup.append(i2)
					# Remove i2 from candidates
					candidates.remove(i2)
			
			# Sort jobGroup
			jobGroup.sort()
			
			# Append it to job sequence
			seq.append(jobGroup)
			
		# Remove processedSaves
		for job in self.jobList:
			if 'processedSaves' in job:
				del job['processedSaves']
		
		return seq

		
	#
	# Retrieval of simulation results
	#
	
	def res_voltage(self, node1, node2=None, resIndex=0):
		"""
		Retrieves the voltage corresponding to *node1* (voltage between 
		nodes *node1* and *node2* if *node2* is also given) from the 
		active result. 
		"""
		if node2 is None:
			return self.activeResult[resIndex][0][node1]
		else:
			return self.activeResult[resIndex][0][node1]-self.activeResult[resIndex][0][node2]
	
	def res_current(self, name, resIndex=0):
		"""
		Retrieves the current flowing through instance *name* from the 
		active result group. 
		"""
		return self.activeResult[resIndex][0][name+':p']
		
	def res_property(self, name, parameter, index=None, resIndex=0):
		"""
		Retrieves property named *parameter* belonging to instance 
		named *name*. The property is retrieved from the active 
		result. 
		
		Note that this works only of the property was saved with a 
		corresponding save directive. 
		"""
		if index is None:
			return self.activeResult[resIndex][0][name+':'+parameter]
		else:
			raise Exception, DbgMsg("SPSI", "Device properties do not have indices in SPECTRE.")
			
	
	def res_noise(self, reference, name=None, contrib=None, resIndex=0):
		"""
		Retrieves the noise spectrum density of contribution *contrib* of 
		instance *name* to the input/output noise spectrum density. *reference* 
		can be ``'input'`` or ``'output'``. If ``'gain'`` is given as 
		*reference* the gain is returned. 
		
		If *name* and *contrib* are not given the output or the equivalent 
		input noise spectrum density is returned (depending on the value of 
		*reference*). 
		
		Partial noise spectra are in V^2/Hz, total spectra are in V/sqrt(Hz). 
		
		The spectrum is obtained from the active result. 
		"""
		# TODO: units of total/partial input/output spectra (V**2/Hz or V**2)
		if name is None:
			# Input/output noise spectrum
			if reference=='input':
				spec=self.activeResult[resIndex][0]['in']
			elif reference=='output':
				spec=self.activeResult[resIndex][0]['out']
			elif reference=='gain':
				spec=self.activeResult[resIndex][0]['gain']
			else:
				raise Exception, "Bad noise reference."
		else:
			# Partial spectrum
			if reference=='input':
				# A=self.activeResult[resIndex][0]['inoise_spectrum']/self.activeResult[resIndex][0]['onoise_spectrum']
				A=self.activeResult[resIndex][0]['gain']
			elif reference=='output':
				A=1.0
			else:
				raise Exception, "Bad noise reference."
			
			spec=self.activeResult[resIndex][0][str(name)+"."+str(contrib)]/A
	
		return spec
		
	def res_scale(self, resIndex=0):
		"""
		Retrieves the default scale of the active result. 
		"""
		scaleName=self.activeResult[resIndex][1]
		return self.activeResult[resIndex][0][scaleName]
	
	def getGenerators(self):
		"""
		Returns a dictionary of functions that retrieve results from the 
		active result group. 
		
		The following functions (dictionary key names) are available:
		
		* ``v`` - a reference to the :meth:`res_voltage` method
		* ``i`` - a reference to the :meth:`res_current` method
		* ``p`` - a reference to the :meth:`res_property` method
		* ``ns`` - a reference to the :meth:`res_noise` method
		* ``scale`` - a reference to the :meth:`res_scale` method
		* ``ipath`` - a reference to the :func:`ipath` function
		"""
		return {
			'v': self.res_voltage, 
			'i': self.res_current, 
			'p': self.res_property, 
			'ns': self.res_noise, 
			'scale': self.res_scale, 
			'ipath': ipath
		}
	
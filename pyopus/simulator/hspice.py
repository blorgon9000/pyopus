"""
.. inheritance-diagram:: pyopus.simulator.hspice
    :parts: 1

**HSPICE interface (PyOPUS subsystem name: HSSI)**

HSPICE is a batch mode simulator. It is capable of changing the circuit's 
topology and its parameters between consecutive simulations without the need to 
restart the simulator with a new input file. HSPICE is completely input-file 
driven and presents no command prompt to the user. 

Save directives do not apply to the AC analysis because the HSPICE ``.probe`` 
simulator directive works only for real values. 

The ``temperature`` parameter has special meaning and represents the circuit's 
temperature in degrees centigrade. 

Analysis command generators return a tuple with teh following members: 
 
0. analysis type (``'dc'``, ``'ac'``, ``'tran'``, ``'noise'``)
1. analysis results file ending (``'sw'``, ``'ac'``, ``'tr'``)
2. analysis directive text

A job sequence in HSPICE is internally list of lists where the inner lists 
contain the indices of jobs belonging to one job group. The user is, however, 
presented with only one job group containing all job indices ordered in the 
manner the jobs will later be simulated (flat job sequence). 

Job sequence optimization minimizes the number of topology changes between 
consecutive jobs. Internally this is represented in the job sequence by job 
groups where all jobs in a group share the same circuit topology. 

One result group can consist of multiple plots resulting from multiple 
invocations of the same analysis (resulting from a parametric sweep). 
See :mod:`pyopus.simulator.hspicefile` module for the details on the result 
files. 

Repeated analyses with a parameter sweep and the collection of ``.measure`` 
directive results are currently not supported. 
"""

# HSPICE simulator interface

# Benchmark result on opamp, HSPICE, Windows XP 32bit
#	131 iterations, best in 113, final cost -0.0692397737822
#	53.236s/57.737s = 92.2% time spent in simulator

# Benchmark result on opamp, HSPICE, Linux AMD64 2CPU
#	131 iterations, best in 113, final cost -0.0692719531626
#	16.046s/18.014s = 89.1% time spent in simulator

# Benchmark result on opamp, HSPICE, Linux AMD64 1CPU
#	131 iterations, best in 113, final cost -0.0692719531626
#	16.046s/18.014s = 89.1% time spent in simulator

from ..misc.debug import DbgMsgOut, DbgMsg
import subprocess
from base import Simulator
from hspicefile import hspice_read
import os
import platform
from sys import exc_info
from traceback import format_exception, format_exception_only
from ..misc.env import environ

__all__ = [ 'ipath', 'save_all', 'save_voltage', 'save_current', 'save_property', 
			'an_op', 'an_dc', 'an_ac', 'an_tran', 'an_noise', 'HSpice' ] 

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
	if *innerHierarchy* is not given *input* is assumed to be the innermost element in teh hierarchy. 
	
	Returns a string representing a hierarchical path. 
	
	If *input* is a list the return value is also a list representing 
	hierarchical paths corresponding to elements in *input*. 
	
	*innerHierarchy* and *outerHierarchy* can also be ordinary strings 
	(equivalent to a list with only one string as a member). 
	
	The *objectType* argument is for compatibility with other simulators. 
	Because HSPICE treats the hierarchical paths of all objects in the same 
	way, the return value does not depend on *objectType*. The available 
	values of *objectType* are ``'inst'``, ``'mod'``, and ``'node'``. 
	
	HSPICE hierarchical paths begin with the outermost instance followed by its
	subinstances. A dot (``.``) is used as the separator between instances in 
	the hierarchy. So ``x2.x1.m1`` is an instance named ``m1`` that is a part 
	of ``x1`` (inside ``x1``) which in turn is a part of ``x2`` (inside ``x2``). 
	
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
		suffStr=''
	else:
		if type(outerHierarchy ) is str:
			prefStr=outerHierarchy+'.'
		else:
			# Reverse outer hierarchy
			prefStr='.'.join(outerHierarchy[::-1])+'.'
	
	# Inner hierarchy is represented by a suffix
	if innerHierarchy is None:
		suffStr=''
	else:
		if type(innerHierarchy) is str:
			suffStr='.'+innerHierarchy
		else:
			 # Reverse inner hierarchy
			suffStr='.'+'.'.join(innerHierarchy[::-1])
	
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

def save_all():
	"""
	Returns a save directive that saves all results the simulator normally 
	saves in its output (in HSPICE these are all node voltages and all currents 
	flowing through voltage sources and inductances). 
	"""
	return [ 'all' ]
	
def save_voltage(what):
	"""
	If *what* is a string it returns a save directive that instructs the 
	simulator to save the voltage of node named *what* in simulator output. 
	If *what* is a list of strings a multiple save directives are returned 
	instructing the simulator to save the voltages of nodes with names given 
	by the *what* list. 
	"""
	compiledList=[]
	
	if type(what) is list:
		input=what
	else:
		input=[what]
		
	for name in input:
		compiledList.append('v('+name+')')
	
	return compiledList
	
def save_current(what):
	"""
	If *what si a string it returns a save directive that instructs the 
	simulator to save the current flowing through instance names *what* in 
	simulator output. If *what* is a list of strings multiple save diretives 
	are returned instructing the simulator to save the currents flowing through 
	instances with names given by the *what* list. 
	"""
	compiledList=[]
	
	if type(what) is list:
		input=what
	else:
		input=[what]
	
	for name in input:
		compiledList.append('i('+name+')')
	
	return compiledList

def save_property(devices, params, indices=None):
	"""
	Saves the properties given by the list of property names in *params* of 
	instances with names given by the *devices* list. Also capable of handling 
	properties that are vectors (although currently SPICE OPUS devices have no 
	such properties). The indices of vector components that need to be saved is 
	given by the *indices* list. 
	
	If *params*, *devices*, and *indices* have n, m, and o memebrs, n*m*o save 
	directives are are returned describing all combinations of device name, 
	property name, and index. 
	
	If *indices* is not given, save directives for scalar device properties 
	are returned. Currently HSPICE devices have no vector properties. 
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
		raise Exception, "HSPICE does not support properties with indices."
	
	return compiledList

#
# Analysis command generators
#

def an_op():
	"""
	Generates the HSPICE simulator directive that invokes the operating point 
	analysis. 
	
	This is achieved with a trick - performing a DC sweep of a parameter named 
	``dummy__`` across only one point. The ``dummy__`` parameter is added to 
	the simulator input file automatically.
	"""
	# Sweep a dummy parameter dummy___, one point where dummy___=0
	return ('dc', 'sw', '.dc dummy___ poi 1 0')

def an_dc(start, stop, sweep, points, name, parameter, index=None):
	"""
	Generates the HSPICE simulator directive that invokes the operating point 
	sweep (DC) analysis. *start* and *stop* give the intial and the final 
	value of the swept parameter. 
	
	*sweep* can be one of 
	
	* ``'lin'`` - linear sweep with the number of points given by *points* 
	* ``'dec'`` - logarithmic sweep with points per decade 
	  (scale range of 1..10) given by *points*
	* ``'oct'`` - logarithmic sweep with points per octave 
	  (scale range of 1..2) given by *points*
	
	*name* gives the name of the instance whose *parameter* is swept. Because 
	HSPICE can sweep only independent voltage and current sources, these two 
	element types are the only ones allowed. Due to this the only allowed 
	value for parameter is ``dc``. 
	
	Because HSPICE knows no such thing as vector parameters, *index* should 
	never be used. 
	
	If *name* is not given a sweep of a circuit parameter (defined with 
	``.param``) is performed. The name of the parameter can be specified with 
	the *parameter* argument. If *parameter* is ``temperature`` a sweep of the 
	circuit's temperature is performed. 
	"""
	if index is None:
		if name is None:
			if parameter=='temperature':
				devStr='temp'
			else:
				devstr=str(parameter)
		else:
			if name[0].lower()!='v' and name[0].lower()!='i':
				raise Exception, "HSPICE can't sweep elements other than independent sources"
			if parameter is not None and parameter.lower()!='dc':
				raise Exception, "HSPICE can sweep only the dc parameter of independent sources"
			devStr=str(name)
	else:
		raise Exception, "HSPICE does not support vector parameter sweep"
	
	if sweep == 'lin':
		anstr='.dc '+devStr+' lin '+str(points)+' '+str(start)+' '+str(stop)
	elif sweep == 'dec':
		anstr='.dc '+devStr+' dec '+str(points)+' '+str(start)+' '+str(stop)
	elif sweep == 'oct':
		anstr='.dc '+devStr+' oct '+str(points)+' '+str(start)+' '+str(stop)
	else:
		raise Exception, "Bad sweep type."
	
	return ('dc', 'sw', anstr)

def an_ac(start, stop, sweep, points):
	"""
	Generats the HSPICE simulator directive that invokes the small signal (AC) 
	analysis. The range of the frequency sweep is given by *start* and *stop*. 
	*sweep* is one of
	
	* ``'lin'`` - linear sweep with the number of points given by *points* 
	* ``'dec'`` - logarithmic sweep with points per decade 
	  (scale range of 1..10) given by *points*
	* ``'oct'`` - logarithmic sweep with points per octave 
	  (scale range of 1..2) given by *points*
	"""
	if sweep == 'lin':
		anstr='.ac lin '+str(points)+' '+str(start)+' '+str(stop)
	elif sweep == 'dec':
		anstr='.ac dec '+str(points)+' '+str(start)+' '+str(stop)
	elif sweep == 'oct':
		anstr='.ac oct '+str(points)+' '+str(start)+' '+str(stop)
	else:
		raise Exception, "Bad sweep type."
	
	return ('ac', 'ac', anstr)

def an_tran(step, stop, start=0.0, maxStep=None, uic=False):
	"""
	Generats the HSPICE simulator directive that invokes the transient analysis. 
	The range of the time sweep is given by *start* and *stop*. *step* is the 
	intiial time step. 
	
	HSPICE does not support an upper limit on the time step. Therefore the 
	*maxStep* argument is ignored. 
	
	If the *uic* flag is set to ``True`` the initial conditions given by 
	``.ic`` simulator directives are used as the first point of the transient 
	analysis instead of the operating point analysis results. 
	"""
	if uic:
		anstr='.tran '+str(step)+' '+str(stop)+' start='+str(start)+' uic'
	else:
		anstr='.tran '+str(step)+' '+str(stop)+' start='+str(start)
	
	return ('tran', 'tr', anstr)

# Pts per summary is supported in HSPICE, but the results go in the text file. We don't want 
# to process them anyway so the results are noise spectra only and we ignore ptsSum. 
# A noise analysis is bundled with an ac analysis.
def an_noise(start, stop, sweep, points, input, outp, outn=None, ptsSum=1):
	"""
	Generats the HSPICE simulator directive that invokes the small signal noise 
	analysis. The range of the frequency sweep is given by *start* and *stop*. 
	*sweep* is one of
	
	* ``'lin'`` - linear sweep with the number of points given by *points* 
	* ``'dec'`` - logarithmic sweep with points per decade 
	  (scale range of 1..10) given by *points*
	* ``'oct'`` - logarithmic sweep with points per octave 
	  (scale range of 1..2) given by *points*
	
	*input* is the name of the independent voltage/current source with ``ac`` 
	parameter set to 1 that is used for calculating the input referred noise. 
	*outp* and *outn* give the voltage that is used as the output voltage. If 
	only *outp* is given the output voltage is the voltage at node *outp*. If 
	*outn* is also given, the output voltage is the voltage between nodes 
	*outp* and *outn*. 
	
	*ptsSum* is supported by HSPICE but the results go to a text file and are 
	not collected after the analysis. If it wasn't ignored we would specify it 
	as the third argument to ``.noise``. 
	
	A HSPICE noise analysis is an addition to the AC analysis. 
	"""
	if outn is None:
		outspec="v("+str(outp)+")"
	else:
		outspec="v("+str(outp)+","+str(outn)+")"
	if sweep=='lin':
		anstr='.ac '+' lin '+str(points)+" "+str(start)+" "+str(stop)
	elif sweep=='dec':
		anstr='.ac '+' dec '+str(points)+" "+str(start)+" "+str(stop)
	elif sweep=='oct':
		anstr='.ac '+' oct '+str(points)+" "+str(start)+" "+str(stop)
	else:
		raise Exception, "Bad sweep type."
	
	return ('noise', 'ac', anstr+'\n.noise '+outspec+' '+input)

	
class HSpice(Simulator):
	"""
	A class for interfacing with the HSPICE batch mode simulator. 
	
	*binary* is the path to the HSPICE simulator binary. If it is not given 
	the ``HSPICE_BINARY`` environmental variable is used as the path to the 
	HSPICE simulator binary. If ``HSPICE_BINARY`` is not defined the binary is 
	assumed to be in the current working directory. 
	
	*args* apecifies a list of additional arguments passed to the simulator 
	binary at startup. 
	
	If *debug* is greater than 0 debug messages are printed at the standard 
	output. If it is above 1 a part of the simulator output (.lis file) is 
	also printed. If *debug* is above 2 full simulator output is printed. 
	
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
	  precedence over the values passed to the :meth:`setInputParameters` method. 
	"""
	def __init__(self, binary=None, args=[], debug=0):
		Simulator.__init__(self, binary, args, debug)
		
		self._compile()
	
	def _compile(self):
		"""
		Prepares internal structures. 
		
		* dictionaries of functions for evaluating save directives and 
		  analysis commands
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
		
		# Default binary based on HSPICE_BINARY and platform
		if self.binary is None:
			if 'HSPICE_BINARY' in environ:
				hspicebinary=environ['HSPICE_BINARY']
			else:
				if platform.system()=='Windows':
					hspicebinary='hspice.exe'
				else:
					hspicebinary='hspice'
			
			self.binary=hspicebinary
	
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
				raise Exception, "Save directives must evaluate to a list of strings."
				
			for save in saveList:
				if type(save) is not str:
					raise Exception, "Save directives must evaluate to a list of strings."
			
			compiledList+=saveList
			
		# Make list memebers unique
		compiledSet=set(compiledList)
		
		# Find 'all' in set, make it the first element of the unique saves list
		if 'all' in compiledSet:
			compiledSet.remove('all')
			haveAll=True
			return ['all']+list(compiledSet)
		else:
			return list(compiledSet)
	
	#
	# Batch simulation
	#
		
	def writeFile(self, i):
		"""
		Prepares the simulator input file for running the *i*-th job group. 
		Because there is only one job group in HSPICE 0 is the only allowed 
		value of *i*. 
		
		Generates files 
		
		* ``simulatorID_analysis.lib`` - lists all analyses, one library 
		  section per analysis
		* ``simulatorID.sp`` - the main simulator input file
		
		These files must be generated every time new input parameter values 
		are set with the :meth:`setInputParaneters` method. 
		
		The ``simulatorID_analysis.lib`` file is a library file with sections 
		named ``anFileEndingIndex`` where ``FileEnding`` is the one returned 
		by analysis command generators and ``Index`` is the consecutive index 
		of the analysis of that type. 
		
		Every section has in ``simulatorID_analysis.lib`` consists of 
		
		* Simulator options (``.options`` simulator directive). 
		* The value of the ``temperature`` parameter in form of a ``.temp`` 
		  simulator directive. 
		* ``.options post=1`` which forces writing the results in binary 
		  output files. 
		* The valus of parameters in the form of ``.param`` dirrectives. The 
		  parameters specified in the corresponding job description take 
		  precedence over input parameter values. 
		* Save directives (``.probe`` simulator directive). If at least one 
		  save directive is specified, the ``all()`` directive is not used, 
		  and the analysis is not an AC analysis the ``.options probe=1`` 
		  directive is added. This instructs the simulator to save only those 
		  results that are specified with save directives. 
		
		The ``simulatorID.sp`` file invokes individual jobs. The first job 
		starts with a ``.title`` simulator directive giving the job name as 
		the title. All other jobs start with an ``.alter`` directive giving 
		their corresponding job names. 
		
		Every ``.title``/``.alter`` directive is followed by ``.del lib`` and 
		``.lib`` directives that include a section of the topology file and 
		the section of the the ``simulatorID_analysis.lib`` file that 
		correspond to the job. 
		
		All output files with simulation results are files with endings 
		comprising 
		
		* the file ending returned by the analysis command generator and 
		* the consecutive index of the analysis. 
		
		All output files are in HSPICE binary file format. 
		
		The function returns the name of the main simulator input file. 
		"""
		# Because the user is always presented with only one job group, 
		# raise exception if i is not 0. 
		if i!=0:
			raise Exception, DbgMsg("HSSI", "Bad job group index (not 0).")
		
		# Write analysis file
		analysisFileName=self.simulatorID+'_analysis.lib'
		
		if self.debug>0:
			DbgMsgOut("HSSI", "Writing analyses to '"+analysisFileName+"'.") 
		
		f=open(analysisFileName, 'w')
		f.write('* HSPICE analysis library\n\n')
		
		# Prepare file ending counter
		anCodeCount={}
		
		# Prepare file ending list for jobs. 
		self.jobIndex2fileEnding=[""]*len(self.jobList)
		
		# Traverse all jobs in flas job sequence. 
		for jobIndex in self.flatJobSequence: 
			# Get job. 
			job=self.jobList[jobIndex]
			
			if self.debug>0:
				DbgMsgOut("HSSI", "  '"+str(job['name'])+"'")
			
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
			
			# Prepare parameters dictionary for local namespace
			self.analysisLocals['param'].clear()
			self.analysisLocals['param'].update(analysisParams)
			
			# Evaluate analysis statement
			(anType, anCode, anCommand)=eval(job['command'], globals(), evalEnv)
			
			# File ending
			if anCode not in anCodeCount:
				anCodeCount[anCode]=0
			else:
				anCodeCount[anCode]+=1
			fileEnding=anCode+str(anCodeCount[anCode])
			
			# Store file ending
			self.jobIndex2fileEnding[jobIndex]=fileEnding 
						
			f.write('* Analysis: '+job['name']+'\n')
			f.write('.lib an'+fileEnding+'\n')
			
			# Write options
			if 'options' in job:
				for (option, value) in job['options'].iteritems():
					if value is True:
						f.write('.options '+str(option)+'=1\n')
					elif value is False:
						f.write('.options '+str(option)+'=0\n')
					else:
						f.write('.options '+str(option)+'='+str(value)+'\n')
			
			# Force writing the results to binary files
			f.write('.options post=1\n')
			
			# Write parameters
			for (name, value) in analysisParams.iteritems():
				if name=='temperature':
					f.write('.temp '+str(value)+'\n')
				else:
					f.write('.param '+name+'='+str(value)+'\n')
			
			# Generate saves 
			if 'saves' in job:
				savesList=self._createSaves(job['saves'], job['variables'])
		
			# Write saves
			# Assume that 'all' is first in compiled saves list
			if len(savesList)>0: 
				# Force storing only saves if 'all' is missing
				if savesList[0]!='all' and anType!='ac': 
					# Save only those quantities that are listed under .probe
					# This can't be done for AC analysis because probe saves only real values
					f.write('.options probe=1\n')
					saves=savesList
				elif anType=='ac':
					# In case of AC analysis ignore saves, save everything
					saves=[]
				else:
					# Skip 'all'
					saves=savesList[1:]
				
				# Are there any saves left
				if len(saves)>0:
					# Write 8 saves at a time
					count=0
					for save in saves:
						if count == 0:
							f.write('.probe '+anType)
						f.write(' '+save)
						count+=1
						if count == 8:
							count=0
							f.write('\n')
					f.write('\n')
			
			# Write analysis
			f.write(anCommand+'\n')
			
			f.write('.endl an'+fileEnding+'\n\n')
			
		f.close()

		# Write main file
		fileName=self.simulatorID+'.sp'
		
		if self.debug>0:
			DbgMsgOut("HSSI", "Writing top level file to '"+fileName+"'.")
		
		f=open(fileName, 'w')
		
		# First line
		f.write('* HSPICE simulator input file.\n\n')
		
		# Add dummy parameter for operating point calculation
		f.write('.param dummy___=0\n\n')
		
		# Add analyses
		first=True
		for jobIndex in self.flatJobSequence:
			# Get job. 
			job=self.jobList[jobIndex]
			
			if self.debug>0:
				DbgMsgOut("HSSI", "  '"+str(job['name'])+"'")
			
			# Set alter or title
			f.write('* Analysis: '+job['name']+'\n')
			if first:
				f.write('.title '+job['name']+'\n')
				
			else:
				f.write('.alter '+job['name']+'\n')
			
			# Write topology
			if first or self.jobIndex2topologyIndex[jobIndex]!=topologyIndex:
				if not first:
					# For any topology change, delete the old topology
					f.write('.del lib \''+self.simulatorID+'_topology.lib\' top'+str(topologyIndex)+'\n')
				topologyIndex=self.jobIndex2topologyIndex[jobIndex]
				f.write('.lib \''+self.simulatorID+'_topology.lib\' top'+str(topologyIndex)+'\n')
			
			# Write analysis
			if not first:
				f.write('.del lib \''+self.simulatorID+'_analysis.lib\' an'+fileEnding+'\n')
			fileEnding=self.jobIndex2fileEnding[jobIndex]
			f.write('.lib \''+self.simulatorID+'_analysis.lib\' an'+fileEnding+'\n\n')
			
			# Handled first analysis
			if first:
				first=False
		
		# Write .end
		f.write('.end\n')
		
		f.close()

		return fileName

	def writeTopology(self):
		"""
		Creates the topology file. The file is named 
		``simulatorID_topology.lib`` and is a library file with one section 
		corresponding to the circuit description of one group of jobs with 
		a common circuit definition. 
		
		Sections of the library are named ``topIndex`` where ``Index`` is the 
		index of the group of jobs in the job list. 
		
		Every section consists of ``.include`` and ``.lib`` simulator 
		directives corresponding to system description modules given in the 
		job description. 
		
		The topology file does not depend on the input parameter values. 
		Therefore it is created only once for every job list that is supplied 
		with the :meth:`setJobList` method. 
		"""
		# Write the topology file, store topology section names in job
		fileName=self.simulatorID+'_topology.lib'
		
		if self.debug>0:
			DbgMsgOut("HSSI", "Writing topology file to '"+fileName+"'.")
		
		f=open(fileName, 'w')
		f.write('* HSPICE topology library\n\n')
		
		# Prepare topology index list for jobs. 
		self.jobIndex2topologyIndex=[0]*len(self.jobList)
		
		# Traverse all job groups
		topologyIndex=0
		for jobGroup in self.jobSequence:
			# Get representative job (first in job group)
			jobIndex=jobGroup[0]
			job=self.jobList[jobIndex] 
			
			f.write('.lib top'+str(topologyIndex)+'\n')
			
			# Go through all definitions
			for definition in job['definitions']:
				if 'section' in definition:
					f.write('.lib \''+definition['file']+'\' '+definition['section']+'\n')
				else:
					f.write('.include \''+definition['file']+'\'\n')
			
			# Store topology index for all jobs in this group. 
			for jobIndex in jobGroup:
				self.jobIndex2topologyIndex[jobIndex]=topologyIndex
			
			f.write('.endl top'+str(topologyIndex)+'\n\n')
			
			# Next topology index
			topologyIndex+=1
		
		f.close()
		
	def cleanupResults(self, i): 
		"""
		Removes all result files that were produced during the simulation of 
		the *i*-th job group. Because the user is always presented with only 
		one job group, 0 is the only allowed value of *i*. 
		
		Simulator input files are left untouched. 
		""" 
		if i!=0:
			raise Exception, DbgMsg("HSSI", "Bad job group index (not 0).")
		
		if self.debug>0:
			DbgMsgOut("HSSI", "Cleaning up.")
		
		# Remove old results files
		for jobIndex in self.flatJobSequence:
			fileEnding=self.jobIndex2fileEnding[jobIndex]
			try:
				os.remove(self.simulatorID+'.'+fileEnding)
			except KeyboardInterrupt:
				DbgMsgOut("HSSI", "Keyboard interrupt.")
				raise
			except:
				None
			
	def runFile(self, fileName):
		"""
		Runs the simulator on the main simulator input file given by 
		*fileName*. 
		
		Returns ``True`` if the simulation finished successfully. 
		This does not mean that any results were produced. 
		It only means that the return code from the simuator was 0 (OK). 
		"""
		if self.debug>0:
			DbgMsgOut("HSSI", "Running file '"+fileName+"'.") 
				
		# Run the file
		spawnOK=True
		p=None
		try:
			# Start simulator, input file is fileName, output file is fileName less last 3 chars (.sp)
			if platform.system()=='Windows':
				# The HSPICE window should not appear
				info = subprocess.STARTUPINFO()
				info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
				info.wShowWindow = subprocess.SW_HIDE
				
				p=subprocess.Popen(
					[self.binary]+self.cmdline+['-i', fileName, '-o', fileName[:-3]], 
					startupinfo=info
				)
			else:
				p=subprocess.Popen(
					[self.binary]+self.cmdline+['-i', fileName, '-o', fileName[:-3]], 
					universal_newlines=True, 
					stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE
				)
				
				# Collect output
				self.messages=p.stdout.read()
			
			# Now wait for the process to finish. If we don't wait p might get garbage-collected before the
			# actual process finishes which can result in a crash of the interpreter. 
			retcode=p.wait()
			
			# In windows nothing goes to stdout so we must read the .lis file
			if platform.system()=='Windows':
				# We read the .lis file only in debug mode
				if self.debug>1:
					f=open(fileName[:-3]+'.lis', 'r')
					self.messages=f.read()
					f.close()
				else:
					self.messages=''
			
			if self.debug>2:
				DbgMsgOut("HSSI", self.messages)
			elif self.debug>1:
				DbgMsgOut("HSSI", self.messages[-400:])
			
			# Check return code. Nonzero return code means that something has gone bad. 
			# At least the simulator says so. 
			if retcode!=0:
				spawnOK=False
				
				if self.debug>0:
					DbgMsgOut("HSSI", "Spawn process FAILED.") 
					
		except KeyboardInterrupt:
			DbgMsgOut("HSSI", "Keyboard interrupt.") 
			
			# Will raise an exception if process exits before kill() is called.
			try:
				p.kill()
			except:
				pass
			
			raise KeyboardInterrupt
		except:
			if self.debug>0:
				ei=exc_info()
				if self.debug>1:
					for line in format_exception(ei[0], ei[1], ei[2]):
						DbgMsgOut("HSSI", "Exception: "+line) 
				else:
					for line in format_exception_only(ei[0], ei[1]):
						DbgMsgOut("HSSI", "Exception: "+line) 

			spawnOK=False
		
		if not spawnOK and self.debug>0:
			DbgMsgOut("HSSI", "  run FAILED")
		
		return spawnOK
	
	def runJobGroup(self, i):
		"""
		Runs the *i*-th job group. Because the user is always presented with 
		only one job group, 0 is the only allowed value of *i*. 
		
		If a fresh job list is detected a new topology file is created by 
		invoking the :meth:`writeTopology` method. Next the analysis library 
		file and the main simulator input file are created by the 
		:meth:`writeFile` method. 
		
		The :meth:`cleanupResults` method removes any old results produced by 
		previous runs of the jobs. 
		
		Finally the :meth:`runFile` method is invoked. Its return value is 
		stored in the :attr:`lastRunStatus` member. 
		
		The function returns a tuple (*jobIndices*, *status*) where 
		*jobIndices* is a list of job indices corresponding to the jobs that 
		were simulated. *status* is the status returned by the :meth:`runFile` 
		method. 
		"""
		# raise exception if i is not 0. 
		if i!=0:
			raise Exception, DbgMsg("HSSI", "Bad job group index (not 0).")
		
		# If job list is fresh, write topology file. Mark job list as not freash. 
		if self.jobListFresh:
			self.writeTopology()
			self.jobListFresh=False
		
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
		Collects the results produces by running jobs with indices given by the 
		*indices* list. *runOK* specifies the status returned by the 
		:meth:`runJobGroup` method which produced the results. 
		
		If *runOK* is ``False`` the result groups of the jobs with indices 
		given by *indices* are set to ``None``. 
		
		A results group corresponding to some job is set to ``None`` if the 
		result file is not successfully loaded. 
		"""
		if runOK is None:
			runOK=self.lastRunStatus
		
		if runOK:
			# Collect results
			count=0
			countOK=0
			if self.debug>1: 
				DbgMsgOut("HSSI", "Reading results")
				
			for jobIndex in indices:
				job=self.jobList[jobIndex]
				fileEnding=self.jobIndex2fileEnding[jobIndex]
			
				if self.debug>1: 
					DbgMsgOut("HSSI", "  from '"+self.simulatorID+'.'+fileEnding+"' for job '"+job["name"]+"'.")
				
				self.results[jobIndex]=hspice_read(self.simulatorID+'.'+fileEnding)
				if self.results[jobIndex] is not None:
					countOK+=1
				count+=1
			
			if self.debug>0:
				DbgMsgOut("HSSI", "  "+str(countOK)+"/"+str(count)+" analyses OK")
		else:
			# Simulator failed, no results
			for jobIndex in indices:
				self.results[jobIndex]=None
			
	def jobGroupCount(self):
		"""
		Returns the number of job groups. 
		"""
		return 1
	
	def jobGroup(self, i):
		"""
		Returns a list of job indices corresponding to the jobs in *i*-th job 
		group. 
		
		Because the user is always presented with only one job group, only 0 is 
		allowed for the value of *i*. 
		"""
		# Because the user is always presented with only one job group, 
		# raise exception if i is not 0. 
		if i!=0:
			raise Exception, DbgMsg("HSSI", "Bad job group index (not 0).")
		
		return self.flatJobSequence

	
	#
	# Job optimization
	#
	
	def unoptimizedJobSequence(self):
		"""
		Returns the unoptimized internal job sequence. If there are n jobs in 
		the job list the following list of lists is returned: 
		``[[0], [1], ..., [n-1]]``. 
		This means we have n job groups with every one of them holding one job. 
		
		Also stores the flat job sequence in the :attr:`flatJobSequence` 
		member. A flat job sequence is a list of jobs appearing in the order 
		in which they will be simulated. In this case the flat job list is 
		[0, 1, ..., n-1]. The flast job sequence is the first and only job 
		group which is presented to the user. 
		"""
		seq=[[0]]*len(self.jobList)
		for i in range(len(self.jobList)):
			seq[i]=[i];
		
		# Store flat job sequence, so we don't have to rebuild it every time. 
		self.flatJobSequence=range(len(self.jobList))
		
		return seq
	
	def optimizedJobSequence(self):
		"""
		Returns the optimized internal job sequence. It has as many job groups 
		as there are different circuit topologies (lists of system description 
		modules) in the job list. Jobs in one job group share the same circuit 
		topology. They are ordered by their indices with lowest job index 
		listed as the first in the group. 
		
		Also stores the flat job sequence in the :attr:`flatJobSequence` 
		member. A flat job sequence is a list of jobs appearing in the order 
		in which they will be simulated. In this case the flat job sequence 
		is actually the flattened version of the optimized internal job 
		sequence. 
		"""
		# Count jobs
		jobCount=len(self.jobList)
		
		# Construct a list of job indices
		candidates=set(range(jobCount))
		
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
				# Compare jobs, join them if the definitions are identical.  
				if (self.jobList[i1]['definitions']==self.jobList[i2]['definitions']):
					# Job i2 can be joined with job i1, add it to jobGroup
					jobGroup.append(i2)
					# Remove i2 from candidates
					candidates.remove(i2)
			
			# Sort jobGroup
			jobGroup.sort()
			
			# Append it to job sequence
			seq.append(jobGroup)
		
		# Store flat job sequence, so we don't have to rebuild it every time. 
		self.flatJobSequence=[]
		for jobGroup in seq:
			self.flatJobSequence.extend(jobGroup)
				
		return seq

	
	#
	# Retrieval of simulation results
	#
	
	def res_voltage(self, node1, node2=None, resIndex=0):
		"""
		Retrieves the voltage corresponding to *node1* (voltage between nodes 
		*node1* and *node2* if *node2* is also given) from the *resIndex*-th 
		simulation result in the active result group. 
		
		Because HSPICE output files always contain the result of only one 
		analysis, *resIndex* should always be 0. 
		"""
		if node2 is None:
			return self.activeResult[resIndex][0][2][0][node1]
		else:
			return self.activeResult[resIndex][0][2][0][node1]-self.activeResult[resIndex][0][2][0][node2]
	
	def res_current(self, name, resIndex=0):
		"""
		Retrieves the current flowing through instance *name* from the 
		*resIndex*-th simulation result in the active result group. 
		
		Because HSPICE output files always contain the result of only one 
		analysis, *resIndex* should always be 0. 
		"""
		return self.activeResult[resIndex][0][2][0]['i('+name]
		
	def res_property(self, name, parameter, index=None, resIndex=0):
		"""
		Retrieves the property named *parameter* belonging to instance named 
		*name*. The property is retrieved from *resIndex*-th plot in the 
		active result group. 
		
		Because HSPICE knows no vector properties *index* is ignored. 
		
		Note that this works only of the property was saved with a 
		corresponding save directive. 
		
		Because HSPICE output files always contain the result of only one 
		analysis, *resIndex* should always be 0. 
		"""
		if index is None:
			return self.activeResult[resIndex][0][2][0][parameter+'('+name]
		else:
			raise Exception, "HSPICE does not support properties with indices"
	
	def res_noise(self, reference, name=None, contrib=None, resIndex=0):
		"""
		Retrieves the noise spectrum density of contribution *contrib* of 
		instance *name* to the input/output noise spectrum density. 
		*reference* can be ``'input'`` or ``'output'``. 
		
		If *name* and *contrib* are not given the output or the equivalent 
		input noise spectrum density is returned (depending on the value of 
		*reference*). 
		
		Because HSPICE output files always contain the result of only one 
		analysis, *resIndex* should always be 0. 
		"""
		# TODO: units of total/partial input/output spectra (V**2/Hz or V**2)
		if name is None:
			# Input/output noise spectrum
			if reference=='input':
				spec=self.activeResult[resIndex][0][2][0]['innoise']
			elif reference=='output':
				spec=self.activeResult[resIndex][0][2][0]['outnoise']
			else:
				raise Exception, "Bad noise reference."
		else:
			# Partial spectrum
			if reference=='input':
				A=self.activeResult[resIndex][0][2][0]['innoise']/self.activeResult[resIndex][0][2][0]['outnoise']
			elif reference=='output':
				A=1.0
			else:
				raise Exception, "Bad noise reference."
			
			spec=self.activeResult[resIndex][0][2][0][str(contrib)+'('+str(name)]*A
	
		return spec
		
	def res_scale(self, resIndex=0):
		"""
		Retrieves the default scale of the *resIndex*-th plot in the active 
		result group. 
		
		Because HSPICE output files always contain the result of only one 
		analysis, *resIndex* should always be 0. 
		"""
		scaleName=self.activeResult[resIndex][1]
		return self.activeResult[resIndex][0][2][0][scaleName]
	
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
	
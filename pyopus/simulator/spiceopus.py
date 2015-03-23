"""
.. inheritance-diagram:: pyopus.simulator.spiceopus
    :parts: 1

**SPICE OPUS batch mode interface (PyOPUS subsystem name: SOSI)**

SPICE OPUS is a free Berkeley SPICE3-based simulator. It is capable of 
interactive operation but this module uses it in batch mode. This means that 
none of the advanced interactive features of SPICE OPUS are used. 

SPICE OPUS in batch mode is not capable of changing the circuit's parameters 
or its topology (system definition) without restarting the simulator and 
loading a new input file. 

An exception to this is the ``temperature`` parameter which represents the 
circuit's temperature in degrees centigrade (``.option temp=...`` simulator 
directive) and can be changed without restarting the simulator. Consequently 
the ``temp`` simulator option is not allowed to appear in the simulator 
options list. 

All simulator options (``.option`` directive) can be changed interactively 
without the need to restart the simulator and load a new input file. This 
leaves very little space for job list optimization. Nevertheles there are 
still some advantages to be gained from an optimized job list. 

A job sequence in SPICE OPUS is a list of lists containing the indices of jobs 
belonging to individual job groups. 

One result group can consist of multiple plots. See 
:mod:`pyopus.simulator.rawfile` module for the details on the result files and 
plots in SPICE OPUS. 
"""

# SPICE OPUS simulator interface

# Benchmark result on opamp, OPUS, Windows XP 32bit AMD64 farm
#	131 iterations, best in 129, final cost -0.106015891203
#	33.576s/36.315s = 89.9% time spent in simulator

# Benchmark result on opamp, OPUS, Windows XP 32bit
#	131 iterations, best in 129, final cost -0.106015891203
#	45.205s/50.302s = 89.9% time spent in simulator

# Benchmark result on opamp, OPUS interactive, Windows XP 32bit
#	131 iterations, best in 129, final cost -0.104213303008
#	(23.776s+13.819s)/41.856s = 89.8% time spent in simulator

# Benchmark result on opamp, OPUS, Linux AMD64
#	131 iterations, best in 129, final cost -0.106015891203
#	34.175s/36.411s = 93.9% time spent in simulator

# Benchmark result on opamp, OPUS interactive, Linux AMD64 farm
#	131 iterations, best in 129, final cost -0.104213303008
#	32.550s/34.411s = 94.6% time spent in simulator


# Comparison of restart and interactive mode in python
# py  100iter		34.978 (with cost evaluation)
# pyi 100iter		25.140s (with cost evaluation)
#
# restart mode spends 9.838s more time (39% extra wrt interactive)
#
# Comparison of interactive mode in nutmeg script (.control) and in python
# pyi 100iter		25.140s (with cost evaluation)
# spi 100iter		22.422s (no cost evaluation)
# 
# python spent 2.718s for its own things (cost evaluation included)
# that is 12% of pure spice time
#
# profile shows that 4.24s per 100iter are spent for cost evaluation

import subprocess
from base import Simulator
from rawfile import raw_read
import os
import platform
from ..misc.env import environ
from ..misc.debug import DbgMsgOut, DbgMsg

__all__ = [ 'ipath', 'save_all', 'save_voltage', 'save_current', 'save_property', 
			'an_op', 'an_dc', 'an_ac', 'an_tran', 'an_noise', 'SpiceOpus' ] 

import pdb

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
	  inside ``x2``. Returns ``'m1:x1:x2'``. 
	* ``ipath('x1', innerHierarchy=['m0', 'x0'])`` - instance ``m0`` inside 
	  ``x0`` inside ``x1``. Returns ``'m0:x0:x1'``. 
	* ``ipath(['m1', 'm2'], ['x1', 'x2']) - instances ``m1`` and ``m2`` inside 
	  ``x1`` inside ``x2``. Returns ``['m1:x1:x2', 'm2:x1:x2']``. 
	* ``ipath(['xm1', 'xm2'], ['x1', 'x2'], 'm0')`` - instances named ``m0`` 
	  inside paths ``xm1:x1:x2``  and ``xm2:x1:x2``. Returns 
	  ``['m0:xm1:x1:x2', 'm0:xm2:x1:x2']``. 
	"""
	# Create outer and inner path
	
	# Outer hierarchy is represented by a suffix
	if outerHierarchy is None:
		suffStr=''
	else:
		if type(outerHierarchy ) is str:
			suffStr=':'+outerHierarchy
		else:
			suffStr=':'+':'.join(outerHierarchy)
	
	# Inner hierarchy is represented by a prefix
	if innerHierarchy is None:
		prefStr=''
	else:
		if type(innerHierarchy) is str:
			prefStr=innerHierarchy+':'
		else:
			prefStr=':'.join(innerHierarchy)+':'
	
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
	saves in its output (in SPICE OPUS these are all node voltages and all 
	currents flowing through voltage sources and inductances). 
	
	Equivalent of SPICE OPUS ``save all`` simulator command. 
	"""
	return [ 'all' ]
	
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
		compiledList.append('v('+name+')')
	
	return compiledList
	
def save_current(what):
	"""
	If *what si a string it returns a save directive that instructs the 
	simulator to save the current flowing through instance names *what* in 
	simulator output. If *what* is a list of strings multiple save diretives 
	are returned instructing the simulator to save the currents flowing 
	through instances with names given by the *what* list. 
	
	Equivalent of SPICE OPUS ``save i(what)`` simulator command. 
	"""
	compiledList=[]
	
	if type(what) is list:
		input=what
	else:
		input=[what]
	
	for name in input:
		compiledList.append(name+'#branch')
	
	return compiledList

def save_property(devices, params, indices=None):
	"""
	Saves the properties given by the list of property names in *params* of 
	instances with names given by the *devices* list. Also capable of handling 
	properties that are vectors (although currently SPICE OPUS devices have no 
	such properties). The indices of vector components that need to be saved 
	is given by the *indices* list. 
	
	If *params*, *devices*, and *indices* have n, m, and o memebrs, n*m*o save 
	directives are are returned describing all combinations of device name, 
	property name, and index. 
	
	If *indices* is not given, save directives for scalar device properties 
	are returned. Currently SPICE OPUS devices have no vector properties. 
	
	Equvalent of SPICE OPUS ``save @device[property]`` (or in case the 
	property is a vector ``save @device[property][index]``) simulator command. 
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
				compiledList.append('@'+name+'['+param+']')
	else:
		if type(indices) is list:
			inputIndices=indices
		else:
			inputIndices=[indices]
	
		for name in inputDevices:
			for param in inputParams:
				for i in inputIndices:
					compiledList.append('@'+name+'['+param+']['+str(i)+']')
	
	return compiledList


#
# Analysis command generators
#

def an_op():
	"""
	Generates the SPICE OPUS simulator command that invokes the operating 
	point analysis. 
	
	Equivalent of SPICE OPUS ``op`` simulator command. 
	"""
	return 'op'

def an_dc(start, stop, sweep, points, name, parameter, index=None):
	"""
	Generates the SPICE OPUS simulator command that invokes the operating point 
	sweep (DC) analysis. *start* and *stop* give the intial and the final value 
	of the swept parameter. 
	
	*sweep* can be one of 
	
	* ``'lin'`` - linear sweep with the number of points given by *points* 
	* ``'dec'`` - logarithmic sweep with points per decade 
	  (scale range of 1..10) given by *points*
	* ``'oct'`` - logarithmic sweep with points per octave 
	  (scale range of 1..2) given by *points*
	
	*name* gives the name of the instance whose *parameter* is swept. If the 
	parameter is a vector parameter *index* gives the integer index (zero 
	based) of the vector's component that will be swept. 
	
	Equivalent of SPICE OPUS ``dc @name[param][index] start stop sweep points`` 
	simulator command. 
	"""
	if index is None:
		if name is None:
			if parameter=='temperature':
				devStr='@@@temp'
			else:
				raise Exception, DbgMsg("SOSI", "Bad sweep parameter.")
		else:
			devStr='@'+str(name)+'['+str(parameter)+']'
	else:
		devStr='@'+str(name)+'['+str(parameter)+']['+str(index)+']'
	
	if sweep == 'lin':
		return 'dc '+devStr+' '+str(start)+' '+str(stop)+' lin '+str(points)
	elif sweep == 'dec':
		return 'dc '+devStr+' '+str(start)+' '+str(stop)+' dec '+str(points)
	elif sweep == 'oct':
		return 'dc '+devStr+' '+str(start)+' '+str(stop)+' oct '+str(points)
	else:
		raise Exception, DbgMsg("SOSI", "Bad sweep type.")

def an_ac(start, stop, sweep, points):
	"""
	Generats the SPICE OPUS simulator command that invokes the small signal 
	(AC) analysis. The range of the frequency sweep is given by *start* and 
	*stop*. *sweep* is one of
	
	* ``'lin'`` - linear sweep with the number of points given by *points* 
	* ``'dec'`` - logarithmic sweep with points per decade 
	  (scale range of 1..10) given by *points*
	* ``'oct'`` - logarithmic sweep with points per octave 
	  (scale range of 1..2) given by *points*
	
	Equivalent of SPICE OPUS ``ac sweep points start stop`` simulator command. 
	"""
	if sweep == 'lin':
		return 'ac lin '+str(points)+' '+str(start)+' '+str(stop)
	elif sweep == 'dec':
		return 'ac dec '+str(points)+' '+str(start)+' '+str(stop)
	elif sweep == 'oct':
		return 'ac oct '+str(points)+' '+str(start)+' '+str(stop)
	else:
		raise Exception, DbgMsg("SOSI", "Bad sweep type.")
		
def an_tran(step, stop, start=0.0, maxStep=None, uic=False):
	"""
	Generats the SPICE OPUS simulator command that invokes the transient 
	analysis. The range of the time sweep is given by *start* and *stop*. 
	*step* is the intiial time step. The upper limit on the time step is given 
	by *maxStep*. If the *uic* flag is set to ``True`` the initial conditions 
	given by ``.ic`` simulator directives and initial conditions specified as 
	instance parameters (e.g. ``ic`` paraneter of capacitor) are used as the 
	first point of the transient analysis instead of the operating point 
	analysis results. 
	
	If *uic* is ``True`` and *maxStep* is not given, the default value 
	*maxStep* is *step*. 
	
	Equivalent of SPICE OPUS ``tran step stop start maxStep [uic]`` simulator 
	command. 
	"""
	if uic:
		if maxStep is None:
			maxStep=step
		return 'tran '+str(step)+" "+str(stop)+" "+str(start)+" "+str(maxStep)+" uic"
	else:
		if maxStep is None:
			return 'tran '+str(step)+" "+str(stop)+" "+str(start)
		else:
			return 'tran '+str(step)+" "+str(stop)+" "+str(start)+" "+str(maxStep)

def an_noise(start, stop, sweep, points, input, outp, outn=None, ptsSum=1):
	"""
	Generats the SPICE OPUS simulator command that invokes the small signal 
	noise analysis. The range of the frequency sweep is given by *start* and 
	*stop*. sweep* is one of
	
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
	*outp* and *outn*. *ptsSum* gives the number of points pwer summary 
	(integrated noise) vector. 
	
	Equivalent of SPICE OPUS 
	``noise outspec input sweep points start stop ptsSum`` 
	simulator command. 
	"""
	if outn is None:
		outspec="v("+str(outp)+")"
	else:
		outspec="v("+str(outp)+","+str(outn)+")"
	if sweep=='lin':
		anstr='noise '+outspec+" "+str(input)+' lin '+str(points)+" "+str(start)+" "+str(stop)+" "+str(ptsSum)
	elif sweep=='dec':
		anstr='noise '+outspec+" "+str(input)+' dec '+str(points)+" "+str(start)+" "+str(stop)+" "+str(ptsSum)
	elif sweep=='oct':
		anstr='noise '+outspec+" "+str(input)+' oct '+str(points)+" "+str(start)+" "+str(stop)+" "+str(ptsSum)
	else:
		raise Exception, DbgMsg("SOSI", "Bad sweep type.")
	
	return anstr+"\nsetplot previous"

	
class SpiceOpus(Simulator):
	"""
	A class for interfacing with the SPICE OPUS simulator in batch mode. 
	
	*binary* is the path to the SPICE OPUS simulator binary. If it is not given 
	the ``OPUSHOME`` environmental variable is used as the path to the SPICE 
	OPUS installation. The simulator is assumed to be in ``OPUSHOME/bin``. 
	If ``OPUSHOME`` is not defined the binary is assumed to be in the current 
	working directory. 
	
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
	"""
	def __init__(self, binary=None, args=[], debug=0):
		Simulator.__init__(self, binary, args, debug)
		
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
			'ipath': ipath
		}
		
		# Local namespace for analysis evaluation
		self.analysisLocals={
			'op': an_op, 
			'dc': an_dc, 
			'ac': an_ac, 
			'tran': an_tran, 
			'noise': an_noise, 
			'ipath': ipath, 
			'param': {}
		}
		
		# Default binary based on OPUSHOME and platform
		if self.binary is None:
			if 'OPUSHOME' in environ:
				opuspath=os.path.join(environ['OPUSHOME'], 'bin') 
			else:
				opuspath='.'
			
			if platform.system()=='Windows':
				opusbinary=os.path.join(opuspath, 'spiceopus.exe')
			else:
				opusbinary=os.path.join(opuspath, 'spiceopus.bin')
				
			self.binary=opusbinary
	
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
		
	def _createSaves(self, saveDirectives):
		"""
		Creates a list of save directives by evaluating the members of the 
		*saveDirectives* list. 
		"""
		compiledList=[]
		
		for saveDirective in saveDirectives:
			# A directive must be a string that evaluates to a list of strings
			saveList=eval(saveDirective, globals(), self.saveLocals)

			if type(saveList) is not list: 
				raise Exception, DbgMsg("SOSI", "Save directives must evaluate to a list of strings.")
				
			for save in saveList:
				if type(save) is not str:
					raise Exception, DbgMsg("SOSI", "Save directives must evaluate to a list of strings.")
			
			compiledList+=saveList
			
		# Make list memebers unique
		return list(set(compiledList))

	#
	# Batch simulation
	#
	
	def writeFile(self, i):
		"""
		Prepares the simulator input file for running the *i*-th job group. 
		
		The file is named ``simulatorID_group_i.cir`` where *i* is the index 
		of the job group. 
		
		All output files with simulation results are .raw files in binary 
		format. 
		
		System description modules are converted to ``.include`` and ``.lib`` 
		simulator directives. 
		
		Simulator options are set with the ``set`` simulator command. 
		Integer, real, and string simulator options are converted with the 
		:meth:`__str__` method before they are written to the file. Boolean 
		options are converted to ``set`` or ``unset`` commands depending on 
		whether they are ``True`` or ``False``. 
		
		The parameters set with the last call to :meth:`setInputParameters` 
		method are joined with the parameters in the job description. The 
		values from the job description take precedence over the values 
		specified with the :meth:`setInputParameters` method. All parameters 
		are written to the input file in form of ``.param`` simulator directives. 
		
		The ``temperature`` parameter is treated differently. It is written to 
		the input file in form if a ``set`` simulator command preceding its 
		corresponding analysis command. 
		
		Save directives are written as a series of ``save`` simulator commands. 
		
		Every analysis command is evaluated in its corresponding environment 
		taking into account the parameter values passed to the 
		:meth:`setInputParameters` method. 
		
		Every analysis is followed by a ``write`` command that stores the 
		results in a file named ``simulatorID_job_j_jobName.raw`` where *j* 
		denotes the job index from which the analysis was generated. *jobName* 
		is the ``name`` member of the job description. 
		
		The function returns the name of the simulator input file it generated. 
		"""
		# Build file name
		fileName=self.simulatorID+"_group"+str(i)+'.cir'
		
		if self.debug>0:
			DbgMsgOut("SOSI", "Writing job group '"+str(i)+"' to file '"+fileName+"'")
			
		f=open(fileName, 'w')
		
		# First line
		f.write('* Simulator input file for job group '+str(i)+'\n\n')
		
		# Job group
		jobGroup=self.jobGroup(i)
		
		# Representative job
		repJob=self.jobList[jobGroup[0]]

		# Include definitions
		for definition in repJob['definitions']:
			if 'section' in definition:
				f.write('.lib \''+definition['file']+'\' '+definition['section']+'\n')
			else:
				f.write('.include \''+definition['file']+'\'\n')
		
		# Write representative options (as .option directives)
		if 'options' in repJob:
			for (option, value) in repJob['options'].iteritems():
				if value is True:
					f.write('.option '+option+'\n')
				else:
					f.write('.option '+option+'='+str(value)+'\n')

		# Prepare representative parameters dictionary. 
		# Case: input parameters get overriden by job parameters - default
		params={}
		params.update(self.inputParameters)
		if 'params' in repJob:
			params.update(repJob['params'])
			
		# Case: job parameters get overriden by input parameters - unimplemented

		# Write representative parameters, handle temperature as simulator option. 
		for (param, value) in params.iteritems():
			if param!="temperature":
				f.write('.param '+param+'='+str(value)+'\n')
			else:
				f.write('.option temp='+str(value)+'\n')

		# Control block
		f.write('\n');
		f.write('.control\n')
		f.write('unset *\n')
		f.write('delete all\n\n')
		f.write('set filetype=binary\n\n')
		
		# Handle analyses
		for j in jobGroup:
			# Get job
			job=self.jobList[j]
			
			# Get job name
			if self.debug>0:
				DbgMsgOut("SOSI", "  job '"+job['name']+"'")
			
			# Prepare analysis params - used for evauating analysis expression. 
			# Case: input parameters get overriden by job parameters - default
			analysisParams={}
			analysisParams.update(self.inputParameters)
			if 'params' in job:
				analysisParams.update(job['params'])
				
			# Case: job parameters get overriden by input parameters - unimplemented
			
			# Analysis commands start here
			f.write('* '+job['name']+'\n')
			
			# Delete old results and save directives. 
			# Do not unset old options. 
			# f.write('unset *\n')
			f.write('destroy all\n')
			f.write('delete all\n')
			
			# Write options for analysis
			if 'options' in job:
				for (option, value) in job['options'].iteritems():
					if value is True:
						f.write('set '+option+'\n')
					elif value is False:
						f.write('unset '+option+'\n')
					else:
						f.write('set '+option+'='+str(value)+'\n')
			
			# Handle temperature parameter
			# Because job parameters 
			if 'temperature' in job['params']:
				f.write('set temp='+str(job['params']['temperature'])+'\n')
				
			# Write saves for analysis
			if 'saves' in job:
				saves=self._createSaves(job['saves'])
				
				count=0
				for save in saves:
					if count == 0:
						f.write('save ')
					f.write(save+' ')
					count+=1
					if count == 10:
						count=0
						f.write('\n')
				f.write('\n')
			
			# Prepare parameters dictionary for local namespace
			self.analysisLocals['param'].clear()
			self.analysisLocals['param'].update(analysisParams)
			
			# Write analysis
			f.write('echo Running '+str(job['name'])+'\n')
			f.write(eval(job['command'], globals(), self.analysisLocals)+'\n')
			f.write('if $(#plots) gt 1\n  set filetype=binary\n  write '+self.simulatorID+'_job'+str(j)+'_'+job['name']+'.raw\nelse\n echo '+str(job['name'])+' analysis failed.\nend\n\n')
			
		# Write quit - no need for it... it is sent to simulator's stdin
		# f.write('set noaskquit\nquit\n')

		# End control block
		f.write('.endc\n')
		
		# End netlist
		f.write('.end\n')
		
		f.close()
		
		return fileName
		
	def cleanupResults(self, i): 
		"""
		Removes all result files that were produced during the simulation of 
		the *i*-th job group. Simulator input files are left untouched. 
		""" 
		if self.debug>0:
			DbgMsgOut("SOSI", "Cleaning up result for job group "+str(i))
			
		jobGroup=self.jobGroup(i)
		
		# Remove old .raw files
		for j in jobGroup:
			job=self.jobList[j]
			try:
				os.remove(self.simulatorID+"_job"+str(j)+'_'+job['name']+'.raw')
			except KeyboardInterrupt:
				DbgMsgOut("SOSI", "Keyboard interrupt")
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
			DbgMsgOut("SOSI", "Running file '"+fileName+"'")
				
		# Run the file
		spawnOK=True
		p=None
		try:
			# Start simulator
			p=subprocess.Popen(
					[self.binary, '-c']+self.cmdline+[fileName], 
					universal_newlines=True, 
					stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE
				)
			
			# Send quit command
			p.stdin.write('set noaskquit\nquit\n')
			# Collect output
			self.messages=p.stdout.read()
			
			if self.debug>2:
				DbgMsgOut("SOSI", self.messages)
			elif self.debug>1:
				DbgMsgOut("SOSI", self.messages[-400:])
			
			# Now wait for the process to finish. If we don't wait p might get garbage-collected before the
			# actual process finishes which can result in a crash of the interpreter. 
			retcode=p.wait()
			
			# Check return code. Nonzero return code means that something has gone bad. 
			# At least the simulator says so. 
			if retcode!=0:
				spawnOK=False
		except KeyboardInterrupt:
			DbgMsgOut("SOSI", "Keyboard interrupt")
			
			# Will raise an exception if process exits before kill() is called.
			try:
				p.kill()
			except:
				pass
			
			raise KeyboardInterrupt
		except:
			spawnOK=False
		
		if not spawnOK and self.debug>0:
			DbgMsgOut("SOSI", "  run FAILED")
			
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
				job=self.jobList[jobIndex]
				self.results[jobIndex]=raw_read(self.simulatorID+"_job"+str(jobIndex)+'_'+job['name']+'.raw')
				if self.results[jobIndex] is not None:
					countOK+=1
				count+=1
			
			if self.debug>0:
				DbgMsgOut("SOSI", "  "+str(countOK)+"/"+str(count)+" jobs OK")
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
		Returns the unoptimized job sequence. If there are n jobs the job list 
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
		* identical simulator parameter values (excluding temperature which is 
		  actually a simulator option), 
		* identical simulator option lists, but not neccessarily identical 
		  option values. 
		
		In other words: job group members are job indices of jobs that differ 
		only in simulator option values. 
		"""
		# Move temperature to options. Raise an error if temp option is found. 
		# This way jobs that have different temperature but are otherwise joinable end up in the same group. 
		# Also add empty dictionaries for missing entries
		for job in self.jobList:
			for option in job['options'].keys():
				if option.lower()=='temp':
					raise Exception, DbgMsg("SOSI", "TEMP option is not allowed. Use temperature parameter.")
			if 'temperature' in job['params']: 
				job['options']['temp']=job['params']['temperature']
				del job['params']['temperature']
		
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
				# Compare jobs, join them if all of the following holds  
				# - definitions are identical
				# - parameters are identical
				# - the list of options is identical, but not neccessarily the values
				if (self.jobList[i1]['definitions']==self.jobList[i2]['definitions'] and
					self.jobList[i1]['params']==self.jobList[i2]['params'] and
					set(self.jobList[i1]['options'].keys())==set(self.jobList[i1]['options'].keys())):
					# Job i2 can be joined with job i1, add it to jobGroup
					jobGroup.append(i2)
					# Remove i2 from candidates
					candidates.remove(i2)
			
			# Sort jobGroup
			jobGroup.sort()
			
			# Append it to job sequence
			seq.append(jobGroup)
			
		# Move temp option to parameters
		for job in self.jobList:
			if 'temp' in job['options']: 
				job['params']['temperature']=job['options']['temp']
				del job['options']['temp']
		
		return seq

		
	#
	# Retrieval of simulation results
	#
	
	def res_voltage(self, node1, node2=None, resIndex=0):
		"""
		Retrieves the voltage corresponding to *node1* (voltage between nodes 
		*node1* and *node2* if *node2* is also given) from the *resIndex*-th 
		plot in the active result group. 
		
		Equivalent to SPICE OPUS expression ``v(node1)`` 
		(or ``v(node1,node2)``). 
		"""
		if node2 is None:
			return self.activeResult[resIndex][0][node1]
		else:
			return self.activeResult[resIndex][0][node1]-self.activeResult[resIndex][0][node2]
	
	def res_current(self, name, resIndex=0):
		"""
		Retrieves the current flowing through instance *name* from the 
		*resIndex*-th plot in the active result group. 
		
		Equivalent to SPICE OPUS expression ``i(name)`` (also ``name#branch``). 
		"""
		return self.activeResult[resIndex][0][name+'#branch']
		
	def res_property(self, name, parameter, index=None, resIndex=0):
		"""
		Retrieves the *index*-th component of property named *parameter* 
		belonging to  instance named *name*. If the property is not a vector, 
		*index* can be ommitted. The property is retrieved from *resIndex*-th 
		plot in the active result group. 
		
		Note that this works only of the property was saved with a 
		corresponding save directive. 
		
		Equivalent to SPICE OPUS expression ``@name[parameter]`` 
		(or ``@name[parameter][index]``). 
		"""
		if index is None:
			return self.activeResult[resIndex][0]['@'+name+'['+parameter+']']
		else:
			return self.activeResult[resIndex][0]['@'+name+'['+parameter+']['+index+']']
	
	def res_noise(self, reference, name=None, contrib=None, resIndex=0):
		"""
		Retrieves the noise spectrum density of contribution *contrib* of 
		instance *name* to the input/output noise spectrum density. *reference* 
		can be ``'input'`` or ``'output'``. 
		
		If *name* and *contrib* are not given the output or the equivalent 
		input noise spectrum density is returned (depending on the value of 
		*reference*). 
		
		The spectrum is obtained from the *resIndex*-th plot in the active 
		result group. 
		"""
		if name is None:
			# Input/output noise spectrum
			if reference=='input':
				spec=self.activeResult[resIndex][0]['inoise_spectrum']
			elif reference=='output':
				spec=self.activeResult[resIndex][0]['onoise_spectrum']
			else:
				raise Exception, "Bad noise reference."
		else:
			# Partial spectrum
			if reference=='input':
				A=self.activeResult[resIndex][0]['inoise_spectrum']/self.activeResult[resIndex][0]['onoise_spectrum']
			elif reference=='output':
				A=1.0
			else:
				raise Exception, "Bad noise reference."
			
			spec=self.activeResult[resIndex][0]["onoise_"+str(name)+"_"+str(contrb)]*A
	
		return spec
		
	def res_scale(self, resIndex=0):
		"""
		Retrieves the default scale of the *resIndex*-th plot in the active 
		result group. 
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
	
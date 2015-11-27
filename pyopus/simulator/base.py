"""
.. inheritance-diagram:: pyopus.simulator.base
    :parts: 1
	
**Base class for simulator objects (PyOPUS subsystem name: SI)**

A **system** is a physical object (e.g. a circuit) whose characteristics and 
responses are evaluated (simulated) by a simulator. 

Simulators are divided into two groups: **batch mode simulators** and 
**interactive simulators**. An interactive simulator presents the user with a 
prompt where commands of a simulator-specific command language can be entered 
to control the simulator (load input file, perform simulation, save result 
file, ...). 

Interactive simulators can be used as batch simulators if they are presented 
with a script written in the simulator control language in at their standard 
input. 

To tell a batch mode simulator what to simulate one must provide a 
**job list**. A job list is a list of jobs where every job specifies one 
simulation. 

A job is a dictionary with the following members:

* ``name`` - the name of the job
* ``definitions`` - a list of system description modules that constitute the 
  system to be simulated 
* ``params`` - a dictionary with the system parameters used for this job. 
  Keys are parameter names while values are parameter values 
* ``options`` - a dictionary with the simulator options used for this job. 
  Keys are option names while values are option values
* ``variables`` - a dictionary containing the Python variables that are 
  available during the evaluation of ``saves`` and ``command``. 
* ``saves`` - a list of strings giving Python expressions that evaluate to 
  lists of save directives specifying what simulated quantities should be 
  stored in simulator output
* ``command`` - a string giving a Python expression that evaluates to the 
  simulator command for the analysis performed by this job

The order in which jobs are given may not be optimal for fastest simulation 
because it can require an excessive number of system parameter/system 
description changes. The process of **job optimization** reorders and groups 
the jobs into **job groups** in such manner that the number of these changes 
is minimized. the sequence of job groups is called **optimized job sequence** 
Optimal job ordering and grouping as well as the way individual groups of jobs 
are handled in simulation depends on the simulator. 

A batch simulator offers the capability to run the jobs in one job group at a 
time. For every job group the simulator is started, jobs simulated, and the 
simulator shut down. Every time a job group is run theuser can provide a 
dictionary of *inputParameters* that complement the parameters defined in the 
``params`` dictionaries of the jobs. If both the *inputParameters* and the 
``params`` dictionary of a job list the value of the same parameter, the value 
in the ``params`` dictionary takes precedence over the value defined in the 
*inputParameters* dictionary. After the simulation in finiched the results can 
be retrieved from the simulator object. 

Interactive simulators offer the user the possibility to send commands to the 
simulator. After the execution of every command the output produced by the 
simulator is returned. Collecting simulator results is left over to the user. 
The simulator can be shut down and restarted whenever the user decides. 
"""

# Abstract siumulator interface
from ..misc.identify import locationID
from ..misc.debug import DbgMsgOut
from os import remove
from dircache import listdir

__all__ = [ 'Simulator' ]

# TODO: stop changing input structures by adding missing defaults
 
class Simulator(object):
	"""
	Base class for simulator classes. 
	
	Every simulator is equipped with a *simulatorID* unique for every simulator 
	in every Python process on every host. The format of the simulatorID is 

	``hostname_processID_simulatorObjectNumber``
	
	A different simulator object number is assigned to every simulator object 
	of a given type in one Python interpreter. 
	
	All intermediate files resulting from the actions performed by a simulator 
	object have their name prefixed by *simulatorID*. 
	
	Every simulator can hold result groups from several jobs. One of these 
	groups is the active result group. 
	
	The *binary* argument gives the full path to the simulator executable. 
	*args* is a list of additional command line parameters passed to the 
	simulator executable at simulator startup. 
	
	If *debug* is greater than 0 debug messages are printed to standard output. 
	"""
	# Class variable that counts simulator instances
	instanceNumber=0
	
	# Define instance variables and initialize instance
	def __init__(self, binary, args=[], debug=0):
		# Simulator binary and command line arguments
		self.binary=binary
		self.cmdline=args
		
		# Debug mode fkag
		self.debug=debug
		
		# Job list, sequence, and freshness flag
		self.jobList=None
		self.jobListFresh=False
		self.jobSequence=None
		
		# Input parameters dictionary
		self.inputParameters={}
		
		# Results of the simulation
		self.results={}
		self.activeResult=None
		
		# Build a unique ID for this simulator
		self.simulatorID=locationID()+"_"+str(Simulator.instanceNumber)
		
		# Increase simulator instance number
		Simulator.instanceNumber+=1
	
	def cleanup(self):
		"""
		Removes all intermediate files created as a result of actions performed 
		by this simulator object (files with filenames that have *simulatorID* 
		for prefix). 		
		"""
		# Get directory
		dirEntries=listdir('.')
		
		# Find all directory entries starting with simulatorID and delete them
		for entry in dirEntries:
			if entry.find(self.simulatorID)==0:
				if self.debug:
					DbgMsgOut("SI", "Removing "+entry)
				try:
					remove(entry)
				except:
					pass
	
	#
	# For interactive simulators
	# 
	
	def simulatorRunning(self):
		"""
		Returns ``True`` if the interactive simulator is running. 
		"""
		return False
	
	def startSimulator(self):
		"""
		Starts the interactive simulator if it is not already running. 
		"""
		pass
	
	def stopSimulator(self):
		"""
		Stops a running interactive simulator. 
		This is done by politely asking the simulator to exit.   
		"""
		pass
	
	#
	# Job list and job list optimization
	#
	
	def setJobList(self, jobList, optimize=True):
		"""
		Sets *jobList* to be the job list for batch simulation. If the 
		``options``, ``params``, ``saves``, or ``variables`` member is 
		missing in any of the jobs, an empty dictionary/list is added to 
		that job. 
		
		The job list is marked as fresh meaning that a new set of simulator 
		input files needs to be created. Files are created the first time a 
		job group is run. 
		
		If *optimize* is ``True`` the optimized job sequence is computed by 
		calling the :meth:`optimizedJobSequence` method. If *optimize* is 
		``True`` an unoptimized job sequence is produced by calling the 
		:meth:`unoptimizedJobSequence` method. 
		"""
		# Add missing members
		for job in jobList:
			if 'options' not in job:
				job['options']={}
			if 'params' not in job:
				job['params']={}
			if 'saves' not in job: 
				job['saves']=[]
			if 'variables' not in job:
				job['variables']={}
		# Store it, mark it as fresh
		self.jobList=jobList
		self.jobListFresh=True
		# Optimize if required
		if optimize:
			self.jobSequence=self.optimizedJobSequence()
		else:
			self.jobSequence=self.unoptimizedJobSequence()
		# Converter from jobIndex to jobGroupIndex
		self.jobGroupIndex={}
		for jobGroupNdx in range(len(self.jobSequence)):
			jobGroup=self.jobSequence[jobGroupNdx]
			for jobIndex in jobGroup:
				self.jobGroupIndex[jobIndex]=jobGroupNdx
		
	def unoptimizedJobSequence(self):
		"""
		Returns the unoptimized job sequence. A job sequence is a list of 
		lists containing indices in the job list passed to the 
		:meth:`setJobList` method. Every inner list corresponds to one job 
		group. 
		"""
		raise Exception, DbgMsg("SI", "Job list optimization procedure is not defined.") 
		
	def optimizedJobSequence(self):
		"""
		Returns the optimized job sequence. Usually the order of job groups 
		and jobs within them is identical to the order of jobs in the job list 
		set with the :meth:`setJobList` method. 
		"""
		raise Exception, DbgMsg("SI", "The procedure for obtaining the unoptimized job list is not defined.") 
		
	def jobGroupCount(self):
		"""
		Returns the number of job groups. 
		"""
		raise Exception, DbgMsg("SI", "The method is not defined.")
	
	def jobGroup(self, i):
		"""
		Returns a structure describing the *i*-th job group. 
		"""
		raise Exception, DbgMsg("SI", "The method is not defined.")
	
	#
	# Batch simulation
	#
	
	def setInputParameters(self, inParams):
		"""
		Sets the dictionary of input parameters to *inParams*. 
		These parameters are used in all subsequent batch simulations. 
		"""
		self.inputParameters=inParams
		
	def runJobGroup(self, i):
		"""
		Runs the jobs in the *i*-th job group. 
		Returns a tuple of the form (*jobIndices*, *status*) where *jobIndices* 
		is a list of indices corresponding to the jobs that ran. *status* is 
		``True`` if everything is OK. 
		"""
		raise Exception, DbgMsg("SI", "The method is not defined.")
	
	def cleanupResults(self, i):
		"""
		Removes all result files that were produced during the simulation of 
		the *i*-th job group. Simulator input files are left untouched. 
		"""
		# The derived class has to override this method
		raise Exception, DbgMsg("SI", "The method is not defined.")
		
	def collectResults(self, indices, runOK=None):
		"""
		Collect the result groups produced by the jobs with indices given by 
		the *indices* list. The *indices* list is the list returned by the 
		:meth:`runJobGroup` method as *jobIndices*. Can also be used to load 
		results of multiple job group runs by combining the *jobIndices* 
		corresponding to the respective job groups. 
		
		*runOK* is the status returned by the :meth:`runJobGroup` method. 
		If *runOK* is not given, the status of the last job that was run is 
		used. If *runOK* is ``False`` ``None`` is set for the result groups 
		corresponding to the jobs specified by *indices*. 
		"""
		raise Exception, DbgMsg("SI", "The method is not defined.")
	
	def resetResults(self):
		"""
		Removes all collected result groups from memory. Sets the active result 
		group to ``None``.  
		"""
		self.results={}
		self.activeResult=None
	
	def activateResult(self, i):
		"""
		Activate the result group corresponding to the *i*-th job. 
		Return the structure with the results of *i*-th job. 
		If the result group is not found, ``None`` is returned and the active 
		result group becomes ``None``. 
		"""
		if i not in self.results:
			self.activeResults=None
		else:
			self.activeResult=self.results[i]
		return self.activeResult
	
	def haveResult(self):
		"""
		Return ``True`` is the active result group is not ``None`` (i.e. a 
		result group was successfully obtained with simulation, loaded, and 
		activated). 
		"""
		return self.activeResult is not None
	
	#
	# Retrieval of simulation results
	#
	
	def getGenerators(self):
		"""
		Returns a dictionary of functions that retrieve results from the active 
		result group. These functions are referred to as **the generators**. 
		They are used for measurement evaluation in 
		:class:`~pyopus.evaluator.performance.PerformanceEvaluator` objects 
		and are used as part of the environment for evaluating the measures. 
		
		The following functions (dictionary key names) are usually available:
		
		* ``v`` - retrieve a node potential or potential difference between two 
		  nodes
		* ``i`` - retrieve a current flowing through an instance (usually a 
		  voltage source or inductor)
		* ``p`` - retrieve an element instance property (assuming that it was 
		  saves during simulation with a correspondign save directive)
		* ``ns`` - retrieve a noise spectrum
		* ``scale`` - retrieve the default scale of the result group
		* ``ipath`` - construct a hierarchical path name of a node/element 
		  instance/model
		"""
		return {}
	
"""
.. inheritance-diagram:: pyopus.parallel.base
    :parts: 1
	
**Base classes for virtual machines (PyOPUS subsystem name: VM)**

A **spawner task** is a task that can spawn new tasks on hosts in the virtual 
machine. All other tasks are **worker tasks**. 


**Mirroring and local storage**

Often tasks in the virtual machine require the use of additional storage (like 
a harddisk) for storing the parallel algorithm's input files which are 
identical for all hosts. These files must therefore be distributed to all 
hosts in a virtual machine before the computation can begin. 

One way to make all virtual machines see the same directory structure is to 
use network filesystems like NFS or SMBFS. In this approach the storage 
physically resides one a computer in the network (file server). This storage 
is exported and mounted by all hosts in the virtual machine. 

Take for instance that folder ``/shareme/foo`` on the file server is exported. 
Hosts in the virtual machine mount this folder under ``/home``. This way all 
hosts see the ``/shareme/foo`` folder located on the physical storage of the 
server as their own folder ``/home``. 

This approach is very simple and ensures that all hosts see the same input 
files in the same place. Unfortunately as the number of hosts grows and the 
amount of data read operatins to the shared folder grows, the network quickly 
becomes saturated and the computational performance of the virtual machine 
dramatically decreases because tasks must wait on slow data read operations. 

A more scalable solution is to use the local storage of every host to store 
the algoruthm's input files. This has the downside that before the computation 
begins these files must be distributed to all hosts. 

On the other hand parallel algorithms often require some local storage for 
storing various intermediate files. If this local storage is in the form of a 
shared folder an additional problem occurs when multiple hosts try to write a 
file with the same name to a physically same place, but with different content. 

A solution to the intermediate file problem is to use every host's phisically 
local storage (e.g. its local harddisk) for storing the intermediate files. 

The solution to these problems in PyOPUS is to use mirroring. Mirroring is 
configured through two environmental variables. ``PARALLEL_LOCAL_STORAGE`` 
specifies the path to the folder where the folders for the local storage of 
input and intermediate files will be created. The ``PARALLEL_MIRRORED_STORAGE`` 
environmental variable specifies a colon (``:``) separated list of paths to the 
directories that are mounted from a common file server. The first such path in 
``PARALLEL_MIRRORED_STORAGE`` on all hosts corresponds to the same directory 
on the file server. The same goes for the second, the third, etc. paths listed 
in ``PARALLEL_MIRRORED_STORAGE``. 

The ``PARALLEL_LOCAL_STORAGE`` and the ``PARALLEL_MIRRORED_STORAGE`` 
environmental variables must be set on all hosts in a virtual machine. It is 
usually set to ``/localhome/USERNAME`` which must physically reside on the 
local machine. ``PARALLEL_MIRRORED_STORAGE`` should be writeable by the user 
that is running the spawned processes. 

Both ``PARALLEL_LOCAL_STORAGE`` and ``PARALLEL_MIRRORED_STORAGE`` can use UNIX 
style user home directory expansion (e.g. ``~user`` expands to ``/home/user``). 


**Path translation in mirroring operations**

Suppose the ``PARALLEL_MIRRORED_STORAGE`` environmental variable is set to 
``/foo:/bar`` on host1 and ``/d1:/d2`` on host2. This means that the following 
directories are equivalent (mounted from the same exported folder on a file 
server)
	
	=======	=======
	host1	host2
	=======	=======
	/foo	/d1
	/bar	/d2
	=======	=======

So ``/foo`` on host1 represents the same physical storage as ``/d1`` on host2. 
Similarly ``/bar`` on host1 represents the same physical storage as ``/d2`` on 
host2. Usually only ``/home`` is common to all hosts in a virtual machine so 
``PARALLEL_MIRRORED_STORAGE`` is set to ``/home`` on all hosts. 

Path translation converts a path on host1 into a path to the physically 
same object (mounted from the same exported directory on the file 
server) on host2. The following is an example of path translation. 

	=============	=============
	Path on host1	Path on host2
	=============	=============
	/foo/a			/d1/a
	/bar/b			/d2/b
	=============	=============
"""

from inspect import ismethod, isclass
from ..misc.debug import DbgMsg, DbgMsgOut
from ..misc.env import environ

from glob import iglob
import os, sys, shutil, time

__all__ = [ 'TaskID', 'HostID', 'Msg', 'MsgTaskExit', 'MsgHostDelete', 'MsgHostAdd', 
			'MsgTaskResult', 'VirtualMachine', 'getNumberOfCores' ] 

# This variable holds a reference to the first imported VM module - for resolving conflicts 
# (e.g. MPI and PVM don't work together)
firstVM=None
			
# All derivative classes must be picklable

# Task identifier definition
class TaskID(object):
	"""
	Basic task identifier class that is used for identifying a task in a 
	virtual machine. :class:`TaskID` objects must support comparison 
	(:func:`__cmp__`), hashing (:func:`__hash__`), 	and conversion to a string 
	(:func:`__str__`). 	
	"""
	@staticmethod
	def bad():
		"""
		A static member function. Called with ``TaskID.bad()``. 
		Returns an invalid task ID. 
		"""
		return TaskID()
		
	def __init__(self):
		pass
	
	def __cmp__(self, other):
		if type(self)==type(other):
			return 0
		elif type(self)<type(other):
			return -1
		else:
			return 1
	
	def __hash__(self):
		return 0
		
	def __str__(self):
		return "NOTASK"
	
	def valid(self):
		"""
		Returns ``True`` if this :class:`TaskID` object is valid. 
		"""
		return False 


# Host identifier definition
class HostID(object):
	"""
	Basic host identifier class that is used for identifying a host in a 
	virtual machine. :class:`HostID` objects must support comparison 
	(:func:`__cmp__`), hashing (:func:`__hash__`), and conversion to a string 
	(:func:`__str__`). 	
	"""
	@staticmethod
	def bad():
		"""
		A static member function. Called with ``HostID.bad()``. 
		Returns an invalid host ID. 
		"""
		return TaskID()
		
	def __init__(self):
		pass
	
	def __cmp__(self, other):
		if type(self)==type(other):
			return 0
		elif type(self)<type(other):
			return -1
		else:
			return 1
	
	def __hash__(self):
		return 0
		
	def __str__(self):
		return "NOHOST"
		
	def valid(self):
		"""
		Returns ``True`` if this :class:`HostID` object is valid. 
		"""
		return False


# Message on virtual machine level of abstraction
class Msg(object):
	"""
	Base class for a message used in task-to-task communication. 
	"""
	pass


# Define some common messages
class MsgTaskExit(Msg):
	"""
	This is message that signals a task has exited. 
	
	The :class:`TaskID` object corresponding to the task is stored in the 
	:attr:`taskID` member. 
	"""
	def __init__(self, taskID):
		Msg.__init__(self)
		self.taskID=taskID
		
		
class MsgHostDelete(Msg):
	"""
	This is message that signals a host has exited from the virtual machine. 
	
	The :class:`HostID` object corresponding to the host is stored in the 
	:attr:`hostID` member. 
	"""
	def __init__(self, hostID):
		Msg.__init__(self)
		self.hostID=hostID


class MsgHostAdd(Msg):
	"""
	This is message that signals new hosts were added to the virtual machine. 
	
	The list of :class:`HostID` objects corresponding to the added hosts is 
	stored in the :attr:`hostIDs` member. 
	"""
	def __init__(self, hostIDs):
		Msg.__init__(self)
		self.hostIDs=hostIDs
		

class MsgTaskResult(Msg):
	"""
	This is message that is sent to the process that spawned a Python function 
	in a virtual machine. The message holds a boolean flag that tells if the 
	function succeeded to run and the return value of the function. 
	
	The boolean flag and the return value can be found in :attr:`success` and 
	:attr:`returnValue` members. 
	"""
	def __init__(self, success, returnValue):
		Msg.__init__(self)
		self.success=success
		self.returnValue=returnValue


# Virtual machine
class VirtualMachine(object):
	"""
	The base class for accessing hosts working in parallel. 
	
	*debug* specifies the debug level. If it is greater than 0 debug messages 
	are printed on the standard output. 
	
	*startupDir* specifies the working directory where the spawned 
	functions will wake up. If set to ``None``, the underlying virtual machine 
	default is used. 
	
	If *translateStartupDir* is ``True`` path translation is applied to 
	*startupDir* just as it is to *mirrorMap*. If mirroring is defined 
	(*mirrorMap* is given) then no translation is performed on *startupDir* 
	In this case *startupDIr* is treated as relative path with respect to 
	the local storage.
	
	*mirrorMap* is a dictionary specifying filesystem objects (files and 
	directories) on the spawner which are to be mirrored (copied) to local 
	storage on the host where the task is spawned. The keys represent 
	paths on the spawner while the values represent the corresponding 
	paths (relative to the local storage dorectory) on the host where the 
	task will be spawned. Keys can use UNIX style globbing (anything the 
	:func:`glob.glob` function can handle is OK). If *mirrorMap* is set to 
	``None``, no mirroring is performed. 
	
	For the mirroring to work the filesystem objects on the spawner must be 
	in the folders specified in the ``PARALLEL_MIRRORED_STORAGE`` 
	environmental variable. This is because mirroring is performed by local 
	copy operations which require the source to be on a mounted network 
	filesystem. 
	
	To find out more about setting the working directory and mirroring see 
	the :meth:`prepareEnvironment` method. 
	"""
	def __init__(self, debug=0, startupDir=None, translateStartupDir=True, mirrorMap=None):
		# Debug level
		self.debug=debug
		
		# Process startupDir
		if translateStartupDir and mirrorMap is not None and startupDir is not None:
			(self.startupDirIndex, self.startupDirSuffix)=self.translateToAbstractPath(startupDir)
		else:
			self.startupDirIndex=None
			self.startupDirSuffix=startupDir
		
		# Process local storage map
		if mirrorMap is None:
			self.mirrorList=None
		else:
			# Build mirror list
			# Start with empty list
			self.mirrorList=[]
			# Go through all entries in mirrorMap 
			for (masterObject, target) in mirrorMap.iteritems():
				(index, suffix)=self.translateToAbstractPath(masterObject)
				
				# Append to self.mirrorList
				self.mirrorList.append((index, suffix, target))
				
				if self.debug:
					DbgMsgOut("VM", "Mirroring '"+masterObject+"' from '"+suffix+"' in ("+str(index)+") to '"+target+"' in local storage.")
	
	@staticmethod
	def slots():
		"""
		Returns the number of slots for tasks in a virtual machine. 
		Every processor represents a slot for one task. 
		"""
		return 0
	
	@staticmethod
	def freeSlots():
		"""
		Returns the number of free slots for tasks in the virtual machine. 
		"""
		return 0
		
	@staticmethod
	def hosts():
		"""
		Returns the list of :class:`HostID` objects representing the nosts in 
		the virtual machine. Works only for hosts that are spawners. 
		"""
		return []
	
	@staticmethod
	def taskID():
		"""
		Returns the :class:`TaskID` object corresponding to the calling task. 
		"""
		return None
	
	@staticmethod
	def hostID():
		"""
		Returns the :class:`HostID` object corresponding to the host on which 
		the caller task runs. 
		"""
		return None
	
	@staticmethod
	def parentTaskID():
		"""
		Returns the :class:`TaskID` object corresponding to the task that 
		spawned the caller task. 
		"""
		return None
	
	@staticmethod
	def formatSpawnerConfig():
		"""
		Formats the configuration information gathered by a spawner task as a 
		string. Works only if called by a spawner task.  
		"""
		return ""
	
	@classmethod
	def dummy(cls):
		pass
	
	# This function prepares a function descriptor for spawning so that member functions can also be spawned
	@classmethod
	def func2desc(cls, func):
		if ismethod(func):
			# Handles instance methods and classmethods 
			# Get self, pack it together with method name
			return (func.im_self, func.__name__)
		else:
			# Plain function, just return the function
			return func
	
	# This function reconstructs a function from a function descriptor
	@classmethod
	def desc2func(cls, desc):
		if type(desc) is tuple:
			# Instance methods and classmethods
			s,n=desc
			if isclass(s):
				# Class, look in __dict__
				return getattr(s, n)
			else:
				# Instance, use __getattribute__
				return s.__getattribute__(n)
		else:
			# Plain function
			return desc
	
	# This must be overriden by every derived class. 
	def spawnFunction(self, function, args=(), kwargs={}, count=-1, targetList=None, sendBack=True):
		"""
		Spawns a *count* instances of a Python *function* on remote hosts and 
		passes *args* and *kwargs* to the function. Spawning a function 
		actually means to start a Python interpreter, import the function, and 
		call it with *args* and *kwargs*. 
		
		If *count* is ``None`` the number of tasks is select in such way that 
		all available slots are filled. 
		
		*function*, *args*, and *kwargs* must be pickleable. 
		
		*targetList* specifies a list of hosts on which the function instances 
		will be spawned. If it is ``None`` all hosts in the virtual machine are 
		candidates for the spawned instances of the function. 
		
		If *sendBack* is ``True`` the spawned tasks return the status and the 
		return value of the function back to the spawner after the function 
		exits. The return value must be pickleable. 
		
		Returns a list of :class:`TaskID` objects representing the spawned tasks. 
		
		Works only if called by a spawner task. 
		"""
		raise Exception, DbgMsg("VM", "Spawning not implemented.")
	
	def checkForIncoming(self):
		"""
		Returns ``True`` if there is a message waiting to be received. 
		"""
		raise Exception, DbgMsg("VM", "Message check not implemented.")
		
	def receiveMessage(self, timeout=-1.0):
		"""
		Receives a *message* (a Python object) and returns a tuple 
		(*senderTaskId*, *message*)
		
		The sender of the *message* can be identified through the 
		*senderTaskId* object of class :class:`TaskID`. 
		
		If *timeout* is negative the function waits (blocks) until some message 
		arrives. If *timeout*>0 seconds pass without receiving a message, an 
		empty tuple is returned. Zero *timeout* performs a nonblocking receive 
		which returns an empty tuple if no message is received. 
		
		In case of an error the return value is ``None``. 
		"""
		raise Exception, DbgMsg("VM", "Reception of messages not implemented.")
	
	def sendMessage(self, destination, message):
		"""
		Sends *message* (a Python object) to a task with :class:`TaskID` 
		*destination*. Returns ``True`` on success. 
		"""
		raise Exception, DbgMsg("VM", "Sending of messages not implemented.")
	
	@staticmethod
	def finalize():
		"""
		Cleans up after a parallel program. 
		Should be called before exit. 
		"""
		pass
		
	def clearLocalStorage(self, timeout=-1.0):
		"""
		This function spawns the :func:`localStorageCleaner` function on all 
		slots in the virtual machine. The spawned instances remove the local 
		storage that was created for the slot. 
		
		*timeout* is applied where needed. Negative values stand for infinite 
		*timeout*. 
		
		This function should never be called if there are tasks running in the 
		virtual machine because it will remove their local storage. 
		
		This function should be called only by the spawner. 
		"""
		# Get hosts
		hostIDs=self.hosts()
		
		# Spawn a cleaner on every host
		taskIDs=[]
		for hostID in hostIDs:
			taskID=self.spawnFunction(localStorageCleaner, kwargs={'vm': self}, count=1, targetList=[hostID])
			taskIDs.extend(taskID)
			
		# Collect return values and task exit messages from all hosts confirming that cleanup is finished. 
		taskIDs=set(taskIDs)
		mark=time.time()
		while len(taskIDs)>0:
			remains=timeout
			if timeout>=0:
				remains=timeout-(time.time()-mark)
				if remains<=0:
					break
			
			recv=self.receiveMessage(remains)
			if recv is not None and len(recv)==2:
				(sourceID, msg)=recv
				if  type(msg) is MsgTaskExit:
					# Remove taskID from set of spawned task IDs. 
					if sourceID in taskIDs:
						taskIDs.remove(sourceID)
				else:
					# Throw away other messages. 
					pass
	
	# These are helper methods  
	def translateToAbstractPath(self, path):
		"""
		Translates a *path* on the local machine to a tuple (*index*, *suffix*) 
		where *index* denotes the index of the path entry in the 
		``PARALLEL_MIRRORED_STORAGE`` environmental variable and *suffix* is 
		the path relative to that entry. 
		
		Note that *suffix* is a relative path even if it starts with ``/``. 
		"""
		# Make path canonical
		canonical=os.path.realpath(path)
		# Mark it as not found in PARALLEL_MIRRORED_STORAGE
		for index in range(len(ParallelMirroredStorage)):
			masterMirroredDir=ParallelMirroredStorage[index]
			if canonical.find(masterMirroredDir)==0:
				found=True
				
				# We have a mirrored direcotry prefix, get relative path. 
				suffix=os.path.relpath(canonical, masterMirroredDir)
				
				return (index, suffix)
		
		# Failed to translate
		raise Exception, DbgMsg("VM", "'"+path+"' not in PARALLEL_MIRRORED_STORAGE.")
		
	def translateToActualPath(self, index, relPath):
		"""
		Translates a *index* and *relPath* to an actual path on the local 
		machine. 
		
		This is the inverse of the :meth:`translateToAbstractPath` method. 
		"""
		if index>0 or index>=len(ParallelMirroredStorage):
			raise Exception, DbgMsg("VM", "PARALLEL_MIRRORED_STORAGE should have at least "+str(index+1)+"members.")
		
		return os.path.join(ParallelMirroredStorage[index], relPath) 
	
	def createLocalStorage(self, subpath):
		"""
		Creates a local storage directory subtree given by *subpath* under 
		``PARALLEL_LOCAL_STORAGE``. 
		
		If a local storage directory with the same name already exists it is 
		suffixed by an underscore and a hexadecimal number. 
		
		Returns the path to the created local storage directory. 
		"""
		if ParallelLocalStorage is None:
			raise Exception, DbgMsg("VM", "PARALLEL_LOCAL_STORAGE is not set.")
		
		# Build storage directory name
		taskStorage=os.path.join(ParallelLocalStorage, subpath)
				
		# Change name if it already exists, add numeric suffix (hex). 
		counter=1
		while os.path.lexists(taskStorage): 
			# Directory exists, try another one (add numeric suffix)
			taskStorage=os.path.join(ParallelLocalStorage, subpath+("_%x" % counter))
			counter+=1
			
		try:	
			os.makedirs(taskStorage)
		except:
			raise Exception, DbgMsg("VM", "Failed to create local storage in '"+taskStorage+"'")
			
		return taskStorage
	
	# Create local storage, mirror, and set working directory.
	# Return local storage path. If no mirroring was performed, return None. 
	def prepareEnvironment(self):
		"""
		Prepares the working environment (working directory and local storage) 
		for a spawned function. This method is called by a spawned task. The 
		mirroring information is received from the spawner (spawner's 
		*mirrorList*) at function spawn time. Spawned task's virtual machine 
		object is namely the spawner's virtual machine object sent to the 
		spawned task. 
		
		* If mirroring is not configured with the :meth:`setSpawn` method, the 
		  working directory is the one specified as *startupDir* with an 
		  appropriate path translation applied to it. If it is ``None`` the 
		  working directory is determined by the undelying virtual machine 
		  library (e.g. MPI). 
		
		* If mirroring is configured with :meth:`setSpawn` (*mirrorList* is not 
		  ``None``), a local storage directory is created by calling 
		  :meth:`createLocalStorage` with *subpath* set to the PID of the 
		  calling process in hexadecimal notation. 
		
		  Next mirroring is performed by traversing all members of the 
		  processed *mirrorList* dictionary received from the spawner which is 
		  a list of tuples of the form (*index*, *suffix*, *target*) where 
		  *index* and *suffix* specify the source filesystem object to mirror 
		  (see the :meth:`translateToAbstractPath` method for the explanation 
		  of *index* and *suffix*) while *target* is the destination where the 
		  object will be copied. 
		  
		  The source can be specified with globbing characters (anything the 
		  :func:`glob.glob` function can handle is OK). 
		  
		  *target* is the path relative to the local storage directory where 
		  the source will be copied. Renaming of the source is not possible. 
		  *target* always specifies the destination directory. 
		  
		  If source is a directory, symbolic links within it are copied as 
		  symbolic links which means that they should be relative and point to 
		  mirrored filesystem objects in order to remain valid after mirroring. 
		  
		  If *startupDir* was given and it is not ``None`` the working 
		  directory is set to path given by *startupDir* that is relative to 
		  the local storage directory. 
		  
		  If *startupDir* is ``None`` the working directory is set to the 
		  local storage directory. 
		  
		Returns the path to the local storage directory. 
		"""
		if self.mirrorList is None: 
			# No mirroring. Change working directory and return. 
			if self.startupDirSuffix is not None:
				if self.startupDirIndex is None:
					startupPath=self.startupDirSuffix
				else:
					startupPath=self.translateToActualPath(self.startupDirIndex, self.startupDirSuffix)
				if self.debug:
					DbgMsgOut("VM", "Changing working directory to '"+startupPath+"'.")
				os.chdir(startupPath)
			return None
		
		# Have mirroring. 
		# Create local storage directory
		if self.debug:
			DbgMsgOut("VM", "Creating local storage.")
			
		taskStorage=self.createLocalStorage("%x" % os.getpid())
		
		if self.debug:
			DbgMsgOut("VM", "Local storage created in '"+taskStorage+"'.")
		
		# Change working directory to local storage
		os.chdir(taskStorage)
		
		# Copy local storage subdirs (master must do some preparation first)
		# Worker instructions for directory copyiing are in the vm object received from parent
		for mirrorDirective in self.mirrorList:
			(index, suffix, target)=mirrorDirective
			
			# Translate
			sourcePath=self.translateToActualPath(index, suffix)
			
			# Debug
			if self.debug:
				DbgMsgOut("VM", "Mirroring '"+sourcePath+"' to '"+target+"'.")
				sys.stdout.flush()
			
			# Do the globbing
			for source in iglob(sourcePath):
				if os.path.isdir(source):
					# Copying a directory. Destination is the directory where the copy of the tree will be created. 
					# Renaming is not possible. Copy symlinks as symlinks. 
					# Get the name of the copied object (copytree wants a destination object name). 
					(srcPath, srcName)=os.path.split(source)
					# Copy tree, create destination directory if it does not exist.
					shutil.copytree(source, os.path.join(target, srcName), symlinks=True)
				else:
					# Copying a file. Destination is the directory where the copy of the file will be created. 
					# Renaming is not possible. Copy file itself, not symlink
					# Get the name of the copied object. 
					(srcPath, srcName)=os.path.split(source)
					# Create destination directory if it does not exist.
					if not os.path.exists(target): 
						os.makedirs(target)
					if not os.path.isdir(target):
						raise Exceptin, DbgMsg("VM", "Mirroring destination exists, but is not a directory.") 
					# Copy file. 
					shutil.copy(source, target)
		
		# See if workDir is given
		if self.startupDirSuffix is not None:
			# startupDir is relative to local storage directory. 
			tmpDir=os.path.join(taskStorage, self.startupDirSuffix)
			if self.debug:
				DbgMsgOut("VM", "Changing working directory to '"+tmpDir+"'.")
			os.chdir(tmpDir)
			
		return taskStorage
	
	def cleanupEnvironment(self, taskStorage):
		"""
		Removes the working environment prepared for a remote task. 
		
		*taskStorage* is the path to the local storage directory. 
		This directory is removed. 
		
		If *taskStorage* is ``None`` nothing is removed. 
		"""
		if taskStorage is None:
			return
		
		if self.debug: 
			DbgMsgOut("VM", "Removing '"+taskStorage+"'.")
			
		# Go to ParallelLocalStorage so we are not in our own way. 
		os.chdir(ParallelLocalStorage)
		
		# Remove task storage folder
		shutil.rmtree(taskStorage, True)
#
# Helper functions
#

def localStorageCleaner(vm=None):
	"""
	This is the function spawned by the 
	:meth:`VirtualMachine.clearLocalStorage` method. 
	
	The function looks in the directory given by the ``PARALLEL_LOCAL_STORAGE`` 
	environmental variable and removes everything in that directory. 
	"""
	# Traverse storageRoot entries. 
	entries=os.listdir(ParallelLocalStorage)
	# Go to ParallelLocalStorage so we are not in our own way. 
	os.chdir(ParallelLocalStorage)
	# Go through entries. 
	for entry in entries:
		completeEntry=os.path.join(ParallelLocalStorage, entry)
		# Ignore errors (delete everythiong we can delete). 
		try:
			if os.path.isdir(completeEntry):
				shutil.rmtree(completeEntry, True)
			else:
				os.remove(completeEntry)
			if vm.debug:
				DbgMsgOut("VM", "Removing '"+completeEntry+"'.")
		except:
			if vm.debug:
				DbgMsgOut("VM", "Failed to remove '"+completeEntry+"'.")

def getNumberOfCores():
	"""
	Returns the number of available CPU cores. 
	
	Works for Linux, Unix, MacOS, and Windows. 
	
	Uses code from Parallel Python (http://www.parallelpython.com). 
	"""
	# Taken from Parallel Python. Thanks. 
	# For Linux, Unix and MacOS
	if hasattr(os, "sysconf"):
		if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
			# Linux and Unix
			ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
			if isinstance(ncpus, int) and ncpus > 0:
				return ncpus
		else:
			# MacOS X
			return int(os.popen2("sysctl -n hw.ncpu")[1].read())
	# For Windows
	if "NUMBER_OF_PROCESSORS" in environ:
		ncpus = int(environ["NUMBER_OF_PROCESSORS"])
		if ncpus > 0:
			return ncpus
	
	# Default
	return 1

	
#
# Initialization - runs once at first module import 
#

# Get local storage from environment - this is the directory residing on a phisically 
# local medium where runs local to the host are performed. The user who owns the
# Python process must have read/write permission to this directory. 
# If it does not exist yet, we try to create it. 
if 'PARALLEL_LOCAL_STORAGE' in environ:
	ParallelLocalStorage=environ['PARALLEL_LOCAL_STORAGE']
	
	try:
		# Expand user home (~, ~name (unix only))
		ParallelLocalStorage=os.path.expanduser(ParallelLocalStorage)
		
		# Normalize path, get canonical path (full path, eliminate symlinks)
		ParallelLocalStorage=os.path.realpath(ParallelLocalStorage)
		
		# Create if not there yet
		if not os.path.exists(ParallelLocalStorage):
			# Not there, create it (along with all missing directories in the path)
			os.makedirs(ParallelLocalStorage)
	
	except KeyboardInterrupt:
		DbgMsgOut("VM", "Keyboard interrupt.")
		raise
		
	except:
		DbgMsgOut("VM", "Failed to process local storage dir "+ParallelLocalStorage+".")
		raise
else:
	ParallelLocalStorage=None

	
# Get mirrored storage from environment - a directory which is common to all hosts
# in the virtual machine. This is a directory that is mounted from a common
# location (nfs exported by a server). Usually this is the /home directory. 
# This is where you start your parallel runs and keep your python scripts. 
# Can be a colon separated list of directories. If one host specifies a
# colon separated list, all must specify a list of same length with
# directories in the same order as they appear across hosts. 
if 'PARALLEL_MIRRORED_STORAGE' in environ:
	ParallelMirroredStorage=environ['PARALLEL_MIRRORED_STORAGE']
		
	# Split list in directories
	ParallelMirroredStorage=ParallelMirroredStorage.split(':')
	
	# Process list
	for i in range(0, len(ParallelMirroredStorage)):
		mirrored=ParallelMirroredStorage[i]
		
		try:
			# Expand user home (~, ~name (unix only))
			mirrored=os.path.expanduser(mirrored)
			
			# Normalize path, get canonical path (full path, eliminate symlinks)
			mirrored=os.path.realpath(mirrored)
		
		except KeyboardInterrupt:
			print "VM: keyboard interrupt"
			raise
		
		except:
			print "VM: failed to process mirrored storage dir", mirrored
		
		ParallelMirroredStorage[i]=mirrored
	del mirrored
else:
	ParallelMirroredStorage=[]
	

	
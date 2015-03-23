"""
.. inheritance-diagram:: pyopus.parallel.pvm
    :parts: 1
	
**A virtual machine based on the PVM library (PyOPUS subsystem name: PVM)**

Attempts to import :mod:`pyopus.parallel.pypvm`. If import fails the PVM 
library is not available on the platform and an arror is raised. 
"""

# PVM launcher module

from ..misc.debug import DbgMsgOut, DbgMsg
from base import VirtualMachine, HostID, TaskID, MsgTaskExit, MsgHostDelete, MsgHostAdd, MsgTaskResult

# Try importing pvm support. If it fails, raise our own exception. 
try:
	import pypvm
except ImportError:
	raise Exception, DbgMsg("PVM", "Failed to import pyPVM module. PVM not supported.")
	
import sys, os, shutil, socket
from time import time
from ..misc.env import environ

import pdb

__all__ = [ 'PVMHostID', 'PVMTaskID', 'PVM' ] 


class PVMHostID(HostID):
	"""
	A PVM host ID class based on the :class:`~pyopus.parallel.base.HostID` 
	class. 
	
	In the PVM library host IDs are integers. Valid host IDs are nonnegative. 
	
	The actual host ID can be accessed as the :attr:`dtid` member. 
	
	The host name is stored in the :attr:`name` member provided that it is 
	successfully obtained from the :attr:`pvmStatus` static member of the 
	:class:`PVM` class. If this fails *name* is used or ``Unknown`` if *name* 
	is ``None``. 
	
	See the :class:`~pyopus.parallel.base.HostID` class for more information. 
	"""
	def bad():
		"""
		A static member function. Called with ``PVMHostID.bad()``. Returns an 
		invalid host ID. 
		"""
		return PVMHostID(-1)

	def __init__(self, dtid, name=None):
		if dtid is None or dtid<0:
			dtid=-1
		self.dtid=dtid
		if name is None:
			self.name="Unknown"
		else: 
			self.name=name
		# Try to resolve dtid to host name. 
		if name is None and PVM.pvmStatus is not None:
			if dtid in PVM.pvmStatus['hosts']:
				self.name=PVM.pvmStatus['hosts'][dtid]['name']
	
	def __cmp__(self, other):
		if self.dtid==other.dtid:
			return 0
		elif self.dtid<other.dtid:
			return -1
		else:
			return 1

	def __hash__(self):
		return self.dtid
		
	def __str__(self):
		if self.valid():
			return "%s(%x)" % (self.name, self.dtid)
		else: 
			return "PVM_NOHOST"
	
	def valid(self):
		"""
		Returns ``True`` if this :class:`PVMHostID` object is valid. 
		"""
		if self.dtid<=0:
			return False
		else:
			return True


class PVMTaskID(TaskID):
	"""
	A PVM task ID class based on the :class:`~pyopus.parallel.base.TaskID` 
	class. 
	
	In the PVM library task IDs are integers. Valid task IDs are nonnegative. 
	
	The actual task ID can be accessed as the :attr:`tid` member. 
	
	See the :class:`~pyopus.parallel.base.TaskID` class for more information. 
	"""
	def bad():
		"""
		A static member function. Called with ``PVMTaskID.bad()``. Returns an 
		invalid task ID. 
		"""
		return PVMTaskID(-1)
		
	def __init__(self, tid):
		if tid is None or tid<0:
			tid=-1
		self.tid=tid
		
	def __cmp__(self, other):
		if self.tid==other.tid:
			return 0
		elif self.tid<other.tid:
			return -1
		else:
			return 1

	def __hash__(self):
		return self.tid
		
	def __str__(self):
		if self.valid():
			return "%x" % self.tid
		else: 
			return "PVM_NOTASK"
			
	def valid(self):
		"""
		Returns ``True`` if this :class:`PVMTaskID` object is valid. 
		"""
		if self.tid<=0:
			return False
		else:
			return True
		
		
class PVM(VirtualMachine): 
	"""
	A virtual machine class based on the PVM library. 
	
	One task is the master and does all the spawning. Others are just spawned 
	workers. 
	
	Assumes homogeneous clusters in terms of operating system. This means that 
	LINUX32 and LINUX64 are homogeneous while LINUX32 and WINDOWS32 are not. 
	
	See the :class:`~pyopus.parallel.base.TaskID` class for more information on 
	the constructor. 
	
	*debug* levels above 1 enable task output forwarding to the spawner (see 
	the :func:`catchout` PVM library function). 
	"""
	
	# Static members - only once per every process that uses PVM. 
	
	# For spawners and workers
	tid=None
	"""
	Static member that appears once per every Python process useing the PVM 
	library. 
	
	Represents the PVM task ID (integer). Can be obtained as ``PVM.tid``. 
	"""
	
	ptid=None
	"""
	Static member that appears once per every Python process using the PVM 
	library. 
	
	Represents the PVM parent task ID (integer) of the task that spawned this 
	task. Can be obtained as ``PVM.tid``. 
	"""
	
	dtid=None
	"""
	Static member that appears once per every Python process using the PVM 
	library. 
	
	Represents the PVM host ID (integer) of the task that spawned this task. 
	Can be obtained as ``PVM.dtid``. 
	"""
	
	# For spawners
	pvmStatus=None
	"""
	Static member that appears once per every Python process using the PVM 
	library. 
	
	Represents the status of the PVM virtual machine. Updated by the 
	:meth:`updateSpawnerInfo` method. Available only to spawners. See 
	:meth:`updateSpawnerInfo` for more information. 
	
	Can be obtained as ``PVM.pvmStatus``. 
	"""
	
	def __init__(self, debug=0, importUser=False):
		VirtualMachine.__init__(self, debug, importUser)
		
		# Turn on catchout() in debug mode
		try:
			if self.debug>1:
				pypvm.catchout(sys.stdout)
			else:
				pypvm.catchout(None)
		except KeyboardInterrupt:
			print "PVM: keyboard interrupt"
			raise
		except:
			pass
	
	def alive(self):
		"""
		Returns ``True`` if the virtual machine is alive. 
		Uses the :func:`mytid` PVM library function for the check (it fails if 
		the VM is down). 
		
		Can't use the more appropriate :func:`config` function because it leaks 
		memory due to a bug in the PVM library). This method can be called many 
		times so memory leaks are a bad thing. 
		"""
		try:
			# pypvm.config() leaks memory probably in libpvm
			# result=pypvm.config()
			
			# mytid() fails of PVM is not alive. 
			result=pypvm.mytid()
			return True
		except KeyboardInterrupt:
			print "PVM: keyboard interrupt"
			raise
		except:
			return False
			
	def waitForVM(self, timeout=-1.0, beatsPerTimeout=50):
		"""
		Waits for the virtual machien to come up (becoem alive). Returns 
		``True`` if it does. 
		
		See the :meth:`~pyopus.parallel.base.VirtualMachine.waitForVM` method 
		of the :class:`~pyopus.parallel.base.VirtualMachine` class for more 
		information. 
		"""
		if timeout<0.0:
			slice=60.0/beatsPerTimeout
		else:
			slice=timeout*1.0/beatsPerTimeout
		
		count=0
		while True:
			alive=self.alive()
			
			if alive or (timeout>=0.0 and count>=beatsPerTimeout): 
				break
			
			sleep(slice)
			count+=1
		
		return alive
	
	def slots(self):
		"""
		Returns the number of slots for tasks in a virtual machine.

		The information on the slots is gathered by the 
		:meth:`updateSpawnerInfo` method. 
		"""
		if PVM.pvmStatus is None:
			raise Exception, DbgMsg("PVM", "Task not configured as spawner.")
			
		hosts=PVM.pvmStatus['hosts']
		n=0
		for dtid, host in hosts.iteritems():
			n+=host['ncpu']
		
		return n
		
	def freeSlots(self):
		"""
		Returns the number of free slots for tasks in the virtual machine. 
		
		The information on the slots is gathered by the 
		:meth:`updateSpawnerInfo` method. 
		"""
		if PVM.pvmStatus is None:
			raise Exception, DbgMsg("PVM", "Task not configured as spawner.")
			
		hosts=PVM.pvmStatus['hosts']
		nfree=0
		for dtid, host in hosts.iteritems():
			n=host['ncpu']-host['task_count']
			if n>0:
				nfree+=n
		
		return nfree
	
	def hosts(self):
		"""
		Returns the list of :class:`PVMHostID` objects representing the nosts 
		in the virtual machine. The information on the hosts is gathered by the 
		:meth:`updateSpawnerInfo` method. 
		
		Works only for hosts that are spawners. 
		"""
		hostList=[]
		if PVM.pvmStatus is not None:
			for (dtid, host) in PVM.pvmStatus['hosts'].iteritems():
				if host['responsive']:
					hostList.append(PVMHostID(dtid, host['name']))
		return hostList
	
	def taskID(self):
		"""
		Returns the :class:`PVMTaskID` object corresponding to the calling task. 
		"""
		return PVMTaskID(PVM.tid)
	
	def hostID(self):
		"""
		Returns the :class:`PVMHostID` object corresponding to the host on 
		which the caller task runs. 
		"""
		return PVMHostID(PVM.dtid, socket.gethostname())
	
	def parentTaskID(self):
		"""
		Returns the :class:`PVMTaskID` object corresponding to the task that 
		spawned the caller task. 
		"""
		return PVMTaskID(PVM.ptid)
	
	def updateWorkerInfo(self):
		"""
		Updates the internal information used by a worker task 
		(static members :attr:`tid`, :attr:`ptid`, and :attr:`dtid`). 
		"""
		# Get my task ID
		try:
			PVM.tid=pypvm.mytid()
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			PVM.tid=None
		
		# Get my parent ID (None for master)
		try:
			PVM.ptid=pypvm.parent()
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			PVM.ptid=None
		
		# Get my host ID
		try:
			PVM.dtid=pypvm.tidtohost(self.tid)
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			PVM.dtid=None
			
		# Clear pvmStatus (we are not a spawner)
		PVM.pvmStatus=None
	
	def updateSpawnerInfo(self, timeout=-1.0):
		"""
		Updates the internal information used by a spawner task (static 
		members :attr:`tid`, :attr:`ptid`, :attr:`dtid`, and 
		:attr:`pvmStatus`). Applies *timeout* in seconds where needed. If 
		*timeout* is negative blocks until all data is collected. 
		
		:attr:`pvmStatus` is a dictionary with three members:
		
		* ``hosts`` - information about the hosts in the virtual machine
		* ``tasks`` - information about the tasks in the virtual machine
		* ``narch`` - the number of architectures in the virtual machine
		
		``hosts`` is a dictionary with ``dtid`` (integer host ID) for key and 
		members which are dictionaries with the following members: 
		
		* ``speed`` - integer giving the relative speed of the host
		* ``arch`` - the architecture of the host (string)
		* ``name`` - the name of the host (string)
		* ``task_count`` - the number of tasks running on the host
		* ``ncpu`` - the number of CPU cores in the host
		* ``responsive`` - a boolean flag indicating that the host is alive
		
		``tasks`` is a dictionary with ``tid`` (integer task ID) for key and 
		members which are dictionaries with the following members:
		
		* ``parent`` - the ``tid`` (integer task ID) of the parent task
		* ``dtid`` - the ``dtid`` (integer host ID) on which the task is running
		* ``flags`` - an integer representing the flags set for the task
		* ``name`` - the name of the task
		* ``pid`` - the process identifier of the task
		
		``hosts`` and ``tasks`` are obtained with the PVM library functions 
		:func:`config` and :func:`tasks`. Because :func:`config` leaks memory 
		(PVM library bug), this method should be called only occasionally. 
		
		The ``ncpu`` and the ``responsive`` dictionary members of the ``hosts`` 
		dictionary are obtained with the :meth:`updateCoreCount` method. Hosts 
		marked as unresponsive are not used for spawning new tasks. 
		
		Returns ``True`` on success. 
		"""
		
		# Update worker info
		self.updateWorkerInfo()
		
		# Get hosts and tasks
		try:
			cfg=pypvm.config()
			tsk=pypvm.tasks()
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			PVM.pvmStatus=None
			return False
			
		# Number of architectures
		narch=cfg[1]
		
		# Host list
		hosts={}
		for host in cfg[2]:
			# Add host description
			hosts[host['dtid']]={
				'speed': host['speed'], 
				'arch': host['arch'], 
				'name': host['hostname'], 
				'task_count': 0, 
				'ncpu': None, 
				'responsive': False #
			}
		
		# Task list
		tasks={}
		for task in tsk:
			tid=task[0]
			dtid=task[2]
			
			# Add task description		
			tasks[tid]={
				'parent': task[1], 
				'dtid': task[2], 
				'flags': task[3], 
				'name': task[4], 
				'pid': task[5]
			}
			
			# Increase task count for a host
			hosts[dtid]['task_count']+=1
		
		# Build pvmStatus - host and task IDs are not PVMHostID nor PVMTaskID objects!
		# This is an internal structure. 
		PVM.pvmStatus={
			'hosts': hosts,
			'tasks': tasks, 
			'narch': narch
		}
		
		# Update core count
		self.updateCoreCount(timeout)
		
		return True

	def formatSpawnerConfig(self):
		"""
		Formats the configuration information gathered by a spawner task as a 
		string. Works only if called by a spawner task.  
		"""
		if PVM.pvmStatus is None:
			raise Exception, DbgMsg("PVM", "Task not configured as spawner.")
			
		hosts=PVM.pvmStatus['hosts']
		tasks=PVM.pvmStatus['tasks']
		
		txt=""
		txt+="Architectures: "+str(self.pvmStatus['narch'])+"\n\n"
		txt+="Hosts and tasks:\n"
		for (dtid, info) in hosts.iteritems():
				if info['ncpu'] is None:
					cpuinfo="??"
				else:
					cpuinfo="%-2d" % info['ncpu']
				
				if info['responsive']:
					reach=""
				else:
					reach="not responsive"
					
				txt+="dtid=%-7x speed=%-7d arch=%s ncpu=%2s %s %s\n" % (dtid, info['speed'], info['arch'], cpuinfo, reach, info['name'])
				
				for (tid, procinfo) in tasks.iteritems():
					if procinfo['dtid']==dtid:
						txt+="\ttid=%-7x flags=%-3d pid=%-7d '%s'\n" % (tid, procinfo['flags'], procinfo['pid'], procinfo['name'])
			
		return txt
		
	def spawnerBarrier(self, timeout=-1.0):
		"""
		Tells the PVM library to notify this task (the spawner) to receive 
		information on new hosts and host failures. 
		
		*timeout* in seconds is used where applicable. Negative values stand 
		for infinite *timeout*. 
		"""
		# Receive notification messages about new hosts. 
		self._monitorNewHost()
		# Update spawner information (hosts, tasks). 
		if self.updateSpawnerInfo(timeout):
			# Receive notification messages about failed hosts. 
			self._monitorHostFailure(PVM.pvmStatus['hosts'].keys())
		
	def spawnFunction(self, function, args=(), kwargs={}, count=None, targetList=None, sendBack=False):
		"""
		Spawns a *count* instances of a Python *function* on remote hosts and 
		passes *args* and *kwargs* to the function. Spawning a function 
		actually means to start a Python interpreter, import the function, and 
		call it with *args* and *kwargs*. 
		
		*targetList* specifies the hosts on which the tasks are started. 
		
		If *sendBack* is ``True`` the status and the return value of the 
		spawned function are sent back to the spawner with a 
		:class:`pyopus.parallel.base.MsgTaskResult` message when the spawned 
		function returns. 
		
		Invokes the :func:`runFunctionWithArgs` function on the remote host for 
		starting the spawned function. 
		
		If spawning succeeds, updates the ``tasks`` and ``hosts`` structures of 
		the :attr:`pvmStatus` member. 
		
		Makes sure the spawned tasks are spread as uniformaly as possible 
		across the hosts. 
		
		Returns a list of :class:`TaskID` objects representing the spawned 
		tasks. 
		
		See the :meth:`~pyopus.parallel.base.spawnFunction` method of the 
		:class:`~pyopus.parallel.base.VirtualMachine` class for more 
		information. 
		"""
		# Get hosts if not specified
		if targetList is None:
			hostList=self.hosts()
		else:
			hostList=targetList
		
		# Order hosts
		hostIDs=self.orderHosts(hostList)

		# No tasks spawned for now
		taskIDlist=[]
		
		# Get module where runFunctionWithArgs resides
		myModule=runFunctionWithArgs.__module__
		
		# Import user module in spawned task, if required. 
		userImportStr=''
		if self.importUser:
			userImportStr='import user;'
		
		# Do we have any hosts to spawn on? 
		if len(hostIDs)>0:
			# Ok, start
			atHostIndex=0
			spawnedCount=0
			# Loop until not enough task are spawned
			while count is None or len(taskIDlist)<count:
				# Cycle through ordered hosts, fill every one of them with tasks. 
				# Add tasks to host until it is fully loaded with tasks. 
				# Then add one task per every host, until the desired number of tasks is reached. 
				
				# Get host name and dtid
				dtid=hostIDs[atHostIndex].dtid
				
				# Get target host information
				targetHost=PVM.pvmStatus['hosts'][dtid]
				
				# Get free slots of a host
				freeSlots=targetHost['ncpu']-targetHost['task_count']
				
				# If count was not specified, stop as soon as we find a host that has no free slots.
				if count is None and freeSlots<=0:
					break
				
				# Spawn and get tid
				tids=pypvm.spawn(getPythonBinary(targetHost['arch']), [ '-c', 
					userImportStr+'from '+myModule+' import runFunctionWithArgs;runFunctionWithArgs()'
				], pypvm.spawnOpts['TaskHost'], targetHost['name'], 1)
				tid=tids[0]
				# Increase spawned task count. 
				spawnedCount+=1 
				
				# Check if pvm level spawn succeeded 
				if tid>0:
					# Turn on task failure monitoring.
					self._monitorTaskFailure(tids)
					
					# Update internal pvm status data
					PVM.pvmStatus['tasks'][tid]={
						'parent': PVM.tid, 
						'dtid': pypvm.tidtohost(tid), 
						'flags': 0, # Unknown
						'name': "", # Unknown
						'pid': -1	# Unknown
					}
				
					# Spawn OK, add to taskID list
					taskIDlist.append(PVMTaskID(tid))
					
					# Increase process count for a host
					targetHost['task_count']+=1
					
					# Send init info (PVM object)
					if not sendPVMMessage(tid, PVMMSGTAG_INIT_WORKER, self):
						# Send init info failed
						# print "send init info failed"
						pass
						
					# Send args
					if not sendPVMMessage(tid, PVMMSGTAG_SPAWN_FUNCTION, 
							(function, args, kwargs, sendBack)
						):
						# Send spawn function failed
						# print "send spawn function failed"
						pass
				else:
					# Spawn failed
					# print "spawn failed"
					pass
				
				# If free slots was <=1 before this spawn, advance to next host
				if freeSlots<=1:
					atHostIndex+=1
					if atHostIndex>=len(hostIDs):
						atHostIndex=0
				
				# Stop if count is specified and we have spawned the desired number of tasks. 
				if count is not None and spawnedCount>=count:
					break

		return taskIDlist
	
	def checkForIncoming(self):
		"""
		Returns ``True`` if there is a message waiting to be received. 
		"""
		return checkForIncoming()
		
	def receiveMessage(self, timeout=-1.0):
		"""
		Receives a *message* (a Python object) and returns a tuple 
		(*senderTaskId*, *message*)
		
		The sender of the *message* can be identified through the 
		*senderTaskId* object of class :class:`PVMTaskID`. 
		
		If *timeout* (seconds) is negative the function waits (blocks) until 
		some message arrives. If *timeout*>0 seconds pass without receiving a 
		message, an empty tuple is returned. Zero *timeout* performs a 
		nonblocking receive which returns an empty tuple if no message is 
		received. 
		
		In case of an error the return value is ``None``. 
		
		Handles transparently all 
		
		* new hosts notification messages
		* failed host notification messages
		* task exit notification messages
		* spawned function return value messages
		
		Discards all other low-level PVM messages that were not sent with the 
		:meth:`sendMessage` method. 
		"""
		# receive any message from any source
		received=receivePVMMessage(timeout)
		
		# None means error. No message. 
		if received is None:
			return None
		
		# Empty tuple means timeout. No message.  
		if len(received)==0:
			return ()
		
		# Unpack tuple
		(source, pvmMsgTag, bufid)=received
		
		# Is it a message? 
		if pvmMsgTag==PVMMSGTAG_MESSAGE:
			# Unpack message
			message=pypvm.upk()
			return (PVMTaskID(source), message)
		elif pvmMsgTag==PVMMSGTAG_FUNCTION_RETURN:
			# Unpack success flag and response
			message=pypvm.upk()
			(success, response)=message
			return (PVMTaskID(source), MsgTaskResult(success, response))
		elif pvmMsgTag==PVMMSGTAG_TASK_EXIT:
			# A task has exited
			tid=pypvm.upkint(1)[0]
			taskID=PVMTaskID(tid)
			if self.debug:
				DbgMsgOut("PVM", "Task "+str(taskID)+" exit detected.")
			# Remove task from task list, update host. 
			if PVM.pvmStatus is not None:
				if tid in PVM.pvmStatus['tasks']:
					dtid=PVM.pvmStatus['tasks'][tid]['dtid']
					del PVM.pvmStatus['tasks'][tid]
					if dtid in PVM.pvmStatus['hosts']:
						PVM.pvmStatus['hosts'][dtid]['task_count']-=1
			# source is 0x80000000
			# Behave as if the source is the task that exited. This will enable us to use filters in EventDrivenMS.
			return (taskID, MsgTaskExit(taskID))
		elif pvmMsgTag==PVMMSGTAG_HOST_DELETE:
			# A host is down
			dtid=pypvm.upkint(1)[0]
			hostID=PVMHostID(dtid)
			if self.debug:
				DbgMsgOut("PVM", "Host "+str(hostID)+" failure detected.")
			# Remove host. 
			if PVM.pvmStatus is not None:
				if dtid in PVM.pvmStatus['hosts']:
					del PVM.pvmStatus['hosts'][dtid]
				# Delete all its tasks (don't have to because we will get TASK_EXIT messages for them). 
				#for tid in PVM.pvmStatus['tasks'].keys():
				#	if PVM.pvmStatus['tasks'][tid][dtid]:
				#		del PVM.pvmStatus['tasks'][tid]
			# source is 0x80000000
			# These are not filtered usually so we do not have fo fake the message source task for filtering. 
			# Anyway what is the source of such a message? 
			return (PVMTaskID(-1), MsgHostDelete(PVMHostID(dtid)))
		elif pvmMsgTag==PVMMSGTAG_HOST_ADD:
			# A host was added
			cnt=pypvm.upkint(1)[0]
			dtids=pypvm.upkint(cnt)
			self._monitorHostFailure(dtids)
			# This one has a catch. The easiest thing to do is to call updateSpawnerInfo(). 
			# Unfortunately pypvm.config() leaks memory so it should be called too often. 
			if PVM.pvmStatus is not None:
				if self.debug:
					DbgMsgOut("PVM", "Updating host and task info.")
				self.updateSpawnerInfo(-1)
			
			# Build list of hostIDs
			hostIDs=[]
			for dtid in dtids:
				hostID=PVMHostID(dtid)
				hostIDs.append(hostID)
				if self.debug:
					DbgMsgOut("PVM", "New host "+str(hostID)+" detected.")
			
			# source is 0x80000000
			# These are not filtered usually so we do not have fo fake the message source task for filtering. 
			# Anyway what is the source of such a message? 
			return (PVMTaskID(-1), MsgHostAdd(hostIDs))
		else: 
			# Unknown message. No message.  
			return None
			
	def sendMessage(self, destination, message):
		"""
		Sends *message* (a Python object) to a task with :class:`PVMTaskID` 
		*destination*. Returns ``True`` on success. 
		"""
		
		return sendPVMMessage(destination.tid, PVMMSGTAG_MESSAGE, message)

		
	# 
	# Private methods
	#
	
	def _monitorTaskFailure(self, tidlist):
		"""
		Enables task exit notification messages sent back to the calling task. 
		The list of `tid numbers (integers) of process whose exit results in a 
		notification message is given by *tidlist*. 
		"""
		try:
			pypvm.notify(pypvm.notifyDict['TaskExit'], PVMMSGTAG_TASK_EXIT, tidlist)
			return True
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			return False
			
	def _monitorHostFailure(self, dtidlist):
		"""
		Enables host failure notification messages sent back to the calling 
		task. The list of `dtid` numbers (integers) of hosts whose failure 
		results in a notification message is given by *dtidlist*. 
		"""
		# Convert to PVM dtid list
		try:
			pypvm.notify(pypvm.notifyDict['HostDelete'], PVMMSGTAG_HOST_DELETE, dtidlist)
			return True
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			return False
	
	def _monitorNewHost(self):
		"""
		Enables new host notification messages sent to the calling task. 
		"""
		try:
			pypvm.notify(pypvm.notifyDict['HostAdd'], PVMMSGTAG_HOST_ADD, [], -1)
			return True
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			return False
	
	
	#
	# PVM specific methods
	# 
	
	def orderHosts(self, hostList):
		"""
		Returns a list of :class:`PVMHostID` objects representing the members 
		of *hostList* list of :class:`PVMHostID` objects sorted by free slots 
		(most free slots first). 
		
		The information on used slots comes from the :attr:`pvmStatus` static 
		member. If a host is marked as not ``responsive``, it is ommitted from 
		the list. 
		"""
		
		hostStatus=PVM.pvmStatus['hosts']
		# Build (free slots, dtid) list
		ltmp=[]
		for host in hostList:
			dtid=host.dtid
			
			# Skip hosts thar are not present
			if dtid not in hostStatus:
				continue
			hostInfo=hostStatus[dtid]
			
			# Skip unresponsive hosts
			if not hostInfo['responsive']:
				continue
			
			# Calculate free slots
			freeSlots=hostInfo['ncpu']-hostInfo['task_count']
			ltmp.append((freeSlots, dtid))
		
		# Least free slots first
		ltmp.sort()
		
		# Build return values
		ids=[]
		for entry in ltmp:
			(slots, dtid)=entry
			ids.append(PVMHostID(dtid))
		
		ids.reverse()
		return ids
	
	def kill(self, taskID):
		"""
		Kills a task represented by the *taskID* :class:`PVMTaskID` object and 
		updates the :attr:`pvmStatus` static member. 
		"""
		
		tid=taskID.tid
		try:
			if pypvm.kill(tid)<0:
				# Error
				return False
			else:
				# OK
				# Reduce number of tasks on a host
				dtid=PVM.pvmStatus['tasks'][tid]['dtid']
				if dtid in PVM.pvmStatus['hosts']:
					PVM.pvmStatus['hosts'][dtid]["task_count"]-=1
				return True
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		except:
			# Error
			return False
	
	def killAll(self): 
		"""
		Kills all tasks listed in the :attr:`pvmStatus` static member and 
		updates the :attr:`pvmStatus` static member. Does not kill the calling 
		task. 
		
		Works only on spawner. 
		"""
		tasks=PVM.pvmStatus['tasks']
		
		for tid in tasks.keys():
			# Don't shoot yourself :)
			if tid==PVM.tid:
				continue
			
			self.kill(PVMTaskID(tid))
	
	def updateCoreCount(self, timeout=-1.0):
		"""
		Updates the CPU core count in the :attr:`pvmStatus` static member. 
		
		The update is performed by spawning the :func:`reportHostInfo` function 
		on all hosts listed in the :attr:`pvmStatus` static member. The 
		function detects the number of CPU cores and sends back a message with 
		the result. Communication (send back) is performed on the lowest level 
		so :meth:`receiveMessage` knows nothing of it. The response messages 
		are collected with a given *timeout* in seconds. Negative timeout means 
		infinite timeout. 
		
		If some hosts do not respond within *timeout* seconds, they are marked 
		as ``unresponsive``. 
		"""
		hosts=PVM.pvmStatus['hosts']
		tasks=PVM.pvmStatus['tasks']
		
		# Spawn tasks that will count the CPUs and report back. 
		spawnedtids=[]
		for dtid in hosts.keys():
			# Get host info
			targetHost=hosts[dtid]
			
			# Get module where reportCoreCount function is defined (this module)
			myModule=reportHostInfo.__module__
			
			# Import user module if required
			if self.importUser:
				userImportStr='import user;'
			else:
				userImportStr=''
			
			# Spawn Python on host where pvmd with given dtid resides, import reportCoreCount() from module, call it
			tids=pypvm.spawn(getPythonBinary(targetHost['arch']), [ '-c', userImportStr+'from '+myModule+' import reportHostInfo;reportHostInfo()' ], pypvm.spawnOpts['TaskHost'], targetHost['name'], 1)
			
			# Only one task spawned, tids is a list with one member
			tid=tids[0]
			
			# If spawning failed (tid<0) mark host as unresponsive
			if tid<=0:
				hosts[dtid]["responsive"]=False
			else:
				hosts[dtid]["responsive"]=True
			
			# Add tid to list of spawned tids
			spawnedtids.append(tid)
		
		# Wait for incoming messages
		timeoutPerTask=timeout*1.0/len(spawnedtids)
		spawnedtids=set(spawnedtids)
		errors=0
		mark=time()
		# Until all reposnes received and no more than 5 errors
		while len(spawnedtids)>0 and errors<=5: 
			# Receive message
			result=receivePVMMessage(timeoutPerTask, PVMMSGTAG_NCPU)
			
			# Process result
			if result is None:
				# Error
				errors+=1
			elif len(result)==0:
				# Timeout, do nothing
				pass
			else:
				# Received a message
				(tid, pvmmsgtag, bufid)=result
				# Get dtid (host)
				dtid=pypvm.tidtohost(tid)
				# Get number of cores
				ncpu=pypvm.upk()
				# Store number of cores
				hosts[dtid]['ncpu']=ncpu
				# Remove from set
				spawnedtids.remove(tid)
			
			# Check for timeout
			if timeout>0 and time()-mark>timeout:
				break
		
		# Mark all hosts that did not respond as unresponsive. 
		for tid in spawnedtids:
			# Get dtid (host)
			dtid=pypvm.tidtohost(tid)
			hosts[dtid]["responsive"]=False
		

#
# Helper functions for reporting the number of CPU cores
#

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

def reportHostInfo():
	"""
	Reports the number of available CPU cores to the parent task (:attr:`ptid` 
	static member). 
	
	This function should never be called by the user. 
	"""
	mytid=pypvm.mytid()
	ptid=pypvm.parent()
	
	ncpu=getNumberOfCores()
	
	sendPVMMessage(ptid, PVMMSGTAG_NCPU, ncpu)
	
	sys.stdout.flush()
	
	# Exit from PVM cleanly. 
	pypvm.exit()	


#
# Message handling at PVM level. 
#
	
def sendPVMMessage(tid, id, pvmMsg):
	"""
	Sends a message (binary string *pvmMsg*) to the task with pvm *tid* 
	(integer). The message is tagged with an integer *id*. 
	
	Should never be called by the user if the communication methods of 
	:class:`PVM` are used. 
	"""
	try:
		pypvm.initsend()
		pypvm.pk(pvmMsg)
		pypvm.send(tid, id)
		return True
	except KeyboardInterrupt:
		print "PVM: keyboard interrupt"
		raise
	except:
		return False
			
def receivePVMMessage(timeout=-1.0, pvmMsgTag=-1, tid=-1):
	"""
	Receives a message (binary string) with tag *pvmMsgTag* (integer) from a 
	task with given *tid* (integer). If *tid* is negative messages from all 
	tasks are accepted. If *pvmMsgTag* is negative messages with all tids are 
	accepted. 
	
	If the message is not received within *timout* seconds, returns an empty 
	tuple. Negative values of *timeout* stands for infinite timeout. 
	
	If a message is received a tuple (*tid*, *msgTag*, *bufID*) is returned 
	where *tid* is the integer task identifier of the sender, *msgTag* is the 
	integer tag of the message, and *bufID* is the number of the buffer where 
	the message is stored (integer). 
	
	The user can unpack the message from the buffer by calling the 
	:func:`~pyopus.parallel.pypvm.upk` function from the 
	:mod:`pyopus.parallel.pypvm` module. 
	
	Returns ``None`` if an error occurs. 
	
	Should never be called by the user if the communication methods of 
	:class:`PVM` are used. 
	"""
	try:
		if timeout>0.0:
			bufID=pypvm.trecv(timeout, pvmMsgTag, tid)
		elif timeout==0.0:
			bufID=pypvm.nrecv(tid, pvmMsgTag)
		else:
			bufID=pypvm.recv(tid, pvmMsgTag)
	except KeyboardInterrupt:
		raise KeyboardInterrupt
	except:
		return None
	
	if bufID>0: 
		(bytes, msgTag, tid)=pypvm.bufinfo(bufID)
		return (tid, msgTag, bufID)
	elif timeout>=0.0 and bufID==0:
		# Timeout, return empty tuple
		return ()
	else:	
		# Error, return None
		return None

def checkForIncoming(pvmMsgTag=-1):
	"""
	Returns ``True`` if a message with tag *pvmMsgTag* is waiting to be 
	received. Negative values of *pvmMsgTag* match a message with an arbitrary 
	tag. 
	"""
	try:
		if pypvm.probe(-1, pvmMsgTag)>0:
			return True
		else:
			return False
	except KeyboardInterrupt:
		print "PVM: keyboard interrupt"
		raise
	except:
		return False

def getPythonBinary(pvmArch):
	"""
	Returns the Python interpreter's executable name for the given 
	architecture. 
	
	Currently works for Linux and Windows. 
	"""
	if pvmArch.find("LINUX")==0:
		return "python"
	else:
		return "python.exe"
	
def runFunctionWithArgs(): 
	"""
	Receives a low-level pvm message with the object of the :class:`PVM` class 
	sent by the spawner (:meth:`PVM.spawnFunction` method). Using the received 
	object it creates local storage, performs mirroring of filesystem objects, 
	and sets up the working directory. 
	
	Next it receives a low-level pvm message with the function, its arguments, 
	and the *sendBack* flag sent by the :meth:`PVM.spawnFunction` method. It 
	calls the functionm with the received arguments and catches any exception 
	in the function. 
	
	If the *sendBack* flag in ``True`` the status (whether there was no 
	exception) and the return value of the function are sent back to the 
	spawner in a low-level pvm message
	
	In the end this function exits the Python interpreter. 
	"""
	mytid=pypvm.mytid()
	ptid=pypvm.parent()
	
	# print "Python path:", os.environ["PYTHONPATH"]
	# print "Directory  :", os.getcwd()
	
	# Receive args (block until received)
	response=receivePVMMessage(-1, PVMMSGTAG_INIT_WORKER, ptid)
	if response is not None and len(response)!=0: 
		(tid, tag, bufid)=response
		
		# Receive OK, get init info
		pvm=pypvm.upk()
		
		# Update worker info
		pvm.updateWorkerInfo()
				
		# Create local storage, mirror, and set work directory
		taskStorage=pvm.prepareEnvironment()
		
	# Receive args (block until received)
	response=receivePVMMessage(-1, PVMMSGTAG_SPAWN_FUNCTION, ptid)
	if response is not None and len(response)!=0: 
		(tid, tag, bufid)=response
		
		# Receive OK, get spawn_info
		spawn_info=pypvm.upk()
		
		# spawn_info is a tuple (function, args, kwargs, send_back)
		(function, args, kwargs, sendBack)=spawn_info
				
		# Assume spawn failed
		succes=False
		response=None
		
		# Call function
		try:
			response=function(*args, **kwargs)
			success=True
		except KeyboardInterrupt:
			DbgMsgOut("PVM", "Keyboard interrupt.")
			raise
		
		# Flush stdout so parent will receive it. 
		sys.stdout.flush()
		
		# Send back response (if requested)
		if sendBack:
			sendPVMMessage(PVM.ptid, PVMMSGTAG_FUNCTION_RETURN, (success, response))
	else:
		# Receive failed, nothing to do as we don't know if we must report back
		pass
	
	# Exit from PVM cleanly. 
	pypvm.exit()
	

# PVM Message tags for low-level PVM messages

PVMMSGTAG_NCPU=1
"""
The message tag for low-level pvm messages reporting the number of CPU cores. 
"""

PVMMSGTAG_TASK_EXIT=21
"""
The message tag for low-level pvm notification messages reporting that a task 
exited. 
"""

PVMMSGTAG_HOST_DELETE=22
"""
The message tag for low-level pvm notification messages reporting a host failed. 
"""

PVMMSGTAG_HOST_ADD=23
"""
The message tag for low-level pvm notification messages reporting a new host 
was added to the virtual machine. 
"""

PVMMSGTAG_INIT_WORKER=30
"""
The message tag for low-level pvm messages carrying the :class:`PVM` class 
object to the host where a Python function will be spawned. 
"""

PVMMSGTAG_SPAWN_FUNCTION=31
"""
The message tag for low-level pvm messages carrying the function, its 
arguments and the *sendBack* flag to the host where a Python function will be 
spawned. 
"""

PVMMSGTAG_FUNCTION_RETURN=32
"""
The message tag for low-level pvm messages carrying the status and the return 
value of the spawned Python function.
"""

PVMMSGTAG_MESSAGE=100
"""
The message tag for low-level pvm messages carrying Python objects sent by the 
:meth:`PVM.sendMessage` method. 
"""

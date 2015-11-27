"""
.. inheritance-diagram:: pyopus.parallel.cooperative
    :parts: 1
	
**Cooperative multitasking OS with task outsourcing (PyOPUS subsystem name: COS)**

This module is based on the :mod:`greenlet` module. Concurrent tasks can be created 
in a UNIX-like fashion (with the :meth:`Spawn` method). The return value of a task 
or multiple tasks is collected with the :meth:`Join` method. Joins can be blocking 
or nonblocking. 

The cooperative multitasking OS takes care of outsourcing the tasks to computing 
nodes if it is permitted to do this and there are computing nodes available. 
Outsourcing uses the virtual machine specified by calling the :meth:`setVM` method. 
If no virtual machine is specified outsourcing is not possible. 

COS makes it possible to implement multilevel parallelism and asynchronous 
algorithms in a simple manner. Parallel algorithms can be run on a single CPU 
by taking advantage of the :mod:`greenlet` module for providing the microthread 
functionality. Every local task is a microthread that runs concurrently with 
other microthreads. The microthreads are cooperatively scheduled by COS. 
Of course such a run is slower than a real parallel run involving multiple 
tasks across multiple processors. 
"""

from greenlet import greenlet
from pyopus.parallel.base import MsgTaskResult, MsgTaskExit
from ..misc import identify

__all__ = [ 'cOS', 'Task', 'SysCall', 'Spawn', 'Yield', 'Join', 'Scheduler', 'OpusOS' ]

# System calls 

# Base class
class SysCall(object):
	"""
	Base class for COS system calls.
	"""
	def handle(self, sched, task):
		raise Exception("COS: This is an abstract class.")

# Spawn a child task
class Spawn(SysCall):
	"""
	System call for spawning a child task. 
	
	* *func* - function defining the task
	* *args* - positional arguments to the task's function
	* *kwargs* - keyword arguments to the task's function
	* *remote* - ``True`` if a remote spawn is requested. Spawns a local 
	  task if the scheduler has no VM object or *remote* is ``False``. 
	* *block* - ``True`` if a remote spawn should block until a slot is 
	  free. Has no effect for local spawns. Nonblocking remote spawn 
	  returns -1 on failure. 
	
	Returns the tid of a spawned task. 
	"""
	def __init__(self, func, args=[], kwargs={}, remote=False, block=True):
		self.func=func
		self.remote=remote
		self.block=block
		self.args=args
		self.kwargs=kwargs
	
	def handle(self, sched, task):
		# Check if we can hope to spawn remotely, if not now then at least sometime in the future
		if self.remote and sched.vm is not None and sched.vm.freeSlots()+len(sched.remoteTasks)>0:
			# Remote spawn
			newTask=sched.new(parent=task, func=self.func, remote=True, args=self.args, kwargs=self.kwargs)
			if newTask is None:
				# No slots for a remote task
				if self.block:
					# Blocking spawn
					# Enqueue this spawn
					sched.enqueueSpawn(self, task)
				else:
					# Not blocking, return negative tid
					task.sendval=-1
					# Schedule spawner task
					sched.schedule(task)
			else:
				# Success, return tid
				task.sendval=newTask.tid
				# Schedule spawner task
				sched.schedule(task)
		else:
			# Local spawn
			newTask=sched.new(parent=task, func=self.func, remote=False, args=self.args, kwargs=self.kwargs)
			task.sendval=newTask.tid
			# Schedule spawner task
			sched.schedule(task)
		
# Yield control to the scheduler
class Yield(SysCall):
	"""
	A system call for yielding the control to the scheduler. 
	
	The scheduler switches the context to the next scheduled microthread.
	"""
	def handle(self, sched, task):
		sched.schedule(task)

# Join a child task
class Join(SysCall):
	"""
	A system call for joining a child task. 
	Return the task's tid and its return value. 
	
	* *tidlist* - list of task IDs that we are waiting on. 
	  Empty list waits on all child tasks. 
	* *block* - ``True`` if the call should block until a task can 
	  be joined
	  
	Returns a dictionary with tid for key holding the return value 
	of the task that was joined. The dictionary has a single entry. 
	
	Returns an empty dictionary if there are no children. 
	
	Failed nonblocking join returns an empty dictionary. 
	"""
	def __init__(self, tidList=[], block=True):
		self.tidList=tidList
		self.block=block
		
	def handle(self, sched, task):
		# Prepare an empty return value for the syscall
		task.sendval={}
		
		# No finished child found
		finishedChild=None
		
		# Empty tid list
		if len(self.tidList)==0:
			# No children
			if task.nchildren+len(task.finishedChildren)==0:
				# Schedule the task again
				sched.schedule(task)
				# Return an empty sendval
				return
			# A finished child exists
			if len(task.finishedChildren)>0:
				# Get child
				childTid=task.finishedChildren.keys()[0]
				finishedChild=task.finishedChildren[childTid]
		else:		
			# Verify if listed children exist, verify if they are really children
			for tid in self.tidList:
				# Waiting on tid 0 (main task) is not allowed 
				if tid==0:
					task.localThrow(Exception, "COS: Waiting on task 0 (main) is not allowed.") 
				if tid in sched.tasks:
					# Child is running
					child=sched.tasks[tid]
				elif tid in task.finishedChildren:
					# Child is finished, get it
					child=task.finishedChildren[tid]
					finishedChild=child
				else:
					task.localThrow(Exception, "COS: Child %d does not exist." % tid) 
				if child.ptid!=task.tid:
					task.localThrow(Exception, "COS: Task %d is not a child of %d." % (tid, task.tid))
		
		# Finished child found
		if finishedChild is not None:
			# Remove it
			del task.finishedChildren[finishedChild.tid]
			# Construct return value and schedule the task
			task.sendval[finishedChild.tid]=finishedChild.retval
			sched.schedule(task)
			return
		
		# If the call is nonblocking, return an empty sendval
		if not self.block:
			# Schedule the task again
			sched.schedule(task)
			# Return an empty sendval
			return
		else:
			# No finished child found, stop task because the call is blocking
			task.waitingOn=self.tidList
			task.status=Task.Swaiting
		

# Task wrapper 

class Task(object):
	"""
	Task wrapper object. Wraps one microthread or one remote task. 
	
	Arguments: 
	
	* *parent* - parent task. Should be ``None`` for teh main task. 
	* *greenlet* - greenlet of a local task
	* *remotetaskID* - ID of a remote task
	* *name* - task name. If ``None`` a name is generated. 
	  from *greenlet* or *remoteTaskID*
	* *args* - positional arguments to the greenlet passed at startup
	* *kwargs* - keyword arguments to the greenlet passed at startup
	
	If *args* and *kwargs* are ``None`` the greenlet is assumed to be 
	already running. 
	
	Members: 
	
	* ``tid`` - task id
	* ``ptid`` - parent task id
	* ``nchildren`` - number of children
	* ``finishedChildren`` - dictionary of finished children waiting to be 
	  joined. The key is the tid of a child. 
	* ``sendval`` - return value passed to the task at switch
	* ``retval`` - value returned by the finished task
	* ``waitingOn`` - list of child task ids the task is waiting on
	* ``status`` - 0 (created), 1 (running), 2 (stopped), 3 (finished)
	"""
	
	Screated=0
	Srunning=1
	Swaiting=2
	Sfinished=3
	
	# 0 is the scheduler, 1 is the main task (created at scheduler startup)
	taskid=1
	
	def __init__(self, parent, greenlet=None, remoteTaskID=None, name=None, func=None, args=[], kwargs={}):
		if greenlet is not None:
			# Local task
			self.remoteTaskID=None
			self.greenlet=greenlet
			if args is None and kwargs is None:
				self.status=Task.Srunning
				#args=[]
				#kwargs={}
			else:
				self.status=Task.Screated
		else:
			# Remote task, assume it is already running
			self.greenlet=None
			self.remoteTaskID=remoteTaskID
			self.name=name if name is not None else str(remoteTaskID)
			# Remote tasks are already running
			self.status=Task.Srunning
		
		self.func=func
		self.args=args
		self.kwargs=kwargs
		self.ptid=parent.tid if parent is not None else None
		
		if name is not None:
			self.name=name
		elif func is not None:
			self.name=str(func)
		elif greenlet is not None:
			self.name=str(greenlet)
		else:
			self.name=str(remoteTaskID)
		self.tid=Task.taskid
		Task.taskid+=1
		if parent is not None:
			parent.nchildren+=1
		self.nchildren=0
		self.finishedChildren={}
		self.sendval=None
		self.retval=None
		self.waitingOn=None
	
	#def __repr__(self):
	#	return str(self.tid)+"("+str(self.remoteTaskID)+")"
		
	def switchToLocal(self):
		"""
		Switch control to the local task (microthread) represented by this object. 
		"""
		# Is the task running
		if self.greenlet is None:
			raise Exception("COS: Cannot switch to remote task tid=%d." % self.tid)
		if self.status==Task.Screated:
			# Send arguments (task startup)
			self.status=Task.Srunning
			retval=self.greenlet.switch(*self.args, **self.kwargs)
		else:
			# Send the value to task
			retval=self.greenlet.switch(self.sendval)
			
		# Check if it is finished
		if self.greenlet.dead:
			self.status=Task.Sfinished
			
		# Reset value that will be sent at next switch
		self.sendval=None
		return retval
	
	def localThrow(self, exception, value):
		"""
		Throws an *exception* with value *value* in the local task represented 
		by this object. 
		"""
		if self.greenlet is None:
			raise Exception("COS: Cannot switch to a remote task.")
		self.greenlet.throw(exception, value)


# Scheduler 

class Scheduler(object):
	"""
	Cooperative multitasking scheduler based on greenlets. 
	
	*vm* - virtual machine abstraction for spawning remote tasks. 
	
	The main loop of the scheduler is entered by calling 
	the scheduler object. 
	"""
	# Initially main task is active
	
	def __init__(self, vm=None):
		# Task ID of scheduler
		self.tid=0
		# Local task queue
		self.ready=[]
		# All tasks that are running (i.e. not finished), tid for key
		self.tasks={}
		# Remote tasks that are not finished, remote task ID for key
		self.remoteTasks={}
		# A queue for spawn system calls waiting on a free remote slot
		self.waitingOnSlot=[]
		# Set VM
		self.setVM(vm)
	
	def countTasks(self):
		"""
		Returns the number of running tasks including the scheduler 
		and the main task. 
		"""
		return len(self.tasks)
	
	def countRemoteTasks(self):
		"""
		Returns the number of running remote tasks. 
		"""
		return len(self.remoteTasks)
	
	def countLocalTasks(self):
		"""
		Returns the number of running local tasks including the scheduler 
		and the main task. 
		"""
		return len(self.tasks)-len(self.remoteTasks)
	
	def setVM(self, vm):
		"""
		Sets the VM abstraction object used for spawning remote tasks. 
		
		Allowed only when no remote tasks are running. 
		Setting a VM object on remote task has no effect. 
		"""
		if len(self.remoteTasks)>0:
			raise Exception("COS: Remote tasks running. Cannot replace VM.")
		
		if vm is not None and vm.parentTaskID().valid():
			# Valid parent task, this is a slave
			# Do not allow spawning remote tasks
			self.vm=None
		elif vm is not None and vm.slots()<2:
			# Need at lest 2 slots for remote spawning (1 for master and 1 for worker)
			self.vm=None
		else:
			self.vm=vm
	
	def new(self, parent, func, args=[], kwargs={}, remote=False):
		"""
		Create a new task. 
		
		*parent* - parent task object
		*func* - function defining the task
		*args* - positional arguments to the task's function
		*kwargs* - keyword arguments to the task's function
		*remote* - ``True`` for a remote task
		"""
		if not remote or self.vm is None: 
			# Create a greenlet
			g=greenlet(func)
			task=Task(parent, greenlet=g, func=func, args=args, kwargs=kwargs)
			self.tasks[task.tid]=task
			self.schedule(task)
		else:
			# Spawn a remote task
			if self.vm.freeSlots()<=0:
				return None
			
			taskIDs=self.vm.spawnFunction(func, 
				args=args, kwargs=kwargs, 
				count=1, sendBack=True)
			
			if len(taskIDs)>0:
				remoteTaskID=taskIDs[0]
			else:
				return None
			
			task=Task(parent, remoteTaskID=remoteTaskID, func=func, args=args, kwargs=kwargs)
			self.tasks[task.tid]=task
			self.remoteTasks[remoteTaskID]=task
			# Do not schedule a remote task
			
		return task
	
	def schedule(self, task):
		"""
		Schedules a local *task* for execution.  
		"""
		if task.greenlet is None:
			raise Exception("Trying to schedule remote task tid=%d." % task.tid)
		self.ready.append(task)
		
	def enqueueSpawn(self, spawnSyscall, spawnerTask):
		"""
		Equeues a spawn system call waiting on a free slot.  
		"""
		self.waitingOnSlot.append((spawnSyscall, spawnerTask))
	
	def __call__(self):
		# Enqueue main task, receives tid=1
		mainTaskGreenlet=greenlet.getcurrent().parent
		if not mainTaskGreenlet:
			raise Exception("COS: Scheduler must run in a separate greenlet.")
		mainTask=Task(parent=None, greenlet=mainTaskGreenlet, name="_main", args=None, kwargs=None)
		self.tasks[mainTask.tid]=mainTask
		self.schedule(mainTask)
		
		# Main loop, exit when there are no tasks left
		while self.tasks:
			# See if there are any messages
			task=None
			retval=None
			if self.vm is not None:
				# Are there any tasks scheduled
				if len(self.ready)>0:
					# Yes, non-blocking receive
					recv=self.vm.receiveMessage(0)
				else:
					# No, blocking receive
					recv=self.vm.receiveMessage(-1)
					
				# Valid message
				if recv is not None and len(recv)==2:
					(srcID, msg)=recv
					# MsgTaskResult received
					if type(msg) is MsgTaskResult and srcID in self.remoteTasks:
						# Find task, set its return value
						task=self.remoteTasks[srcID]
						task.retval=msg.returnValue
					elif type(msg) is MsgTaskExit:
						if srcID in self.remoteTasks:
							# Get remote task
							task=self.remoteTasks[srcID]
							# Remove it from list of remote tasks
							del self.remoteTasks[srcID]
							# Mark it as finished
							task.status=Task.Sfinished
						# We have a free slot. Check for waiting spawn syscalls. 
						if len(self.waitingOnSlot)>0:
							# Get it
							spawnSyscall, spawnerTask=self.waitingOnSlot.pop(0)
							# Check if the spawner task is still running
							if spawnerTask.tid in self.tasks:
								# Handle it
								spawnSyscall.handle(self, spawnerTask) 
						
			# No messages, no remote task finished
			# Try with local tasks
			if task is None and len(self.ready)>0:  
				# Get next local task
				task=self.ready.pop(0)
				
				# Switch to local task
				identify.tid=task.tid
				retval=task.switchToLocal()
				identify.tid=self.tid
				
				# If task is finished, store its return value
				if task.status==Task.Sfinished:
					task.retval=retval
			
			# Still no task? Do it again. 
			if task is None:
				continue
			
			# Is the task finished?
			if task.status==Task.Sfinished:
				# Get parent
				parent=self.tasks[task.ptid]
				
				# Remove it from the list of running tasks
				del self.tasks[task.tid]
				
				# Update children count of parent task
				parent.nchildren-=1
				
				# Is the parent waiting on the task (Join system call)? 
				if (
					parent.status==task.Swaiting and 
					(len(parent.waitingOn)==0 or task.tid in parent.waitingOn)
				):
					# Set return value for parent
					parent.sendval={ task.tid: task.retval }
					# Schedule parent
					parent.status=Task.Srunning
					self.schedule(parent)
				else:
					# Parent is not waiting on the task, add to parent's finished children dictionary
					parent.finishedChildren[task.tid]=task
			elif isinstance(retval, SysCall):
				# System call from task, handle it and prepare return value
				retval.handle(self, task)
				# Do not schedule the task, leave that to the syscall handler
			else:
				# Nothing special, schedule local task again
				if task.remoteTaskID is None:
					self.schedule(task)
				# Do nothing for remote tasks
				
				

# Cooperative OS wrapper

class OpusOS(object):
	"""
	Cooperative multitasking OS class. 
	
	The user should import the only instance of this class represented 
	by the :data:`cOS` variable. 
	
	*vm* - virtual machine abstraction for spawning remote tasks. 
	
	If *vm* is ``None`` remote spawning is disabled. 
	"""
	scheduler=None
	
	def __init__(self, vm=None):
		self._createScheduler(vm)
	
	def setVM(self, vm=None):
		"""
		Sets the virtual machine object. 
		
		Allowed only when there are no remote tasks running. 
		
		This is not a system call and does not 
		yield execution the the scheduler. 
		"""
		OpusOS.scheduler.setVM(vm)
		
	def _createScheduler(self, vm):
		if OpusOS.scheduler is not None:
			raise Exception("COS: There can be only one OpusOS object.")
		# Create scheduler
		OpusOS.scheduler=Scheduler(vm)
		# Create scheduler greenlet
		OpusOS.schedulerGreenlet=greenlet(OpusOS.scheduler)
		# Swith to scheduler to start it
		OpusOS.schedulerGreenlet.switch()
		
	# For pickling at remote spawn
	def __getstate__(self):
		# Pack the state and the vm object
		state=self.__dict__.copy()
		return state, OpusOS.scheduler.vm
	
	# For unpickling at remote spawn
	def __setstate__(self, stateIn):
		# Unpack the state and the vm object
		state, vm = stateIn
		# Set state
		self.__dict__.update(state)
		# Create scheduler object if there is none yet
		if OpusOS.scheduler is None:
			# Create scheduler if this is the first object of this type in this process
			self._createScheduler(vm)
	
	# Functions
	@staticmethod
	def freeSlots():
		"""
		Returns the number of free slots in a vm. 
		If there is no vm, returns -1. 
		
		This is not a system call and does not 
		yield execution the the scheduler. 
		"""
		if OpusOS.scheduler.vm is not None:
			return OpusOS.scheduler.vm.freeSlots()
		return -1
	
	@staticmethod
	def slots():
		"""
		Returns the number of slots in a vm. 
		If there is no vm, returns -1. 
		
		This is not a system call and does not 
		yield execution the the scheduler. 
		"""
		if OpusOS.scheduler.vm is not None:
			return OpusOS.scheduler.vm.slots()
		return -1
	
	@staticmethod
	def getTid():
		"""
		Returns the task id of the running microthread. 
		
		This is not a system call and does not 
		yield execution the the scheduler. 
		"""
		return OpusOS.scheduler.activeTask
		
	# System calls. These function yield execution to the scheduler. 
	@staticmethod
	def Yield():
		"""
		Invokes the *Yield* system call. 
		
		See :class:`Yield`. 
		"""
		return OpusOS.schedulerGreenlet.switch(Yield())
	
	@staticmethod
	def Spawn(*args, **kwargs):
		"""
		Invokes the *Spawn* system call. 
		
		See :class:`Spawn`. 
		"""
		return OpusOS.schedulerGreenlet.switch(Spawn(*args, **kwargs))
	
	@staticmethod
	def Join(*args, **kwargs):
		"""
		Invokes the *Join* system call. 
		
		See :class:`Join`.
		"""
		return OpusOS.schedulerGreenlet.switch(Join(*args, **kwargs))
	
	# Asynchronous dispatch
	@staticmethod
	def dispatch(jobList, collector=None, remote=True, buildResultList=None):
		"""
		Dispatches multiple jobs and collects the results. 
		If *remote* is ``True`` the jobs are dispatched 
		asynchronously across the available computing nodes. 
		
		A job is a tuple of the form 
		``(callable, args, kwargs, extra)``. 
		A job is evaluated by invoking the callable with given 
		``args`` and ``kwargs``. If ``kwargs`` is omitted only 
		positional arguments are passed. If ``args`` is also 
		omitted the ``callable`` is invoked without arguments. 
		The return value of the ``callable`` is the job result. 
		Extra data (``extra``) can be stored in the optional 
		entries after ``kwargs``. This data is not passed to 
		the callable. 
		
		*jobList* is an iterator object (i,e, list) that holds the 
		jobs. It can also be a generator that produces jobs. 
		A job may not be ``None``. 
		
		*collector* is an optional unprimed coroutine. When a 
		job is finished its result is sent to the *collector* 
		in the form of a tuple ``(index, job, result)`` where 
		``index`` is the index of the ``job``. Collector's 
		task is to collect the results in a user-defined data 
		structure. 
		
		By catching the ``GeneratorExit`` exception in the 
		*collector* a postprocessing step can be performed on 
		the collected results. 
		
		Returns the list of results (in the order of generated 
		jobs). When a *collector* is specified the results are 
		not collected in a list unless *buildResultList* is set 
		to ``True``. 
		"""
		# Prime the collector
		if collector is not None:
			collector.next()
		
		# Prepare results list
		# Collect results in a list
		if collector is None or buildResultList is True:
			resList=[]
			
		jobs={}
		ii=0
		while True:
			# Get next job
			try:
				job=jobList.next()
				
				# Get parts
				f=job[0]
				args=job[1] if len(job)>1 else []
				kwargs=job[2] if len(job)>2 else {}
				
				# Spawn
				tid=OpusOS.Spawn(f, args=args, kwargs=kwargs, remote=remote, block=True)
				jobs[tid]=ii,job
				ii+=1
				
			except StopIteration:
				job=None
			
			# Join a job, block if one of the following holds
			# - no more jobs to spawn
			# - there are less than 2 slots in the VM
			# - there are no free slots in the VM
			block=(job is None) or (OpusOS.slots()<2) or (OpusOS.freeSlots()<=0)
			jr=OpusOS.Join(block=block)
			for tid, retval in jr.iteritems():
				# Extract job
				jj,jjob=jobs[tid]
				del jobs[tid]
				
				# Send result to the collector
				if collector is not None:
					collector.send((jj, jjob, retval))
				
				# Collect results in a list
				if collector is None or buildResultList is True:
					# Make space
					if len(resList)<=jj:
						resList.extend([None]*(jj-len(resList)+1))
						
					# Store
					resList[jj]=retval
				
			# Displatched all jobs and nothing to join left
			if not jr and job is None:
				break
		
		# Shut down the collector
		if collector is not None:
			collector.close()
			
		# Return result list 
		if collector is None or buildResultList is True:
			return resList
		else:
			return None
	
	@staticmethod
	def dispatchSingle(function, args=[], kwargs={}, remote=True):
		"""
		Dispatches a single task defined by *function*, *args*, and 
		*kwargs*. If *remote* is ``True`` the task is dispatched to 
		a remote computing node. 
		
		This function is used for moving a task to a remote processor 
		and waits for the results. It is not very useful in the sense 
		that it does not introduce any parallelism. 
		
		Returns the return value of the function. 
		"""
		tid=OpusOS.Spawn(function, args, kwargs, remote=remote, block=True)
		return OpusOS.Join()[tid]
	
	@staticmethod
	def finalize():
		"""
		Performs cleanup. Calls the :meth:`finalize` method of the vm. 
		"""
		if OpusOS.scheduler.vm is not None:
			OpusOS.scheduler.vm.finalize()
	
# OS object

cOS=OpusOS()
"Cooperative multitasking OS instance for accessing its functionality."

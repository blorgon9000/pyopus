"""
.. inheritance-diagram:: pyopus.parallel.evtdrvms
    :parts: 1
	
**Master-slave event-driven algorithm model (PyOPUS subsystem name: EDMS)**

This is a master-slave algorithm model that first spawns slaves across computing 
nodes and then communicates with them via messages. An algorithm is described 
with messages and responses of actors (master and slaves) to the received messages. 
This module is obsolete and will be removed. It does not support multilevel 
parallelism. Use the :mod:`pyopus.parallel.cooperative` module instead. 

In **master-slave** algorithms one task is the **master task** while all other 
tasks are **slave tasks**. The master is usually the task that sends jobs to 
slaves and collects the results sent back by slave tasks. 

An **event-driven algorithm** is described with a set of 
**message handler functions**. Every message handler function takes (handles) 
one type of messages, processes them, and returns new messages and destinations 
(task identifier objects) for these messages. The produced messages are sent to 
their destinations. 

A **message filter** is a function that takes a message source identifier 
object as its argument and returns ``True`` if the message should be handled or 
``False`` if it should be discarded. Discarded messages are never processed.

A **dispatcher** is a function which detects the message type, applies the 
corresponding filter, and if the filter returns ``True`` hands the message over 
to the corresponding message handler function. 

Every task in an event-driven algorithm is running a so-called **event loop** 
which collects incoming messages and dispatches them to corresponding message 
handler functions. usually the master task's event loop is more complicated 
than the event loop of slave tasks. 

We denote by **local mode** the mode of operation where there are no slave 
tasks present and all messages have the same destination - the master. This 
means that the master hands out the work and the master also does all the work. 

If implemented correctly every parallel algorithm can be transformed in such 
way that it is capable of running in local mode (on a single CPU core). 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import Msg, MsgTaskExit, MsgHostAdd, MsgHostDelete
import sys

__all__ = [ 'MsgSlaveStarted', 'MsgStartup', 'MsgHalt', 'MsgIdle', 'EventDrivenMS' ] 

class MsgSlaveStarted(Msg):
	"""
	A message sent by a slave to the master immediately after it is started. 
	The :attr:`taskID` member holds the task identifier object of the slave. 
	"""
	def __init__(self, taskID):
		Msg.__init__(self)
		self.taskID=taskID

class MsgStartup(Msg):
	"""
	This message is received by a master/slave immediately after startup 
	(before the event loop is entered). 
	
	:attr:`taskID` is the identifier object of the task that receives the 
	message. If the virtual machine is down :attr:`taskID` is ``None``. 
	
	The :attr:`isMaster` member is ``True`` if the task that receives this 
	message is the master. 
	
	The :attr:`localMode` member is ``True`` if the task that receives this 
	message starts in local mode. 
	"""
	def __init__(self, taskID, isMaster=False, localMode=False):
		Msg.__init__(self)
		self.taskID=taskID
		self.isMaster=isMaster
		self.localMode=localMode

class MsgHalt(Msg):
	"""
	When this message is received the event loop exits in next iteration. 
	"""
	def __init__(self):
		Msg.__init__(self)

class MsgIdle(Msg):
	"""
	These messages are generated when a task is marked as idle. They are also 
	generated in local mode when master processes all messages and no responses 
	are produced. They are needed if we want an algorithm to work in both 
	master-slave and local mode. 
	
	If the :attr:`taskID` member is ``None``, this is a local Idle message 
	(master is running in local mode and it has become idle). Otherwise the 
	:attr:`taskID` is the task identifier object of the idle task. 
	"""
	def __init__(self, taskID):
		Msg.__init__(self)
		self.taskID=taskID

# Event-driven master-slave algorithm 
class EventDrivenMS(object):
	"""
	The :class:`EventDrivenMS` class provides an event-driven algorithm model. 
	It takes care of starting up the slave tasks. If there are not enough hosts 
	available for starting slave tasks :class:`EventDrivenMS` can also run in 
	so-called **local mode** where only the master event loop is running with 
	all messages having the same destination - the master (denoted by using 
	``None`` instead on a task identifier object for destination). The opposite 
	of local mode is the *master-slave* mode. 
	
	For an algorithm to be capable of running in both master-slave and in local 
	mode the chain of messages must be interrupted from time-to-time (after all 
	input messages have been dispatched, none of the message handlers may 
	produce any output message). At such points in time :class:`EventDrivenMS` 
	checks if there is enough hosts available for starting slave tasks. If 
	there are enough hosts, slave tasks are started and the local mode is 
	replaced by master-slave mode. 
	
	*vm* is the :class:`VirtualMachine` object through which 
	:class:`EventDrivenMS` performs the spawning and communicates with slaves. 
	
	*maxSlaves* denotes the desired number of slaves. The actual number of 
	slaves spawned depends on the number of free slots in the virtual machine. 
	If there are *n* free slots the number of spawned slaves is at most *n*.  
	
	*minSlaves* denotes the minimal number of slaves needed for the parallel 
	algorithm to function in master-slave mode. If the number of slaves falls 
	below *minSlaves* the algorithm is run in local mode. If local mode is not 
	allowed the algorithm fails. 
	
	If *debug* is greater than 0, debug messages are printed to standard 
	output. 
	
	If *slaveIdleMessages* is ``True`` a :class:`MsgIdle` message is generated 
	in the master's event loop and dispatched to the master every time a slave 
	is marked as idle. A slave is marked as idle immediately after it is 
	started and marked as busy a message is sent to it. An event handler called 
	from the master's event loop can mark a slave as (not) idle by calling the 
	:meth:`markSlaveIdle` method. 
	
	A :class:`MsgIdle` message is generated once for every started slave. 
	Despite the fact that it can be replaced by the :class:`MsgSlaveStarted` 
	message it is still useful. If a user installs 	a custom 
	:class:`MsgSlaveStarted` handler and forgets to call the original one, 
	things stop working because the default :class:`MsgSlaveStarted` handler 
	should take care of slave management, but is now bypassed. It is safer to 
	use :class:`MsgSlaveIdle` which is not used for slave management. 
	
	If *localIdleMessages* is set to ``True`` and we are running in local mode 
	a :class:`MsgIdle` message is generated every time all messages incoming to 
	the master are handled and the message handlers produce no output messages. 
	This is needed for the algorithm to be able to switch from local back to 
	master-slave mode (happens at such idle moments). Without local 
	:class:`MsgIdle` messages local run is not possible. Marking a master 
	running in local mode as idle by calling the :meth:`markSlaveIdle` method 
	with ``None`` as the slave task identifier object has no effect. 
	
	For every slave task there is a **local storage dictionary** in the master 
	task that holds the data used by the master for managing the slave. The 
	dictionary can be retrieved by specifying the slave task identifier object 
	to the :meth:`taskStorage` method. For local mode there is also a local 
	storage dictionary available. It can be retrieved by specifying ``None`` as 
	the task identifier object. 
	"""
	def __init__(self, vm=None, maxSlaves=None, minSlaves=0, debug=0, 
					slaveIdleMessages=True, localIdleMessages=False):
		# Debug events flag
		self.debugEvt=debug
		
		# Desired number of slaves. None means use all free slots in the VM.
		self.maxSlaves=maxSlaves
		
		# Minimal number of slaves. If the number falls below this value, we stop. 
		self.minSlaves=minSlaves
		
		# Generate Idle messages for slaves (in master loop). 
		# A slave is marked as busy when a message is sent to it. 
		# Message handlers can mark a slave as idle. In such case a slaveIdle message 
		# will be generated in the beginning of the next master loop iteration. 
		self.slaveIdleMessages=slaveIdleMessages
		
		# Generate local Idle messages (in master loop when running in local mode)
		# These messages are generated in the master loop. They start the chain of messages. 
		# Handlers must produce a single message in response. If multiple messages are 
		# produced an exception is raised. When a handler produces no response messages a new
		# local Idle message is generated. 
		# This flag must be true for master loop to work in local mode. 
		self.localIdleMessages=localIdleMessages
		
		# Private members
		# Flag that signals we must exit the event loop
		self.exitLoop=False
		
		# A task becomes spawned immediately after ot is spawned. 
		# A task becomes started when SlaveStarted message is received. 
		# Idle/busy is controlled by message handlers. 
		# When a task becomes started it is markes as idle. 
		
		# Set of spawned tasks
		self.spawnedSet=set()
		# Set of started tasks
		self.startedSet=set()
		# Set of idle tasks
		self.idleSet=set()
		
		# Task storage 
		self.storage={}
		# Add None host's storage (local mode) 
		self.storage[None]={}
		
		# Slave count
		self.nSlaves=0
		
		# Virtual machine. If we have no virtual machine, set maxSlaves and minSlaves to 0. 
		self.vm=vm
		if self.vm is None:
			self.minSlaves=0
			self.maxSlaves=0
			
		# Local mode if off
		self.localMode=False
		
		# Empty response table
		self.handler={}
		
		# Add message handlers
		self.fillHandlerTable()
		
		# Are we master
		self.isMaster=False
	
	# Destructor - not allowed due to circular references
	# def __del__(self):
	# 	self.shutdownSlaves()
	
	# For pickling
	def __getstate__(self):
		state=self.__dict__.copy()
		del state['handler']
		
		return state
	
	# For unpickling
	def __setstate__(self, state):
		self.__dict__.update(state)
		
		self.handler={}
		self.fillHandlerTable()
		
	def fillHandlerTable(self): 
		"""
		Sets up the message handler and filters table. 
		Override this method in derived classes, but don't forget to call the 
		parent's :meth:`fillhandlerTable` method before making any calls to the 
		:meth:`addHandler` method. 
		"""
		# Our messages (from started tasks)
		self.addHandler(MsgHalt, self.handleHalt)										# From any task 
		self.addHandler(MsgSlaveStarted, self.handleSlaveStarted, self.allowSpawned)	# From spawned tasks only 
		self.addHandler(MsgStartup, self.handleStartup, self.allowLocal)				# Local only 
		self.addHandler(MsgIdle, self.handleIdle, self.allowLocal)						# Local only 
		
		# VM messages (notifications)
		# We handle TaskExit and HostAdd. At host failure/deletion we get TaskExit messages for all of its tasks. 
		self.addHandler(MsgTaskExit, self.handleTaskExit, self.allowSpawned)	# From spawned tasks only 
		self.addHandler(MsgHostAdd, self.handleHostAdd)							# From any task
		self.addHandler(MsgHostDelete, self.handleHostDelete)					# From any task

	def addHandler(self, messageType, handler, filter=None):
		"""
		Adds a new entry for messages of the type *messageType* (should be a 
		class) to the message handler and filter table. The handler and filter 
		for the *nmessageType* are given by *handler* and *filter*. if *filter* 
		is ``None`` all messages of the type *messageType* are handled. 
		
		A message handler is a function that takes two arguments:
		
		* the task identifier object of the task that produced the message and
		* the message object. 
		
		Message handlers return a list of zero or more tuples of the form 
		(*destination*, *message*) giving the output messages and their 
		destinations. 
		
		A filter is a function that takes one argument (the message source task 
		identifier object) and returns ``True`` if the message should be 
		handled or ``False`` otherwise. 
		
		If a handler/filter for a *messageType* is installed multiple times, 
		the last installed handler/filter pair is the one that is used for 
		handling the messages. 
		"""
		self.handler[messageType]=(handler, filter)

	# Message filters
	# ----
	# Message filters return True (message can be handled) or False (message must not be handled)
	# The input argument is source id.
	
	def allowSpawned(self, source):
		"""
		A filter that allows the handling of messages coming for spawnned tasks. 
		Handling of messages generated in local mode (*source* = ``None``) is 
		not allowed. 
		""" 
		if source in self.spawnedSet:
			return True
		else:
			return False
			
	def allowStarted(self, source):
		"""
		A filter that allows the handling of messages comming from spawnned 
		tasks from which we already received the :class:`MsgSlaveStarted` 
		message. Also allows the messages generated in local mode 
		(*source* = ``None``) to be handled. 
		""" 
		if source is None or source in self.startedSet:
			return True
		else:
			return False
	
	def allowLocal(self, source):
		"""
		A filter that allows the handling of messages generated in local mode 
		(*source* = ``None``). Messages from all other sources are discarded. 
		"""
		if source is None:
			return True
		else:
			return False
	
	# Message handlers
	# ----

	def handleHalt(self, source, message):
		"""
		Message handler for the :class:`MsgHalt` message. 
		
		Sets the :attr:`exitLoop` variable to ``True`` 
		indicating that the event loop should exit. 
		Produces no response message. 
		"""
		if self.debugEvt:
			DbgMsgOut("EDMS", "Handling halt message from "+str(source)+".")
			
		# Set exitLoop flag
		self.exitLoop=True
		
		# No response
		return []
	
	def handleSlaveStarted(self, source, message):
		"""
		Message handler for the :class:`MsgSlaveStarted` message. 
		
		Marks the slave identified by the :attr:`taskID` member of the message 
		as started and idle. Creates local storage for the slave. 
		
		Produces no response message. 
		"""
		if self.debugEvt:
			DbgMsgOut("EDMS", "Handling SlaveStarted message from "+str(source)+".")
			
		# Mark as started and idle. 
		taskID=message.taskID
		self.markSlaveStarted(taskID)
		self.markSlaveIdle(taskID, True)
		
		# Create task storage
		self.storage[taskID]={}
		
		# No response
		return []
	
	def handleStartup(self, source, message):
		"""
		Message handler for the :class:`MsgStartup` message. 
		
		This handler can find out if it was called from the master event loop 
		by examining the :attr:`isMaster` member of the *message*. 
		
		If the :attr:`taskID` member of the *message* holds the task identifier 
		object of the task that received the message. ``None`` 
		**does not indicate** that the master is running in local mode. 
		
		The :attr:`localMode` member of the *message* is ``True`` if the task 
		starts in local mode. 
		
		This is an empty handler (should be overriden by derived classes. 
		
		It should not produces any response message. If it does the responses 
		are discarded. 
		"""
		if self.debugEvt:
			DbgMsgOut("EDMS", "Handling Startup message.")
			
		return []
	
	def handleIdle(self, source, message):
		"""
		Message handler for the :class:`MsgIdle` message. 
		
		If the :attr:`taskID` member of the *message* is ``None`` the message 
		was sent because the master is idle in local mode. In all other cases 
		:attr:`taskID` member holds the	task identifier object of the idle 
		slave. 
		
		This is an empty handler (should be overriden by derived classes. 
		It produces no response message. 		
		"""
		if self.debugEvt:
			DbgMsgOut("EDMS", "Handling Idle message.")
			
		return []
		
	def handleTaskExit(self, source, message):
		"""
		Message handler for the :class:`~pyopus.parallel.base.MsgTaskExit` 
		message (virtual machine notification). 
		
		It removes the task that exited from the list of slaves. Then it tries 
		to spawn new slaves (if there are enough free slots). 
		
		It produces no response message. 
		"""
		if self.debugEvt:
			DbgMsgOut("EDMS", "Task exit detected for "+str(message.taskID)+".")
		
		# Delete task 
		found=self.removeSlave(message.taskID)
			
		# Try re-spawning
		self.spawnSlaves()
		
		# No response
		return []
		
	def handleHostAdd(self, source, message):
		"""
		Message handler for the :class:`~pyopus.parallel.base.MsgHostAdd` 
		message (virtual machine notification). 
		
		It tries to spawn new slaves.  
		
		It produces no response message. 
		"""
		for hostID in message.hostIDs:
			if self.debugEvt:
				DbgMsgOut("EDMS", "New host "+str(hostID)+" detected.")
		
		# Try spawning
		self.spawnSlaves()
		
		# No response 
		return []
	
	def handleHostDelete(self, source, message):
		"""
		Message handler for the :class:`~pyopus.parallel.base.MsgHostDelete` 
		message (virtual machine notification). 
		
		Does nothing, because the tasks that were running on the host that 
		failed will produce :class:`~pyopus.parallel.base.MsgTaskExit` messages. 
		
		It produces no response message. 
		"""
		if self.debugEvt:
			DbgMsgOut("EDMS", "Host "+str(message.hostID)+" failure detected.")
		
		# Do not spawn. We don't have more free slots :)
		
		# No response 
		return []
	
	# Task storage management (master-only)
	# ----
	
	def taskStorage(self, taskID):
		"""
		Returns the local storage dictionary of the task corresponding to the 
		*taskID* object. The local storage for spawned slaves is created when a 
		:class:`MsgSlaveStarted` message is handled and deleted when a 
		:class:`~pyopus.parallel.base.MsgTaskExit` message is handled. 
		
		There is also a local storage for the master running in local mode. It 
		is obtained by passing ``None`` for *taskID*. 
		
		If no storage is available for the task with *taskID* it returns 
		``None``. 
		"""
		if taskID in self.storage:
			return self.storage[taskID]
		else:
			return None
			
	# Slaves management (master-only)
	# ----
	
	def removeSlave(self, taskID):
		"""
		Removes a slave task identified by the *taskID* object from the list of 
		slaves and deletes its local storage. 
		
		Returns ``True`` if a task identified by *taskID* was present when the 
		function was called. 
		"""
		# Remove from all sets
		found=False
		if taskID in self.spawnedSet:
			self.spawnedSet.remove(taskID)
			found=True
		if taskID in self.startedSet:
			self.startedSet.remove(taskID)
		if taskID in self.idleSet:
			self.idleSet.remove(taskID)
			
		# Remove task storage
		if taskID in self.storage: 
			del self.storage[taskID]
		
		if found:
			self.nSlaves-=1
			if self.debugEvt:
				DbgMsgOut("EDMS", "%d slave(s) left." % self.nSlaves)
		else:
			if self.debugEvt:
				DbgMsgOut("EDMS", "Task "+str(taskID)+" not found in spawned tasks set.")
		
		return found
	
	def markSlaveSpawned(self, taskID):
		"""
		Marks a slave task as spawned. 
		"""
		self.spawnedSet.add(taskID)
		self.nSlaves+=1
			
	def markSlaveStarted(self, taskID):
		"""
		Marks a slave task as started. 
		"""
		if taskID in self.spawnedSet:
			self.startedSet.add(taskID)
	
	def markSlaveIdle(self, taskID, idle=True):
		"""
		Marks a slave task as idle if *idle* is ``True``. Otherwise it marks it 
		as not idle (busy). 
		"""
		if taskID in self.spawnedSet:
			if idle:
				self.idleSet.add(taskID)
			else:
				self.idleSet.discard(taskID)
			
	def shutdownSlaves(self):
		"""
		Stops all slave tasks by sending them a :class:`MsgHalt` message. It 
		waits for the :class:`MsgTaskExit` message to be received from every 
		task. 
		"""
		# If VM is not available, do nothing. 
		if self.vm is None:
			return
		
		# Ask the slaves politely to stop. 
		for taskID in self.spawnedSet:
			if self.debugEvt:
				DbgMsgOut("EDMS", "Sending Halt message to "+str(taskID)+".")
			self.vm.sendMessage(taskID, MsgHalt())
		
		# Wait for TaskExit messages.
		while len(self.spawnedSet)>0:
			recv=self.vm.receiveMessage(-1)
			if recv is None or recv is ():
				continue
			(taskID, message)=recv
			if type(message) is MsgTaskExit:
				if self.debugEvt:
					DbgMsgOut("EDMS", "Received TaskExit message from "+str(message.taskID)+".")
				self.removeSlave(message.taskID)
	
	def spawnSlaves(self):
		"""
		Spawns slaves on the hosts in the virtual machine. 
		
		Retrieves the number of free slots from the virtual machine. The upper 
		limit on the number of slaves is set by *maxSlaves*. If *maxSlaves* is 
		``None`` the upper limit is equal to teh number of free slots. 
		
		The spawning is performed without specifying the hosts on which the 
		tasks shoukld be spawned. The choice of slots is left over to the 
		virtual machine. 
		
		A slave is started by spawning the :func:`runSlave` function and 
		passing it a refernce to *self* (object of class 
		:class:`EventDrivenMS`). A successfully spawned slave is added to the 
		list of spawned slaves. It becomes operational (gets its own local 
		storage) when a :class:`MsgSlaveStarted` message is received from the 
		slave indicating that it has entered the slave event loop. 
		"""
		# Do nothing if vm is not available
		if self.vm is None:
			return 
	
		# Get number of free slots
		freeSlots=self.vm.freeSlots()
		
		# Calculate number of slaves to spawn
		if self.maxSlaves is None:
			# Fill all free slots
			nSpawn=freeSlots
		else:
			# Have a desired number of slaves
			# Calculate the number of slaves to spawn. 
			nSpawn=self.maxSlaves-self.nSlaves
			
		# Do nothing if there is nothing to do :)
		if nSpawn<=0:
			return 
			
		if self.debugEvt:
			DbgMsgOut("EDMS", "Trying to spawn %d slave(s) across %d free slot(s)." % (nSpawn, freeSlots))
		
		# Spawn
		taskIDs=self.vm.spawnFunction(function=runSlave, args=[self], count=nSpawn)
		
		if self.debugEvt:
			DbgMsgOut("EDMS", "Spawned %d slaves:" % len(taskIDs))
			
		# Add TaskIDs to spawnedSet
		for taskID in taskIDs:
			self.markSlaveSpawned(taskID)
			if self.debugEvt:
				DbgMsgOut("EDMS", "  "+str(taskID))
	
	# Message handler and event loops
	# ----
	
	def handleMessage(self, source, message): 
		"""
		Handles a *message* received from *source*. 
		
		First it checks if there is an entry for the message type in the 
		message handler table. If no entry is found the message is discarded. 
		
		Next it applies a filter to the message, provided that a filter is set 
		for the corresponding message type. If the filter returns ``False`` the 
		message is discarded. 

		Finally the handler is called. The *message* and the *source* are
		passed to the handler. The list of responses (tuples of the form 
		(*destination*, *message*)) is collected from the handler and returned. 
		If the handler returns ``None`` an empty response list is assumed. 
		
		Returns the response list or ``None`` if the message was discarded. 
		
		Note that messages may come from sources that were not spawned by the 
		master. Most notably these are workers spawned by the previous master. 
		Therefore filters should always be used. 
		"""
		# Message is unhandled by default. 
		responses=None
		
		# See in the response table
		messageType=type(message)
		if messageType in self.handler:
			# Found it, get handler and filter
			(handler, filter)=self.handler[messageType]
			sys.stdout.flush()
			# Filter source. If no filter is installed handle the message. 
			if filter is None or filter(source):
				# Call handler and collect responses
				if handler is not None:
					responses=handler(source, message)
				else:
					# None handler produces no responses
					responses=[]
				# Replace None with an empty list
				if responses is None:
					responses=[]
		
		return responses
	
	def masterEventLoop(self):
		"""
		This is the master's event loop. Users should call this function to 
		start the event-driven algorithm. 
		
		First it sets the :attr:`exitLoop` member to ``False``. 
		
		Then it checks if a virtual machine is available. If it is, slaves are
		spawned. If the number of slaves is below *minSlaves* the algorithm 
		tries to run in local mode.
		
		An exception is raised if the alogrithm tries to run in local mode and 
		local idle messages are disabled (*localIdleMessages* is ``False``). 
		
		If a virtual machine is available slaves are spawned. 
		The initial mode of operation (local or master-slave) is set. 
		
		A :class:`MsgStartup` message is generated with *isMaster* set to 
		``True`` and *localMode* indicating the mode the algorithm starts in. 
		The message is mmediately handled and the resposes are discarded. 
		
		Next the main loop is entered. Every iteration of this loop does the 
		following
		
		1. Re-checks if we are in local mode or master-slave mode. 
		2. If a virtual machine is available performs the following steps
		
		   1. If not in local mode, no incoming messages are pending, and 
		      *slaveIdleMessages* is ``True``, generate :class:`MsgIdle` 
		      messages for idle slaves and handle them. Send out the responses. 
		   2. If in local mode do a nonblocking receive. In master-slave mode 
		      do a blocking receive. 
		   3. Handle the received message and send out the responses. 
		
		3. If we are in local mode and *localIdleMessages* is ``True`` performs 
		   the following steps
		   
		   1. Generate a local :class:`MsgIdle` message 
		      (*taskID* set to ``None``) and put it in *message*
		   2. Handle *message* and collect responses. 
		   3. If there is more than one response, raise an exception. 
		   4. If there are no responses go to step 6. 
		   5. Set *message* to the first (and only) response and go back to 
		      step 2. 
		   6. Done.
		   
		The loop exits when it detects the :attr:`exitLoop` member is ``True``. 
		
		After the loop exits the slaves are shut down by calling the 
		:meth:`shutdownSlaves` method. 
		"""
		# Reset exitLoop flag
		self.exitLoop=False
		
		# We are master
		self.isMaster=True

		# Local mode at last check
		localModeOld=None
		
		# Do we have a virtual machine?
		if self.vm:
			# Get task ID
			taskID=self.vm.taskID()
			# Spawn slaves 
			self.spawnSlaves()
			# Check slave count
			if self.nSlaves<self.minSlaves:
				self.localMode=True
			else:
				self.localMode=False
		else:
			# No VM
			taskID=None
			self.localMode=True
			
		# Check if local mode is required and local Idle messages are allowed
		if self.localMode and not self.localIdleMessages:
			raise Exception, DbgMsg("EDMS", "Started in local mode, but local idle messages are disabled.")
			
		keyboardInterrupt=False
		try:
			# Generate a startup message and handle it immediately. Discard the responses. 
			self.handleMessage(None, MsgStartup(taskID, True, self.localMode))
			
			# Main loop
			while not self.exitLoop:
				# Update mode of operation. 
				# If number of slaves is less than minSlaves or 0, we are in local mode. 
				self.localMode=self.nSlaves<self.minSlaves or self.nSlaves==0
				
				if localModeOld!=self.localMode: 
					if self.localMode:
						# Entered local mode but local Idle messages are not allowed. 
						if not self.localIdleMessages:
							raise Exception, DbgMsg("EDMS", "Must enter local mode but local idle messages are not allowed.") 
					if self.debugEvt:
						if self.localMode:
							DbgMsgOut("EDMS", "Entering local mode with %d slaves." % self.nSlaves)
						else:
							DbgMsgOut("EDMS", "Entering master-slave mode with %d slaves." % self.nSlaves)
				
				# Remember previous value of localMode
				localModeOld=self.localMode
				
				# Do we have a vm
				if self.vm is not None:
					# We have a vm. 
					# If we are not in local mode, Idle messages are allowed, and no messages are pending
					# we must generate Idle messages for all idle slaves. 
					if not self.localMode and self.slaveIdleMessages and not self.vm.checkForIncoming():
						responses=[]
						for taskID in self.idleSet:
							# Emulate Idle messages from None with idle task's ID. 
							response=self.handleMessage(None, MsgIdle(taskID))
							
							# If the message was handled
							if response is not None:
								responses+=response
							
							# Check if loop is to exit (for)
							if self.exitLoop:
								break
						
						# Send out responses
						for response in responses:
							# Unpack
							(destination, message)=response
							# Send message
							if self.vm.sendMessage(destination, message):
								# Success, mark recipient as busy
								self.markSlaveIdle(destination, idle=False)
							else:
								# Failed to send
								if self.debugEvt:
									DbgMsgOut("EDMS", "Failed to send response to "+str(destination)+". Ignoring failure.")
					
						# Check if loop is to exit
						if self.exitLoop:
							break
					
					# We do a nonblocking receive in local mode. 
					# This way we handle incoming messages while in local mode. 
					if self.localMode: 
						# Local mode, do a nonblocking receive, just in case a message is incoming. 
						received=self.vm.receiveMessage(0.0)
					else:
						# Blocking receive when we are in master-slave mode. 
						received=self.vm.receiveMessage(-1.0)
						
					# Did we receive anything? 
					if received is not None and received is not ():
						# Handle received message
						responses=self.handleMessage(*received)
						
						# Was the message handled. 
						if responses is not None:
							# Send out responses
							for response in responses:
								# Unpack 
								(destination, message)=response
								# Send message
								if self.vm.sendMessage(*response):
									# Success, mark recipient as busy
									self.markSlaveIdle(destination, idle=False)
								else:
									# Failed to send
									if self.debugEvt:
										DbgMsgOut("EDMS", "Failed to send response to "+str(destination)+". Ignoring failure.")
				
				# Are we in local mode and are local Idle messages allowed? 
				if self.localMode and self.localIdleMessages:
						# Yes, they are, we can do things locally. 
						# Generate local Idle message. 
						message=MsgIdle(None)
						
						# Until we get an empty response we follow the chain of messages
						while message is not None:
							# Handle response
							responses=self.handleMessage(None, message)
							
							# Without an Idle message handler we get None as response. 
							if responses is None or len(responses)==0:
								# Unhandled message, or no response. Throw it away and end chain
								message=None
							else:
								# If a handler produces more than one response message, the algorithm can't run locally. 
								# Multiple responses could result in a message explosion. 
								if len(responses)>1:
									raise Exception, DbgMsg("EDMS", "More than one response to '"+str(type(message))+"' in local mode. Possible message explosion.") 
								
								# Take first (and only) response. 
								(source, message)=responses[0]
								
							# Check if loop is to exit
							if self.exitLoop:
								break
								
		except KeyboardInterrupt:
			# Catch keyboard interrupt
			DbgMsgOut("EDMS", "Keyboard interrupt.")
			keyboardInterrupt=True
		except:
			# Catch everything else, cleanly shut down slaves, reraise
			if len(self.spawnedSet)>0:
				self.shutdownSlaves()
			raise
			
		# Shutdown slaves
		try:
			# Do we have any (spawned slaves)? 
			if len(self.spawnedSet)>0:
				self.shutdownSlaves()
			# If the algorithm stopped due to a keyboard interrupt, reraise the exception
			if keyboardInterrupt:
				raise KeyboardInterrupt
		except:
			# Catch exceptions thrown during shutdownSlaves(), reraise keyboard interrupt
			if keyboardInterrupt:
				raise KeyboardInterrupt
			DbgMsgOut("EDMS", "Failed to clean up VM.")
		
	def slaveEventLoop(self):
		"""
		This is the slave event loop. Users should never call this function. It 
		is invoked automatically when a slave task is spawned. 
		
		First it sets the :attr:`exitLoop` member to ``False``. 
		
		Next it removes all slave information inherited from the master's 
		:class:`EventDrivenMS` object. 
		
		A :class:`MsgSlaveStarted` message is sent to the parent task (the 
		master) with the :attr:`taskID` member set to the identifier object of 
		the task that called this function. 
		
		Next a :class:`msgStartup` message is generated with the :attr:`taskID` 
		member set to the task identifier object of the task that called this 
		function, the :attr:`isMaster` member set to ``False`` and the 
		:attr:`localMode` member set to ``False``. The message is immediately 
		handled and the resposes are discarded. 
		
		The main loop is entered. The loop performs the following steps 
		
		1. Receive a message (blocking)
		2. Handle it. 
		3. Send out the responses. 
		
		The loop exits when it detects the :attr:`exitLoop` member is ``True``. 
		"""
		# Reset exitLoop flag
		self.exitLoop=False
		
		# Clear spawned, started, and idle set. Clear task storage. 
		self.spawnedSet=set()
		self.startedSet=set()
		self.idleSet=set()
		self.storage={}
		
		# Send started message to master
		if self.vm.sendMessage(self.vm.parentTaskID(), MsgSlaveStarted(self.vm.taskID())) is False:
			# Sending failed. Something is wrong. Stop. 
			if self.debugEvt:
				DbgMsgOut("EDMS", "Failed to send SlaveStarted message. Exiting.")
			return
		
		# Generate a startup message and send it to yourself. 
		# Message contains our taskID (slave). 
		# Handle it immediately. 
		responses=self.handleMessage(None, MsgStartup(self.vm.taskID(), False, False))
		
		# Message handling loop
		while not self.exitLoop:
			# Wait for message
			received=self.vm.receiveMessage(-1.0)
			
			# Handle error and timeout
			if received is None or len(received)==0:
				continue
				
			# Unpack 
			(source, message)=received
			
			# Handle message
			responses=self.handleMessage(source, message)
			
			# Was the message handled? 
			if responses is not None:
				# Send responses
				for response in responses:
					if self.vm.sendMessage(*response) is False:
						# Error, stop sending
						if self.debugEvt:
							DbgMsgOut("EDMS", "Failed to send response to "+str(response[0])+". Ignoring failure.")
	
def runSlave(slaveObject): 
	"""
	This is a function which is spawned when master is spawning a new slave. 
	*slaveObject* is a copy of the master's :class:`EventDrivenMS` object. 
	"""
	# Run slave's event loop
	slaveObject.slaveEventLoop()
	
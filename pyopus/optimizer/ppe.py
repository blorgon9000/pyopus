"""
.. inheritance-diagram:: pyopus.optimizer.ppe
    :parts: 1

**Parallel point evaluator (PyOPUS subsystem name: PPE)**

Evaluates mutually independent points in parallel. Can be used for parallel 
implementations of Monte-Carlo analysis, and parametric sweeps. The set of 
points for evaluation is specified by means of point generator objects. 
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from base import Optimizer
from ..parallel.evtdrvms import EventDrivenMS, MsgSlaveStarted, MsgStartup, MsgIdle
from ..parallel.base import Msg

from numpy import random, isfinite, tile, where, abs, unique, arange, zeros, concatenate
from time import sleep, time

__all__ = [ 'PointGenerator', 'GenUniform', 'GenNormal', 'GenPointSet', 
			'MsgEvaluateChunk', 'MsgResult', 'ParallelPointEvaluator' ] 


class PointGenerator(object):
	"""
	A generic point generator class
	
	Point generators are callable. The calling convention is ``object(n)`` 
	where *n* is the number of points to generate. The return value is a tuple 
	of two arrays (*indices*, *chunk*). *indices* is a 1-dimensional array with 
	the indices of the *n* generated points while *chunk* is a 2-dimensional 
	array with first index specifying the point and the second index specifying 
	the point's components. 
	
	The generator returns ``None`` for *chunk* when no more points can be 
	generated. 
	
	The generator keeps track of point indices so every generated point 
	receives a unique index strting from 0. The indices of the points in the 
	first chunk are :math:`0..n-1`. Points in the second chunk have indices 
	:math:`n..2n-1`
	"""
	def __init__(self):
		self.ndim=None
		self.index=0
	
	def __call__(self, n=1):
		return (None, None)
	
	def reset(self):
		"""
		Resets the generator to its initial state. This means that the point 
		index becomes 0. 
		"""
		self.index=0


class GenUniform(PointGenerator):
	"""
	A point generator for generating uniformly distributed random points from 
	a rectangle specified by lower and upper bounds on point's components. 
	
	*xlo* and *xhi* are the 1-dimensional arrays of lower and upper bounds with 
	matching lengths. The lengths of *xlo* and *xhi* define the dimension of 
	space from which the points are taken. 
	"""
	def __init__(self, xlo, xhi):
		PointGenerator.__init__(self)
		
		self.xlo=xlo
		self.xhi=xhi
		
		if len(xlo.shape)!=1 or len(xhi.shape)!=1:
			raise Exception, DbgMsg("PPE", "Lower/upper bound must be a vector.") 
		
		if xlo.shape[0]!=xhi.shape[0]:
			raise Exception, DbgMsg("PPE", "Lower/upper bound must have matching shape.") 
			
		self.ndim=xlo.shape[0]
	
	def __call__(self, n=1): 
		xlo=tile(self.xlo, (n, 1))
		xhi=tile(self.xhi, (n, 1))
		
		indices=arange(self.index, self.index+n)
		self.index+=n
		
		return (indices, xlo+(xhi-xlo)*random.rand(n, self.ndim))
		

class GenNormal(PointGenerator):
	"""
	A point generator for generating points with normally distributed 
	components. 
	
	*xmean* and *xstddev* are the 1-dimensional arrays of mean values and 
	standard deviations of point's components. The length of *stddev* must be 
	the same as the length of *mean*.
	
	If a truncation factor is specified with *truncate* the points that lie 
	outside :math:`mean \\pm truncate \\cdot stddev` are regenerated until they 
	lie within that interval. *truncate* must be a 1-dimensional array with 
	length 1 (applied to all point components) or its length must be the same 
	as the length of *mean* and *stddev*. 
	"""
	def __init__(self, xmean, xstdev, xtruncate=None):
		PointGenerator.__init__(self)
		
		self.xmean=xmean
		self.xstdev=xstdev
		self.xtruncate=xtruncate
		
		if len(xmean.shape)!=1: 
			raise Exception, DbgMsg("PPE", "Mean must be a vector.") 
		
		if len(xstdev.shape)!=1:
			raise Exception, DbgMsg("PPE", "Standard deviation must be a vector.") 
		
		if xtruncate is not None and xtruncate<=0:
			raise Exception, DbgMsg("PPE", "Truncation factor must be positive.")
		
		if xmean.shape[0]!=xstdev.shape[0]:
			raise Exception, DbgMsg("PPE", "Mean/stdev must have matching shape.") 
		
		self.ndim=xmean.shape[0]
	
	def __call__(self, n=1):
		xmean=tile(self.xmean, (n, 1))
		xstdev=tile(self.xstdev, (n, 1))
		
		chunk=random.standard_normal((n, self.ndim))
		
		# Perform truncation if required (if xtruncate in not None)
		if self.xtruncate is not None:
			while True:
				# Bad point indices
				indices=where(abs(chunk)>abs(xtruncate))[0]
				
				# Everything OK?
				if indices.size==0:
					break
				
				# Get unique indices
				indices=unique(indices)
				
				# Regenerate
				chunk[indices,:]=random.standard_normal((indices.size, self.ndim))
		
		indices=arange(self.index, self.index+n)
		self.index+=n
		
		return (indices, xmean+chunk*xstdev)
		
	
class GenPointSet(PointGenerator):
	"""
	A point generator that generates the poits specified by array *pointSet*. 
	The first index in the array is the point index while the second index is 
	the point component index. The points are returned in the same order as 
	they appear in *pointSet*. 
	
	When the set is exhausted the generator returns ``None``. the :meth:`reset` 
	method resets the generator to its initial state and starts over generating 
	points at the first point in *pointSet*. 
	"""
	def __init__(self, pointSet):
		PointGenerator.__init__(self)
		
		self.pointSet=pointSet
		
		self.npts=pointSet.shape[0]
		self.ndim=pointSet.shape[1]
		
	def __call__(self, n=1):
		nTake=n
		if self.index+n>self.npts:
			nTake=self.npts-self.index
		
		if nTake<=0:
			return (None, None)
		
		indices=arange(self.index, self.index+nTake)
		self.index+=nTake
		
		return (indices, self.pointSet[indices,:])


class MsgEvaluateChunk(Msg):
	"""
	A message sent by a master to a slave requesting the evaluation of the 
	points given by *chunk*. The indices of the points are given by *indices*. 
	"""
	def __init__(self, indices, xchunk):
		Msg.__init__(self)
		self.indices=indices
		self.xchunk=xchunk


class MsgResult(Msg):
	"""
	A message sent by a slave reporting the cost function values *fchunk* 
	calculated for the points in *xchunk*. *annotations* is a list of 
	annotations corresponding to the points in the chunk. 
	"""
	def __init__(self, xchunk, fchunk, annotations=None):
		Msg.__init__(self)
		self.xchunk=xchunk
		self.fchunk=fchunk
		self.annotations=annotations

# generators is a list of point generators. The components of points that are evaluated
# are generated by the generators. The dimension of points is the sum of dimensions of
# all generators in the list. 
class ParallelPointEvaluator(Optimizer, EventDrivenMS):
	"""
	Parallel point evaluator class
	
	Evaluates *function* on a set of points generated by a list of 
	*generators*. If the *generators* list has more than one member then 
	every generated point is obtained by concatenating the components 
	produces by individual generators in the same order as they appear in 
	the *generators* list. 
	
	*maxiter* specifies the number of points to evaluate. If some generator 
	produces ``None`` when it is called (indicating it ran out of points) 
	the algorithm is stopped regardless of *maxiter*. 
	
	*fstop* (if it is not ``None``) stops the algorithm when a point is 
	found where the value of *function* is below *fstop*. It can be used to 
	implement Monte-Carlo optimization that stops when a goal (*fstop*) is 
	reached. 
	
	If a virtual machine object is passed as the *vm* argument the 
	evaluations are spread out across the virtual machine and performed in 
	parallel. 
	
	If the evaluation of a single point takes too little time the parallel 
	algorithm can be very inefficient (the communication delay dominates 
	over computation time). Setting *chunkSize* (the number of points that 
	are evaluated by one task) to a larger value improves the efficiency. 
	
	The results can be collected by a plugin (see for instance the 
	:class:`pyopus.evaluator.cost.CostCollector` class). The order in which 
	the results are obtained is not nccessarilly the order in which they 
	are roduced by the generators. Therefore every point receives a unique 
	index when it is generated and this index is available in the 
	:attr:`index` member of the :class:`ParallelPointEvakuator` object 
	when a point is registered (plugins are called). 
	
	The algorithm is capable of handling notification messages received when a 
	slave fails or a new host joins the virtual machine. In the first case the 
	work that was performed by the failed slave is reassigned. In the second 
	case new slaves are spawned. The latter is performed by the handler in the 
	:class:`EventDrivenMS` class. Work is assigned to new slaves when the 
	algorithm detects that they are idle. 
	
	See the :class:`~pyopus.optimizer.base.Optimizer` and the 
	:class:`~pyopus.parallel.evtdrvms.EventDrivenMS` classes for more 
	information. 
	"""
	def __init__(self, function, generators, debug=0, fstop=None, maxiter=None, 
					chunkSize=1, vm=None, maxSlaves=None, minSlaves=0):
		Optimizer.__init__(self, function, debug, fstop, maxiter)
		
		if debug>1:
			debugEvt=1
		else:
			debugEvt=0
		
		# This one will call the fillHandlerTable() method of ParallelMonteCarlo. 
		# Emulate SlaveIdle messages for idle slaves
		# Emulate local Idle messages (local run)
		EventDrivenMS.__init__(self, vm=vm, maxSlaves=maxSlaves, minSlaves=minSlaves, debug=debugEvt, 
			slaveIdleMessages=True, localIdleMessages=True)

		# Generators
		self.generators=generators
		
		# Get dimension
		self.ndim=0
		for gen in self.generators:
			self.ndim+=gen.ndim
		
		# Chunk size
		self.chunkSize=chunkSize
		
		# List of failed chunks
		self.failedChunks=[]
		
		# Number of chunks in evaluation
		self.chunksInEvaluation=0
		
		# All chunks generated 
		self.noMoreChunks=False
		
		# How many points were generated until now
		self.generatedCount=0
		
		self.startedCount=0
		self.tRun=0.0
		self.runIter=0
	
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
		
	# Initialize response table
	def fillHandlerTable(self):
		"""
		Fills the handler table of the 
		:class:`~pyopus.parallel.evtdrvms.EventDrivenMS` object with the 
		handlers that take care of the parallel point evaluator's messages. 
		"""
		# Parent class' messages
		EventDrivenMS.fillHandlerTable(self)
		
		# Idle message is listed in EventDrivenMS.fillHandlerTable()
		# Here we only override its handler. 
		
		# Master's messages, allow only from started tasks
		self.addHandler(MsgResult, self.handleResult, self.allowStarted)
				
		# Slave's messages, allow from all (default)
		self.addHandler(MsgEvaluateChunk, self.handleEvaluateChunk)
	
	# Generate chunk, return (indices, chunk) tuple
	def generateChunk(self, n=1):
		"""
		Generates a chunk of *n* points with the generators listen in the 
		*generators* list. 
		
		Returns a tuple (*indices*, *points*) where *indices is a 1-dimensional 
		array of indices corresponding to *points*. The first index in the 
		*points* array is the index of a point within the array and the second 
		index is the index of the point's component. 
		"""
		chunk=zeros((n, self.ndim))
		
		atdim=0
		
		ilist=[]
		clist=[]
		
		shape=None
		for gen in self.generators:
			(indices, subchunk)=gen(n)
			
			# Do we have to stop
			if subchunk is None:
				return (None, None)
			
			ilist.append(indices)
			clist.append(subchunk)
			
			# Check shape
			if shape is None:
				shape=subchunk.shape[0]
			elif shape!=subchunk.shape[0]:
				# Subchunks don't match in shape
				return (None, None)
		
		return (ilist.pop(), concatenate(clist, 1))
	
	#
	# Message handlers
	#
	
	def handleSlaveStarted(self, source, message):
		"""
		Handles the MsgSlaveStarted message received by the master from a 
		freshly started slave. Does nothing special, just calls the inherited 
		handler from the :class:`EventDrivenMS` class. 
		"""
		# Timing for n>0 workers
		self.startedCount+=1
		if len(self.startedSet)==len(self.spawnedSet):
			self.tRun=time()
			self.runIter=self.niter
		
		# Call parent's method
		return EventDrivenMS.handleSlaveStarted(self, source, message)
	
	def handleTaskExit(self, source, message):
		"""
		Handles a notification message reporting that a slave has failed 
		(is down). 
		
		Stores the work that was performed at the time the slave failed so it 
		can later be reassigned to some other slave. 
		"""
		# Get ID
		id=message.taskID
		
		# Mark point as unevaluated, remove it from the set of points being evaluated
		taskStorage=self.taskStorage(id)
		if 'job' in taskStorage: 
			job=taskStorage['job']
			self.failedChunks.append(job)
			
			self.chunksInEvaluation-=1
		
			# Debug message
			if self.debug:
				indices=job[0]
				DbgMsgOut("PPE", "Task "+str(id)+" down. Reassigned chunk "+str(indices.min())+".."+str(indices.max()))
			
		# Call parent's method that will remove the task's structures
		EventDrivenMS.handleTaskExit(self, source, message)
		
	def handleStartup(self, source, message): 
		"""
		Handles a :class:`MsgStartup` message received by master/slave when the 
		event loop is entered. 
		
		Performs some initializations on the master and prevents the plugins 
		in the slave from printing messages. 
		"""
		# Make triggers quiet on slaves
		if not message.isMaster:
			# We are a slave
			# Make triggers quiet - we are the worker, we only forward stuff to the master
			for plugin in self.plugins:
				plugin.setQuiet(True)
		else:
			# We are master
			
			# Reset generators
			for gen in self.generators:
				gen.reset()
			
			# Empty failed chunks list
			self.failedChunks=[]
			
			# Number of chunks in evaluation is 0
			self.chunksInEvaluation=0
			
			# We still have chunks to generate
			self.noMoreChunks=False
			
			# How many points were generated
			self.generatedCount=0
			
		# Call parent's method
		return EventDrivenMS.handleStartup(self, source, message)
	
	def handleIdle(self, source, message):
		"""
		Handles a :class:`MsgIdle` message. 
		
		Distributes work to the slaves by sending them :class:`MsgEvaluateChunk`
		messages. 
		"""
		# Timing for 0 workers
		if self.niter==0:
			self.tRun=time()
		
		# See if there are any failed chunks
		if len(self.failedChunks)>0:
			# Have failed chunks
			(indices, chunk)=self.failedChunks.pop()
		elif not self.noMoreChunks: 
			# Generate new chunk
			
			# Calculate chunk size
			if self.maxiter is not None: 
				nGenerate=self.maxiter-self.generatedCount
			else:
				nGenerate=self.chunkSize
			
			# Limit chunk size
			if nGenerate>self.chunkSize:
				nGenerate=self.chunkSize
			
			# Do we have anything to generate
			if nGenerate>0:
				(indices, chunk)=self.generateChunk(nGenerate)
			else:
				(indices, chunk)=(None, None)
			
			# Stop producing points? 
			if chunk is None:
				self.noMoreChunks=True
			
				if self.debug:
					DbgMsgOut("PPE", "Stopped producing points.") 
		else:
			# Stopped generating chunks
			(indices, chunk)=(None, None)
		
		if chunk is None:
			# Set stop condition if no more chunks are to be generated,
			#   the set of failed chunks is empty, 
			#   and no chunks are in evaluation. 
			if self.noMoreChunks and len(self.failedChunks)==0 and self.chunksInEvaluation==0:
				self.stop=True
				
				if self.debug:
					DbgMsgOut("PPE", "Finished.") 
			
			# Exit loop on stop
			if self.stop:
				self.exitLoop=True
			
			# No messages to send
			self.markSlaveIdle(message.taskID, True)
			return []
		else:
			# Remember what the task is doing
			self.taskStorage(message.taskID)['job']=(indices, chunk)
			
			# Increase the number of chunks in evaluation
			self.chunksInEvaluation+=1
			
			# Increase the number of generated points
			self.generatedCount+=indices.size
			
			if self.debug:
				DbgMsgOut("PPE", "Chunk "+str(indices.min())+".."+str(indices.max())+" sent to "+str(message.taskID))
					
			# Dispatch chunk
			return [(message.taskID, MsgEvaluateChunk(indices, chunk))]
		
	def handleEvaluateChunk(self, source, message):
		"""
		Handles :class:`MsgEvaluateChunk` messages received by slaves. 
		Evaluates the received points and sends back a :class:`MsgResult` 
		message. 
		"""
		
		# Evaluate
		indices=message.indices
		xchunk=message.xchunk
		
		# Chunk size
		npts=xchunk.shape[0]
		
		# Results
		fchunk=zeros(npts)
		annotations=[]
		
		# Evaluate
		for ndx in range(npts):
			x=xchunk[ndx,:]
			self.index=indices[ndx]
			fchunk[ndx]=self.fun(x)
			annotations.append(self.annotations)
		
		return [(source, MsgResult(xchunk, fchunk, annotations))]
		
	def handleResult(self, source, message): 
		"""
		Handles a :class:`MsgResult` message and takes care of the evaluation 
		result. 
		
		Marks the slave that sent the message as idle so that new work can be 
		assigned to it by the :class:`MsgIdle` message handler. 
		"""
		
		xchunk=message.xchunk
		fchunk=message.fchunk
		annotations=message.annotations	# This is a list of annotations
			
		if self.debug:
			DbgMsgOut("PPE", "Received results from "+str(source))
			
		# Decrease the number of chunks in evaluation
		self.chunksInEvaluation-=1
		
		# Get indices, clear local storage
		storage=self.taskStorage(source)
		indices=storage['job'][0]
		del storage['job']
		
		# If the message was sent from None (local mode) this stuff got handled in self.fun()
		if source is not None:
			# Handle new results
			npts=fchunk.shape[0]
			
			for ndx in range(npts):
				self.index=indices[ndx]
				self.newResult(xchunk[ndx,:], fchunk[ndx], annotations[ndx])
			
		# Mark slave as idle - end message chain
		self.markSlaveIdle(source, True)
		
		# No response to this message
		return []
	
	def run(self):
		"""
		Runs the optimization algorithm. 
		"""
		# Reset stop flag of the Optimizer class
		self.stop=False
		
		self.startedCount=0
		self.tRun=time()
		
		# Start master event loop
		EventDrivenMS.masterEventLoop(self)
		
		self.tRun=time()-self.tRun
		self.runIter=self.niter-self.runIter
		
		if self.debug:
			DbgMsgOut("PPE", "Evaluations: "+str(self.runIter)+" time: "+str(self.tRun))
			DbgMsgOut("PPE", str(self.runIter/self.tRun)+" evaluations/s")

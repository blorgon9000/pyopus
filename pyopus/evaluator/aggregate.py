"""
.. inheritance-diagram:: pyopus.evaluator.aggregate
    :parts: 1
	
**Parformance aggregation module (PyOPUS subsystem name: AG)**

**Normalization** is the process where a performance measure is scaled in such 
way that values not satifying the goal result in normalized values smaller than 
0. If the performance measure value exceeds the goal the normalized value is 
greater than 0. 

**Shaping** is the process where the normalized performance is shaped. 
Usually positive values (corresponding to performance not satifying the goal) 
are shaped differently than negative values (corresponding to performance 
satifying the goal). Shaping results in a **shaped contribution** for 
every corner. 

**Corner reduction** is the process where **shaped contributions** of 
individual corners are incorporated into the aggregate function. There are 
several ways how to achieve this. For instance one could incorporate only the 
contribution of the corner in which worst peroformance is observed, or on the 
other hand one could incorporate the mean contribution of all corners. 

The main data structure is the **aggregate function description** which is a list 
of **component descriptions**. 
Every **component description** is a dictionary with the following members: 

* ``measure`` - the name of the performance meeasure on which the aggregate  
  function's component is based. 
* ``norm`` - an object performing the normalization of the performance measure. 
* ``shape`` - an object performing the shaping of normalized performance measure. 
  Defaults to ``Slinear2(1.0, 0.0)``. 
* ``reduce`` - an object performing the corner reduction of the cotributions. 
  Defaults to ``Rworst()``. 
  
The **ordering of parameters** is a list of parameter names that defines the 
order in which parameter values apper in a parameter vector.

A **parameter vector** is a list or array of parameter values where the values 
are ordered according to a given **ordering of input parameters**. 
"""

from ..optimizer.base import Reporter, Stopper, Plugin, CostCollector, Annotator
from ..misc.debug import DbgMsgOut, DbgMsg
from auxfunc import paramList, paramDict
from numpy import concatenate, array, ndarray, where, zeros, sum, abs, array, ndarray
import sys

normalizationNames = [ 'Nabove', 'Nbelow', 'Nbetween' ]
shaperNames = [ 'Slinear2' ]
reductionNames = [ 'Rexcluded', 'Rworst', 'Rmean' ]

__all__ = normalizationNames + shaperNames + reductionNames + [ 
		'formatParameters', 'Aggregator', 
	] 

def formatParameters(param, inputOrder=None, nParamName=15, nNumber=15, nSig=6):
	"""
	Returns string representation of a parameter dictionary *param* where the 
	ordering of parameters is specified by *inputOrder*. The width of parameter 
	name and parameter value formatting is given by *nParamName* and *nNumber*, 
	while *nSig* specifies the number of significant digits. 
	"""
	output=''
	
	if inputOrder is None:
		inputOrder=param.keys()
		inputOrder.sort()
	
	first=True
	for paramName in inputOrder:
		if not first:
			output += "\n"
		else:
			first=False
		output += '%*s: %*.*e' % (nParamName, paramName, nNumber, nSig, param[paramName])
	
	return output

# Basic normalization class	
class Nbase(object):
	"""
	Basic normalization class. Objects of this class are callable. The calling 
	convention of the object is ``object(value)`` where *value* is a scalar or 
	an array of performance measure values. When called with a scalar the 
	return value is a scalar. When called with an array the return value is an 
	array of the same shape where every component is treated as if it was a 
	scalar. 
	
	The return value is greater than 0 if the passed value fails to satisfy the 
	goal. It is less than 0 if the passed value exceeds the goal. Exceeding the 
	goal by *norm* means that the return value is -1.0. Failing to satify the 
	goal by *norm* results in a return value of 1.0. If the value passed at 
	call is ``None``, the return value is equal to *failure*. 
	
	If *norm* is not given the default normalization is used which is equal to 
	*goal* or 1.0 if *goal* is equal to 0.0. If *goal* is a vector, *norm* 
	must either be a vector of the same size or a scalar in which case it 
	applies to all components of the *goal*. 
	"""
	def __init__(self, goal, norm=None, failure=10000.0):
		self.goal=array(goal)

		if norm is None:
			if self.goal.size==1:
				self.norm=abs(self.goal) #/10.0
				if self.norm==0:
					self.norm=1.0
			else:
				self.norm=abs(self.goal) #/10.0
				self.norm[where(self.norm==0.0)]=1.0
		else:
			self.norm=norm

		self.failure=failure
	
	def __call__(self, value):
		pass
		
	def worst(self, values, total=False):
		"""
		Returns the worst performance value across all corners (the one with 
		the largest normalized value). The values across corners are given in 
		the *values* array where first array index is the corner index. 
		
		If the array has more than 1 dimension the worst value is sought along 
		the first dimension of the array. This means that if *value* is of 
		shape (n, m1, m2, ...) then the return value is of shape (m1, m2, ...). 
		The return value is an array of performance measure values. 
		
		If *total* is ``True`` the worst value is sought acros the whole array 
		and the return value is a scalar worst performance value. 
		"""
		pass
	
	def worstCornerIndex(self, values, corners, total=False):
		"""
		Returns the index corresponding to the corner where the performance 
		measure takes its worst value (the one with the largest normalized 
		value). The values across corners are given in the *values* array where 
		first array index is the corner index. 
		
		If the array has more than 1 dimension the worst value is sought along 
		the first dimension of the array. This means that if *value* is of 
		shape (n, m1, m2, ...) then the return value is of shape (m1, m2, ...). 
		The return value as an array of corner indices. 
		
		The corner indices corresponding to the first dimension of *values* are 
		given by *corners*. 
		
		If *total* is ``True`` the worst value is sought across the whole array 
		and the return value is a scalar worst corner index. 
		"""
		pass
	
	def report(self, name, nName=12, nGoal=12, nSigGoal=3):
		"""
		Formats the goal as a string of the form 
		
		*name* *normalization_symbol* *goal* where *name* 
		
		is the name of the performance measure. The *normalization_symbol* 
		depends on the type of normalization (derived class). 
		
		*nName* and *nGoal* specify the width of performance measure name and 
		goal formatting. *nSigGoal* is the number of significant digits of the 
		*goal* in the formatted string. 
		"""
		pass

		
# Normalization for targets of the form value>goal. 
class Nabove(Nbase):
	"""
	Performance normalization class requiring the performance to to be above 
	the given *goal*. See :class:`Nbase` for more information. 
	"""
	def __init__(self, goal, norm=None, failure=10000.0):
		Nbase.__init__(self, goal, norm, failure)

	def worst(self, values, total=False):
		"""
		Find the worst *value*. See :meth:`Nbase.worst` method for more 
		information. 
		"""
		if not total:
			return values.min(0)
		else:
			return values.min()
	
	def worstCornerIndex(self, values, corners, total=False):
		"""
		Find the worst corner index. See :meth:`Nbase.worstCornerIndex` method 
		for more information. 
		"""
		if not total:
			return corners[values.argmin(0)]
		else:
			return corners.ravel()[values.argmin()]
	
	def __call__(self, value):
		return (self.goal-value)/self.norm
	
	def report(self, name, nName=12, nGoal=12, nSigGoal=3):
		"""
		Format the goal as a string. The output is a string of the form 
		
		*name*  >  *goal*
		
		See :meth:`Nbase.report` method for more information. 
		"""
		if self.goal.size!=1:
			return "%*s  >  %-*s" % (nName, name, nGoal, 'vector')
		else:
			return "%*s  >  %-*.*e" % (nName, name, nGoal, nSigGoal, self.goal)

			
# Normalization for targets of the form value<goal. 
class Nbelow(Nbase):
	"""
	Performance normalization class requiring the performance to to be below 
	the given *goal*. See :class:`Nbase` for more information. 
	"""
	def __init__(self, goal, norm=None, failure=10000.0):
		Nbase.__init__(self, goal, norm, failure)
		
	def worst(self, values, total=False):
		"""
		Find the worst *value*. See :meth:`Nbase.worst` method for more 
		information. 
		"""
		if not total:
			return values.max(0)
		else:
			return values.max()
	
	def worstCornerIndex(self, values, corners, total=False):
		"""
		Find the worst corner index. See :meth:`Nbase.worstCornerIndex` method 
		for more information. 
		"""
		if not total:
			return corners[values.argmax(0)]
		else:
			return corners.ravel()[values.argmax()]
		
	def __call__(self, value):
		return (value-self.goal)/self.norm
	
	def report(self, name, nName=12, nGoal=12, nSigGoal=3):
		"""
		Format the goal as a string. The output is a string of the form 
		
		*name*  <  *goal*
		
		See :meth:`Nbase.report` method for more information. 
		"""
		if self.goal.size!=1:
			return "%*s  <  %-*s" % (nName, name, nGoal, 'vector')
		else:
			return "%*s  <  %-*.*e" % (nName, name,nGoal, nSigGoal, self.goal)


# Normalization for targets of the form goal<value<goalHigh. 
class Nbetween(Nbase):
	"""
	Performance normalization class requiring the performance to to be above 
	*goal* and below *goalHigh*. See :class:`Nbase` for more information. 
	This class is deprecated. Use two contributions instead (one with 
	Nbelow and one with Nabove). 
	"""
	def __init__(self, goal, goalHigh, norm=None, failure=10000.0):
		Nbase.__init__(self, goal, norm, failure)
		
		self.goalHigh=array(goalHigh)
		
		if (self.goal>self.goalHigh).any():
			raise Exception, DbgMsg("AG", "Lower bound is above upper bound.")
			
		if norm is None:
			if self.goal.size==1:
				self.norm=abs(self.goalHigh-self.goal) # /10.0
				if self.norm==0:
					self.norm=1.0
			else:
				self.norm=abs(self.goalHigh-self.goal) # /10.0
				self.norm[where(self.norm==0.0)]=1.0
		else:
			self.norm=norm

	def worst(self, values, total=False):
		"""
		Find the worst *value*. See :meth:`Nbase.worst` method for more 
		information. 
		"""
		# Distance from center
		distance=abs(values-(self.goal+self.goalHigh)/2)
		
		if values.size==1:
			return values
		else:
			if not total:
				return values[distance.argmax(0)]
			else:
				return values[distance.argmax()]
		
	def worstCornerIndex(self, values, corners, total=False):
		"""
		Find the worst corner index. See :meth:`Nbase.worstCornerIndex` method 
		for more information. 
		"""
		# Distance from center
		distance=abs(values-(self.goal+self.goalHigh)/2)
		
		if not total:
			return corners[distance.argmax(0)]
		else:
			return corners.ravel()[distance.argmax()]
	
	def __call__(self, value):
		center=(self.goal+self.goalHigh)/2
		return where(value<center, (self.goal-value)/self.norm, (value-self.goalHigh)/self.norm)
	
	def report(self, name, nName=12, nGoal=12, nSigGoal=3):
		"""
		Format the goal as a string. The output is a string of the form 
		
		*name* < > (*goal* + *goalHigh*)/2
		
		See :meth:`Nbase.report` method for more information. 
		"""
		if self.goal.size!=1:
			return "%*s < > %-*s" % (nName, name, nGoal, 'vector')
		else:
			return "%*s < > %-*.*e" % (nName, name, nGoal, nSigGoal, (self.goal+self.goalHigh)/2)
			

# Linear two-segment shaping 
class Slinear2(object):
	"""
	Two-segment linear shaping. Normalized performances above 0 (failing 
	to satify the goal) are multiplied by *w*, while the ones below 0 
	(satisfying the goal) are multiplied by *tw*. This shaping has a 
	discontinuous first derivative at 0. 
	
	Objects of this class are callable. The calling comvention is 
	``object(value)`` where *value* is an array of normalized performance 
	measures. 
	"""
	def __init__(self, w=1.0, tw=0.0):
		self.w=w
		self.tw=tw
	
	def __call__(self, normMeasure):
		goodI=where(normMeasure<=0)
		badI=where(normMeasure>0)
		
		shapedMeasure=normMeasure.copy()
		
		shapedMeasure[goodI]*=self.tw
		shapedMeasure[badI]*=self.w
		
		return shapedMeasure
		

# Basic corner reduction class
# Reducing contributions from multiple corners to one. 
# The corners are the first dimension of the input array. 
class Rbase(object):
	"""
	Basic corner reduction class. Objects of this class are callable. The 
	calling convention of the object is ``object(shapedMeasure)`` where 
	*shapedMeasure* is an array of shaped contributions. The return value 
	is a scalar. 
	"""
	def __init__(self):
		pass
		
	def __call__(self, shapedMeasure):
		pass
	
	def flagSuccess(self, fulfilled):
		"""
		Return a string that represents a flag for marking performance measures 
		that satify the goal (*fulfilled* is ``True``) or fail to satisfy the 
		goal (*fulfilled* is ``False``). 
		"""
		pass
		
	def flagFailure(self):
		"""
		Return a string that represents the flag for marking performance 
		measures for which the process of evaluation failed (their value is 
		``None``). 
		"""
		pass

# Excludes contribution, i.e. always returns 0
class Rexcluded(Rbase):
	"""
	Corner reduction class for excluding the performance measure from the 
	aggregate function. Objects of this class are callable and return 0. 
	See :class:`Rbase` for more information. 
	"""
	def __init__(self):
		Rbase.__init__(self)
		
	def __call__(self, shapedMeasure):
		return array(0.0)
	
	# Returns a characters for output. 
	# ' ' if fulfilled is true, '.' if fulfilled is false. 
	def flagSuccess(self, fulfilled):
		"""
		Return a string that represents a flag for marking performance 
		measures. A successfully satisfied goal is marked by ``' '`` while a 
		failure is marked by ``'.'``. 
		
		See :meth:`Rbase.flagSuccess` method for more information. 
		"""
		if fulfilled:
			return ' '
		else:
			return '.'
	
	# Returns a character for failure ('x'). 
	def flagFailure(self):
		"""
		Return a string that represents the flag for marking performance 
		measures for which the process of evaluation failed (their value is 
		``None``). Returns ``'x'``. 
		
		See :meth:`Rbase.flagFailure` method for more information. 
		"""
		return 'x'

# Returns largest contribution across corners 
class Rworst(Rbase):
	"""
	Corner reduction class for including only the worst performance measure 
	across all corners. Objects of this class are callable and return the 
	larget contribution. 
	
	See :class:`Rbase` for more information. 
	"""
	def __init__(self):
		None
	
	# Reduce along first axis (corners)
	def __call__(self, shapedMeasure):
		return shapedMeasure.max(0)
	
	def flagSuccess(self, fulfilled):
		"""
		Return a string that represents a flag for marking performance 
		measures. A successfully satisfied goal is marked by ``' '`` while a 
		failure is marked by ``'o'``. 
		
		See :meth:`Rbase.flagSuccess` method for more information. 
		"""
		if fulfilled:
			return ' '
		else:
			return 'o'
	
	def flagFailure(self):
		"""
		Return a string that represents the flag for marking performance 
		measures for which the process of evaluation failed (their value is 
		``None``). Returns ``'X'``. 
		
		See :meth:`Rbase.flagFailure` method for more information. 
		"""
		return 'X'

# Returns mean contribution. 
class Rmean(Rbase):
	"""
	Corner reduction class for including only the mean contribution across 
	all corners. Objects of this class are callable and return the mean of 
	contributions passed at call. 
	
	See :class:`Rbase` for more information. 
	"""
	def __init__(self):
		None
	
	# Reduce along first axis (corners)
	def __call__(self, shapedMeasure):
		return shapedMeasure.mean(0)
	
	def flagSuccess(self, fulfilled):
		"""
		Return a string that represents a flag for marking performance 
		measures. A successfully satisfied goal is marked by ``' '`` while a 
		failure is marked by ``'o'``. 
		
		See :meth:`Rbase.flagSuccess` method for more information. 
		"""
		if fulfilled:
			return ' '
		else:
			return 'o'
	
	def flagFailure(self):
		"""
		Return a string that represents the flag for marking performance 
		measures for which the process of evaluation failed (their value is 
		``None``). Returns ``'X'``. 
		
		See :meth:`Rbase.flagFailure` method for more information. 
		"""
		return 'X'

		
normalizations=set([])
for name in normalizationNames:
	normalizations.add(eval(name))

shapers=set([])
for name in shaperNames:
	shapers.add(eval(name))

reductions=set([])
for name in reductionNames:
	reductions.add(eval(name))


class Aggregator:
	"""
	Aggregator class. Objects of this class are callable. The calling 
	convention is ``object(paramVector)`` where *paramvector* is a list or an 
	array of input parameter values. The ordering of input parameters is given 
	at object construction. The return value is the value of the aggregate 
	function. 
	
	*perfEval* is an object of the 
	:class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	class which is used for evaluating the performance measures of the system. 
	*inputOrder* is the ordering of system's input parameters. 
	*definition* is the aggregate function description. 
	
	If *debug* is set to a value greater than 0, debug messages are generated 
	at the standard output. 
	
	Objects of this class store the details of the last evaluated aggregate 
	function value in the :attr:`results` member which is a list (one member 
	for every aggregate function component) of dictionaries with the following 
	members:
	
	* ``worst`` - the worst value of corresponding performance mesure across 
	  corners where the performance measure was computed. This is the return 
	  value of the normalization object's :meth:`Nbase.worst` method when 
	  called with with *total* set to ``True``. ``None`` if performance measure 
	  evaluation fails in at least one corner
	* ``worst_vector`` - a vector with the worst values of the performance 
	  measure. If the performance measure is a scalar this is also a scalar. 
	  If it is an array of shape (m1, m2, ...) then this is an array of the 
	  same shape. This is the return value of the normalization object's 
	  :meth:`Nbase.worst` method with *total* set to ``False``. ``None`` if 
	  performance measure evaluation fails in at least one corner. 
	* ``worst_corner`` - the index of the corner in which the worst value of 
	  performance measure occurs. If the performance measure is an array of 
	  shape (m1, m2, ..) this is still a scalar which refers to the corner 
	  index of the worst performance measure across all components of the 
	  performance measure in all corners. This is the return value of the 
	  normalization object's :meth:`Nbase.worstCornerIndex` method with 
	  *total* set to ``True``. If the performance evaluation fails in at least 
	  one corner this is the index of one of the corners where the failure 
	  occurred. 
	* ``worst_corner_vector`` - a vector of corner indices where the worst 
	  value of the performance measure is found. If the performance measure is 
	  a scalar this vector has only one component. If the performance measure 
	  is an array of shape (m1, m2, ...) this is an array with the same shape. 
	  This is the return value of the normalization object's 
	  :meth:`Nbase.worstCornerIndex` method with *total* set to ``False``. 
	  If the evaluation of a performance measure fails in at least one corner 
	  this vector holds the indices of corners in which the failure occured. 
	* ``contribution`` - the value of the contribution to the aggregate 
	  function. This is always a number, even if the evaluation of some 
	  performance measures fails (see *failure* argument to the constructor of 
	  normalization objects - e.g. :class:`Nbase`). 
	* ``fulfilled`` - ``True`` if the corresponding performance measure is 
	  successfully evaluated in all of its corresponding corners and all 
	  resulting values satisfy the corresponding goal. ``False`` otherwise. 
	
	Corner indices refer to corners in the *cornerList* member of the object
	which is a list of names of corners defined in *perfEval* (see 
	:class:`~pyopus.evaluator.performance.PerformanceEvaluator`). 
	
	The :attr:`paramVector` member holds the input parameter values passed at 
	the last call to this object. 
	"""
	# Constructor
	def __init__(self, perfEval, definition, inputOrder=None, debug=0):
		# Performance evaluator
		self.perfEval=perfEval
		
		# List of names of all defined corners
		self.cornerList=perfEval.corners.keys()
		
		# Dictionary for converting corner name to global corner index
		cornerName2index={}
		ii=0
		for corner in self.cornerList:
			cornerName2index[corner]=ii
			ii+=1
		
		# List of measures that appear in components
		measureList=set()
		for comp in definition:
			measureList.add(comp['measure'])
		
		# Conversion tables from local corner index to global corner index	
		self.localCI2globalCI={}
		for measureName in measureList:
			measure=self.perfEval.measures[measureName]
			if 'corners' in measure:
				corners=measure['corners']
			else:
				corners=self.cornerList
			globalIndex=[]
			for corner in corners:
				globalIndex.append(cornerName2index[corner])
			self.localCI2globalCI[measureName]=array(globalIndex)
		
		# Debug mode flag
		self.debug=debug
		
		# Problem definition
		self.inputOrder=inputOrder
		self.costDefinition=definition
		
		# Verify definition, set defaults
		ii=0
		for contribution in definition:
			name=contribution['measure']
			if 'norm' in contribution and type(contribution['norm']) not in normalizations:
				raise Exception, DbgMsg("AG", "Bad normalization for contribution %d (%s)" % (ii, name))
			if 'shape' in contribution:
				if type(contribution['shape']) not in shapers:
					raise Exception, DbgMsg("AG", "Bad shaper for contribution %d (%s)" % (ii, name))
			else:
				contribution['shape']=Slinear2(1.0, 0.0)
			if 'reduce' in contribution:
				if type(contribution['reduce']) not in reductions:
					raise Exception, DbgMsg("AG", "Bad reduction for contribution %d (%s)" % (ii, name))
			else:
				contribution['reduce']=Rworst()
			ii+=1
		
		# Input parameters
		self.paramVector=None
		
		# Results of the aggregate function evaluation
		self.results=None
		
	def __call__(self, paramVector):
		if self.debug:
			DbgMsgOut("AG", "Evaluation started.")
		
		# Store parameters
		self.paramVector=array(paramVector)
		
		# Create parameter dictionary
		params=paramDict(paramVector, self.inputOrder)
		
		if self.debug:
			DbgMsgOut("AG", "  Evaluating measures.")
		
		# Evaluate performance
		performances, anCount = self.perfEval(params)
		
		if self.debug:
			DbgMsgOut("AG", "  Processing")
		
		# Evaluate aggregate function
		results=[]
		cf=0;
		
		# Loop through all components of the aggregate function
		for component in self.costDefinition: 
			measureName=component['measure']
			
			# Get performance across corners
			performance=performances[measureName]
			
			# Measure info from measures dictionary of the performance evaluator
			measure=self.perfEval.measures[measureName]
			# If a measure has no corners defined use the list of all corner names 
			if 'corners' in measure:
				measureCornerList=measure['corners']
			else:
				measureCornerList=self.cornerList
				
			if self.debug:
				DbgMsgOut("AG", "    "+str(measureName))
				
			# If measure is a vector with m elements, it behaves as m independent measurements
			# The worst_vector value of a measure across corners is a vector of worst values of  
			# individual vector components across corners. It is a scalar for scalar measures. 
			# The worst value is the worst value in the the worst_vector. 
			
			# The worst_corner_vector is a vector of corner indices 
			# (based on the corner ordering in the cornerList member)
			# where every index corresponds to one component of the worst_vector. 
			# If the worst value occurs in several corners at the same time the index of the corner
			# that appears first is stored in worst_corner_vector. 
			# The worst_corner is the name of the corner with the lowest index and most appearances
			# in the worst_corner_vector.  
			
			# Collect measure values vector across corners
			failedCorners=[]
			resultCorners=[]
			resultVector=[]
			normWorst=None
			normMean=0
			for index in range(0, len(measureCornerList)):
				cornerName=measureCornerList[index]
				value=performance[cornerName]
				if value is None:
					failedCorners.append(array([index]))
				else:
					resultCorners.append(index)
					resultVector.append(array(value))
			
			# If measure is a vector (numpy array) resultVector is a list of n-d
			# arrays (they do not have to be of same size, if the size does not
			# match then the last element is multiplied until full size is reached).
			# Joining them means we obtain another array with n+1 dimensions (the
			# first dimension is the corner index and the remaining dimensions are
			# measure vector indices).
			max_dim = 0
			for index in range(len(resultVector)):
				if max_dim < resultVector[index].size:
					max_dim = resultVector[index].size
			for index in range(len(resultVector)):
				if max_dim > resultVector[index].size:
					tmp = zeros(max_dim - resultVector[index].size) + resultVector[index][-1]
					resultVector[index] = concatenate((resultVector[index], tmp))
			resultVector=array(resultVector)
			resultCorners=array(resultCorners)

			# Total number of corners
			nFailedCorners=len(failedCorners)
			nGoodCorners=len(resultCorners)
			nCorners=nFailedCorners+nGoodCorners

			# If a measure failed in some corner, the worstValueVector is simply None
			# and the worstCornerVector is the list of corners where failure occured.
			
			# Get worst value and corner(s)
			if len(failedCorners)>0:
				worstValueVector=None
				worstCornerVector=array(failedCorners)
				worstValue=None
				worstCorner=worstCornerVector[0]
			else:
				# Worst value vector (across corners)
				worstValueVector=component['norm'].worst(resultVector)
				
				# Worst corner vector
				worstCornerVector=component['norm'].worstCornerIndex(resultVector, resultCorners)
				
				# Worst value
				worstValue=component['norm'].worst(worstValueVector, True)
				
				# Worst corner
				worstCorner=component['norm'].worstCornerIndex(worstValueVector, worstCornerVector, True)
			
			# Warning... corners where measurements were successfull come first, followed
			# by corners where measurements failed. The ordering is not the same as in
			# measure['corners'].
			
			# Calculate normalized measure values
			normMeasureFailed=zeros(nFailedCorners)
			normMeasureGood=zeros(nGoodCorners)
			# Add failed corners
			normMeasureFailed[:]=component['norm'].failure
			
			# Add remaining corners (the ones where measure didn't fail)
			normMeasureGood=component['norm'](resultVector)
			
			# Check if the measure is fulfilled (in all corners)
			if len(failedCorners)<=0:
				if normMeasureGood.max()<=0:
					fulfilled=True
				else:
					fulfilled=False
			else:
				fulfilled=False
			
			# Shape normalized measure values
			shapedMeasureFailed=component['shape'](normMeasureFailed)
			shapedMeasureGood=component['shape'](normMeasureGood)
			
			# Reduce multiple corners to a single value
			# The failed part is just added up
			cfPartFailed=shapedMeasureFailed.sum()
			
			# This is still a vector if the measure is a vector
			if nGoodCorners>0:
				reduced=component['reduce'](shapedMeasureGood)
				if reduced.size>1:
					cfPartGood=reduced.sum()
				else:
					cfPartGood=reduced
			else:
				cfPartGood=0.0
			
			# Add up the shaped part and the failed part
			cfPart=cfPartGood+cfPartFailed
			
			# Convert corner indices from measure corner index to global corner index
			convTable=self.localCI2globalCI[measureName]
			worstCorner=convTable[worstCorner]
			
			if type(worstCornerVector) is ndarray:
				for ii in range(worstCornerVector.size):
					worstCornerVector[ii]=convTable[worstCornerVector[ii]]
			else:
				worstCornerVector=convTable[worstCornerVector]
				
			# Put it in results structure
			thisResult={}
			thisResult['worst']=worstValue
			thisResult['worst_corner']=worstCorner
			thisResult['worst_vector']=worstValueVector
			thisResult['worst_corner_vector']=worstCornerVector
			thisResult['contribution']=cfPart
			thisResult['fulfilled']=fulfilled
			results.append(thisResult)
			
			cf+=cfPart
			
		self.results=results
		
		return cf
		
	def allFulfilled(self):
		"""
		Returns ``True`` if the performance measures corresponding to all 
		aggregate function components that were evaluated with the last call 
		to this object were successfully evaluated and fulfill their 
		corresponding goals. All components are taken into account, even 
		those using the :class:`Rexcluded` corner reduction. 
		"""
		for result in self.results:
			if not result['fulfilled']:
				return False
		
		return True
	
	def allBelowOrAtZero(self):
		"""
		Returns ``True`` if all components of the aggregate function computed 
		with the last call to this object are not greater than zero. 
		Assumes that the following holds: 
		
		The return value is ``True``, if all performance measures corresponding 
		to aggregate function components not using the :class:`Rexcluded` 
		corner reduction satisfy their goals; assuming that
		
		* normalization produces positive values for satisfied goals and 
		  negative values for unsatisfied goals
		* normalization returns a positive value in case of a failure to 
		  evaluate a performance measure (*failed* is greater than 0)
		* aggregate function shaping is nondecreasing and is greater than zero 
		  for positive normalized performance measures
		"""
		for result in self.results:
			if result['contribution']>0:
				return False
		
		return True
			
	def formatResults(self, nTargetSpec=29, nMeasureName=12, nNumber=12, nSig=3, nCornerName=6):
		"""
		Formats a string representing the results obtained with the last call 
		to this object. Only the worst performance across corners along with 
		the corresponding aggregate function component value is reported. 
		Generates one line for every aggregate function component. 
		
		*nTargetSpec* specifies the formatting width for the target 
		specification (specified by the	corresponding normalization object) of 
		which *nMeasureName* is used for the name of the performance measure. 
		*nNumber* and *nSig* specify the width of the formatting and the number 
		of the significant digits for the aggregate function contribution. 
		*nCornerName* specifies the width of the formatting for the worst 
		corner name. 
		"""
		output=""
		
		first=True
		for component, result in zip(self.costDefinition, self.results):
			if not first:
				output+="\n"
			else:
				first=False
			measureName=component['measure']
			measure=self.perfEval.measures[measureName]
			
			# Format measurement target
			targetSpec=component['norm'].report(measureName, nMeasureName, nNumber, nSig)
			
			# Format worst value text
			if 'corners' in measure:
				cornerCount=len(measure['corners'])
			else:
				cornerCount=len(self.cornerList)
			if result['worst'] is None:
				failedCount=len(result['worst_corner_vector'])
				statusText=component['reduce'].flagFailure()
				worstText="%-*s" % (nNumber, ("%d/%d" % ((cornerCount-failedCount), cornerCount)))
				cornerText="%*s" % (nCornerName, " ")
			else:
				failedCount=0
				statusText=component['reduce'].flagSuccess(result['fulfilled'])
				worstText="%*.*e" % (nNumber, nSig, result['worst'])
				cornerText="%*s" % (nCornerName, self.cornerList[result['worst_corner']])
				
			# Format contribution text
			contribText="%.*g" % (nSig, result['contribution'])
			
			if len(targetSpec)>nTargetSpec:
				output+=targetSpec+"\n"+("%*s | " % (nTargetSpec, ""))+statusText+" "+worstText+" "+cornerText+" : "+contribText
			else:
				output+=targetSpec+" | "+statusText+" "+worstText+" "+cornerText+" : "+contribText
		
		return output
	
	def formatParameters(self, x=None, nParamName=15, nNumber=15, nSig=6):
		"""
		Formats a string corresponding to the parameters passed at the last 
		call to this object. Generates one line for every parameter. If *x* is 
		specified it is used instead of the stored parameter vector. 
		*nParamName* and *nNumber* specify the width of the formatting for the 
		parameter name and its value. *nSig* specifies the number of 
		significant digits. 
		"""
		output=''
		
		if x is None:
			paramVec=self.paramVector
		else:
			paramVec=x
			
		return formatParameters(paramDict(paramVec, self.inputOrder), self.inputOrder, nParamName, nNumber, nSig)
	
	# Return annotator plugin. 
	def getAnnotator(self):
		"""
		Returns an object of the :class:`CostAnnotator` class which can be used 
		as a plugin for iterative algorithms. The plugin takes care of aggregate 
		function details (:attr:`results` member) propagation from the machine 
		where the evaluation of the aggregate function takes place to the machine 
		where the evaluation was requested (usually the master). 
		"""
		return AggregatorAnnotator(self)
	
	# Return collector plugin. 
	def getCollector(self, chunkSize=10):
		"""
		Returns an object of the :class:`CostCollector` class which can be 
		used as a plugin for iterative algorithms. The plugin gathers input 
		parameter and aggregate function values across iterations of the 
		algorithm. 
		
		*chunkSize* is the chunk size used when allocating space for stored 
		values (10 means allocation takes place every 10 iterations). 
		"""
		return CostCollector(self, chunkSize)
		
	# Return stopper plugin that stops optimization when all requirements are satisfied. 
	def getStopWhenAllSatisfied(self):
		"""
		Returns an object of the :class:`StopWhenAllSatisfied` class which can 
		be used as a plugin for iterative algorithms. The plugin signals the 
		iterative algorithm to stop when all contributions obtained with 
		the last call to this :class:`Aggregator` object are smaller than 
		zero (when the :meth:`allBelowOrAtZero` method returns ``True``). 
		"""
		return StopWhenAllSatisfied(self)
		
	# Return reporter plugin. 
	def getReporter(self, reportParameters=True):
		"""
		Returns an object of the :class:`ReportCostCorners` class which can be 
		used as a plugin for iterative algorithms. Every time an iterative 
		algorithm calls this :class:`Aggregator` object the reporter is 
		invoked and prints the details of the aggregate function components. 
		"""
		return AggregatorReporter(self, reportParameters)
		
# Default annotator for Aggregator
class AggregatorAnnotator(Annotator):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Annotator` iterative 
	algorithm plugin class. This is a callable object whose job is to
	
	* produce an annotation (details of the aggregate function value) stored 
	  in the *aggregator* object
	* update the *aggregator* object with the given annotation 
	
	Annotation is a copy of the :attr:`results` member of *aggregator*. 
	
	Annotators are used for propagating the details of the aggregate function 
	from the machine where the evaluation takes place to the machine where the 
	evaluation was requested (usually the master). 
	"""
	def __init__(self, aggregator):
		self.ce=aggregator
	
	def produce(self):
		return self.ce.results
	
	def consume(self, annotation):
		self.ce.results=annotation
		
# Stopper that stops the algorithm when all requirements are satisfied
class StopWhenAllSatisfied(Stopper, AggregatorAnnotator):		
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Stopper` iterative 
	algorithm plugin class that stops the algorithm when the 
	:meth:`Aggregator.allBelowOrAtZero` method of the *Aggregator* object 
	returns ``True``. 
	
	This class is also an annotator that collects the results at remote 
	evaluation and copies them to the host where the remote evaluation was 
	requested. 
	"""
	def __init__(self, aggregator): 
		Stopper.__init__(self)
		AggregatorAnnotator.__init__(self, aggregator)
		self.aggregator=aggregator
		
	def __call__(self, x, f, opt): 
		opt.stop=opt.stop or self.aggregator.allBelowOrAtZero()
		
		
# Reporter for reporting the results of aggregate function evaluation
# Reports details of every best-yet cf improvement
# One report per iterSpacing iterations without best-yet cf improvement
class AggregatorReporter(Reporter, AggregatorAnnotator):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Reporter` iterative 
	algorithm plugin class that reports the details of the last evaluated 
	aggregate function value obtained by the *aggregator* object. Uses 
	the :meth:`Aggregator.reportParameters` and 
	:meth:`Aggregator.formatResults` methods of *aggregator* for 
	obtaining the output that is printed at first iteration and every time 
	the aggregate function value decreases. 
	
	If *reportParameters* is ``True`` the report includes the corresponding 
	input parameter values. 
	
	This class is also an annotator that collects the results at remote 
	evaluation and copies them to the host where the remote evaluation was 
	requested. 
	"""
	
	def __init__(self, aggregator, reportParameters=True):
		Reporter.__init__(self)
		AggregatorAnnotator.__init__(self, aggregator)
		
		self.aggregator=aggregator
		self.reportParameters=reportParameters
		
	def __call__(self, x, f, opt):
		# Print basic information for every iteration
		Reporter.__call__(self, x, f, opt)
		
		# Print details for first iteration and for every improvement
		details=(opt.f is None) or (opt.niter==opt.bestIter)
		
		# Print details. This requires an annotation either to be received or created. 
		if details and not self.quiet:
			# Report details
			
			# Print parameters
			msg=self.aggregator.formatParameters(x)
			print(msg)
			
			# Print performance
			msg=self.aggregator.formatResults()
			print(msg)
	

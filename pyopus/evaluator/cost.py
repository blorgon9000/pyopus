# Cost evaluation module
"""
.. inheritance-diagram:: pyopus.evaluator.cost
    :parts: 1
	
**Cost function management module (PyOPUS subsystem name: CE)**

**Normalization** is the process where a performance measure is scaled in such 
way that values not satifying the goal result in normalized values smaller than 
0. If the performance measure value exceeds the goal the normalized value is 
greater than 0. 

**Cost shaping** is the process where the normalized performance is shaped. 
Usually positive values (corresponding to performance not satifying the goal) 
are shaped differently than negative values (corresponding to performance 
satifying the goal). Cost shaping results in a **shaped cost contribution** for 
every corner. 

**Corner reduction** is the process where **shaped cost contributions** of 
individual corners are incorporated into the cost function. There are several 
ways how to achieve this. For instance one could incorporate only the 
contribution of the corner in which worst peroformance is observed, or on the 
other hand one could incorporate the mean contribution of all corners. 

The main data structure is the **cost function description** which is a list 
of **cost function component descriptions**. 
Every **cost function component description** is a dictionary with the following 
members: 

* ``measure`` - the name of the performance meeasure on which the cost 
  function component is based. 
* ``goal`` - a string that evaluates to an object representing the 
  normalization of the performance
  measureof. 
* ``shape`` - a string that evaluates to an object representing the cost 
  shaping of normalized performance measure. 
  Defaults to ``CSlinear2(1.0, 0.0)``. 
* ``reduce`` - a string that evaluates to an object representing the corner 
  reduction of shaped cost cotributions. Defaults to ``CCworst()``. 
  
The cost function is a function of the input parameters. The **cost function 
input description** is a dictionary of **parameter descriptions** with 
parameter name as key. Every **parameter description** is a dictionary with 
the following members: 

* ``init`` - the initial value of the parameter 
* ``lo`` - the upper bound on the parameter 
* ``hi`` - the lower bound on the parameter
* ``step`` - the step in which the parameter is incremented

A **parameter dictionary** is a dictionary with parameter name as key holding 
the values of parameters. 

The **ordering of parameters** is a list of parameter names that defines the 
order in which parameter values apper in a parameter vector.

A **parameter vector** is a list of parameter values where the values are 
ordered according to a given **ordering of input parameters**. 
"""

from ..optimizer.base import Reporter, Stopper, Plugin
from ..misc.debug import DbgMsgOut, DbgMsg
from numpy import concatenate, array, ndarray, where, zeros, sum, abs
import sys

__all__ = [ 'formatParameters', 'parameterDictionary', 'parameterVector', 'parameterSetup', 
	'MNbase', 'MNabove', 'MNbelow', 'CSlinear2', 'CCbase', 'CCexcluded', 'CCworst', 'CCmean', 
	'CostEvaluator', 'CostAnnotator', 'CostCollector', 'ReportCostCorners' ] 

import pdb

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
		
	for paramName in inputOrder:
		output += '%*s: %*.*e\n' % (nParamName, paramName, nNumber, nSig, param[paramName])
	
	return output

def parameterDictionary(inputOrder, parameterVector): 
	"""
	Returns a parameter dictionary corresponding to *parameterVector* where the 
	ordering of parameters is given by *inputOrder*. *parameterVector* can be a 
	list of values or an array. 
	"""
	if type(parameterVector) is ndarray:
		pv=list(parameterVector)
	elif type(parameterVector) is list: 
		pv=parameterVector
	else:
		raise Exception, DbgMsgOut("CE", "Parameter vector is not a vector.")
	
	params={}
	for index in range(0, len(pv)):
		params[inputOrder[index]]=pv[index]
		
	return params 

def parameterVector(inputOrder, paramDict):
	"""
	Return a list of parameter values in order *inputOrder* corresponding to 
	*paramDict* parameter dictionary. 
	"""
	x=[]
	for paramName in inputOrder: 
		x.append(paramDict[paramName])
	return x

def parameterSetup(inputOrder, costInput): 
	"""
	Returns tuple (*initVec*, *loVec*, *hiVec*, *stepVec*) specifying the lists 
	of initial, low, high, and stepvalues for every parameter. Parameter order 
	is given by the *inputOrder* while the values are taken from the cost 
	function input description given by *costInput*. 
	"""
	initVec=[]
	loVec=[]
	hiVec=[]
	stepVec=[]
	for paramName in inputOrder: 
		initVec.append(costInput[paramName]['init'])
		loVec.append(costInput[paramName]['lo'])
		hiVec.append(costInput[paramName]['hi'])
		stepVec.append(costInput[paramName]['step'])
	
	return (initVec, loVec, hiVec, stepVec)
	

# Basic normalization class	
class MNbase(object):
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
	*goal*/10 or 1.0 if *goal* is equal to 0.0. If *goal* is a vector, *norm* 
	must either be a vector of the same size or a scalar is which case it 
	applies to all components of the *goal*. 
	"""
	def __init__(self, goal, norm=None, failure=10000.0):
		self.goal=array(goal)

		if norm is None:
			if self.goal.size==1:
				self.norm=abs(self.goal)/10.0
				if self.norm==0:
					self.norm=1.0
			else:
				self.norm=abs(self.goal)/10.0
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
class MNabove(MNbase):
	"""
	Performance normalization class requiring the performance to to be above 
	the given *goal*. See :class:`MNbase` for more information. 
	"""
	def __init__(self, goal, norm=None, failure=10000.0):
		MNbase.__init__(self, goal, norm, failure)

	def worst(self, values, total=False):
		"""
		Find the worst *value*. See :meth:`MNbase.worst` method for more 
		information. 
		"""
		if not total:
			return values.min(0)
		else:
			return values.min()
	
	def worstCornerIndex(self, values, corners, total=False):
		"""
		Find the worst corner index. See :meth:`MNbase.worstCornerIndex` method 
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
		
		See :meth:`MNbase.report` method for more information. 
		"""
		if self.goal.size!=1:
			return "%*s  >  %-*s" % (nName, name, nGoal, 'vector')
		else:
			return "%*s  >  %-*.*e" % (nName, name, nGoal, nSigGoal, self.goal)

			
# Normalization for targets of the form value<goal. 
class MNbelow(MNbase):
	"""
	Performance normalization class requiring the performance to to be below 
	the given *goal*. See :class:`MNbase` for more information. 
	"""
	def __init__(self, goal, norm=None, failure=10000.0):
		MNbase.__init__(self, goal, norm, failure)
		
	def worst(self, values, total=False):
		"""
		Find the worst *value*. See :meth:`MNbase.worst` method for more 
		information. 
		"""
		if not total:
			return values.max(0)
		else:
			return values.max()
	
	def worstCornerIndex(self, values, corners, total=False):
		"""
		Find the worst corner index. See :meth:`MNbase.worstCornerIndex` method 
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
		
		See :meth:`MNbase.report` method for more information. 
		"""
		if self.goal.size!=1:
			return "%*s  <  %-*s" % (nName, name, nGoal, 'vector')
		else:
			return "%*s  <  %-*.*e" % (nName, name,nGoal, nSigGoal, self.goal)


# Normalization for targets of the form goal<value<goalHigh. 
class MNbetween(MNbase):
	"""
	Performance normalization class requiring the performance to to be above 
	*goal* and below *goalHigh*. See :class:`MNbase` for more information. 
	This class is deprecated. Use two cost contributions instead (one with 
	MNbelow and one with MNabove). 
	"""
	def __init__(self, goal, goalHigh, norm=None, failure=10000.0):
		MNbase.__init__(self, goal, norm, failure)
		
		self.goalHigh=array(goalHigh)
		
		if (self.goal>self.goalHigh).any():
			raise Exception, DbgMsg("CE", "Lower bound is above upper bound.")
			
		if norm is None:
			if self.goal.size==1:
				self.norm=abs(self.goalHigh-self.goal)/10.0
				if self.norm==0:
					self.norm=1.0
			else:
				self.norm=abs(self.goalHigh-self.goal)/10.0
				self.norm[where(self.norm==0.0)]=1.0
		else:
			self.norm=norm

	def worst(self, values, total=False):
		"""
		Find the worst *value*. See :meth:`MNbase.worst` method for more 
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
		Find the worst corner index. See :meth:`MNbase.worstCornerIndex` method 
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
		
		See :meth:`MNbase.report` method for more information. 
		"""
		if self.goal.size!=1:
			return "%*s < > %-*s" % (nName, name, nGoal, 'vector')
		else:
			return "%*s < > %-*.*e" % (nName, name, nGoal, nSigGoal, (self.goal+self.goalHigh)/2)
			

# Linear two-segment cost shaping 
class CSlinear2(object):
	"""
	Two-segment linear cost shaping. Normalized performances above 0 (failing 
	to satify the goal) are multiplied by *w*, while the ones below 0 
	(satisfying the goal) are multiplied by *tw*. This cost shaping has a 
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
# Reducing cost contributions from multiple corners to one. 
# The corners are the first dimension of the input array. 
class CCbase(object):
	"""
	Basic corner reduction class. Objects of this class are callable. The 
	calling convention of the object is ``object(shapedMeasure)`` where 
	*shapedMeasure* is an array of shaped cost contributions. The return value 
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

# Excludes cost contribution, i.e. always returns 0
class CCexcluded(CCbase):
	"""
	Corner reduction class for excluding the performance measure from the cost 
	function. Objects of this class are callable and return 0. See 
	:class:`CCbase` for more information. 
	"""
	def __init__(self):
		CCbase.__init__(self)
		
	def __call__(self, shapedMeasure):
		return array(0.0)
	
	# Returns a characters for output. 
	# ' ' if fulfilled is true, '.' if fulfilled is false. 
	def flagSuccess(self, fulfilled):
		"""
		Return a string that represents a flag for marking performance 
		measures. A successfully satisfied goal is marked by ``' '`` while a 
		failure is marked by ``'.'``. 
		
		See :meth:`CCbase.flagSuccess` method for more information. 
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
		
		See :meth:`CCbase.flagFailure` method for more information. 
		"""
		return 'x'

# Returns largest contribution across corners 
class CCworst(CCbase):
	"""
	Corner reduction class for including only the worst performance measure 
	across all corners. Objects of this class are callable and return the 
	larget cost contribution. 
	
	See :class:`CCbase` for more information. 
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
		
		See :meth:`CCbase.flagSuccess` method for more information. 
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
		
		See :meth:`CCbase.flagFailure` method for more information. 
		"""
		return 'X'

# Returns mean contribution. 
class CCmean(CCbase):
	"""
	Corner reduction class for including only the mean cost contribution across 
	all corners. Objects of this class are callable and return the mean of cost 
	contribution passed at call. 
	
	See :class:`CCbase` for more information. 
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
		
		See :meth:`CCbase.flagSuccess` method for more information. 
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
		
		See :meth:`CCbase.flagFailure` method for more information. 
		"""
		return 'X'
		

# Cost evaluator class
class CostEvaluator:
	"""
	Cost evaluator class. Objects of this class are callable. The calling 
	convention is ``object(paramVector)`` where *paramvector* is an array of 
	input parameter values. The ordering of input parameters is given at object 
	construction. The return value is the value of the cost function. 
	
	*perfEval* is an object of the 
	:class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	class which is used for evaluating the performance measures of the system. 
	*inputOrder* is the ordering of system's input parameters. *costDefinition* 
	is the cost function description. 
	
	If *debug* is set to a value greater than 0, debug messages are generated 
	at the standard output. 
	
	Objects of this class store the details of the last evaluated cost function 
	value in the :attr:`results` member which is a list (one member for every 
	cost function component) of dictionaries with the following members:
	
	* ``worst`` - the worst value of corresponding performance mesure across 
	  corners where the performance measure was computed. This is the return 
	  value of the normalization object's :meth:`MNbase.worst` method when 
	  called with with *total* set to ``True``. ``None`` if performance measure 
	  evaluation fails in at least one corner
	* ``worst_vector`` - a vector with the worst values of the performance 
	  measure. If the performance measure is a scalar this is also a scalar. 
	  If it is an array of shape (m1, m2, ...) then this is an array of the 
	  same shape. This is the return value of the normalization object's 
	  :meth:`MNbase.worst` method with *total* set to ``False``. ``None`` if 
	  performance measure evaluation fails in at least one corner. 
	* ``worst_corner`` - the index of the corner in which the worst value of 
	  performance measure occurs. If the performance measure is an array of 
	  shape (m1, m2, ..) this is still a scalar which refers to the corner 
	  index of the worst performance measure across all components of the 
	  performance measure in all corners. This is the return value of the 
	  normalization object's :meth:`MNbase.worstCornerIndex` method with 
	  *total* set to ``True``. If the performance evaluation fails in at least 
	  one corner this is the index of one of the corners where the failure 
	  occurred. 
	* ``worst_corner_vector`` - a vector of corner indices where the worst 
	  value of the performance measure is found. If the performance measure is 
	  a scalar this vector has only one component. If the performance measure 
	  is an array of shape (m1, m2, ...) this is an array with the same shape. 
	  This is the return value of the normalization object's 
	  :meth:`MNbase.worstCornerIndex` method with *total* set to ``False``. 
	  If the evaluation of a performance measure fails in at least one corner 
	  this vector holds the indices of corners in which the failure occured. 
	* ``contribution`` - the value of the contribution to the cost function. 
	  This is always a number, even if the evaluation of some performance 
	  measures fails (see *failure* argument to the constructor of 
	  normalization objects - e.g. :class:`MNbase`). 
	* ``fulfilled`` - ``True`` if the corresponding performance measure is 
	  successfully evaluated in all of its corresponding corners and all 
	  resulting values satisfy the corresponding goal. ``False`` otherwise. 
	
	Corner indices refer to corners as they are defined by the *corners* 
	argument a list of corner descriptions) to the constructor of *perfEval* 
	object (see :class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	class). 
	
	The :attr:`paramVector` member holds the input parameter values passed at 
	the last call to this object. 
	"""
	# Constructor
	def __init__(self, perfEval, inputOrder, costDefinition, debug=0):
		# Performance evaluator
		self.perfEval=perfEval
		
		# Debug mode flag
		self.debug=debug
		
		# Problem definition
		self.inputOrder=inputOrder
		self.costDefinition=costDefinition
		
		# Input parameters
		self.paramVector=None
		
		# Results of the cost evaluation
		self.results=None
		
		# Compile for faster evaluation
		if self.debug:
			DbgMsgOut("CE", "Compiling.")
		
		self._compile()
	
	# Compile stuff for faster evaluation
	def _compile(self):
		"""
		Prepares internal structures for faster processing. 
		This function should never be called by the user. 
		"""
		# Defaults, compilation
		self.compiled=[]
		for contrib in self.costDefinition: 
			measureName=contrib['measure']
			if self.debug:
				DbgMsgOut("CE", "  processing "+str(measureName))
				
			compiledContrib={}
			
			# Goal has no default
			compiledContrib['goal']=eval(contrib['goal'])
			
			# Shape default
			if 'shape' in contrib:
				compiledContrib['shape']=eval(contrib['shape'])
			else:
				compiledContrib['shape']=CSlinear2(1.0, 0.0)
			
			# Corner reduction default
			if 'reduce' in contrib:
				compiledContrib['reduce']=eval(contrib['reduce'])
			else:
				compiledContrib['reduce']=CCworst()
			
			self.compiled.append(compiledContrib)
			
	# For pickling
	def __getstate__(self):
		state=self.__dict__.copy()
		del state['compiled']
		
		return state
	
	# For unpickling
	def __setstate__(self, state):
		self.__dict__.update(state)
		
		self._compile()
		
	def __call__(self, paramVector):
		if self.debug:
			DbgMsgOut("CE", "Evaluation started.")
		
		# Store parameters
		self.paramVector=paramVector
		
		# Create parameter dictionary
		params=parameterDictionary(self.inputOrder, paramVector)
		
		if self.debug:
			DbgMsgOut("CE", "  Evaluating measures.")
		
		# Evaluate performance
		performances=self.perfEval(params)
		
		if self.debug:
			DbgMsgOut("CE", "  Processing")
		
		# Evaluate cost function
		results=[]
		cf=0;
		
		# Loop through all components of the cost function
		for component, compiled in zip(self.costDefinition, self.compiled): 
			measureName=component['measure']
			
			# Get performance across corners
			performance=performances[measureName]
			
			# Measure ifno from measures dictionary of the performance evaluator
			measure=self.perfEval.measures[measureName]
			cornerList=measure['corners']
			
			if self.debug:
				DbgMsgOut("CE", "    "+str(measureName))
				
			# If measure is a vector with m elements, it behaves as m independent measurements
			# The worst_vector value of a measure across corners is a vector of worst values of  
			# individual vector components across corners. It is a scalar for scalar measures. 
			# The worst value is the worst value in the the worst_vector. 
			
			# The worst_corner_vector is a vector of corner indices (based on the corner ordering 
			# of the particular measure in the measures dictionary of the performance evaluator)
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
			for index in range(0, len(cornerList)):
				cornerName=cornerList[index]
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
				worstValueVector=compiled['goal'].worst(resultVector)
				
				# Worst corner vector
				worstCornerVector=compiled['goal'].worstCornerIndex(resultVector, resultCorners)
				
				# Worst value
				worstValue=compiled['goal'].worst(worstValueVector, True)
				
				# Worst corner
				worstCorner=compiled['goal'].worstCornerIndex(worstValueVector, worstCornerVector, True)
			
			# Warning... corners where measurements were successfull come first, followed
			# by corners where measurements failed. The ordering is not the same as in
			# measure['corners'].
			
			# Calculate normalized measure values
			normMeasureFailed=zeros(nFailedCorners)
			normMeasureGood=zeros(nGoodCorners)
			# Add failed corners
			normMeasureFailed[:]=compiled['goal'].failure
			
			# Add remaining corners (the ones where measure didn't fail)
			normMeasureGood=compiled['goal'](resultVector)
			
			# Check if the measure is fulfilled (in all corners)
			if len(failedCorners)<=0:
				if normMeasureGood.max()<=0:
					fulfilled=True
				else:
					fulfilled=False
			else:
				fulfilled=False
			
			# Shape normalized measure values
			shapedMeasureFailed=compiled['shape'](normMeasureFailed)
			shapedMeasureGood=compiled['shape'](normMeasureGood)
			
			# Reduce multiple corners to a single value
			# The failed part is just added up
			cfPartFailed=shapedMeasureFailed.sum()
			
			# This is still a vector if the measure is a vector
			if nGoodCorners>0:
				reduced=compiled['reduce'](shapedMeasureGood)
				if reduced.size>1:
					cfPartGood=reduced.sum()
				else:
					cfPartGood=reduced
			else:
				cfPartGood=0.0
			
			# Add up the shaped part and the failed part
			cfPart=cfPartGood+cfPartFailed

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
		Returns ``True`` if the performance measures corresponding to all cost 
		function components that were evaluated with the last call to this 
		object were successfully evaluated and fulfill their corresponding 
		goals. All cost function components are taken into account, even those 
		with the :class:`CCexcluded` corner reduction. 
		"""
		for result in self.results:
			if not result['fulfilled']:
				return False
		
		return True
	
	def allBelowOrAtZero(self):
		"""
		Returns ``True`` if all cost contributions obtained with the last call 
		to this object are not greater than zero. Assuming that the following 
		holds: 
		
		* normalization produces positive values for satisfied goals and 
		  negative values for unsatisfied goals
		* normalization returns a positive value in case of a failure to 
		  evaluate a performance measure (*failed* is greater than 0)
		* cost function shaping is nondecreasing and is greater than zero for 
		  positive normalized performance measures
		
		the return value is ``True``, if all performance measures that appear 
		in cost function components not using the :class:`CCexcluded` corner 
		reduction satisfy their goals.
		"""
		for result in self.results:
			if result['contribution']>0:
				return False
		
		return True
			
	def formatResults(self, nTargetSpec=29, nMeasureName=12, nNumber=12, nSig=3, nCornerName=6):
		"""
		Formats a string representing the results obtained with the last call 
		to this object. Only the worst performance across corners along with 
		the corresponding cost function component values is reported. Generates 
		one line for every cost function component. 
		
		*nTargetSpec* specifies the formatting width for the target 
		specification (specified by the	corresponding normalization object) of 
		which *nMeasureName* is used for the name of the performance measure. 
		*nNumber* and *nSig* specify the width of the formatting and the number 
		of the significant digits for the cost function contribution. 
		*nCornerName* specifies the width of the formatting for the worst 
		corner name. 
		"""
		output=""
		
		for component, compiled, result in zip(self.costDefinition, self.compiled, self.results):
			measureName=component['measure']
			measure=self.perfEval.measures[measureName]
			
			# Format measurement target
			targetSpec=compiled['goal'].report(measureName, nMeasureName, nNumber, nSig)
			
			# Format worst value text
			cornerCount=len(measure['corners'])
			if result['worst'] is None:
				failedCount=len(result['worst_corner_vector'])
				statusText=compiled['reduce'].flagFailure()
				worstText="%-*s" % (nNumber, ("%d/%d" % ((cornerCount-failedCount), cornerCount)))
				cornerText="%*s" % (nCornerName, " ")
			else:
				failedCount=0
				statusText=compiled['reduce'].flagSuccess(result['fulfilled'])
				worstText="%*.*e" % (nNumber, nSig, result['worst'])
				cornerText="%*s" % (nCornerName, measure['corners'][result['worst_corner']])
				
			# Format contribution text
			contribText="%.*g" % (nSig, result['contribution'])
			
			if len(targetSpec)>nTargetSpec:
				output+=targetSpec+"\n"+("%*s | " % (nTargetSpec, ""))+statusText+" "+worstText+" "+cornerText+" : "+contribText+"\n"
			else:
				output+=targetSpec+" | "+statusText+" "+worstText+" "+cornerText+" : "+contribText+"\n"
		
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
			
		return formatParameters(parameterDictionary(self.inputOrder, paramVec), self.inputOrder, nParamName, nNumber, nSig)
	
	# Return annotator plugin. 
	def getAnnotator(self):
		"""
		Returns an object of the :class:`CostAnnotator` class which can be used 
		as a plugin for iterative algorithms. The plugin takes care of cost 
		function details (:attr:`results` member) propagation from the machine 
		where the evaluation of the cost function takes place to the machine 
		where the evaluation was requested (usually the master). 
		"""
		return CostAnnotator(self)
	
	# Return collector plugin. 
	def getCollector(self, chunkSize=10):
		"""
		Returns an object of the :class:`CostCollector` class which can be 
		used as a plugin for iterative algorithms. The plugin gathers input 
		parameter and cost function values across iterations of the algorithm. 
		
		*chunkSize* is the chunk size used when allocating space for stored 
		values (10 means allocation takes place every 10 iterations). 
		"""
		return CostCollector(self, chunkSize)
		
	# Return stopper plugin that stops optimization when all requirements are satisfied. 
	def getStopWhenAllSatisfied(self):
		"""
		Returns an object of the :class:`StopWhenAllSatisfied` class which can 
		be used as a plugin for iterative algorithms. The plugin signals the 
		iterative algorithm to stop when all cost contributions obtained with 
		the last call to this :class:`CostEvaluator` object are smaller than 
		zero (when the :meth:`allBelowOrAtZero` method returns ``True``). 
		"""
		return StopWhenAllSatisfied(self)
		
	# Return reporter plugin. 
	def getReporter(self, reportParameters=True):
		"""
		Returns an object of the :class:`ReportCostCorners` class which can be 
		used as a plugin for iterative algorithms. Every time an iterative 
		algorithm calls this :class:`CostEvaluator` object the reporter is 
		invoked and prints the details of the cost function components. 
		"""
		return ReportCostCorners(self, reportParameters)
		
		
# Stopper that stops the algorithm when all requirements are satisfied
class StopWhenAllSatisfied(Stopper):		
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Stopper` iterative 
	algorithm plugin class that stops the algorithm when the 
	:meth:`CostEvaluator.allBelowOrAtZero` method of the *costEvaluator* object 
	returns ``True``. 
	"""
	def __init__(self, costEvaluator): 
		Stopper.__init__(self)
		
		self.costEvaluator=costEvaluator
		
	def __call__(self, x, f, opt, annotation=None): 
		if annotation is None: 
			opt.stop=opt.stop or self.costEvaluator.allBelowOrAtZero()
		else:
			opt.stop=opt.stop or annotation
		
		return opt.stop

		
# Reporter for reporting the results of cost evaluation
# Reports details of every best-yet cf improvement
# One report per iterSpacing iterations without best-yet cf improvement
class ReportCostCorners(Reporter):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Reporter` iterative 
	algorithm plugin class that reports the details of the last evaluated cost 
	function value obtained by the *costEvaluator* object. Uses the 
	:meth:`CostEvaluator.reportParameters` and 
	:meth:`CostEvaluator.formatResults` methods of *costEvaluator* for 
	obtaining the output that is printed at first iteration and every time the 
	cost function value decreases. 
	
	If *reportParameters* is ``True`` the report includes the corresponding 
	input parameter values. 
	
	The :class:`ReportCostCorners` takes care of annotations (i.e. once it is 
	installed in an iterative algorithm no further :class:`CostAnnotator` is 
	needed). 
	"""
	
	def __init__(self, costEvaluator, reportParameters=True):
		Reporter.__init__(self)
		
		self.costEvaluator=costEvaluator
		self.reportParameters=reportParameters
		
	def __call__(self, x, f, opt, annotation=None):
		# Print basic information for every iteration
		Reporter.__call__(self, x, f, opt, annotation)
		
		# Print details for first iteration and for every improvement
		details=(opt.f is None) or (opt.niter==opt.bestIter)
		
		# If annotation was given, use it
		if annotation is not None:
			self.costEvaluator.results=annotation
		else:
			# Produce annotation, but only if details are going to be reported. 
			# We can save a lot on communication here. 
			if details:
				annotation=self.costEvaluator.results
			else:
				annotation=None
		
		# Print details. This requires an annotation either to be received or created. 
		if details and not self.quiet:
			# Report details
			
			# Print parameters
			msg=self.costEvaluator.formatParameters(x)
			print(msg)
			
			# Print performance
			msg=self.costEvaluator.formatResults()
			print(msg)
			
		return annotation
	
	
# Default annotator for cost evaluator
class CostAnnotator(Plugin):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Plugin` iterative 
	algorithm plugin class. This is a callable object whose job is to
	
	* produce an annotation (details of the cost function value) stored in the 
	  *costEvaluator* object (when invoked with ``None`` for *annotation*)
	* update the *costEvaluator* object with the given annotation 
	  (when invoked with an *annotation* that is not ``None``)
	
	Annotation is a copy of the :attr:`results` member of *costEvaluator*. 
	
	Annotators are used for propagating the details of the cost function from 
	the machine where the evaluation takes place to the machine where the 
	evaluation was requested (usually the master). 
	"""
	def __init__(self, costEvaluator):
		self.ce=costEvaluator
	
	def __call__(self, x, f, opt, annotation=None):
		if annotation is None:
			return self.ce.results.copy()
		else:
			self.ce.results=annotation

			
# Cost function and input parameter collector
class CostCollector(Plugin):
	"""
	A subclass of the :class:`~pyopus.optimizer.base.Plugin` iterative 
	algorithm plugin class. This is a callable object invoked at every 
	iteration of the algorithm. It collects the input parameter vector 
	(n components) and the cost function value. 
	
	Let niter denote the number of stored iterations. The input parameter 
	values are stored in the :attr:`xval` member which is an array of shape 
	(niter, n) while the cost function values are stored in the :attr:`fval` 
	member (array with shape (niter)). 
	
	Some iterative algorithms do not evaluate iterations sequentially. Such 
	algorithms denote the iteration number with the :attr:`index` member. If 
	the :attr:`index` member is not present in the iterative algorithm object 
	the internal iteration counter of the :class:`CostCollector` is used. 
	
	The first index in the *xval* and *fval* arrays is the iteration index. If 
	iterations are not performed sequentially these two arrays may contain gaps 
	where no valid input parameter or cost function value is found. The gaps 
	are denoted by the *valid* array (of shape (niter)) where zeros denote a 
	gap. 
	
	*xval*, *fval*, and *valid* arrays are resized in chunks of size 
	*chunkSize*. 
	"""
	def __init__(self, chunkSize=100): 
		Plugin.__init__(self)
		
		self.xval=None
		self.fval=None
		self.n = chunkSize
		self.memLen = chunkSize
		
	def __call__(self, x, f, opt, annotation=None):
		if self.xval is None:
			#allocate space in memory
			self.xval = zeros([self.n,len(x)])
			self.fval = zeros([self.n])
			self.valid = zeros([self.n])
			self.localindex = 0 
		
		if 'index' in opt.__dict__:
			index = opt.index 
		else:
			index = self.localindex
			self.localindex += 1
							
		#check if the index is inside the already allocated space -> if not allocate new space in memory
		while index >= self.memLen: 
			self.xval = concatenate((self.xval, zeros([self.n,len(x)])), axis=0)
			self.fval = concatenate((self.fval, zeros([self.n])), axis=0)
			self.valid = concatenate((self.valid, zeros([self.n])), axis=0)
			self.memLen += self.n
				
		#write data
		self.xval[index] = x
		self.fval[index] = f
		self.valid[index] += 1 
					
		return None

	def finalize(self): 
		"""
		Removes the space beyond the recorded iteration with highest iteration 
		number. This space was reserved for the last chunk, but the highest 
		recorded iteration may not be the one recorded at the end of the chunk. 
		
		Must be called after the iterative algorithm is stopped. 
		"""
		
		nonZeroIndex = self.valid.nonzero()
		lastIndex=nonZeroIndex[0].max()+1
		self.xval = self.xval[0:lastIndex][:]
		self.fval = self.fval[0:lastIndex]
		self.valid = self.valid[0:lastIndex]

	def reset(self):
		"""
		Clears the :attr:`xval`, :attr:`fval`, and :attr:`valid` members.
		"""
		
		self.xval = None
		self.fval = None
		self.valid = None
		self.memLen = self.n

"""
.. inheritance-diagram:: pyopus.design.cbd
    :parts: 1

**Corners-based design (PyOPUS subsystem name: CBD)**

Finds the circuit parameters for which the performances satisfy the 
requirements across all specified corners. 
""" 

from ..optimizer import optimizerClass
from ..optimizer.base import Reporter, Annotator
from ..misc.debug import DbgMsgOut, DbgMsg
from ..evaluator.performance import PerformanceEvaluator, updateAnalysisCount
from ..evaluator.aggregate import *
from ..evaluator.auxfunc import listParamDesc, paramDict, paramList, dictParamDesc
import numpy as np
import itertools
from copy import deepcopy

__all__ = [ 'generateCorners', 'CornerBasedDesign' ] 

class AnCountAnnotator(Annotator):
	"""
	Transfers analysis counts from remote tasks to local tasks. 
	"""
	def __init__(self, pe):
		self.pe=pe
	
	def produce(self):
		return self.pe.analysisCount
	
	def consume(self, annotation):
		self.pe.analysisCount=annotation
		
def generateCorners(paramSpec={}, modelSpec={}):
	"""
	Generates corners reflecting all combinations of parameter ranges 
	given by *paramSpec* and device models given by *modelSpec*. 
	
	*cornerSpec* is a dictionary with parameter name for key. Every 
	entry holds a list of possible parameter values. 
	
	*modelSpec* is a dictionary with model name for key. Every entry 
	holds a list of possible modules describing that device. 
	
	The corner name is constructed as c<number> with first corner 
	named c0. 
	
	Example::
	
	  corners=generateCorners(
	    paramSpec={
	      'vdd': [1.6, 1.8, 2.0], 
	      'temperature': [0.0, 25, 100.0]
	    }, 
	    modelSpec={
	      'mos': ['tm', 'wp', 'ws', 'wo', 'wz'], 
	      'bipolar': ['weak', 'strong']
	    }
	  )
	  
	  # Corners is a dictionary with c<number> as key and 
	  # corner description as value. The above statement
	  # results in 3*3*5*2=90 corners named c0-c89 representing 
	  # the cartesian product of possible parameter values for 
	  # vdd and temperature, and possible modules describing the 
	  # model of mos and bipolar. 
	  # These corners can be passed directly to a PerformanceEvaluator
	  # object. 
	"""
	modelNames=modelSpec.keys()
	paramNames=paramSpec.keys()
	
	lst=[]
	for name in modelNames:
		lst.append(modelSpec[name])
	for name in paramNames:
		lst.append(paramSpec[name])
	
	cl={}
	ii=0
	for c in itertools.product(*lst):
		c=list(c)
		corner={ 'params': {}, 'modules': [] }
		for name in modelNames:
			corner['modules'].append(c.pop(0))
		for name in paramNames:
			corner['params'][name]=c.pop(0)
		cl['c'+str(ii)]=corner
		ii+=1
	
	return cl

class optReporter(Reporter):
	def __init__(self, aggregator):
		Reporter.__init__(self)
		self.aggregator=aggregator
		
	def reset(self):
		Reporter.reset(self)
		
	def __call__(self, x, ft, opt):
		if opt.niter==opt.bestIter and not self.quiet:
			DbgMsgOut("CBD", "iter %d, f=%e" % (opt.niter, opt.f))
			DbgMsgOut("CBD", formatParameters(paramDict(x, self.aggregator.inputOrder)))
			DbgMsgOut("CBD", self.aggregator.formatResults())
		
class CornerBasedDesign(object):
	"""
	*paramSpec* is the design parameter specification dictionary 
	with ``lo`` and ``hi`` members specifying the lower and the 
	upper bounds. 
	
	See :class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	for details on *heads*, *analyses*, *measures*, *corners*, and 
	*variables*. 
	
	Fixed parameters are given by *fixedParams* - a dictionary 
	with parameter name for key and parameter value for value. 
	Alternatively the value can be a dictionary in which case the 
	``init`` member specifies the parameter value. 
	
	If *fixedParams* is a list the members of this list must be 
	dictionaries describing parameters. The set of fixed parameters 
	is obtained by merging the information from these dictionaries. 
	
	The performance constraints are specified as the ``lower`` and 
	the ``upper`` member of the measurement description disctionary. 
	
	*norms* is a dictionary specifying the norms for the 
	performance measures. Every norm is by default equal to the 
	specification (lower, upper). If the specification is zero, 
	the norm is 1.0. *norms* overrides this default. Norms are 
	used in the construction of the :class:`~pyopus.evaluator.aggregate.Aggregator` 
	object. 
	
	*failurePenalty* is the penalty assigned to a failed performance 
	measure. It is used in the construction of the 
	:class:`~pyopus.evaluator.aggregate.Aggregator` object used 
	by the optimization algorithm. 
	
	*tradeoffs* can be a number or a dictionary. It defines the 
	values of the tradeoff weights for the 
	:class:`~pyopus.evaluator.aggregate.Aggregator` object used by 
	the optimization algorithm. By default all tradeoff weights 
	are 0.0. A number specifies the same tradeoff weight for all 
	performance measures. To specify tradeoff weights for individual 
	performance measures, use a dictionay. If a performance measure 
	is not specified in the dictionary 0.0 is used. 
	
	Setting *stopWhenAllSatisfied* to ``True`` makes the optimizer 
	stop as soon as all design requirements (i.e. performance 
	constraints) are satisfied. Setting it to ``False`` makes the 
	algorithm stop when its stopping condition is satisfied. 
	When set to ``None`` the behavior depends on the value of 
	the *tradeoffs* parameter. If it is set to 0, the optimizer 
	stops when all design requirements are satisfied. Otherwise 
	the behavior is the same as for ``False``. 
	
	*initial* is the dictionary specifying the initial point for 
	the optimizer. If not given, the initial point is equal to 
	the mean of the lower and the upper bound. 
	
	*method* can be ``local`` or ``global`` and specifies the type 
	of optimization algorithm to use. Currently the local method 
	is QPMADS and the global method is PSADE. 
	
	If *forwardSolution* is ``True`` the solution of previous 
	pass is used as the initial point for the next pass. 
	
	*stepTol* is the step size at which the local optimizer is 
	stopped. 
	
	The difference between the upper and the lower bound an a 
	parameter is divided by *stepScaling* to produce the initial 
	step length for the local optimizer. 
	
	*evaluatorOptions* specifies the option overrides for the 
	:class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	object. 
	
	*aggregatorOptions* specifies the option overrides for the 
	:class:`~pyopus.evaluator.aggregate.Aggregator` object. 
	
	*optimizerOptions* specifies the option overrides for the 
	optimizer. 
	
	Setting *spawnerLevel* to a value not greater than 1 distributes 
	the full corner evaluation across available computing nodes. 
	It is also passed to the optimization algorithm, if the algorithm 
	supports this parameter. 
	
	This is a callable object with no arguments. The return value 
	is a tuple comprising a dictionary with the final values of the 
	design parameters, the :class:`~pyopus.evaluator.aggregate.Aggregator` 
	object used for evaluating the final result across all corners, 
	and the *analysisCount* dictionary. 
	
	Objects of this type store the number of analyses performed 
	during the last call to the object in the *analysisCount* 
	member. 
	"""
		
	def __init__(
		self, paramSpec, heads, analyses, measures, corners=None, 
		fixedParams={}, variables={}, 
		norms=None, failurePenalty=1e6, tradeoffs=0.0, 
		stopWhenAllSatisfied=None, 
		initial=None, method='local', forwardSolution=True, 
		stepTol=0.001, stepScaling=4.0, 
		evaluatorOptions={}, aggregatorOptions={}, optimizerOptions={}, debug=0, 
		spawnerLevel=1
	):
		self.heads=heads
		self.analyses=analyses
		self.measures=measures
		self.corners=corners
		self.variables=variables
		self.norms=norms
		self.failurePenalty=failurePenalty
		self.tradeoffs=tradeoffs
		self.stopWhenAllSatisfied=stopWhenAllSatisfied
		self.paramSpec=paramSpec
		self.initial=initial
		self.method=method
		self.forwardSolution=forwardSolution
		self.debug=debug
		self.evaluatorOptions=evaluatorOptions
		self.aggregatorOptions=aggregatorOptions
		self.optimizerOptions=optimizerOptions
		self.stepTol=stepTol
		self.stepScaling=stepScaling
		self.spawnerLevel=spawnerLevel
		
		# Process fixed parameters
		self.fixedParams={}
		if fixedParams is not None:
			if type(fixedParams) is list or type(fixedParams) is tuple:
				lst=fixedParams
			else:
				lst=[fixedParams]
			for entry in lst:
				nameList=entry.keys()
				if len(nameList)>0 and type(entry[nameList[0]]) is dict:
					# Extract init
					self.fixedParams.update(
						paramDict(
							listParamDesc(entry, nameList, 'init'), 
							nameList
						)
					)
				else:
					self.fixedParams.update(entry)
		
		# Parameter names and counts
		self.paramNames=self.paramSpec.keys()
		self.paramNames.sort()
		self.nParam=len(self.paramNames)
		
		# Parameter ranges
		self.paramLo=np.array(listParamDesc(self.paramSpec, self.paramNames, "lo"))
		self.paramHi=np.array(listParamDesc(self.paramSpec, self.paramNames, "hi"))
		
		# Initial values
		if initial is None:
			self.paramInit=(self.paramLo+self.paramHi)/2
		else:
			self.paramInit=np.array(paramList(initial, self.paramNames))
	
	def __call__(self):
		# Evaluator
		fullPe=PerformanceEvaluator(
			self.heads, self.analyses, self.measures, self.corners, 
			self.fixedParams, self.variables, 
			spawnerLevel=self.spawnerLevel, 
			**self.evaluatorOptions
		)
		
		# Aggregator
		aggDef=[]
		measureNames=self.measures.keys()
		measureNames.sort()
		for measureName in measureNames:
			measure=self.measures[measureName]
			if 'lower' in measure:
				if self.norms is not None and measureName in self.norms:
					# Use given norm
					norm=self.norms[measureName]
				else:
					# Use bound as norm, if zero use 1.0
					norm=np.abs(measure['lower']) if measure['lower']!=0.0 else 1.0
				if type(self.tradeoffs) is dict:
					# Dictionary of tradeoffs
					tradeoff=self.tradeoffs[measureName] if measureName in self.tradeoffs else 0.0
				else:
					# One tradeoff for all
					tradeoff=self.tradeoffs
				entry={
					'measure': measureName, 
					'norm': Nabove(measure['lower'], norm, self.failurePenalty), 
					'shape': Slinear2(1.0, tradeoff), 
					'reduce': Rworst()
				}
				aggDef.append(entry)
			if 'upper' in measure:
				if self.norms is not None and measureName in self.norms:
					# Use given norm
					norm=self.norms[measureName]
				else:
					# Use bound as norm, if zero use 1.0
					norm=np.abs(measure['upper']) if measure['upper']!=0.0 else 1.0
				if type(self.tradeoffs) is dict:
					# Dictionary of tradeoffs
					tradeoff=self.tradeoffs[measureName] if measureName in self.tradeoffs else 0.0
				else:
					# One tradeoff for all
					tradeoff=self.tradeoffs
				entry={
					'measure': measureName, 
					'norm': Nbelow(measure['upper'], norm, self.failurePenalty), 
					'shape': Slinear2(1.0, tradeoff), 
					'reduce': Rworst()
				}
				aggDef.append(entry)
		
		fullAgg=Aggregator(fullPe, aggDef, self.paramNames, *self.aggregatorOptions)
		
		# Initial parameters, step size
		atParam=self.paramInit.copy()
		lastStep=1.0
		
		# Prepare a copy of measures
		measures=deepcopy(self.measures)
		
		# Set up empty corner lists
		for measureName, measure in measures.iteritems():
			measure['corners']=[]
			
		# Set up analysis counters
		self.analysisCount={}
		
		atPass=1
		while True:
			# Evaluate all corners at current point
			if self.debug:
				DbgMsgOut("CBD", "Pass %d, full corner evaluation" % atPass)
			
			# Evaluate all corners
			fullAgg(atParam)
			
			# Clean up 
			fullPe.finalize()
			
			# Colect results
			res=fullAgg.results
			
			# Update analysis counts
			updateAnalysisCount(self.analysisCount, fullPe.analysisCount)
			
			if self.debug:
				DbgMsgOut("CBD", "Pass %d, corner summary" % atPass)
				DbgMsgOut("CBD", fullAgg.formatResults())
			
			# Get worst corners, update measures
			if self.debug:
				DbgMsgOut("CBD", "Pass %d, updating corner lists" % atPass)
			
			cornerAdded=False
			for ii in range(len(aggDef)):
				component=aggDef[ii]
				measureName=component['measure']
				measure=measures[measureName]
				# No tradeoffs, performance measure satisfies goal, and it has at least one corner listed
				if self.tradeoffs==0.0 and res[ii]['contribution']<=0.0 and len(measure['corners'])>0:
					# Skip this performance measure
					continue
				# Add a corner for this performance measure
				worstCorner=fullAgg.cornerList[res[ii]['worst_corner']]
				if 'corners' not in measure:
					measure['corners']=[]
				if worstCorner not in measure['corners']:
					measure['corners'].append(worstCorner)
					cornerAdded=True
				if self.debug:
					DbgMsgOut("CBD", "  %s: %s" % (measureName, str(measure['corners'])))
				# Add a corner to all dependencies
				if 'depends' in measure and measure['analysis'] is None:
					for depName in measure['depends']:
						dep=measures[depName]
						if 'corners' not in dep:
							dep['corners']=[]
						if worstCorner not in dep['corners']:
							dep['corners'].append(worstCorner)
							cornerAdded=True
						if self.debug:
							DbgMsgOut("CBD", "  dep %s: %s" % (depName, str(dep['corners'])))
			
			# If no corner added, we are done
			if not cornerAdded:
				if self.debug:
					DbgMsgOut("CBD", "Corners unchanged, stopping")
				break
			
			if self.debug:
				DbgMsgOut("CBD", "Pass %d, optimization" % atPass)
			
			# Construct new evaluator and aggregator
			pe=PerformanceEvaluator(
				self.heads, self.analyses, measures, self.corners, 
				self.fixedParams, self.variables, 
				**self.evaluatorOptions
			)
			agg=Aggregator(pe, aggDef, self.paramNames, *self.aggregatorOptions)
			
			# Prepare optimizer
			if self.method=='local':
				scale=(self.paramHi-self.paramLo)/self.stepScaling
				optimizerDefaults={
					#'stepBase': 8.0, 'meshBase': 32.0, 'initialMeshDensity': 32.0, 
					#'maxStep': 1.0, 'stopStep': 0.01, 
					'startStep': lastStep, 
					'stepBase': 4.0, 'meshBase': 16.0, 'initialMeshDensity': 2.0*20, # 16384.0,
					'maxStep': 1.0, 'stopStep': self.stepTol,
					'protoset': 0, # minimal=0, maximal=1
					'unifmat': 5, # 5 = nxn Sobol
					'generator': 2, # UniMADS
					'rounding': 0, 'modelOrdering': True, 'lastDirectionOrdering': True, 
					'roundOnFinestMeshEntry': True, 
					'speculativePollSearch': True, 'speculativeModelSearch': False, 
					'model': True, 
					'HinitialDiag': 0.0, 
					'boundSnap': True, 'boundSlide': True, 
					'qpFeasibilityRestoration': False, 
					'stepCutAffectsUpdate': True, 'speculativeStepAffectsUpdate': True, 
					'rho':16, 'rhoNeg': 1.0, 
					'linearUpdate': True, 'simplicalUpdate': True, 'powellUpdate': False, 
					'boundStepMirroring': False, 
					'linearUpdateExtend': False, 
					'forceRegression': True, 
					# 'debug': 2, 
					'maxiter': None, 'hmax': 100.0, 
					'cache': True, 
				}
				optimizerDefaults.update(self.optimizerOptions)
				opt=optimizerClass("QPMADS")(
					agg, self.paramLo, self.paramHi, 
					scaling=scale, 
					**optimizerDefaults
				)
			else:
				optimizerDefaults={
				}
				optimizerDefaults.update(self.optimizerOptions)
				opt=optimizerClass("ParallelSADE")(
					agg, self.paramLo, self.paramHi, 
					spawnerLevel=self.spawnerLevel, 
					**optimizerDefaults
				)
			
			# Install aggregator annotator (for parallel optimization algorithms)
			opt.installPlugin(agg.getAnnotator())
			
			# Install reporter 
			if self.debug:
				opt.installPlugin(optReporter(agg))
			
			# Install annotator for sending analysis count to the spawning host
			opt.installPlugin(AnCountAnnotator(pe))
			
			# Stop as soon as 0 is found if no tradeoffs are required
			if (
				(self.stopWhenAllSatisfied is None and self.tradeoffs==0.0) or 
				self.stopWhenAllSatisfied is True
			):
				opt.installPlugin(agg.getStopWhenAllSatisfied())
			
			# Set initial point
			if self.forwardSolution:
				opt.reset(atParam)
			else:
				opt.reset(self.paramInit)
			
			# Run
			opt.run()
			
			# Clean up 
			pe.finalize()
			
			# Collect result
			atParam=opt.x
			
			# Update analysis counts
			updateAnalysisCount(self.analysisCount, pe.analysisCount, opt.niter)
			
			atPass+=1
		
		# Convert parameters to dictionary
		self.atParam=paramDict(atParam, self.paramNames)
		self.passes=atPass-1
		
		if self.debug:
			DbgMsgOut("CBD", "Analysis count: %s" % str(self.analysisCount))
			DbgMsgOut("CBD", "Result:")
			DbgMsgOut("CBD", formatParameters(self.atParam))
		
		return (self.atParam, fullAgg, self.analysisCount)


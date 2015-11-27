"""
.. inheritance-diagram:: pyopus.design.yt
    :parts: 1

**Corners-based design (PyOPUS subsystem name: YT)**

Finds the circuit's design parameters for which the worst-case performances 
(within *beta* of the statistical parameters origin) satisfy the design 
requirements. *beta* specifies the target yield through the following 
equation. 

Y = 0.5 *beta* ( 1 + erf( *beta* / sqrt(2) ) ) 
""" 

from ..optimizer import optimizerClass
from ..optimizer.base import Reporter, Annotator
from ..misc.debug import DbgMsgOut, DbgMsg
from ..evaluator.performance import updateAnalysisCount
from ..evaluator.aggregate import formatParameters
from cbd import CornerBasedDesign
from wc import WorstCase
from ..evaluator.auxfunc import listParamDesc, paramDict, paramList, dictParamDesc
import numpy as np
import itertools
from copy import deepcopy

__all__ = [ 'YieldTargeting' ] 

		
class YieldTargeting(object):
	"""
	*paramSpec* is the design parameter specification dictionary 
	with ``lo`` and ``hi`` members specifying the lower and the 
	upper bound.
	
	*statParamSpec* is the statistical parameter specification 
	dictionary with ``lo`` and ``hi`` members specifying the lower 
	and the upper bound. 
	
	*opParamSpec* is the operating parameter specification 
	dictionary with ``lo`` and ``hi`` members specifying the lower 
	and the upper bound. The nominal value is specified by the 
	``init`` member. 
	
	See :class:`PerformanceEvaluator` for details on *heads*, 
	*analyses*, *measures*, *corners*, and *variables*. 
	
	Fixed parameters are given by *fixedParams* - a dictionary 
	with parameter name for key and parameter value for value. 
	Alternatively the value can be a dictionary in which case the 
	``init`` member specifies the parameter value. 
	
	If *fixedParams* is a list the members of this list must be 
	dictionaries describing parameters. The set of fixed parameters 
	is obtained by merging the information from these dictionaries. 
	
	*beta* is the spehere radius within which the worst case is 
	sought. It defines the target yield. 
	
	*wcSpecs* is the list of worst cases to compute in the form 
	of a list of (name, type) pairs where name is the performance 
	measure name an type is either ``upper`` or ``lower``. If a 
	specification is just a string it represents the performance 
	measure name. In that case the type of the specification is 
	obtained from the *measures* structure (the ``lower`` and the 
	``upper`` member of a performance measure description dictionary). 
	If *wcSpecs* is not specified the complete list of all 
	performance measures is used and the presence of the ``lower`` 
	and the ``upper`` member in the performance measure 
	description dictionary specifies the type of the worst case 
	that is considered in the process of yield targetting. 
	
	See :class:`~pyopus.design.cbd.CornerBasedDesign` for details on 
	*norms*, *failurePenalty*, *tradeoffs*, and *stopWhenAllSatisfied*. 
	
	*initial* is a dictionary of initial design parameter values. 
	If not aspecified the mean of the lower and the upper bound are 
	used. 
	
	If *initialNominalDesign* is ``True`` an initial design in the 
	nominal corner is performed using :class:`CornerBasedDesign` 
	class and the resulting design parameter values are used as 
	the initial point for yield targeting. *initial* is used as 
	the strating point for the nominal design. 
	
	If *forwardSolution* is ``True`` the solution of previous 
	pass is used as the initial point for the next pass. 
	
	*initialCbdOptions* is a dictionary of options passed to the 
	:class:`~pyopus.design.cbd.CornerBasedDesign` object at its 
	creation in initial nominal design. These options define the 
	behavior of the sizing across corners. If not given *cbdOptions* 
	are used. 
	
	*firstPassCbdOptions* is a dictionary of options passed to the 
	:class:`~pyopus.design.cbd.CornerBasedDesign` object at its 
	creation in the first pass. These options define the behavior 
	of the sizing across corners in the first pass. If not specified 
	*cbdOptions* are used. 
	
	*cbdOptions* is a dictionary of options passed to the 
	:class:`~pyopus.design.cbd.CornerBasedDesign` object at its 
	creation. These options define the behavior of the sizing across 
	corners. 
	
	*wcOptions* is a dictionary of options passed to the 
	:class:`~pyopus.design.wc.WorstCase` object. These options define 
	the behavior of the worst case analysis. 
	
	*cornerTol* is the relative tolerance for determining if two 
	corners are almost equal. The absolute tolerance for the norm of 
	the vector of statistical parameters is obtained by multiplying 
	this value with *beta*. The absolute tolerance for operating 
	parameters is obtained by multiplying the low-high range with 
	*cornerTol*. 
	
	*angleTol* is the angular tolerance in degrees for determining 
	if two sets of statistical parameters are almost equal. 
	
	*debug* turns on debugging. 
	
	Setting *spawnerLevel* to a value not greater than 1 distributes 
	the evaluations across available computing nodes. This argument 
	is forwarded to the :class:`~pyopus.design.cbd.CornerBasedDesign` 
	and the :class:`~pyopus.design.wc.WorstCase` objects. 
	
	This is a callable object with no arguments. The return value 
	is a tuple comprising a dictionary with the final values of the 
	design parameters, the :class:`~pyopus.evaluator.aggregate.Aggregator` 
	object used for evaluating the final result across all relevant 
	corners, the :class:`~pyopus.design.wc.WorstCase` object used for 
	computingthe final worst case performance, and the total dictionary 
	holding the number of all analyses performed. 		
	
	Objects of this type store the number of analyses performed 
	during the last call to the object in the :attr:`analysisCount` 
	member. The last :class:`~pyopus.evaluator.aggregate.Aggregator` 
	object and the last :class:`~pyopus.design.wc.WorstCase` object 
	are stored in the :attr:`aggregator` and the :attr:`wc` member. 
	The resulting set of design parameters is stored in the 
	:attr:`atParam` member. 
	"""
		
	def __init__(
		self, paramSpec, statParamSpec, opParamSpec, 
		heads, analyses, measures, corners=None, 
		fixedParams={}, variables={}, beta=3.0, wcSpecs=None, 
		norms=None, failurePenalty=1e6, tradeoffs=0.0, 
		stopWhenAllSatisfied=None, 
		initial=None, initialNominalDesign=True, 
		forwardSolution=True, 
		initialCbdOptions=None, 
		firstPassCbdOptions=None, 
		cbdOptions={}, 
		wcOptions={}, cornerTol=0.01, angleTol=10, 
		debug=0, spawnerLevel=1
	):
		self.heads=heads
		self.analyses=analyses
		self.measures=measures
		self.corners=corners
		self.variables=variables
		self.paramSpec=paramSpec
		self.statParamSpec=statParamSpec
		self.opParamSpec=opParamSpec
		self.initial=initial
		self.initialNominalDesign=initialNominalDesign
		self.forwardSolution=forwardSolution
		self.debug=debug
		self.initialCbdOptions=initialCbdOptions
		self.firstPassCbdOptions=firstPassCbdOptions
		self.cbdOptions=cbdOptions
		self.wcOptions=wcOptions
		
		self.norms=norms
		self.failurePenalty=failurePenalty
		self.tradeoffs=tradeoffs
		self.stopWhenAllSatisfied=stopWhenAllSatisfied
		
		self.initial=initial
		self.beta=beta
		if wcSpecs is not None:
			self.wcSpecs=wcSpecs
		else:
			self.wcSpecs=self.measures.keys()
			
		self.spawnerLevel=spawnerLevel
		self.cornerTol=cornerTol
		self.angleTol=angleTol
		
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
		
		self.opNames=self.opParamSpec.keys()
		self.opNames.sort()
		self.nOp=len(self.opNames)
		
		self.statNames=self.statParamSpec.keys()
		self.statNames.sort()
		self.nStat=len(self.statNames)
		
		# Parameter ranges
		self.paramLo=np.array(listParamDesc(self.paramSpec, self.paramNames, "lo"))
		self.paramHi=np.array(listParamDesc(self.paramSpec, self.paramNames, "hi"))
		
		self.opLo=np.array(listParamDesc(self.opParamSpec, self.opNames, "lo"))
		self.opHi=np.array(listParamDesc(self.opParamSpec, self.opNames, "hi"))
		
		# Initial values
		if initial is None:
			self.paramInit=(self.paramLo+self.paramHi)/2
		else:
			self.paramInit=np.array(paramList(initial, self.paramNames))
			
		# Nominal op values
		self.opNominal=np.array(listParamDesc(self.opParamSpec, self.opNames, "init"))
	
	def cornersClose(self, c1, c2):
		c1s=np.array(paramList(c1, self.statNames))
		c2s=np.array(paramList(c2, self.statNames))
		c1o=np.array(paramList(c1, self.opNames))
		c2o=np.array(paramList(c2, self.opNames))
		
		# Distance
		ds1=(c1s**2).sum()**0.5
		ds2=(c2s**2).sum()**0.5
		statdif=np.abs(ds1-ds2)
		
		# Angle
		if ds1==0.0 or ds2==0.0: 
			angle=0.0
		else:
			angle=np.arccos((c1s*c2s).sum()/(ds1*ds2))/np.pi*180
		
		# Op difference (relative)
		opdif=np.abs(c1o-c2o)/(self.opHi-self.opLo)
		
		if statdif<=self.beta*self.cornerTol and angle<=self.angleTol and (opdif<=self.cornerTol).all():
			return True
		else:
			return False
		
	def __call__(self):
		# Measures that have corners
		#   subject to wc - add new corners
		#   not subject to wc - do not add new corners
		# Measures without corners
		#   subject to wc - add new corners
		#   not subject to wc - use all corners
		
		self.atParam=None
		self.wcresult=None
		self.analysisCount=None
		self.aggregator=None
		analysisCount={}
		
		# Prepare wc specs if none specified
		if self.wcSpecs is None:
			wcSpecs=self.measures.keys()
		
		# Nominal corner
		p={}
		p.update(
			paramDict(self.opNominal, self.opNames)
		)
		p.update(
			paramDict(np.zeros(self.nStat), self.statNames)
		)
		nom_corner={
			'_nominal': {
				'params': p
			}
		}
		
		# Initial nominal design
		if self.initialNominalDesign:
			if self.debug:
				DbgMsgOut("YT", "Initial nominal design")
			
			# Construct corners
			if self.corners is not None:
				corners=deepcopy(self.corners)
			else:
				corners={}
			corners.update(nom_corner)
			
			# Construct measures - add nominal corner to all measures subject to wc
			measures=deepcopy(self.measures)
			for wcSpec in self.wcSpecs:
				if type(wcSpec) is tuple:
					wcName=wcSpec[0]
					if wcSpes[1] not in measures[wcName]:
						raise Exception(DbgMsg("YT", "Measure %s has no %s specification." % (wcName, wcSpec[1])))
				else:
					wcName=wcSpec
				if 'corners' not in measures[wcName]:
					measures[wcName]['corners']=[]
				measures[wcName]['corners'].append('_nominal')
			
			# Corner-based design 
			cbdOptions={
				'norms': self.norms, 
				'failurePenalty': self.failurePenalty, 
				'tradeoffs': self.tradeoffs, 
				'stopWhenAllSatisfied': self.stopWhenAllSatisfied, 
				'initial': self.initial
			}
			
			if self.initialCbdOptions is not None:
				cbdOptions.update(self.initialCbdOptions)
			else:
				cbdOptions.update(self.cbdOptions)
			
			cbd=CornerBasedDesign(
				self.paramSpec, self.heads, self.analyses, measures, corners, 
				self.fixedParams, self.variables, spawnerLevel=self.spawnerLevel, 
				**cbdOptions
			)
			initialDesignParams, aggregator, anCount = cbd()
			updateAnalysisCount(analysisCount, anCount)
			if self.debug:
				DbgMsgOut("YT", aggregator.formatResults())
				DbgMsgOut("YT", "Analysis count: %s" % str(anCount))
				DbgMsgOut("YT", "Result:")
				DbgMsgOut("YT", formatParameters(initialDesignParams))
			
		else:
			# No initial nominal design
			if self.initial is None:
				# Use the default mean of lo and hi
				initialDesignParams=paramDict((self.paramLo+self.paramHi)/2, self.paramNames)
			else:
				# Use specified initial design parameters
				initialDesignParams=self.initial
		
		designParams=initialDesignParams
		
		cornerLists={} # (name,type) for key, list of corners for value
		atPass=1
		while True:
			# Compute worst case
			if self.debug:
				DbgMsgOut("YT", "Computing worst case, pass %d" % (atPass))
			
			wc=WorstCase(
				self.heads, self.analyses, self.measures, 
				self.statParamSpec, self.opParamSpec, variables=self.variables, 
				fixedParams=designParams, 
				beta=self.beta, 
				spawnerLevel=self.spawnerLevel, 
				**self.wcOptions
			)
			wcresults, anCount = wc(self.wcSpecs)
			updateAnalysisCount(analysisCount, anCount)
			if self.debug:
				DbgMsgOut("YT", wc.formatResults())
				DbgMsgOut("YT", "Analysis count: %s" % str(anCount))
			
			# Update corner sets
			if self.debug:
				DbgMsgOut("YT", "Updating corner sets")
					
			cornersChanged=False
			newCorner=False
			for res in wcresults:
				wcName=res['name']
				wcType=res['type']
				key=(wcName, wcType)
				if key not in cornerLists:
					cornerLists[key]=[]
				
				# If worst case satisfies the requirement and we have at least one corner
				# for this design requirement, go to next result
				if (
					len(cornerLists[key])>0 and (
						(res['type']=='lower' and res['wc']>=self.measures[wcName]['lower']) or
						(res['type']=='upper' and res['wc']<=self.measures[wcName]['upper'])
					)
				):
					continue
				
				corner={}
				corner.update(res['op'])
				corner.update(res['stat'])
				
				# Replace first similar corner or add new corner
				replacedCorner=False
				newCornerList=[]
				for originalCorner in cornerLists[key]:
					if self.cornersClose(corner, originalCorner) and not replacedCorner:
						newCornerList.append(corner)
						replacedCorner=True
					else:
						newCornerList.append(originalCorner)
				
				# Add new corner
				if not replacedCorner: 
					newCornerList.append(corner)
					newCorner=True
				
				# Store new corner list
				cornerLists[key]=newCornerList
				
				if self.debug:
					if replacedCorner:
						DbgMsgOut("YT", "  Replaced corner for %s, %s; corner count = %d" % (wcName, wcType, len(cornerLists[key])))
					else:
						DbgMsgOut("YT", "  Added corner for %s, %s; corner count = %d" % (wcName, wcType, len(cornerLists[key])))
				
				# At this point the new corner either replaced an old one or was added
				cornersChanged=True
			
			# newCorner implies cornersChanged
			# not cornersChanged implies not newCorner
			
			# Stop if corners did not change
			if not cornersChanged:
				if self.debug:
					DbgMsgOut("YT", "Corners unchanged, stopping")
				break
			
			# Stop if no new corners found (only similar corners)
			if not newCorner:
				if self.debug:
					DbgMsgOut("YT", "No significantly different new corner found, stopping.")
				break
			
			# Construct corners and measures for cbd
			if self.corners is not None:
				corners=deepcopy(self.corners)
			else:
				corners={}
			measures=deepcopy(self.measures)
			
			for key, cornerList in cornerLists.iteritems():
				wcName, wcType = key
				ii=0
				for cornerParams in cornerList:
					typeStr='l' if wcType=='lower' else 'u'
					cornerName="_"+wcName+"_"+typeStr+str(ii)
					corners.update({cornerName: { 'params': cornerParams }})
					if 'corners' not in measures[wcName]:
						measures[wcName]['corners']=[]
					measures[wcName]['corners'].append(cornerName)
					# Append corner to dependencies
					if 'depends' in measures[wcName]:
						for dependentMeasure in measures[wcName]['depends']:
							if 'corners' not in measures[dependentMeasure]: 
								measures[dependentMeasure]['corners']=[]
							cornerSet=set(measures[dependentMeasure]['corners'])
							cornerSet.add(cornerName)
							measures[dependentMeasure]['corners']=list(cornerSet)
							
					ii+=1
				
			# Corner-based design
			if self.debug:
				DbgMsgOut("YT", "Sizing across corners, pass %d" % (atPass))
			
			cbdOptions={
				'norms': self.norms, 
				'failurePenalty': self.failurePenalty, 
				'tradeoffs': self.tradeoffs, 
				'stopWhenAllSatisfied': self.stopWhenAllSatisfied, 
			}
			
			# Copy the solution from last pass or use the initial solution
			if self.forwardSolution:
				cbdOptions['initial']=designParams
			else:
				cbdOptions['initial']=initialDesignParams
			
			if self.firstPassCbdOptions is not None and atPass==1:
				cbdOptions.update(self.firstPassCbdOptions)
			else:
				cbdOptions.update(self.cbdOptions)
				
			cbd=CornerBasedDesign(
				self.paramSpec, self.heads, self.analyses, measures, corners, 
				self.fixedParams, self.variables, 
				spawnerLevel=self.spawnerLevel, **cbdOptions
			)
			designParams, aggregator, anCount = cbd()
			updateAnalysisCount(analysisCount, anCount)
			if self.debug:
				DbgMsgOut("YT", aggregator.formatResults())
				DbgMsgOut("YT", "Analysis count: %s" % str(anCount))
				DbgMsgOut("YT", "Result:")
				DbgMsgOut("YT", formatParameters(designParams))
			
			atPass+=1
			
		# Convert parameters to dictionary
		self.atParam=designParams
		self.wc=wc
		self.analysisCount=analysisCount
		self.aggregator=aggregator
		self.passes=atPass-1
		
		if self.debug:
			DbgMsgOut("YT", "Total analysis count: %s" % str(self.analysisCount))
			DbgMsgOut("YT", "Passes: %d" % self.passes)
		
		return (self.atParam, self.aggregator, self.wc, self.analysisCount)


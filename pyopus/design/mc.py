"""
.. inheritance-diagram:: pyopus.design.mc
    :parts: 1

**Monte Carlo analysis (PyOPUS subsystem name: MC)**

Estimates the yield taking into account operating and statistical parameters. 
If at least one operating parameter is specified the yield of the worst 
performance across the statistical parameters is computed. 

The bounds on the performance are specified with the ``lower`` and the 
``upper`` entry of the measurement description dictionary. 

Statistical parameters are assumed to be independent with zero mean and 
variance one. 

Can also be used for estimating the worst case performance. 
""" 

from wc import WorstCase
from ..evaluator.performance import updateAnalysisCount
from ..evaluator.auxfunc import paramDict, paramList
from ..misc.debug import DbgMsgOut, DbgMsg
from ..parallel.cooperative import cOS
import numpy as np

__all__ = [ 'MonteCarlo' ] 

	
class MonteCarlo(object):
	"""
	See :class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	for details on *heads*, *analyses*, *measures*, and *variables*. 
	
	Statistical parameters and operating parameters are given by 
	dictionaries *statParamDesc* and *opParamDesc*. These 
	dictionaries have parameter name for key and parameter 
	property dictionary with *lo* and *hi* specifying the lower 
	and the upper bound. The nominal value is given by the 
	``init`` member. 
	
	Fixed parameters are given by *fixedParams* - a dictionary 
	with parameter name for key and parameter value for value. 
	Alternatively the value can be a dictionary in which case the 
	``init`` member specifies the parameter value. 
	
	If *fixedParams* is a list the members of this list must be 
	dictionaries describing parameters. The set of fixed parameters 
	is obtained by merging the information from these dictionaries. 
	
	Setting *debug* to a value greater than 0 turns on debug messages. 
	The verbosity is proportional to the specified number. 
	
	Setting *debug* to a value greater than 0 turns on debug messages. 
	The verbosity is proportional to the specified number. 
	
	*nSamples* is the number of Monte Carlo samples. 
	
	By setting *storeStatSamples* to ``True`` storing of the vector 
	of statistical parameters in the :attr:`statSamples` member is 
	enabled. The first index is the sample index, while the second 
	index is the parameter index. Parameters are ordered according 
	to the *statNames* list. 
	
	*storeOpParams* turns on storing of the worst operating 
	parameters in the results structure. 
	
	*storeWcEvals* turns on storing the number of performance 
	measure evaluations in the results structure. 
	
	*wcOptions* are the worst case analysis options passed to the 
	:class:`~pyopus.design.wc.WorstCase` object performing the 
	worst case analysis in the space of the operating parameters. 
	
	This is a callable object with at most one argument. If given 
	the argument is a list of entries. Every entry is 
	
	* a tuple of the form (name,type), where name is the measure name
	  and type is ``lower`` or ``upper``
	* a string specifying the measure name. In this case the type 
	  of the performance costraint for which the Monte-Carlo analysis 
	  is performed is  given by the presence of the ``lower`` and the 
	  ``upper`` entries in the performance measure's description. 
	  A separate yield is computed for ``lower`` and ``upper``. 
	
	If no argument is specified, all yields corresponding to 
	lower/upper bounds of all performance *measures* are computed. 
	
	Results are stored in a results dictionary with pairs of the 
	form (name,type) for key. Values are a dictionaries with the 
	following members:
	
	* ``samples`` - a vector of peformance measure values
	* ``op`` - 2-dimensional array with worst operating 
	  parameter values. The first index is the sample index. The 
	  second index is the parameter index. The parameters are 
	  ordered according to the *opNames* member of the object. 
	  This member is available if ``storeOpParams* is set to ``True``. 
	* ``evals`` - array with the number of performance measure 
	  evaluations corresponding to individual samples. Available 
	  if *storeWcEvals* is set to ``True``. 
	* ``feasible`` - number of feasible samples (samples satisfying 
	  the performance constraint). 
	* ``failed`` - number of failed samples for which the evaluation \
	  failed. 
	* ``yield`` - yield obtained by dividing teh number of feasible 
	  samples with *nSamples*
	
	The results are stored in the :attr:`results` member. The number of 
	analyses performed during the last call to the :class:`MonteCarlo` 
	object are stored in the :attr:`analysisCount` member. 
	
	A call to an object of this class returns a tuple holding 
	the results structure and the analysisCount dictionary. 
	
	The :attr:`totalCheck` member holds an array of booleans. 
	Every entry corresponds to one evaluated sample. ``True`` 
	means that a sample satisfies all performance constraints. 
	
	The :attr:`totalYield` member holds the total yield obtained 
	by the analysis (share of the samples that satisfy all 
	performance constraints). 
	
	The :attr:`statSamples` member is a 2-dimensional array holding the 
	stored statistical parameter samples. The first index is the 
	sample index while the second index is the statistical parameter 
	index. The ordering of statistical parameters is given by the 
	*statNames* member. This member is available if *storeStatSamples* 
	is set to ``True``. 
	
	Setting *spawnerLevel* to a value not bigger than 1 distributes 
	the evaluations across available computing nodes. 
	"""
	def __init__(
		self, heads, analyses, measures, 
		statParamDesc, opParamDesc, fixedParams={}, variables={}, debug=0, 
		nSamples=1000, storeStatSamples=False, storeOpParams=False, 
		storeWcEvals=False, wcOptions={}, 
		spawnerLevel=1
	):
		self.heads=heads
		self.analyses=analyses
		self.measures=measures
		self.variables=variables
		self.statParamDesc=statParamDesc
		self.opParamDesc=opParamDesc
		self.debug=debug
		self.nSamples=nSamples
		self.storeStatSamples=storeStatSamples
		self.storeOpParams=storeOpParams
		self.storeWcEvals=storeWcEvals
		self.wcOptions=wcOptions
		
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
		self.opNames=self.opParamDesc.keys()
		self.opNames.sort()
		self.nOp=len(self.opNames)
		self.statNames=self.statParamDesc.keys()
		self.statNames.sort()
		self.nStat=len(self.statNames)
		
		# Random generator
		self.gen=None
	
	def jobGenerator(self):
		gen=np.random.RandomState(0)
		for ii in range(self.nSamples):
			if self.debug:
				DbgMsgOut("MC", "Generating sample #%d" % ii)
			xs=gen.normal(size=self.nStat)
			yield (self.jobProcessor, [xs])
	
	def jobProcessor(self, atStatx):
		return self.compute(atStatx)
		
	def jobCollector(self, results, analysisCount, statSamples, cumulativeCheck):
		try:
			while True:
				index, job, retval = (yield)
				atStatx = job
				wcResult, anCount = retval
				if self.debug:
					DbgMsgOut("MC", "Sample #%d evaluated" % index)
				if self.storeStatSamples:
					statSamples[index,:]=atStatx[:]
				updateAnalysisCount(analysisCount, anCount)
				for res in wcResult:
					key=(res['name'], res['type'])
					
					# Construct performance measure vector if needed
					if key not in results:
						results[key]={
							'samples': np.zeros(self.nSamples)
						}
					
					# Store op parameter values
					if self.storeOpParams and self.nOp>0:
						if 'op' not in results[key]:
							results[key]['op']=np.zeros((self.nSamples, self.nOp))
						results[key]['op'][index][:]=np.array(paramList(res['op'], self.opNames))[:]
					
					# Store wc evaluations
					if self.storeWcEvals: 
						if 'evals' not in results[key]:
							results[key]['evals']=np.zeros(self.nSamples)
						results[key]['evals'][index]=res['evals']
						
					# Get WC
					results[key]['samples'][index]=res['wc'] if res['wc'] is not None else np.NaN
		except GeneratorExit:
			# Postprocessing
			for key,resDict in results.iteritems():
				mcName, mcType = key
				vec=resDict["samples"]
				
				# Test condition
				if mcType in self.measures[mcName]:
					if self.debug:
						DbgMsgOut("MC", "Postprocessing %s" % str(key))
					
					if mcType=="lower":
						check=(vec>=self.measures[mcName][mcType])
					else:
						check=(vec<=self.measures[mcName][mcType])
				else:
					# No condition specified, skip
					if self.debug:
						DbgMsgOut("MC", "Skipped postprocessing of %s" % str(key))
					continue
				
				# Treat NaN as failed
				nanCheck=np.isnan(check)
				check=((~nanCheck) & check)
				
				# Count
				nSuccess=check.sum()
				results[key]['feasible']=nSuccess
				results[key]['failed']=nanCheck.sum()
				
				# Yield
				results[key]['yield']=nSuccess*1.0/self.nSamples
				
				# Update cumulative check
				cumulativeCheck&=check
		
	def __call__(self, mcSpecs=None):
		self.mcSpecs=mcSpecs
		if self.mcSpecs is None:
			self.mcSpecs=self.measures.keys()
		
		# Prepare storage for statistical parameter samples
		if self.storeStatSamples:
			statSamples=np.zeros((self.nSamples, self.nStat))
		else:
			statSamples=None
		
		# Prepare storage for cumulative check
		cumulativeCheck=np.ones(self.nSamples, dtype='bool')
			
		self.results={}
		self.analysisCount={}
		results={}
		analysisCount={}
		
		cOS.dispatch(
			jobList=self.jobGenerator(), 
			collector=self.jobCollector(results, analysisCount, statSamples, cumulativeCheck), 
			remote=self.spawnerLevel<=1
		)
		
		# Total yield
		self.totalCheck=cumulativeCheck
		self.totalYield=cumulativeCheck.sum()*1.0/self.nSamples
		self.statSamples=statSamples
		self.results=results
		self.analysisCount=analysisCount
		
		if self.debug>1:
			DbgMsgOut("MC", "Analysis count: %s" % str(self.analysisCount))
			DbgMsgOut("MC", "Results:")
			DbgMsgOut("MC", self.formatResults())
		
		return self.results, self.analysisCount
	
	# Evaluate a single point
	def compute(self, atStatx):
		# Construct a dictionary of statistical parameters
		fixedParams=paramDict(atStatx, self.statNames)
		
		# Merge with fixed parameters
		fixedParams.update(self.fixedParams)
		
		# Prepare options for worst case analysis
		wcOptions={}
		wcOptions.update(self.wcOptions)
		
		# Worst case analysis across op parameters
		wc=WorstCase(
			self.heads, self.analyses, self.measures, 
			statParamDesc={}, opParamDesc=self.opParamDesc, 
			fixedParams=fixedParams, variables=self.variables, 
			**wcOptions
		)
		wc(self.mcSpecs)
		
		return wc.results, wc.analysisCount
	
	def formatResults(self, nMeasureName=10, nResult=14, nPrec=5, nSamplePrec=4):
		"""
		Formats the results as a string. 
		
		*nMeasureName* specifies the formatting width for the 
		performance measure name. 
		
		*nResult* and *nPrec* specify the formatting width and the 
		number of significant digits for the performance measure 
		values. 
		
		*nPrec* specifies the number of significant digits for the 
		results. 
		
		*nSamplePrec* specifies the formatting width for the sample 
		count.
		"""
		txt=""
		
		specs=[]
		for spec in self.mcSpecs:
			if type(spec) is tuple:
				specs.append(spec)
			else:
				if (spec, "upper") in self.results and "yield" in self.results[(spec,"upper")]:
					specs.append((spec,"upper"))
				if (spec, "lower") in self.results and "yield" in self.results[(spec,"lower")]:
					specs.append((spec,"lower"))
		for spec in specs:
			mcName, mcType = spec
			txt+="%*s" % (nMeasureName, mcName)
			txt+=" < " if mcType=="upper" else " > "
			txt+="%*.*e" % (nResult, nPrec, self.measures[mcName][mcType])
			txt+=" yield=%*.*e" % (nResult, nPrec, self.results[spec]["yield"])
			txt+=" = 1-%*.*e" % (nResult, nPrec, (1.0-self.results[spec]["yield"]))
			txt+=" feasible=%*d" % (nSamplePrec, self.results[spec]["feasible"])
			txt+=" failed=%*d" % (nSamplePrec, self.results[spec]["failed"])
			txt+="\n"
		
		txt+="\ntotal yield=%*.*e" % (nResult, nPrec, self.totalYield)
		txt+=" = 1-%*.*e" % (nResult, nPrec, (1.0-self.totalYield))
		txt+="\n"
		
		return txt
	
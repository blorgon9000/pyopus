"""
.. inheritance-diagram:: pyopus.design.wc
    :parts: 1

**Worst case analysis (PyOPUS subsystem name: WC)**

Computes the worst case performance of a circuit. Statistical parameters are 
assumed to be independent with zero mean and standard deviation equal to 1. 
""" 

from ..optimizer import optimizerClass
from ..optimizer.base import Plugin, Reporter
from ..misc.debug import DbgMsgOut, DbgMsg
from sensitivity import Sensitivity
from ..evaluator.performance import PerformanceEvaluator, updateAnalysisCount
from ..evaluator.aggregate import *
from ..evaluator.auxfunc import listParamDesc, paramDict, paramList, dictParamDesc
from ..parallel.cooperative import cOS
import numpy as np

__all__ = [ 'WorstCase' ] 


def fmtF(f, nResult=14, nPrec=5):
	if f is None:
		return "%*s" % (nResult, "-")
	else:
		return "%*.*e" % (nResult, nPrec, f)

class StatConstr(object):
	def __init__(self, n, beta, beta0=0.0):
		"""
		Constraint on statistical parameter distance (below beta). 
		
		First *n* parameters are statistical parameters. 
		
		*beta0* is the projected distance of statistical parameters 
		that are not being optimized. 
		"""
		self.n=n 
		self.beta=beta
		self.beta0=beta0
		
	def __call__(self, x):
		return np.array([(x[:self.n]**2).sum()+self.beta0**2-self.beta**2])

class StatConstrJac(object):
	def __init__(self, n, beta, beta0=0.0):
		"""
		Jacobian of statistical parameter constraint. 
		
		First *n* parameters are statistical parameters. 
		
		*beta0* is the projected distance of statistical parameters 
		that are not being optimized. 
		"""
		self.n=n 
		self.beta=beta
		self.beta0=beta0
	
	def __call__(self, x):
		J=2*x
		J[self.n:]=0
		J=J.reshape((1,x.size))
		return J
	
class optReporter(Reporter):
	def __init__(self, nsph, evaluator, wcName, lower, beta0=0.0):
		Reporter.__init__(self)
		self.nsph=nsph
		self.evaluator=evaluator
		self.wcName=wcName
		self.worst=None
		self.hworst=None
		self.lower=lower
		self.beta0=beta0
		
	def reset(self):
		Reporter.reset(self)
		self.worst=None
		self.hworst=None
		
	def __call__(self, x, ft, opt):
		if self.nsph>0:
			dist=((x[:self.nsph]**2).sum()+self.beta0**2)**0.5
		perf=self.evaluator.results[self.wcName]['default']
		if opt.niter==opt.bestIter:
			self.worst=perf
			if type(ft) is tuple:
				self.hworst=opt.aggregateConstraintViolation(ft[1])
		str="iter=%d perf=%s " % (opt.niter, fmtF(perf))
		if self.nsph>0:
			str+="dist=%e " % (dist) 
		if self.hworst is not None:
			str+="hworst=%e " % (self.hworst)
		str+="worst=%s " % (fmtF(self.worst))
		if 'step' in opt.__dict__:
			str+="%e " % (opt.step) 
		str+=self.wcName
		DbgMsgOut("WC", str)
		
class evalCollector(Plugin):
	"""
	Collects the values of the performance across iterations. 
	
	The values are stored in the ``history`` member. 
	"""
	def __init__(self, evaluator, wcName):
		Plugin.__init__(self)
		self.evaluator=evaluator
		self.wcName=wcName
		self.history=[]
	       
	def __call__(self, x, ft, opt):
		self.history.append(self.evaluator.results[self.wcName]['default'])
	
	def reset(self):
		self.history=[]
		
class wcStopper(Plugin):
	"""
	Stops the algorithm when the angle between -x and the gradient of f 
	becomes smaller than a threshold given by relative tolerance 
	*angleTol* and the constraint function is close enough to *beta* 
	within relative tolerance *ctol*. 
	
	*n* is the number of statistical parameters
	
	*beta0* is the projected distance of statistical parameters 
	that are not being optimized. 
	
	*beta* is the constraint distance. 
	"""
	def __init__(self, constrTol, angleTol, maxStep, name, n, beta, beta0=0.0, debug=0):
		Plugin.__init__(self)
		self.constrTol=constrTol
		self.angleTol=angleTol
		self.maxStep=maxStep
		self.debug=debug
		self.name=name
		self.n=n
		self.beta=beta
		self.beta0=beta0
		
	def reset(self):
		self.it=None
	
	def optimality(self, x, ft, opt):
		if (
			self.constrTol is not None and self.angleTol is not None and 
			opt.xgo is not None and opt.modg is not None and opt.modH is not None and self.n>0
		):
			cdelta=np.abs(np.array([(x[:self.n]**2).sum()+self.beta0**2])**0.5-self.beta)
			cabstol=self.beta*self.constrTol
			g=(opt.modg.reshape(x.size)+np.dot(opt.modH, (x-opt.xgo).reshape((x.size,1))).reshape(x.size))[:self.n]
			# Use best point
			xv=-x[:self.n]
			l1=(g**2).sum()**0.5
			l2=(xv**2).sum()**0.5
			if l1==0.0 or l2==0.0:
				angle=0.0
			else:
				angle=np.arccos((g*xv).sum()/l1/l2)*180/np.pi
			return cdelta, angle 
		else:
			return None, None
		
	def __call__(self, x, ft, opt):
		# Tolerance
		cabstol=self.beta*self.constrTol
		
		stop=False
		
		# Check best point
		#cdelta, angle = self.optimality(opt.x, opt.f, opt)
		#if cdelta is not None and opt.step<=self.maxStep and cdelta<cabstol and angle<self.angleTol:
			#stop=True
			#self.it=opt.bestIter
			#self.x=opt.x
		#if self.debug and cdelta is not None:
			#DbgMsgOut("STOP best", "h=%.8e angle=%f %s" % (cdelta, angle, self.name))
			#if stop:
				#DbgMsgOut("STOP", "Stopping %s at %d" % (self.name, opt.niter))
		#if stop:
			#opt.stop=opt.stop or stop
			#return None
			
		# Check current point
		#cdelta, angle = self.optimality(x, ft, opt)
		#if cdelta is not None and opt.step<=self.maxStep and cdelta<cabstol and angle<self.angleTol:
			#stop=True
			#self.it=opt.niter
			#self.x=x
		#if self.debug and cdelta is not None:
			#DbgMsgOut("STOP current", "h=%.8e angle=%f %s" % (cdelta, angle, self.name))
			#if stop:
				#DbgMsgOut("STOP", "Stopping %s at %d" % (self.name, opt.niter))
		
		# Check origin
		cdelta, angle = self.optimality(opt.xgo, opt.fo, opt)
		if cdelta is not None and opt.step<=self.maxStep and cdelta<cabstol and angle<self.angleTol:
			stop=True
			self.it=opt.ito
			self.x=opt.xgo
		if self.debug>1 and cdelta is not None:
			DbgMsgOut("STOP origin", "h=%.8e angle=%f %s" % (cdelta, angle, self.name))
			if stop:
				DbgMsgOut("STOP", "Stopping %s at %d" % (self.name, opt.niter))
				
		opt.stop=opt.stop or stop
		return None
	
class WorstCase(object):
	"""
	See :class:`PerformanceEvaluator` for details on *heads*, 
	*analyses*, *measures*, and *variables*. 
	
	Statistical parameters and operating parameters are given by 
	dictionaries *statParamDesc* and *opParamDesc*. These 
	dictionaries have parameter name for key and parameter 
	property dictionary with *lo* and *hi* specifying the lower 
	and the upper bound. The nominal value of the operating 
	parameters is given by the ``init`` member. 
	
	Fixed parameters are given by *fixedParams* - a dictionary 
	with parameter name for key and parameter value for value. 
	Alternatively the value can be a dictionary in which case the 
	``init`` member specifies the parameter value. 
	
	If *fixedParams* is a list the members of this list must be 
	dictionaries describing parameters. The set of fixed parameters 
	is obtained by merging the information from these dictionaries. 
	
	Setting *debug* to a value greater than 0 turns on debug messages. 
	The verbosity is proportional to the specified number. 
	
	*beta* is the maximal distance from the origin in the space 
	of statistical parameters. If it is a vector or a list the worst 
	cases corresponding to all values are computed. 
	
	If *linearWc* is set to ``True`` the initial point for 
	statistical parameters is chosen using linear worst case 
	analysis. Otherwise the initial point is at the origin. 
	
	If *alternating* is setto ``True`` the worst case is computed by 
	alternating optimizations in the operating and statistical 
	parameter space. 
	
	*maxPass* is the maximum number of algorithm passes (main 
	optimization runs). 
	
	*wcStepTol* specifies the step tolerance for the optimization 
	in the space of operating parameters. 
	
	*stepTol* is the step tolerance for stopping the main 
	optimization algorithm. 
	
	*constrTol* is the constraint violation tolerance for stopping 
	the algorithm. 
	
	*angleTol* is the gradient angle tolerance in degrees for 
	stopping the algorithm. 
	
	*stepScaling* is the divider of the lo-hi range for computing 
	the problem scaling. 
	
	*perturbationScaling* is the divider for the lo-hi range for 
	computing the perturbation used in sensitivity computation. 
	
	*screenThreshold* is the threshold for parameter screening. 
	
	*maximalStopperStep* is the maximal step for which the main 
	optimization stopper is invoked to check the alignment of 
	gradients. 
	
	*evaluatorOptions* specifies the option overrides for the 
	circuit evaluator. 
	
	*sensOptions* specifies the option overrides for the 
	sensitivity analysis. 
	
	*screenOptions* specifies the option overrides for the 
	screening. 
	
	*opOptimizerOptions* specifies the option overrides for the 
	optimizer used in the space of operating parameters. 
	
	*optimizerOptions* specifies the option overrides for the 
	main optimizer. 
	
	
	This is a callable object with an optional argument specifying 
	which worst cases to compute. If given, the argument must be a 
	list of entries. Every entry is 
	
	* a tuple of the form (``name``,``type``), where ``name`` is 
	  the measure name and ``type`` is ``lower`` or ``upper``
	* a string specifying the measure name. In this case the type 
	  of comuted worst case is given by the presence of the 
	  ``lower`` and the ``upper`` entries in the measure's 
	  description. 
	  
	If no argument is specified, all worst cases corresponding to 
	lower/upper bounds of all *measures* are computed. These bounds 
	are specified as the ``lower`` and the ``upper`` member of the 
	measurement description dictionary. 
	
	The results are stored in the :attr:`results` member. They are 
	represented as a list of dictionaries with the following members:
	
	* ``name`` - name of the performance measure
	* ``beta`` - targeted distance of statistical parameters
	* ``type`` - ``lower`` or ``upper``
	* ``passes`` - number of algorithm passes
	* ``evals`` - number of evaluations
	* ``status`` - ``OK`` or ``FAILED``
	* ``nominal`` - nominal performance
	* ``nominal_wcop`` - performance at nominal statistical 
	  parameters and initial worst case operating parameters
	* ``linwc`` - performance at the linearized worst case point
	* ``wc`` - worst case performance (at the worst case point)
	* ``op`` - operating parameter values at the worst case point
	* ``stat`` - statistica parameter values at the worst case point 
	* ``dist`` - distance of worst case point from the origin of
	  the statistical parameters
	
	Status FAILED means that the algorithm failed to converge in 
	*maxPass* passes. 
	 
	Objects of this type store the number of analyses performed 
	during the last call to the object in a dictionary stored in 
	the :attr:`analysisCount` member. 
	
	The return value of a call to an object of this class is a 
	tuple holding the results structure and the *analysisCount* 
	dictionary. 
	
	"""
	def __init__(
		self, heads, analyses, measures, 
		statParamDesc, opParamDesc, fixedParams={}, variables={}, debug=0, 
		beta=3.0, linearWc=True, alternating=True, 
		maxPass=20, wcStepTol=0.01, stepTol=0.01, constrTol=0.01, angleTol=15, 
		stepScaling=4, perturbationScaling=64, screenThreshold=0.01, maximalStopperStep=0.5, 
		evaluatorOptions={}, sensOptions={}, screenOptions={}, 
		opOptimizerOptions={}, optimizerOptions={}, 
		spawnerLevel=1
	):
		self.heads=heads
		self.analyses=analyses
		self.measures=measures
		self.statParamDesc=statParamDesc
		self.opParamDesc=opParamDesc
		self.debug=debug
		self.beta=beta
		self.linearWc=linearWc
		self.alternating=alternating
		self.evaluatorOptions=evaluatorOptions
		self.sensOptions=sensOptions
		self.screenOptions=screenOptions
		self.opOptimizerOptions=opOptimizerOptions
		self.optimizerOptions=optimizerOptions
		self.maxPass=maxPass
		self.wcStepTol=wcStepTol
		self.stepTol=stepTol
		self.constrTol=constrTol
		self.angleTol=angleTol
		self.stepScaling=stepScaling # 16
		self.perturbationScaling=perturbationScaling # 64 # 5 # 8 # 5
		self.screenThreshold=screenThreshold # 0.01 # 0.01 # 0.0005
		self.maximalStopperStep=maximalStopperStep # 0.5 # 0.1 # 0.1
		self.variables=variables
		
		self.spawnerLevel=spawnerLevel
		
		# Process fixed parameters
		self.fixedParams={}
		if fixedParams is not None and len(fixedParams)>0:
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
		
		# Parameter ranges
		self.opLo=np.array(listParamDesc(self.opParamDesc, self.opNames, "lo"))
		self.opHi=np.array(listParamDesc(self.opParamDesc, self.opNames, "hi"))
		self.statLo=np.array(listParamDesc(self.statParamDesc, self.statNames, "lo"))
		self.statHi=np.array(listParamDesc(self.statParamDesc, self.statNames, "hi"))
		
		# Nominal values
		self.opNom=np.array(listParamDesc(self.opParamDesc, self.opNames, "init"))
	
	def jobGenerator(self, wcSpecs=None):
		# Prepare jobs
		if wcSpecs is None:
			wcSpecs=self.measures.keys()
		
		isList=False
		try:
			nbeta=len(self.beta)
			isList=True
		except:
			nbeta=1
		ii=0
		for jj in range(nbeta):
			if not isList:
				jj=None
			for wcSpec in wcSpecs:
				# wcSpec is a tuple
				if type(wcSpec) is tuple:
					wcName, wcType = wcSpec
					yield (ii, jj, wcName, wcType)
					ii+=1
				else:
					# Not a tuple, do those wc specs that are given in measures
					wcName=wcSpec
					# A job is (wcIndex, betaIndex, name, isLower)
					if 'upper'in self.measures[wcName]:
						yield (self.jobProcessor, [ii, jj, wcName, "upper"])
						ii+=1
					if 'lower'in self.measures[wcName]:
						yield (self.jobProcessor, [ii, jj, wcName, "lower"])
						ii+=1
			if not isList:
				break
	
	def jobProcessor(self, index, betaIndex, wcName, wcType):
		if betaIndex is None:
			beta=self.beta
		else:
			beta=self.beta[betaIndex]
		return self.compute(beta, wcName, wcType=="lower")
		
	def jobCollector(self, results, analysisCount):
		# Collect results
		while True:
			index, job, retval = (yield)
			summary, anCount = retval
			updateAnalysisCount(analysisCount, anCount)
			if len(results)<=index:
				results.extend([None]*(index+1-len(results)))
			results[index]=summary
			if self.debug>1:
				DbgMsgOut("WC", self.formatResults(results, details=True))
			
	def __call__(self, wcSpecs=None):
		# Clean up results
		self.results=[]
		self.analysisCount={}
		
		# Initialize
		results=[]
		analysisCount={}
		
		cOS.dispatch(
			jobList=self.jobGenerator(wcSpecs), 
			collector=self.jobCollector(results, analysisCount), 
			remote=self.spawnerLevel<=1
		)
		
		# Store results
		self.results=results
		self.analysisCount=analysisCount
		
		if self.debug>1:
			DbgMsgOut("WC", "Analysis count: %s" % str(analysisCount))
			DbgMsgOut("WC", "Results:")
			DbgMsgOut("WC", self.formatResults(details=True))
		
		return self.results, self.analysisCount
	
	# Worst case across op parameters
	def opWorstCase(self, ev, wcName, startStep=1.0, atStatx=None, lower=True):
		# Reset analysis counter
		analysisCount={}
		
		# Initial point for operating parameters
		fixedParams={}
		
		# Prepare list of fixed parameters
		fixedParams.update(self.fixedParams)
		
		# Add statistical parameters if given
		if atStatx is not None:
			fixedParams.update(
				paramDict(atStatx, self.statNames)
			)
		else:
			# Set statistical parameters to 0 if not given
			fixedParams.update(
				paramDict(np.zeros(self.nStat), self.statNames)
			)
		
		# Prepare function
		ev.setParameters(fixedParams)
		
		fun=Aggregator(
			ev, [{
				'measure': wcName, 
				'norm': Nabove(0.0, 1.0, 1e6) if lower else Nbelow(0.0, 1.0, 1e6), 
				'shape': Slinear2(-1.0,-1.0), 
				'reduce': Rworst()
			}], self.opNames
		)
			
		# No op starting point, do corners first
		
		# Evaluate extremes
		if self.debug:
			DbgMsgOut("WC", "  Evaluating extreme op variations and nominal point, %s, %s" % (wcName, ('lower' if lower else 'upper')))
		
		pset=np.zeros((2*self.nOp+1,self.nOp))
		for ii in range(self.nOp):
			# Positive variation
			pset[1+2*ii][:]=self.opNom
			pset[1+2*ii][ii]=self.opHi[ii]
			# Negative variation
			pset[2+2*ii][:]=self.opNom
			pset[2+2*ii][ii]=self.opLo[ii]
		
		# Add nominal
		pset[0][:]=self.opNom
			
		# Evaluate (TODO: do this in parallel)
		history=np.zeros(pset.shape[0])
		for ii in range(pset.shape[0]):
			res, anCount = ev(paramDict(pset[ii][:], self.opNames))
			updateAnalysisCount(analysisCount, anCount)
			history[ii]=res[wcName]['default']
			if self.debug>1:
				DbgMsgOut("WC", "%d: %s=%e" % (ii+1, wcName, history[ii]))
		
		nevals=pset.shape[0]
		
		# Extract nominal 
		nomPerf=history[0]
		result={'nominal': nomPerf}
		
		# Are there any op parameters
		if self.nOp==0:
			# No op parameters, stop
			if self.debug:
				DbgMsgOut("WC", "  No op parameters.")
			
			result['nominal_wcop']=nomPerf
			return result, np.array([]), nevals
		
		# Extract norm
		nmax=np.array(history).max()
		nmin=np.array(history).min()
		norm=nmax-nmin
		
		if norm==0.0:
			norm=1.0
		# print "norm", norm
		
		# Construct worst corner (starting point for op param optimization)
		if self.debug:
			DbgMsgOut("WC", "  Constructing worst op parameters, %s" % wcName)
			
		worstOpx=self.opNom.copy()
		for ii in range(self.nOp):
			pos=history[1+2*ii]
			neg=history[2+2*ii]
			if lower:
				if pos<nomPerf and pos<neg:
					worstOpx[ii]=self.opHi[ii]
				elif neg<nomPerf and neg<pos:
					worstOpx[ii]=self.opLo[ii]
			else:
				if pos>nomPerf and pos>neg:
					worstOpx[ii]=self.opHi[ii]
				elif neg>nomPerf and neg>pos:
					worstOpx[ii]=self.opLo[ii]
		
		# print "initial", worstOpx
		
		# Confirm by optimization
		if self.debug:
			DbgMsgOut("WC", "  Confirming worst op parameters, %s" % wcName)
			
		# Prepare function
		ev.setParameters(fixedParams)
		fun=Aggregator(
			ev, [{
				'measure': wcName, 
				'norm': Nabove(nomPerf, norm, 1e6) if lower else Nbelow(nomPerf, norm, 1e6), 
				'shape': Slinear2(-1.0,-1.0), 
				'reduce': Rworst()
			}], self.opNames
		)
		
		optimizerDefaults={
			# 'stepBase': 2, 'meshBase': 16, 'initialMeshDensity': 8.0,
			## 'stepBase': 2, 'meshBase': 4, 'initialMeshDensity': 32, 
			## 'maxStep': 1, 'stopStep': 0.1,
			'startStep': startStep, 
			'stepBase': 4, 'meshBase': 16, 'initialMeshDensity': 2.0**20, # 16384, 
			'maxStep': 1.0, 'stopStep': self.wcStepTol, 
			'protoset': 0, # minimal=0, maximal=1
			'unifmat': 5, # 5=nxn Sobol
			'generator': 2, # UniMADS
			'rounding': 0, 'modelOrdering': True, 
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
			'maxiter': None, # 'hmax': 100.0, 
			'cache': True, 
			# 'sufficientDescent': sd
		}
		optimizerDefaults.update(self.opOptimizerOptions)
		opt=optimizerClass("QPMADS")(
			fun, self.opLo, self.opHi, 
			scaling=(self.opHi-self.opLo)/self.stepScaling, debug=0,  # 4
			**optimizerDefaults
		)
		collector=evalCollector(ev, wcName)
		if self.debug>1:
			opt.installPlugin(optReporter(0, ev, wcName, lower))
		opt.installPlugin(collector)
		opt.reset(worstOpx)
		opt.run()
		nevals+=opt.niter
		updateAnalysisCount(analysisCount, ev.analysisCount, opt.niter)
		
		# Extract performance at worst op parameters
		result['nominal_wcop']=collector.history[opt.bestIter-1]
		
		return result, opt.x, analysisCount, nevals
	
	# Sensitivity computation
	def sensitivityAndScreening(self, ev, atStatx, atOpx, wcName, lower, skipOp=False):
		ev.setParameters({})
		
		sensNames=self.statNames
		if not skipOp:
			sensNames=sensNames+self.opNames
		
		absPerturb=np.array((self.statHi-self.statLo)/self.perturbationScaling)
		if not skipOp:
			absPerturb=np.hstack((
				absPerturb, 
				(self.opHi-self.opLo)/self.perturbationScaling 
			))
			
		sensDefaults={
			"diffType": 1, 
			'absPerturb': absPerturb
		}
		sensDefaults.update(self.sensOptions)
		
		# Set init to atOpx and atStatx
		opParamDesc=dictParamDesc(
			{
				"lo": self.opLo, 
				"hi": self.opHi, 
				"initial": atOpx
			}, 
			self.opNames
		)
		statParamDesc=dictParamDesc(
			{
				"lo": self.statLo, 
				"hi": self.statHi, 
				"initial": atStatx
			}, 
			self.statNames
		)
		
		if not skipOp:
			ev.setParameters(self.fixedParams)
			sens=Sensitivity(
				ev, [statParamDesc, opParamDesc], sensNames, 
				**sensDefaults
			)
			propDiff,delta,anCount=sens(np.hstack((atStatx, atOpx)))
		else:
			ev.setParameters([self.fixedParams, paramDict(atOpx, self.opNames)])
			sens=Sensitivity(
				ev, [statParamDesc], sensNames, 
				**sensDefaults
			)
			propDiff,delta,anCount=sens(atStatx)
			
		propDiff=propDiff[wcName]['default']
		
		if self.debug:
			DbgMsgOut("WC", "  Screening, %s" % wcName)
		
		screenDefaults={
			'contribThreshold': self.screenThreshold # 0.01
		}
		screenDefaults.update(self.screenOptions)
		ilist=sens.screen(**screenDefaults)
		inNdx=ilist[wcName]['default']["in"]
		outNdx=ilist[wcName]['default']["out"]	
		
		inNdxStat=list(set(inNdx).intersection(range(self.nStat)))
		inNdxStat.sort()
		inNdxOp=list(np.array(list(set(inNdx).intersection(range(self.nStat,self.nStat+self.nOp))))-self.nStat)
		inNdxOp.sort()
		
		outNdxStat=list(set(outNdx).intersection(range(self.nStat)))
		outNdxStat.sort()
		outNdxOp=list(np.array(list(set(outNdx).intersection(range(self.nStat,self.nStat+self.nOp))))-self.nStat)
		outNdxOp.sort()
		
		if self.debug>1:
			DbgMsgOut("WC", "  Keeping %d parameters" % len(inNdx))
			for ii in inNdx:
				DbgMsgOut("WC", "    "+sensNames[ii])
		
		return propDiff, delta, sens.analysisCount, sens.neval, inNdxStat, inNdxOp
	
	def wcOptimization(self, atStatx, atOpx, lastStep, inNdxStat, inNdxOp, outNdxStat, outNdxOp, ev, beta, beta0, nomPerf, norm, wcName, lower=True):
		# Prepare fixed parameters
		fixedParams={}
		# Copy fixed parameters
		fixedParams.update(self.fixedParams)
		# Screened out statistical parameters initial point
		fixedParams.update(
			paramDict(atStatx, self.statNames, outNdxStat)
		)
		# Screened out op parameters initial point
		fixedParams.update(
			paramDict(atOpx, self.opNames, outNdxOp)
		)
		
		# Prepare optimization parameter names, bounds
		screenedStatNames=[]
		screenedOpNames=[]
		nScrStat=len(inNdxStat)
		for ii in inNdxStat:
			screenedStatNames.append(self.statNames[ii])
		for ii in inNdxOp:
			screenedOpNames.append(self.opNames[ii])
		paramNames=screenedStatNames+screenedOpNames
		
		# Prepare bounds
		paramLo=np.hstack((self.statLo[inNdxStat], self.opLo[inNdxOp]))
		paramHi=np.hstack((self.statHi[inNdxStat], self.opHi[inNdxOp]))
		
		# Join worst statistical and worst op parameters as initial point
		paramIni=np.hstack((
			atStatx[inNdxStat], atOpx[inNdxOp]
		))
		
		# Param scaling for op parameters
		scale=(paramHi-paramLo)/self.stepScaling
		
		# Function
		ev.setParameters(fixedParams)
		fun=Aggregator(
			ev, [{
				'measure': wcName, 
				'norm': Nabove(nomPerf, norm, 1e6) if lower else Nbelow(nomPerf, norm, 1e6), 
				'shape': Slinear2(-1.0,-1.0), 
				'reduce': Rworst()
			}], paramNames
		)
		
		# Constraint
		con=StatConstr(nScrStat, beta, beta0)
		
		# Constraint Jacobian
		cJ=StatConstrJac(nScrStat, beta, beta0)
		
		# Prepare optimizer
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
			fun, paramLo, paramHi, 
			constraints=con, clo=np.array([-np.inf]), chi=np.array([0.0]), cJ=cJ, scaling=scale, 
			**optimizerDefaults
		)
		collector=evalCollector(ev, wcName)
		
		# Install plugins
		if self.debug>1:
			opt.installPlugin(optReporter(nScrStat, ev, wcName, lower, beta0))
		opt.installPlugin(collector)
		
		stopper=wcStopper(
			constrTol=self.constrTol, angleTol=self.angleTol, maxStep=self.maximalStopperStep, 
			name=wcName, n=nScrStat, beta=beta, beta0=beta0, debug=self.debug
		)
		opt.installPlugin(stopper)
		
		opt.reset(paramIni)
		opt.run()
		
		# Use stopper's iteration
		if stopper is not None and stopper.it is not None:
			it=stopper.it
			x=stopper.x
		else:
			it=opt.bestIter
			x=opt.x
			
		# Extract statistical parameters
		solStat=paramDict(x[:nScrStat], screenedStatNames)
		solStat.update(paramDict(atStatx, self.statNames, outNdxStat))
		atStatx=np.array(paramList(solStat, self.statNames))
		
		# Extract operating parameters
		solOp=paramDict(x[nScrStat:], screenedOpNames)
		solOp.update(paramDict(atOpx, self.opNames, outNdxOp))
		atOpx=np.array(paramList(solOp, self.opNames))
		
		# Extract worst case 
		worstPerf=collector.history[it-1]
		linPerf=collector.history[0]
		analysisCount={}
		nevals=opt.niter
		updateAnalysisCount(analysisCount, ev.analysisCount, opt.niter)
		
		result={
			'wc': worstPerf, 
			'stat': solStat, 
			'op': solOp, 
			'dist': (atStatx**2).sum()**0.5, 
		}
		if self.linearWc:
			result['linwc']=linPerf
		
		return result, atStatx, atOpx, opt.step, analysisCount, nevals
	
	def compute(self, beta, wcName, lower=True):
		# Reset analysis counter
		analysisCount={}
		
		# Blank result structure
		result={
			'name': wcName, 
			'beta': beta, 
			'type': 'lower' if lower else 'upper', 
			'passes': 0, 
			'evals': 0, 
			'status': None, 
			'nominal': None, 
			'nominal_wcop': None, 
			'linwc': None, 
			'wc': None, 
			'op': None, 
			'stat': None, 
			'dist': None
		}
		
		if self.debug:
			DbgMsgOut("WC", "Running %s, %s at beta=%e" % (wcName, result['type'], beta))
		
		# Construct evaluator
		ev=PerformanceEvaluator(
			self.heads, self.analyses, self.measures, variables=self.variables, 
			activeMeasures=[wcName], debug=0, spawnerLevel=self.spawnerLevel-1
		)
		
		# Intial statistical parameters
		atStatx=np.zeros(self.nStat)
		
		# Evaluations counter
		nevals=[]
		
		# Worst op parameters
		#result1, atOpx, anCount, nev = cOS.dispatchSingle(
		#	self.opWorstCase, args=[ev, wcName], kwargs={'lower': lower}, remote=self.spawnerLevel<=2
		#)
		result1, atOpx, anCount, nev = self.opWorstCase(ev, wcName, lower=lower)
		updateAnalysisCount(analysisCount, anCount)
		nevals.append(nev)
		result.update(result1)
		
		# Initial worst performance
		atWorstCase=result['nominal_wcop']
		# print "initial", atWorstCase, atOpx
		
		atPass=0
		inNdxStat=[]
		inNdxOp=[]
		lastStep=0.25
		while True:
			# Assume op parameters did not change
			opChanged=False
			
			# Confirm op parameters
			if self.alternating and atPass>0:
				# print "before", atWorstCase, atOpx
			
				# Start with small step
				#result1, newAtOpx, anCount, nev = cOS.dispatchSingle(
				#	self.opWorstCase, 
				#	args=[ev, wcName], 
				#	kwargs={'startStep': 1.0/4**2, 'atStatx': atStatx, 'lower': lower}, 
				#	remote=self.spawnerLevel<=2
				#)
				result1, newAtOpx, anCount, nev = self.opWorstCase(
					ev, wcName, startStep=1.0/4**2, atStatx=atStatx, lower=lower
				)
				updateAnalysisCount(analysisCount, anCount)
				nevals.append(nev)
				# print "after", atWorstCase, atOpx
				
				# Get new worst case
				newWorstCase=result1['nominal_wcop']
				
				if (
						((lower and newWorstCase<atWorstCase) or (not lower and newWorstCase>atWorstCase)) and 
						(np.abs(newWorstCase-atWorstCase)/np.abs(result['nominal']-atWorstCase)>=self.constrTol)
				): 
					opChanged=True
					atOpx=newAtOpx
					atWorstCase=newWorstCase
				
				if self.debug:
					# print atOpx
					# print newAtOpx
					if opChanged:
						DbgMsgOut("WC", "  OP parameters changed")
					else:
						DbgMsgOut("WC", "  OP parameters unchanged")
			
			# Do we have stat parameters
			if self.nStat==0:
				# No stat parameters, stop
				if self.debug:
					DbgMsgOut("WC", "  No stat parameters.")
			
				atPass+=1
				result['status']="OK"
				result['stat']=np.array([])
				result['op']=paramDict(atOpx, self.opNames)
				result['wc']=result['nominal_wcop']
				break
				
			# Sensitivity and screening
			if self.debug:
				DbgMsgOut("WC", "  Computing sensitivity, %s, pass %d" % (wcName, atPass+1))
			
			# Skip op sensitivity when alternating
			# No need to spawn this one, it spawns its own subtasks
			propDiff, delta, anCount, nev, inStat, inOp = self.sensitivityAndScreening(ev, atStatx, atOpx, wcName, lower, skipOp=self.alternating)
			updateAnalysisCount(analysisCount, anCount)
			nevals.append(nev)
			screenChanged=not (set(inStat).issubset(set(inNdxStat)) and set(inOp).issubset(set(inNdxOp)))
			
			# No parameters left after screening in first pass, stop
			if len(inStat)+len(inOp)==0 and atPass==0:
				# No stat parameters, stop
				if self.debug:
					DbgMsgOut("WC", "  No parameters left after initial screening, stopping.")
			
				atPass+=1
				result['status']="OK"
				result['stat']=paramDict(np.zeros(self.nStat), self.statNames)
				result['op']=paramDict(atOpx, self.opNames)
				result['wc']=result['nominal_wcop']
				result['dist']=0.0
				break
			
			# Is the set of screened parameters unchanged
			if not screenChanged and not opChanged:
				if self.debug:
					DbgMsgOut("WC", "  Stopping")
				result['status']="OK"
				break
						
			# New parameters set - accumulate
			newNdxStat=set(inNdxStat).union(inStat)
			newNdxOp=set(inNdxOp).union(inOp)
			
			# Report
			if self.debug>1:
				DbgMsgOut("WC", "  Parameter set (%d)" % (len(newNdxStat)+len(newNdxOp)))
				for ii in newNdxStat:
					flag=" "
					if ii in inStat and ii not in inNdxStat:
						flag="*"
					DbgMsgOut("WC", "    "+flag+self.statNames[ii])
				for ii in newNdxOp:
					flag=" "
					if ii in inOp and ii not in inNdxOp:
						flag="*"
					DbgMsgOut("WC", "    "+flag+self.opNames[ii])
				
			# Update
			inNdxStat=newNdxStat
			inNdxOp=newNdxOp
			
			# Complement
			outNdxStat=list(set(range(self.nStat)).difference(inNdxStat))
			outNdxOp=list(set(range(self.nOp)).difference(inNdxOp))
			
			# Make it a list and sort it
			inNdxStat=list(inNdxStat)
			inNdxOp=list(inNdxOp)
			inNdxStat.sort()
			inNdxOp.sort()
			
			# Compute the norm of the performance measure 
			norm=np.abs(beta*((propDiff/delta)[:self.nStat]**2).sum()**0.5)
			if norm==0.0:
				norm=1.0
			
			# Compute linear worst case for statistical parameters, use it as initial point in first pass
			if atPass==0 and self.linearWc and len(inNdxStat)>0:
			# if self.linearWc and len(inNdxStat)>0: 
				# Sensitivities to statistical parameters
				sens=(propDiff/delta)
				ssens=sens[:self.nStat]
				# beta0
				# beta0=(np.array(atStatx)[outNdxStat]**2).sum()**0.5
				# Consider only screened parameters
				ssens[outNdxStat]=0.0
				# Compute linear worst case of statistical parameters
				# Multiply by 1-1e-7 to make sure initial point is feasible
				# atStatxLin=(1.0-1e-7)*((beta**2-beta0**2)**0.5)*ssens/(ssens**2).sum()**0.5
				ssensNorm=(ssens**2).sum()**0.5
				if ssensNorm!=0.0:
					atStatxLin=(1.0-1e-7)*beta*ssens/(ssens**2).sum()**0.5 
				else:
					atStatxLin=np.zeros(self.nStat)
				if lower:
					atStatxLin=-atStatxLin
				# atStatx[inNdxStat]=atStatxLin[inNdxStat]
				atStatx=atStatxLin
				
			# Compute beta0 corresponding to screened out statistical parameters
			beta0=(np.array(atStatx)[outNdxStat]**2).sum()**0.5
			
			# Main optimization
			if self.debug:
				DbgMsgOut("WC", "  Optimization, %s, pass %d" % (wcName, atPass+1))
			
			#result1, atStatx, atOpx, lastStep, anCount, nev = cOS.dispatchSingle(
			#	self.wcOptimization, 
			#	args=[atStatx, atOpx, lastStep*4, inNdxStat, inNdxOp, outNdxStat, outNdxOp, ev, beta, beta0, result['nominal'], norm, wcName, lower], 
			#	remote=self.spawnerLevel<=2
			#)
			result1, atStatx, atOpx, lastStep, anCount, nev = self.wcOptimization(
				atStatx, atOpx, lastStep*4, inNdxStat, inNdxOp, outNdxStat, outNdxOp, ev, beta, beta0, result['nominal'], norm, wcName, lower
			)
			updateAnalysisCount(analysisCount, anCount)
			nevals.append(nev)
			
			# Get new worst case
			atWorstCase=result1['wc']
			
			# Ignore linear wc result for all but the first pass
			if atPass>0 or not self.linearWc:
				del result1['linwc']
			result.update(result1)
		
			# Increase pass count
			atPass+=1
			
			if atPass>=self.maxPass:
				if self.debug:
					DbgMsgOut("WC", "  Maximum number of passes reached, stopping (%s)" % wcName)
				result['status']="FAILED"
				break
			
			# print wcName, atPass, ":", nevals, " ..", np.array(nevals).sum()
			
		result['passes']=atPass
		result['evals']=np.array(nevals).sum()
		# print nevals
		
		return result, analysisCount
	
	def formatResults(self, results=None, nMeasureName=10, nResult=14, nPrec=5, nEvalPrec=4, details=False):
		"""
		Formats the results as a string. 
		
		*results* is the results structure. If not given the 
		:attr:`results` member is used. 
		
		*nMeasureName* specifies the formatting width for the 
		performance measure name. 
		
		*nResult* and *nPrec* specify the formatting width and the 
		number of significant digits for the performance measure 
		values. 
		
		*nEvalPrec* is the number of spaces reserved for the 
		evaluations count. 
		
		If *details* is ``True`` the nominal performance and the 
		performace at the linear worst case are also formatted. 
		"""
		if results is None:
			results=self.results
		txt=""
		first=True
		for res in results:
			if res is None:
				continue
			if not first:
				txt+="\n"
			else:
				first=False
			measure=self.measures[res['name']]
			txt+="%*s" % (nMeasureName, res['name'])
			txt+=" < " if res['type']=='upper' else " > "
			txt+=fmtF(res["wc"], nResult, nPrec)
			if res['type'] in measure:
				if (
					(res['type']=='lower' and res['wc']>=measure['lower']) or
					(res['type']=='upper' and res['wc']<=measure['upper']) 
				):
					txt+="   "
				else:
					txt+=" o "
				txt+="goal=%s" % fmtF(measure[res['type']])
			else:
				txt+=" ? goal=%s"  % fmtF(None)
			if details:
				txt+=" nom="+fmtF(res["nominal"], nResult, nPrec)
				txt+=" lin="+fmtF(res["linwc"], nResult, nPrec)
			txt+=" d="+fmtF(res["dist"], nResult, nPrec)
			txt+=" neval=%*d" % (nEvalPrec, res['evals'])
			txt+=" %d" % res['passes']
			txt+=" "+res['status']
		
		return txt

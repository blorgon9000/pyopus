"""
.. inheritance-diagram:: pyopus.design.wcd
    :parts: 1

**Worst case distance analysis (PyOPUS subsystem name: WCD)**

Computes the worst case distances of the circuit's performances. 
Statistical parameters are assumed to be independent with 
zero mean and standard deviation equal to 1. 
""" 

from wc import WorstCase
from ..optimizer import optimizerClass
from ..optimizer.base import Plugin, Reporter
from ..misc.debug import DbgMsgOut, DbgMsg
from ..evaluator.performance import PerformanceEvaluator, updateAnalysisCount
from ..evaluator.aggregate import *
from ..evaluator.auxfunc import listParamDesc, paramDict, paramList, dictParamDesc
from ..parallel.cooperative import cOS
import numpy as np

import matplotlib.pyplot as plt

__all__ = [ 'WorstCaseDistance' ] 


class Function(object):
	def __init__(self, n, beta0=0.0):
		"""si
		Function - distance from origin. 
		
		First *n* parameters are statistical parameters. 
		"""
		self.n=n 
		self.beta0=beta0
		
	def __call__(self, x):
		return np.array((x[:self.n]**2).sum())+self.beta0**2
	
class Function_gH(object):
	def __init__(self, n):
		"""
		Returns the gradient and the Hessian of the distance from 
		the origin. 
		
		First *n* parameters are statistical parameters. 
		"""
		self.n=n 
	
	def __call__(self, x):
		g=2*x
		g[self.n:]=0.0	# Derivatives wrt operating parameters are 0
		Hdiag=x.copy()
		Hdiag[:self.n]=2.0
		Hdiag[self.n:]=0.0 # Derivatives wrt operating parameters are 0
		
		return g, np.diag(Hdiag)

class optReporter(Reporter):
	def __init__(self, evaluator, wcName, lower):
		Reporter.__init__(self)
		self.evaluator=evaluator
		self.wcName=wcName
		self.worst=None
		self.hworst=None
		self.lower=lower
		
	def reset(self):
		Reporter.reset(self)
		self.worst=None
		self.hworst=None
		
	def __call__(self, x, ft, opt):
		perf=self.evaluator.results[self.wcName]['default']
		if opt.niter==opt.bestIter:
			self.worst=perf
			if type(ft) is tuple:
				self.hworst=opt.aggregateConstraintViolation(ft[1])
		str="iter=%d perf=%e " % (opt.niter, perf)
		if self.hworst is not None:
			str+=" hworst=%e" % (self.hworst)
		str+=" worst=%e" % (self.worst)
		str+=" "+self.wcName
		if 'step' in opt.__dict__:
			str+=" step=%e" % (opt.step)
		DbgMsgOut("WCD", str)
		
class wcdReporter(Reporter):
	def __init__(self, nsph, evaluator, beta0, wcdSign, wcName, goal, lower):
		Reporter.__init__(self)
		self.nsph=nsph
		self.evaluator=evaluator
		self.beta0=beta0
		self.wcdSign=wcdSign
		self.wcName=wcName
		self.goal=goal
		self.wcd=None
		self.wcdperf=None
		self.hworst=None
		self.lower=lower
		
	def reset(self):
		Reporter.reset(self)
		self.worst=None
		self.hworst=None
		
	def __call__(self, x, ft, opt):
		dist=self.wcdSign*(x[:self.nsph]**2+self.beta0**2).sum()**0.5
		perf=self.evaluator.results[self.wcName]['default']
		if type(ft) is tuple:
			h=opt.aggregateConstraintViolation(ft[1])
		else:
			h=None
		if opt.niter==opt.bestIter:
			self.wcd=dist
			self.hworst=h
			self.wcdperf=perf	
		str="iter=%d dist=%.4f h=%e h_worst=%e wcd=%.4f" % (opt.niter, dist, h, self.hworst, self.wcd)
		if 'step' in opt.__dict__:
			str+=" step=%e" % (opt.step)
		str+=" "+self.wcName
		DbgMsgOut("WCD", str)
		
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

class wcdStopper(Plugin):
	"""
	Stops the algorithm when the angle between x and the gradient of c 
	becomes smaller than a threshold given by relative tolerance 
	*angleTol* and the constraint function is close enough to 0
	within tolerance *constrTol*. 
	
	*n* is the number of statistical parameters
	"""
	def __init__(self, constrTol, angleTol, maxStep, name, n, wcdSign, debug=0):
		Plugin.__init__(self)
		self.constrTol=constrTol
		self.angleTol=angleTol
		self.n=n
		self.wcdSign=wcdSign
		self.maxStep=maxStep
		self.debug=debug
		self.name=name
		
	def reset(self):
		self.it=None
	
	def optimality(self, x, ct, opt):
		if (
			self.constrTol is not None and self.angleTol is not None and 
			opt.xgo is not None and opt.modJ is not None and self.n>0
		):
			cdelta=np.abs(ct)
			g=(opt.modJ.reshape(x.size))[:self.n]
			xv=x[:self.n]
			l1=(g**2).sum()**0.5
			l2=(xv**2).sum()**0.5
			if l1==0.0 or l2==0.0:
				angle=0.0
			else:
				angle=np.arccos(self.wcdSign*(g*xv).sum()/l1/l2)*180/np.pi
			return cdelta, angle 
		else:
			return None, None
	
	def __call__(self, x, ft, opt):
		stop=False
		
		# Check origin
		cdelta, angle = self.optimality(opt.xgo, opt.co, opt)
		if cdelta is not None and opt.step<=self.maxStep and cdelta<self.constrTol and angle<self.angleTol:
			stop=True
			self.it=opt.ito
			self.x=opt.xgo
		if self.debug>1 and cdelta is not None:
			DbgMsgOut("STOP origin", "h=%.8e angle=%f %s" % (cdelta, angle, self.name))
			if stop:
				DbgMsgOut("STOP", "Stopping %s at %d" % (self.name, opt.niter))
				
		opt.stop=opt.stop or stop
		return None

class WorstCaseDistance(WorstCase):
	"""
	See the :class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	class for details on *heads*, *analyses*, *measures*, and *variables*. 
	
	See the :class:`~pyopus.design.wc.WorstCase` class for the explanation
	of *statParamDesc*, *opParamDesc*, *fixedParams*, *maxPass*, 
	*wcStepTol*, *stepTol*, *constrTol*, *angleTol*, *stepScaling*, 
	*perturbationScaling*, *screenThreshold*, *maximalStopperStep*, 
	*evaluatorOptions*, *sensOptions*, *screenOptions*, 
	*opOptimizarOptions*, *optimizerOptions*, and *spawnerLevel* 
	options. 
	
	Setting *debug* to a value greater than 0 turns on debug messages. 
	The verbosity is proportional to the specified number. 
	
	If *linearWc* is set to ``True`` the initial point in the space  
	of the statistical parameters is computed using linear worst case 
	distance analysis. 
	
	This is a callable object with an optional argument specifying 
	which worst case distances to compute. If given, the argument 
	must be a list of entries. Every entry is 
	
	* a tuple of the form (name,type), where name is the measure name
	  and type is ``lower`` or ``upper``
	* a string specifying the measure name. In this case the type 
	  of comuted worst case distance is given by the presence of the 
	  ``lower`` and the ``upper`` entries in the measure's 
	  description. 
	  
	If no argument is specified, all worst case distances 
	corresponding to lower/upper bounds of all *measures* are 
	computed. 
	
	The results are stored in the :attr:`results` member. They are 
	represented as a list of dictionaries with the following members:
	
	* ``name`` - name of the performance measure
	* ``type`` - ``lower`` or ``upper``
	* ``nominal`` - nominal performance
	* ``nominal_wcop`` - performance at nominal statistical 
	  parameters and initial worst case operating parameters
	* ``linwc`` - performance at the linearized worst case 
	  distance point
	* ``wc`` - performance at the worst case distance point
	* ``scr_stat`` - list of statistical parameters (screened)
	* ``scr_op`` - list of op parameters (screened)
	* ``lindist`` - linearized worst case distance
	* ``dist`` - worst case distance
	* ``op`` - operating parameter values at the worst case 
	  distance point
	* ``stat`` - statistical parameter values at the worst case 
	  distance point
	* ``evals`` - number of evaluations
	* ``status`` - ``OK``, ``FAILED``, ``OUTSIDE+``, ``OUTSIDE-``, 
	  ``SENS+``, or ``SENS-``
	
	Status FAILED means that the algorithm failed to converge in 
	*maxPass* passes. 
	
	Status OUTSIDE+ means that the algorithm faield to find a 
	point satisfying the WCD search constraint for positive 
	WCD. This means that the WCD is a large positive value. 
	
	Status OUTSIDE- means that the algorithm faield to find a 
	point satisfying the WCD search constraint for negative 
	WCD. This means that the WCD is a large negative value. 
	
	SENS+ and SENS- are similar to OUTSIDE+ and OUTSIDE-, except 
	that they occur when zero sensitivity to statistical parameters 
	is detected. 
	
	Objects of this type store the number of analyses performed 
	during the last call to the object in a dictionary stored in 
	the :attr:`analysisCount` member. 
	
	The return value of a call to an object of this class is a 
	tuple holding the results structure and the analysis count 
	dictionary. 
	"""
		
	def __init__(
		self, heads, analyses, measures, 
		statParamDesc, opParamDesc, fixedParams={}, variables={}, debug=0, 
		linearWc=True, alternating=True, 
		maxPass=20, wcStepTol=0.01, stepTol=0.01, constrTol=0.01, angleTol=15, 
		stepScaling=4, perturbationScaling=64, screenThreshold=0.01, maximalStopperStep=0.5, 
		evaluatorOptions={}, sensOptions={}, screenOptions={}, 
		opOptimizerOptions={}, optimizerOptions={}, 
		spawnerLevel=1
	):
		WorstCase.__init__(
			self, heads=heads, analyses=analyses, measures=measures, 
			statParamDesc=statParamDesc, opParamDesc=opParamDesc, 
			fixedParams=fixedParams, variables=variables, debug=debug, 
			linearWc=linearWc, alternating=alternating, 
			maxPass=maxPass, wcStepTol=wcStepTol, stepTol=stepTol, 
			constrTol=constrTol, angleTol=angleTol, 
			stepScaling=stepScaling, perturbationScaling=perturbationScaling, 
			screenThreshold=screenThreshold, maximalStopperStep=maximalStopperStep, 
			evaluatorOptions=evaluatorOptions, 
			sensOptions=sensOptions, screenOptions=screenOptions, 
			opOptimizerOptions=opOptimizerOptions, optimizerOptions=optimizerOptions, 
			spawnerLevel=spawnerLevel
		)
		
	def jobGenerator(self, wcSpecs=None):
		# Prepare jobs
		if wcSpecs is None:
			wcSpecs=self.measures.keys()
		
		ii=0
		for wcSpec in wcSpecs:
			# wcSpec is not a tuple, do those wc specs that are given in measures
			wcName=wcSpec
			# A job is (index, name, wcType)
			if 'upper'in self.measures[wcName]:
				yield (self.jobProcessor, [ii, wcName, 'upper'])
				ii+=1
			if 'lower'in self.measures[wcName]:
				yield (self.jobProcessor, [ii, wcName, 'lower'])
				ii+=1
	
	def jobProcessor(self, index, name, wcType):
		target=self.measures[name][wcType]
		return self.compute(name, target, wcType=="lower")
		
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
				DbgMsgOut("WCD", self.formatResults(results))
	
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
			DbgMsgOut("WCD", "Analysis count: %s" % str(self.analysisCount))
			DbgMsgOut("WCD", "Results:")
			DbgMsgOut("WCD", self.formatResults(details=True))
			
		return self.results, self.analysisCount
	
	def wcdOptimization(self, atStatx, atOpx, lastStep, inNdxStat, inNdxOp, outNdxStat, outNdxOp, ev, beta0, wcdSign, target, norm, wcName, lower=True):
		# Linear worst case distance
		linDist=(atStatx**2).sum()**0.5
		
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
				
		# Join statistical and op parameters as initial point
		paramIni=np.hstack((
			atStatx[inNdxStat], atOpx[inNdxOp]
		))
		
		# Param scaling for op parameters
		scale=(paramHi-paramLo)/self.stepScaling
		
		# Function (distance)
		fun=Function(nScrStat)
		
		# Prepare gradient and Hessian
		gH=Function_gH(nScrStat)
		
		# Constraint
		ev.setParameters(fixedParams)
		# Positive constraint = infeasible
		agg=Aggregator(
			ev, [{
				'measure': wcName, 
				'norm': Nabove(target, norm, 1e6) if lower else Nbelow(target, norm, 1e6), 
				'shape': Slinear2(1.0,1.0), 
				'reduce': Rworst()
			}], paramNames
		)
		
		# TODO
		# Wrap in an array (this is not picklable, needs to be replaced when a parallel algorithm is used)
		con=lambda x: np.array([agg(x)])
		
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
		
		# Constraint depends on the wcd type
		if wcdSign>0:
			# Look in infeasible region
			clo=np.array([0])
			chi=np.array([np.Inf])
		else:
			# Look in feasible region
			clo=np.array([-np.Inf])
			chi=np.array([0])
			
		opt=optimizerClass("QPMADS")(
			fun, paramLo, paramHi, fgH=gH, 
			constraints=con, clo=clo, chi=chi, scaling=scale, debug=0, 
			**optimizerDefaults
		)
		collector=evalCollector(ev, wcName)
		
		# Install plugins
		if self.debug>1:
			opt.installPlugin(wcdReporter(nScrStat, ev, beta0, wcdSign, wcName, target, lower))
		opt.installPlugin(collector)
		# 5%, 5 deg tol
		stopper=wcdStopper(
			constrTol=self.constrTol, angleTol=self.angleTol, maxStep=self.maximalStopperStep, 
			name=wcName, n=nScrStat, wcdSign=wcdSign, debug=self.debug
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
			'dist': wcdSign*(atStatx**2).sum()**0.5, 
			'stat': solStat, 
			'op': solOp, 
		}
		if self.linearWc:
			result['linwc']=linPerf
			result['lindist']=linDist
		
		return result, atStatx, atOpx, opt.step, analysisCount, nevals
	
	def compute(self, wcName, target, lower=True):
		# Reset analysis counter
		analysisCount={}
		
		if self.debug:
			if lower:
				str="lower"
			else:
				str="upper"
			DbgMsgOut("WCD", "Running %s, %s" % (wcName, str))
		
		# Construct evaluator
		ev=PerformanceEvaluator(self.heads, self.analyses, self.measures, variables=self.variables, activeMeasures=[wcName])
		
		# Intial statistical parameters
		atStatx=np.zeros(self.nStat)
		
		# Evaluations counter
		nevals=[]
		
		# Worst op parameters
		#result, atOpx, anCount, nev = cOS.dispatchSingle(
		#	self.opWorstCase, 
		#	args=[ev, wcName], 
		#	kwargs={'lower': lower}, 
		#	remote=self.spawnerLevel<=2
		#)
		result, atOpx, anCount, nev = self.opWorstCase(ev, wcName, lower=lower)
		updateAnalysisCount(analysisCount, anCount)
		nevals.append(nev)
		
		# Initial worst performance
		atWorstCase=result['nominal_wcop']
		
		# Determine the type of the WCD problem
		alternating=self.alternating
		if (lower and atWorstCase>=target) or (not lower and atWorstCase<=target):
			wcdSign=1
		else:
			wcdSign=-1
			# Do not optimize alternating op and stat parameters when wcd is negative
			alternating=False
		if self.debug:
			if wcdSign>0:
				DbgMsgOut("WCD", "Positive wcd expected for %s, %s" % (wcName, str))
			else:
				DbgMsgOut("WCD", "Negative wcd expected for %s, %s" % (wcName, str))
		
		atPass=0
		inNdxStat=[]
		inNdxOp=[]
		lastStep=0.25
		zeroSens=False
		while True:
			# Assume op parameters did not change
			opChanged=False
			
			# Confirm op parameters
			if alternating and atPass>0:
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
						DbgMsgOut("WCD", "  OP parameters changed")
					else:
						DbgMsgOut("WCD", "  OP parameters unchanged")
				
			# Sensitivity and screening
			if self.debug:
				DbgMsgOut("WCD", "  Computing sensitivity, %s, pass %d" % (wcName, atPass+1))
			
			if wcdSign>0:
				useType=lower
			else:
				useType=not lower
				
			# Skip op sensitivity when alternating
			# No need to spawn this one, it spawns its own subtasks
			propDiff, delta, anCount, nev, inStat, inOp = self.sensitivityAndScreening(ev, atStatx, atOpx, wcName, useType, skipOp=alternating)
			updateAnalysisCount(analysisCount, anCount)
			nevals.append(nev)
			screenChanged=not (set(inStat).issubset(set(inNdxStat)) and set(inOp).issubset(set(inNdxOp)))
			
			# Set of screened parameters unchanged and op parameters values unchanged ... stop
			if not screenChanged and not opChanged:
				if self.debug:
					DbgMsgOut("WCD", "  Stopping")
				result['status']="OK"
				break
			
			# New parameter set - accumulate
			newNdxStat=set(inNdxStat).union(inStat)
			newNdxOp=set(inNdxOp).union(inOp)
			# New parameter set - replace
			# newNdxStat=set(inStat)
			# newNdxOp=set(inOp)
			
			# Report
			if self.debug>1:
				DbgMsgOut("WCD", "  Parameter set (%d)" % (len(newNdxStat)+len(newNdxOp)))
				for ii in newNdxStat:
					flag=" "
					if ii in inStat and ii not in inNdxStat:
						flag="*"
					DbgMsgOut("WCD", "    "+flag+self.statNames[ii])
				for ii in newNdxOp:
					flag=" "
					if ii in inOp and ii not in inNdxOp:
						flag="*"
					DbgMsgOut("WCD", "    "+flag+self.opNames[ii])
				
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
			
			# Use beta=3.0 for norm computation
			norm=np.abs(3.0*((propDiff/delta)[:self.nStat]**2).sum()**0.5)
			if norm==0.0:
				norm=1.0
			
			# Compute sensitivity to statistical parameters
			sens=(propDiff/delta)
			ssens=sens[:self.nStat]
			
			# Verify zero sensitivity
			nrm=(ssens**2).sum()**0.5
			if (
				nrm==0.0 or 
				np.abs(target-atWorstCase)/nrm>=((self.statHi-self.statLo)**2).sum()**0.5/2
			):
				zeroSens=True
			else:
				zeroSens=False
				
			# Compute linear worst case distance point
			if atPass==0 and self.linearWc and len(inNdxStat)>0:
			# if self.linearWc and len(inNdxStat)>0: 
				# Consider only screened parameters
				ssens[outNdxStat]=0.0
				# Compute sensitivity norm
				snorm=(ssens**2).sum()**0.5
				if snorm!=0.0:
					# Compute linear step length
					dl=(target-atWorstCase)/snorm
					# Limit step length to 10
					if dl>10.0:
						dl=10.0
					if dl<-10.0:
						dl=-10.0
					# Compute step
					atStatdxLin=dl*ssens/snorm
				else:
					# Zero sensitivity
					atStatdxLin=np.zeros(self.nStat)
					
				atStatxLin=atStatx+atStatdxLin
				# Slide along boundary
				atStatxLin=np.where(atStatxLin>self.statHi, self.statHi, atStatxLin)
				atStatxLin=np.where(atStatxLin<self.statLo, self.statLo, atStatxLin)
				# Move
				atStatx=atStatxLin
			
			# Compute beta0 corresponding to screened out statistical parameters
			beta0=(np.array(atStatx)[outNdxStat]**2).sum()**0.5
			
			# Main optimization
			if self.debug:
				DbgMsgOut("WCD", "  Optimization, %s, pass %d" % (wcName, atPass+1))
			
			#result1, atStatx, atOpx, lastStep, anCount, nev = cOS.dispatchSingle(
			#	self.wcdOptimization, 
			#	args=[atStatx, atOpx, lastStep*4, inNdxStat, inNdxOp, outNdxStat, outNdxOp, ev, beta0, wcdSign, target, norm, wcName, lower], 
			#	remote=self.spawnerLevel<=2
			#)
			result1, atStatx, atOpx, lastStep, anCount, nev = self.wcdOptimization(
				atStatx, atOpx, lastStep*4, inNdxStat, inNdxOp, outNdxStat, outNdxOp, ev, beta0, wcdSign, target, norm, wcName, lower
			)
			updateAnalysisCount(analysisCount, anCount)
			nevals.append(nev)
			
			# Get new worst case
			atWorstCase=result1['wc']
			
			# Ignore linear wc result for all but the first pass
			if atPass>0 or not self.linearWc:
				del result1['linwc']
				del result1['lindist']
			result.update(result1)
		
			# Increase pass count
			atPass+=1
			
			if atPass>=self.maxPass:
				if self.debug:
					DbgMsgOut("WCD", "  Maximum number of passes reached, stopping (%s)" % wcName)
				result['status']="FAILED"
				break
			
			# print wcName, atPass, ":", nevals, " ..", np.array(nevals).sum()
			
		result['name']=wcName
		result['target']=target
		result['passes']=atPass
		if lower:
			result['type']="lower"
		else:
			result['type']="upper"
		result['evals']=np.array(nevals).sum()
		
		# Verify if result is outside search region
		if result['status']=='OK':
			# Check if it is in the search region
			constraint=(result['wc']-result['target'])/norm
			if result['type']=='lower':
				constraint=-constraint
			if wcdSign<0:
				constraint=-constraint
			# Values above -constrTol are OK
			if constraint<-self.constrTol:
				result['status']='OUTSIDE'+('+' if wcdSign>0 else '-')
		
		# Verify zero sensitivity
		if zeroSens:
			result['status']='SENS'+('+' if wcdSign>0 else '-')
			
		return result, analysisCount
		
	def formatResults(self, results=None, nMeasureName=10, nResult=14, nPrec=5, nEvalPrec=4, details=False):
		"""
		Formats the results as a string. 
		
		*nMeasureName* specifies the formatting width for the 
		performance measure name. 
		
		*nResult* and *nPrec* specify the formatting width and the 
		number of significant digits for the performance measure 
		values. 
		
		*nEvalPrec* is the number of spaces reserved for the 
		evaluations count. 
		
		If *details* is ``True`` the nominal performance, the  
		performance the worst case operating parameters and the nominal 
		statistical parameters, and the linear worst case distance 
		are also formatted. 
		"""
		if results is None:
			results=self.results
			
		txt=""
		for res in results:
			if res is None:
				continue
			txt+="%*s" % (nMeasureName, res['name'])
			if res['type']=="upper":
				txt+=" < "
			else:
				txt+=" > "
			txt+="%*.*e" % (nResult, nPrec, res['target'])
			if details:
				txt+=" nom=%*.*e" % (nResult, nPrec, res['nominal'])
				txt+=" wcop=%*.*e" % (nResult, nPrec, res['nominal_wcop'])
				if self.linearWc:
					txt+=" lin_wcd=%*.*e" % (nResult, nPrec, res['lindist'])
			txt+=" wc=%*.*e wcd=%*.*e neval=%*d" % (nResult, nPrec, res['wc'], nResult, nPrec, res['dist'], nEvalPrec, res['evals'])
			txt+=" %d" % res['passes']
			txt+=" "+res['status']+"\n"
		
		return txt
	
"""
.. inheritance-diagram:: pyopus.design.sensitivity
    :parts: 1

**Sensitivity analysis (PyOPUS subsystem name: SENS)**

Uses finite differences to estimate the sensitivities of performance 
measures to input parameters. 
""" 

from ..misc.debug import DbgMsgOut, DbgMsg
from ..evaluator.performance import PerformanceEvaluator, updateAnalysisCount
from ..evaluator.auxfunc import listParamDesc, paramDict
from ..parallel.cooperative import cOS
import numpy as np


__all__ = [ 'Sensitivity' ] 

class Sensitivity(object):
	"""
	Callable object that computes sensitivities. 
	
	See :class:`~pyopus.evaluator.performance.PerformanceEvaluator` 
	for details on *evaluator*. 
	
	The parameters are given by a dictionary *paramDesc* where 
	parameter name is the key. One entry in this dictionary 
	corresponds to one parameter. Every entry is a dictionary 
	with the following members 
	
	  * lo .. lower bound
	  * hi .. upper bound
	
	*paramDesc* can be a list in which case the complete set of 
	parameters is the union of parameters specified by list 
	memebers. 
	
	*paramNames* specifies the order of parameters in the 
	sensitivity vector. 
	
	Setting *debug* to a value greater than 0 turns on debug messages. 
	The verbosity is proportional to the specified number. 
	
	*diffType* is the finite difference type 
	(0=central, 1=forward, 2=backward)
	
	*relPerturb* specifies the relative perturbation with respet 
	to the allowed parameter span given by lo and hi. 
	
	*absPerturb* overrides *relPerturb*. It specifies the absolute
	perturbation of the parameters. 
	
	If *relPerturb* (or *absPerturb*) is scalar the same 
	perturbation is applied to all parameters. If it is a vector 
	its components specify the perturbations of individual 
	parameters in the order given by *paramNames*. 
	
	If *spawnerLevel* is not greater than 1 the evaluations of 
	perturbed points are dispatched to available computing nodes. 
	
	The number of circuit evaluations is stored in the neval member. 
	
	Calling convention: obj(x0) where *x0* is a vector of parameters
	specifying the point where the sensitivity should be computed. 
	The components represent parameters in the order specified by
	*paramNames*. The sensitivities are computed at *x0*. 
	
	Returns a tuple with performance deltas as the first member, 
	the parameter perturbations as the second member, and the 
	*analysisCount* dictionary as the third member. 
	
	The performance deltas are returned in a double dictionary 
	where the first index is the performance measure name and the 
	second index is the corner name. 
	
	Every performance delta is a vector whose components correspond 
	to parameters in the order given by *paramNames*. The vector 
	holds the perturbations for one performance measure in one corner. 
	If the computation of a performance delta for some parameter 
	perturbation fails (i.e. any of the points required for computing the 
	performance delta fail to produce a valid result) the corresponding 
	component of the vector is ``NaN``. 
	
	Dividing performance deltas by parameter perturbations 
	results in the sesitivities. This holds regardless of the type of 
	perturbation specified with *diffType*. 
	
	The performance deltas are stored in the *results* member. 
	The parameter perturbations are stored in the *delta* member. 
	
	Objects of this type store the number of analyses performed 
	during the last call to the object in the *analysisCount* 
	member. 
	"""
	def __init__(self, evaluator, paramDesc, paramNames, debug=0, 
		diffType=0, relPerturb=0.1, absPerturb=None, 
		spawnerLevel=1
	):
		self.diffType=diffType	# 0=central difference, 1=forward difference, 2=backward difference
		self.relPerturb=relPerturb	# Relative to lo-hi range of a parameter
		self.absPerturb=absPerturb	# Absolute perturbation, overrides relPerturb
		
		self.evaluator=evaluator
		self.paramNames=paramNames
		self.debug=debug
		
		self.spawnerLevel=spawnerLevel
		
		# Build the list of parameters by merging dictionaries
		if type(paramDesc) is list or type(paramDesc) is tuple:
			self.paramDescDict={}
			for entry in paramDesc:
				self.paramDescDict.update(entry)
		else:
			self.paramDescDict=paramDesc
	
	def jobGenerator(self, originVec, useDiffType, vplus, vminus):
		for ii in range(len(self.paramNames)):
			yield (self.jobProcessor, [ii, originVec, useDiffType[ii], vplus[ii], vminus[ii]])
			
	def jobProcessor(self, ii, originVec, useDiffType, plus, minus):
		analysisCount={}
		neval=0
		
		if self.debug:
			DbgMsgOut("SENS", "Perturbing parameter "+str(ii)+" ("+str(self.paramNames[ii])+")")
		
		# Plus perturbation
		plusPerf=None
		if useDiffType==0 or useDiffType==1:
			if self.debug:
				DbgMsgOut("SENS", "  positive perturbation")
			par=originVec.copy()
			par[ii]=plus
			plusPerf, anCount = self.evaluator(paramDict(par, self.paramNames))
			neval+=1
			updateAnalysisCount(analysisCount, anCount)
			
		# Minus perturbation
		minusPerf=None
		if useDiffType==0 or useDiffType==2:
			if self.debug:
				DbgMsgOut("SENS", "  negative perturbation")
			par=originVec.copy()
			par[ii]=minus
			minusPerf, anCount = self.evaluator(paramDict(par, self.paramNames))
			neval+=1
			updateAnalysisCount(analysisCount, anCount)
			
		return plusPerf, minusPerf, analysisCount, neval
	
	def jobCollector(self, results, analysisCount, originPerf):
		while True:
			index, job, retval = (yield)
			
			jj, originVec, useDiffType, plus, minus = job[1]
			plusPerf, minusPerf, anCount, neval = retval
			
			updateAnalysisCount(analysisCount, anCount)
			self.neval+=neval
			
			# Compute sensitivity
			# Go through all measures
			for measureName in self.evaluator.getActiveMeasures():
				measure=self.evaluator.measures[measureName]
				
				# Get corner names
				if plusPerf is not None:
					cornerNames=plusPerf[measureName].keys()
				else:
					cornerNames=minusPerf[measureName].keys()
				
				# Go through all corners
				for cornerName in cornerNames:
					# Prepare results entry, NaN means failed sensitivity
					if cornerName not in results[measureName]:
						results[measureName][cornerName]=np.zeros(len(self.paramNames))
						results[measureName][cornerName].fill(np.NaN)
			
					if self.debug:
						DbgMsgOut("SENS", "  sensitivity of '%s' in corner '%s' to '%s'." % (measureName, cornerName, self.paramNames[jj]))
						
					if (
						useDiffType==0 and
						plusPerf[measureName][cornerName] is not None and
						minusPerf[measureName][cornerName] is not None 
					):
						# Central
						results[measureName][cornerName][jj]=(plusPerf[measureName][cornerName]-minusPerf[measureName][cornerName])/2
					elif (
						useDiffType==1 and
						plusPerf[measureName][cornerName] is not None and
						originPerf[measureName][cornerName] is not None 
					):
						# Forward
						results[measureName][cornerName][jj]=(plusPerf[measureName][cornerName]-originPerf[measureName][cornerName])
						# Do we need to move origin
					elif (
						useDiffType==2 and
						minusPerf[measureName][cornerName] is not None and
						originPerf[measureName][cornerName] is not None 
					):
						# Backward
						results[measureName][cornerName][jj]=(originPerf[measureName][cornerName]-minusPerf[measureName][cornerName])
						# Do we need to move origin
						
					if self.debug: 
						if np.isnan(results[measureName][cornerName][jj]):
							DbgMsgOut("SENS", "    FAILED")
						else:
							DbgMsgOut("SENS", "    OK")
		
	def __call__(self, x0):
		# Reset analysis count
		analysisCount={}
		self.neval=0
		
		# Clear results
		self.results={}
		self.delta=[]
		
		# List of parameter names, extract lo, hi, and nominal vector
		pVal=listParamDesc(self.paramDescDict, self.paramNames)
		vnom=np.array(x0)
		vhi=np.array(pVal["hi"])
		vlo=np.array(pVal["lo"])
		
		if self.absPerturb is None:
			# Relative
			delta=(np.array(pVal["hi"])-np.array(pVal["lo"]))*np.array(self.relPerturb)
		else:
			# Absolute
			delta=np.zeros(len(self.paramNames))+np.array(self.absPerturb)
		
		vplus=vnom+delta
		vminus=vnom-delta
		
		if (vnom>vhi).any() or (vnom<vlo).any():
			raise Exception, DbgMsg("SENS", "Initial point violates bounds")
		
		if ((vplus>vhi)&(vminus<vlo)).any():
			raise Exception, DbgMsg("SENS", "Perturbation too large")
		
		useDiffType=np.zeros(len(self.paramNames))-1
		if self.diffType==0:
			# Central difference
			useDiffType=np.where((vplus<=vhi)&(vminus>=vlo), 0, useDiffType)
			# Switch to backward difference
			if ((vplus>vhi)&(vminus>=vlo)).any():
				if self.debug:
					DbgMsgOut("SENS", "Switching from central to backward difference for some parameters.")
				useDiffType=np.where((vplus>vhi)&(vminus>=vlo), 2, useDiffType)
			# Switch to forward difference
			if ((vminus<vlo)&(vplus<=vhi)).any():
				if self.debug:
					DbgMsgOut("SENS", "Switching from central to forward difference for some parameters.")
				useDiffType=np.where((vminus<vlo)&(vplus<=vhi), 1, useDiffType)
		elif self.diffType==1:
			# Forward difference
			useDiffType=np.where((vplus<=vhi), 1, useDiffType)
			# Switch to backward difference
			if ((vplus>vhi)&(vminus>=vlo)).any():
				if self.debug:
					DbgMsgOut("SENS", "Switching from forward to backward difference for some parameters.")
				useDiffType=np.where((vplus>vhi)&(vminus>=vlo), 2, useDiffType)
		else:
			# Backward difference
			useDiffType=np.where((vminus>=vlo), 2, useDiffType)
			# Switch to forward difference
			if ((vminus<vlo)&(vplus<=vhi)).any():
				if self.debug:
					DbgMsgOut("SENS", "Switching from backward to forward difference for some parameters.")
				useDiffType=np.where((vminus<vlo)&(vplus<=vhi), 1, useDiffType)
		
		if (useDiffType!=0).any():
			# Need nominal values
			if self.debug:
				DbgMsgOut("SENS", "Evaluating nominal performance")
			nomPerf, anCount = cOS.dispatchSingle(
				self.evaluator, args=[paramDict(vnom, self.paramNames)], 
				remote=self.spawnerLevel<=1
			)
			self.neval+=1
			updateAnalysisCount(analysisCount, anCount)
		else:
			nomPerf=None
			
		# Set origin and origin performance
		originPerf=nomPerf
		originVec=vnom.copy()
		
		# Prepare results structure entries for measures
		results={}
		for measureName in self.evaluator.getActiveMeasures():
			results[measureName]={}
		
		cOS.dispatch(
			jobList=self.jobGenerator(originVec, useDiffType, vplus, vminus), 
			collector=self.jobCollector(results, analysisCount, originPerf), 
			remote=self.spawnerLevel<=1
		)
		
		# Return deltas and perturbations
		self.results=results
		self.delta=delta
		
		self.analysisCount=analysisCount
		
		if self.debug>1: 
			DbgMsgOut("SENS", "Analysis count: %s" % str(self.analysisCount))
		
		return results, delta, self.analysisCount
	
	def screen(self, contribThreshold=0.01, cumulativeThreshold=0.25, useSens=False, squared=True):
		"""
		Performs parameter screening. Returns a double dictionary with 
		performance measure name as the first index and corner name as 
		the second index. Every entry is a dictionary containing two vectors of 
		indices. The first vector (named out) lists the indices of 
		parameters with small influence. The second one (named in) lists 
		the remaining indices. The indices are ordered by 
		absolute sensitivity from the smallest to the largest. 
		
		A parameter is considered to have low influence if its relative 
		contribution is below *contribThreshold*. The relative cumulative 
		contribution of the set of parameters with low influence may 
		not exceed *cumulativeThreshold*. 
		
		If *useSens* is set to ``True`` the performance measure 
		semnsitivities are used instead of deltas. This makes sense only 
		if all parameters are expressed in the same units. 
		
		If *squared* is set to ``False`` the absolute sensitivities are 
		used instead of squared sensitivities in the computation of the 
		relative influence. 
		"""
		# Go through all measures / corners
		indices={}
		for measureName in self.evaluator.getActiveMeasures():
			measure=self.evaluator.measures[measureName]
			
			# Prepare indices structure entry
			indices[measureName]={}
			
			# Go through all corners
			for cornerName in self.results[measureName].keys():
				# Prepare indices structure entry
				indices[measureName][cornerName]=None
				
				# Collect sensitivity data
				evSens=self.results[measureName][cornerName]
				if useSens:
					evSens=evSens/self.delta
				# Use absolute values
				pSens=np.abs(evSens)
				# Are all sensitivities finite?
				if not np.isfinite(pSens).all():
					# No, no parameter is marked as insignificant
					outIndex=np.array([])
					inIndex=range(len(self.paramNames))
					if self.debug:
						DbgMsgOut("SENS", "Infinite or NaN sensitivity, all parameters are significant.")
				else:
					# Yes
					# Order sensitivities from smallest to largest
					ii=np.argsort(pSens)
					pSensOrd=pSens[ii]
					if pSensOrd.max()<=0.0:
						# Zero sensitivity, no relevant parameters
						if self.debug:
							DbgMsgOut("SENS", "Zero sensitivity, all parameters are significant.")
						inIndex=np.array(range(pSensOrd.size))
						outIndex=np.array([])
					else:
						# Square
						if squared:
							pSensOrd=pSensOrd**2
						# Compute normalized cumulative sum (relative)
						pSensCS=pSensOrd.cumsum()
						pSensCS/=pSensCS[-1]
						# Compute relative contributions
						pSensRel=pSensOrd/pSensOrd.sum()
						# Find threshold 
						thri=np.where((pSensCS>cumulativeThreshold)|(pSensRel>contribThreshold))[0]
						# Extract indices
						if len(thri)<=0:
							# Crossing not found
							outIndex=np.array([])
							inIndex=ii
						else:
							# Crossing found
							outIndex=ii[:thri[0]]
							inIndex=ii[thri[0]:]
					
					if self.debug:
						DbgMsgOut("SENS", "Screening report for '%s' in corner '%s'" % (measureName, cornerName))
						DbgMsgOut("SENS", "Unscreened parameter with lowest influence: "+str(self.paramNames[ii[thri[0]]]))
						DbgMsgOut("SENS", "%30s %12s %12s %12s" % ("name", "sens", "rel", "cum"))
						for jj in range(len(self.paramNames)):
							DbgMsgOut("SENS", "%30s %12.2e %12.2e %12.2e" % (
									self.paramNames[ii[jj]],
									evSens[ii[jj]],
									pSensRel[jj],
									pSensCS[jj]
								)
							)
							
				indices[measureName][cornerName]={
					'out': outIndex, 
					'in': inIndex
				}
		
		return indices
	
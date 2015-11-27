"""
.. inheritance-diagram:: pyopus.problems.mwbm
    :parts: 1
	
**Derivative-free optimization test functions by More and Wild. 
(PyOPUS subsystem name: MWBM)**

All 22 basic test functions [mw]_ in this module are maps from :math:`R^n` to 
:math:`R`. All functions are unconstrained. Every function comes from the 
CUTEr test set and is specified by a tuple (mgh_name, n, m, s), where n is the 
dimension, m is the number of component functions, and s specified the initial 
point scaling, i.e.  

.. math::
  x_0 = 10^{s} x_{0, problem}

Every problem can be smooth, piecewise-smooth, deterministically noisy or 
stochastically noisy. 

The CUTEr problems used by this suite are implemented in the _mwbm module. 
This module does not depend on CUTEr. 

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. It depends only on the cpi and 
the _mwbm modules. 

Beside test functions this module also includes support for data profile 
generation from function value history. 

.. [mw] More, J.J., Wild S.M., Benchmarking Derivative-Free Optimization 
        Algorithms. SIAM Journal on Optimization, vol. 20, pp.172-191, 2009. 
"""

from cpi import CPI, MemberWrapper
import numpy as np
try:
	import _mwbm
except:
	print("Failed to import _mwbm module. MWBM problems will not be available.")

try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y

__all__ = [ 
	'MWBM', 
]

class MWBM(CPI):
	"""
	Test functions from the More-Wild suite. 
	
	*num* is the problem number (0-52). 
	
	*problemType* is the type of the problem (0=smooth, 1=piecewise smooth, 
	2=deterministically noisy, 3=stochastically noisy). 
	
	*epsilon* is the noise level for *problemType* 2 and 3. 
	
	The number of the underlying function (0-21) is stored in :attr:`probNdx`. 
	The dimension, the number of component functions, and the initial point 
	scaling	are stored in the :attr:`n`, :attr:`m`, and :attr:`s` members. 
	
	The full name of the problem is in the :attr:`name` member. It is of the 
	form name_m_s. 
	
	The initial point can be obtained from the :attr:`initial` member. The 
	function value is returned by the :meth:`f` method. 
	
	The gradient and the best known minimum of the function are not available. 
	"""
	
	namelist=[
		"LinearFullRank", #0
		"LinearRank1", #1
		"LinearRank1ZCR", # "LinearRank1ZeroColumnsAndRows", #2
		"Rosenbrock", #3
		"HelicalValley", #4
		"PowellSingular", #5
		"FreudensteinAndRoth", #6		multimin smooth
		"Bard", #7
		"KowalikAndOsborne", #8
		"Meyer", #9
		"Watson", #10
		"Box3D", #11
		"JennrichAndSampson", #12
		"BrownAndDennis", #13
		"Chebyquad", #14
		"BrownAlmostLinear", #15		multimin smooth
		"Osborne1", #16
		"Osborne2", #17
		"Bdqrtic", #18
		"Cube", #19
		"Mancino", #20
		"Heart8" #21
	]
	"List of problem names"
	
	# problem, n, m, s
	descriptors=[
			[1,		9,		45,		0], # 0
			[1,		9,		45,		1], 
			[2,		7,		35,		0], 
			[2,		7,		35,		1], # 3
			[3,		7,		35,		0], 
			[3,		7,		35,		1], 
			[4,		2,		 2,		0], 
			[4,		2,		 2,		1], 
			[5,		3,		 3,		0], 
			[5,		3,		 3,		1], 
			[6,		4,		 4,		0], # 10 - multimin
			[6,		4,		 4,		1], # 11 - multimin
			[7,		2,		 2,		0], 
			[7,		2,		 2,		1], 
			[8,		3,		15,		0], 
			[8,		3,		15,		1], 
			[9,		4,		11,		0], 
			[10,	3,		16,		0], 
			[11,	6,		31,		0], 
			[11,	6,		31,		1], 
			[11,	9,		31,		0], 
			[11,	9,		31,		1], 
			[11,	12,		31,		0], 
			[11,	12,		31,		1], 
			[12,	3,		10,		0], 
			[13,	2,		10,		0], 
			[14,	4,		20,		0], 
			[14,	4,		20,		1], 
			[15,	6,		6,		0], # 28 - multimin
			[15,	7,		7,		0], # 29 - multimin
			[15,	8,		8,		0], # 30 - multimin
			[15,	9,		9,		0], # 31 - multimin
			[15,	10,		10,		0], # 32 - multimin
			[15,	11,		11,		0], # 33 - multimin
			[16,	10,		10,		0], 
			[17,	5,		33,		0], 
			[18,	11,		65,		0], 
			[18,	11,		65,		1], 
			[19,	8,		8,		0], 
			[19,	10,		12,		0], 
			[19,	11,		14,		0], 
			[19,	12,		16,		0], 
			[20,	5,		5,		0], 
			[20,	6,		6,		0], 
			[20,	8,		8,		0], 
			[21,	5,		5,		0], 
			[21,	5,		5,		1], 
			[21,	8,		8,		0], 
			[21,	10,		10,		0], 
			[21,	12,		12,		0], 
			[21,	12,		12,		1], 
			[22,	8,		8,		0], 
			[22,	8,		8,		1],    
	]
	"Problem descriptors (problem number, n, m, s)"
	
	# These values are not reliable, available only for the smooth problems
	fminTab=[
		36,
		36, 
		8.3803,
		8.3803,
		9.8806, 
		9.8806, 
		0.0, 
		0.0, 
		0.0, 
		0.0, # 10
		0.0, 
		0.0, 
		48.984, 
		48.984, 
		8.2149e-3, 
		8.2149e-3, 
		3.0751e-4, # 17
		87.9458, 
		2.28767e-3, # 19 Watson 6
		2.28767e-3, 
		1.39976e-6, # Watson 9
		1.39976e-6, 
		4.72238e-10, # 23 Watson 12
		4.72238e-10, 
		0.0, # 25 Box3D
		124.362, 
		85822.2, # 27 Brown and Dennis
		85822.2, 
		0.0, 
		0.0, 
		3.51687e-3, # 31 Chebyquad 8
		0.0, 
		4.7727e-3, # 33 Chebyquad 10 ?
		2.7998e-3, # 34 Chebyquad 11 ?
		1.0, # 35 Brown almost linear ?
		5.46489e-5, # 36 Osborne 1
		4.01377e-2, # 37 Osborne 2
		4.01377e-2,
		1.0239e1,
		1.8281e1, # 40 Bdqrtic
		2.2261e1, 
		2.6273e1, 
		3.2009e-03, 
		3.6501e-03, 
		4.5742e-03, 
		0.0, # 46 Mancino
		0.0, 
		0.0, 
		0.0, 
		0.0, 
		0.0, 
		1.9699e-13, # 52 Heart8
		1.6658e1 # 53
	]
	
	type1set=set([7,8,12,15,16,17])
	
	def __init__(self, num, problemType=0, epsilon=1e-3):
		self.num=num
		self.problemType=problemType
		self.epsilon=epsilon
		
		descriptor=self.descriptors[num]
		
		self.probNdx=descriptor[0]-1
		self.n=descriptor[1]
		self.m=descriptor[2]
		self.s=descriptor[3]
		
		self.name="%s_%d_%d" % (self.namelist[self.probNdx], self.m, self.s)
		
		self.initial=_mwbm.dfoxs(self.probNdx, self.n, 10.0**self.s)
		
		self.fmin=None
		
	def f(self, x):
		if self.problemType==1:
			# Piecewise-smooth
			if self.num in self.type1set:
				x1=np.where(x>=0, x, 0.0)
			else:
				x1=x
			return np.abs(_mwbm.dfovec(self.m, x1, self.probNdx)).sum()
		elif self.problemType==2:
			# Deterministically noisy
			xmax=np.abs(x).max()
			xnorm1=np.abs(x).sum()
			xnorm2=((x**2).sum())**0.5
			phi=0.9*np.sin(100*xnorm1)*np.cos(100*xmax)+0.1*np.cos(xnorm2)
			phi=phi*(4*phi**2-3)
			return (self.epsilon*phi+1.0)*((_mwbm.dfovec(self.m, x, self.probNdx)**2).sum())
		elif self.problemType==3:
			# Stochastically noisy
			# Here we use np.random.uniform instead of surn01.f
			return ((
				_mwbm.dfovec(self.m, x, self.probNdx)
				*(self.epsilon*np.random.uniform(-1.0, 1.0, self.m)+1.0)
			)**2).sum()
		else:
			# Smooth
			return (_mwbm.dfovec(self.m, x, self.probNdx)**2).sum()
		
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		xmin, fmin, and g are not available. 
		
		The info member of the returned dictionary is itself a dictionary with 
		the following members:
		
			* ``m`` - m parameter of the MWBM problem
			* ``s`` - s parameter of the MWBM problem
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=0)
		itf['name']=self.name
		itf['x0']=self.initial
		itf['f']=MemberWrapper(self, 'f')
		
		# No reliable minima available at this point
		#if self.problemType==0:
		#	itf['fmin']=self.fminTab[self.num]
		
		itf['info']={
			'm': self.m, 
			's': self.s
		}
		
		return self.fixBounds(itf)
	
def dataProfile(hist, evalNorm, tau=0.1, fmin=None, iProblem=None, iLimit=None, inLimit=None):
	"""
	Processes the function value history for multiple solvers and problems. 
	Returns the data profiles of solvers. 
	
	* *hist* -- history for all solvers (list). Every entry is a list
	  holding the function value histories (NumPy arrays) for 
	  the problems. 
	* *evalNorm* -- a vector of norms used for normalizing the number of 
	  evaluations, one per problem. 
	* *tau* -- convergence criterion, tolerance = tau*(f0-flowest)
	* *fmin* -- a vector of function minima. If not given the best values found 
	  across all solvers are used. 
	* *iProblem* -- limits the analysis to problems with indices given by iProblem
	* *iLimit* -- considers only function evaluations up to iLimit
	* *inLimit* -- considers only function evaluations up to iLimit*evalNorm
	
	If both *iLimit* and *inLimit* are specified *iLimit* is used. 
				
	Return value is a list of data profiles. Every data profile is a tuple 
	cosisting of the normalized function evaluation vector, the corresponding 
	share of the problems solved within accuracy given by *tau*, and an array 
	of problem indices corresponding to the events (points in time when 
	a problem is solved). 
	"""
	# Get the number of solvers
	nsolv=len(hist)
	
	# Get the number of problems
	if iProblem is not None:
		nprob=len(iProblem)
	else:	
		nprob=len(hist[0])
		iProblem=range(nprob)
	
	# Verify that all solvers have the same number of problems
	n0=len(hist[0])
	for solver in hist:
		if len(solver)!=n0:
			raise Exception, DbgMsg("MWBM", "All solvers must have the same number of problems.")
	
	if iLimit is not None:
		maxI=evalNorm/evalNorm*iLimit
	elif inLimit is not None:
		maxI=evalNorm*inLimit
	else:
		maxI=None
	
	# Find lowest function value for every problem (if it was not given)
	if fmin is None:
		fmin=np.zeros(n0)
		for ii in iProblem:
			flow=None
			for solver in hist:
				if len(solver[ii])>0:
					mi=len(solver[ii])
					if maxI is not None and maxI[ii]<mi:
						mi=maxI[ii]
					fmin1=np.nanmin(solver[ii][:mi])
					if flow is None or fmin1<flow:
						flow=fmin1
			if flow is None:
				raise Exception, DbgMsg("MWBM", ("Cannot determine lowest function value for function %d." % ii))
			
			fmin[ii]=flow
	
	
	# Generate data profiles
	dataProfiles=[]
	for ii in range(nsolv):
		solver=hist[ii]
		
		# Prepare events storage. Event is the normalized evaluation at which some problem converges. 
		evts=[]
		probis=[]
		
		# Go through all functions
		cnt=0
		for jj in iProblem:
			# History 
			fhist=solver[jj]
			
			# Function value at which convergence is assumed
			fconv=fmin[jj]+tau*(fhist[0]-fmin[jj])
			
			# Add event
			if np.nanmin(fhist)<=fconv:
				mi=len(fhist)
				if maxI is not None and maxI[jj]<mi:
					mi=maxI[jj]
				flags=np.nonzero(fhist[:mi]<=fconv)[0]
				if len(flags)>0:
					isolved=np.nanmin(flags)
					evts.append(isolved*1.0/evalNorm[jj])
				
					probis.append(jj)
		
		# Sort events
		evts=np.array(evts)
		ndx=np.argsort(evts)
		evts=evts[ndx]
		
		probis=np.array(probis)
		probis=probis[ndx]
		
		# Generate data profile
		nsolved=np.arange(1,len(evts)+1)*1.0/nprob
		
		dataProfiles.append((np.array(evts), nsolved, probis))
	
	return dataProfiles
	
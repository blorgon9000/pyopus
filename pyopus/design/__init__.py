"""
**Design automation module**

Provides functions and classes for computing sensitivity, 
sizing a design across corners, worst case performance, 
worst case distance, yield targeting, and Monte Carlo analysis.  
""" 

from scipy.special import erf, erfinv
from numpy import ceil

def wcd2yield(beta):
	"""
	Computes the yield that corresponds to the worst case distance *beta*. 
	"""
	return 0.5*(1+erf(beta/2**0.5))

def yield2wcd(y):
	"""
	Computes the worst case distance that corresponds to yield *y*. 
	"""
	return erfinv(2*y-1)*2**0.5

def yieldSigma(y, nSamples):
	"""
	Computes the standard deviation of estimated yield *y* computed with 
	*nSamples* Monte Carlo samples. 
	"""
	return (y*(1-y)/(nSamples-1))**0.5

def nSamples(y, deltaY, confidence=0.99):
	"""
	Computes the number of Monte Carlo samples needed for obtaining a 
	yield estimate that is within +-*deltaY* of *y* with confidence 
	level given by *confidence*. 
	"""
	k_gamma=erfinv(confidence)*2**0.5
	return ceil(y*(1-y)*k_gamma**2/deltaY**2)

# Measurements module 
"""
.. inheritance-diagram:: pyopus.evaluator.measure
    :parts: 1
	
**Performance measure extraction module**

All functions in this module do one of the following things:

* return an object of a Python numeric type (int, float, or complex)
* return an n-dimensional array (where n can be 0) of int, float or 
  complex type
* return None to indicate a failure
* raise an exception to indicate a more severe failure

All signals x(t) are defined in tabular form which means that a signal is 
fully defined with two 1-dimensional arrays of values of the same 
size. One array is the **scale** which represents the values of the scale (t) 
while the other column represents the values of the **signal** (x) corresponding 
to the scale points. 
"""

from numpy import array, floor, ceil, where, hstack, unique, abs, arctan, pi, NaN, log10
from scipy import angle, unwrap
# import matplotlib.pyplot as plt

__all__ = [ 'Deg2Rad', 'Rad2Deg', 'dB2gain', 'gain2dB', 'XatI', 'IatXval', 'filterI', 'XatIrange', 
			'dYdI', 'dYdX', 'integYdX', 'DCgain', 'DCswingAtGain', 'ACcircle', 'ACtf', 'ACmag', 
			'ACphase', 'ACgain' ,'ACbandwidth', 'ACphaseMargin', 'ACgainMargin', 
			'Tdelay', 'Tshoot', 'Tovershoot', 'Tundershoot', 'TedgeTime', 'TriseTime', 'TfallTime', 
			'TslewRate', 'TsettlingTime', 'Poverdrive' ]

#------------------------------------------------------------------------------
# Conversions

def Deg2Rad(degrees):
	"""
	Converts degrees to radians. 
	"""
	return degrees/180.0*pi

	
def Rad2Deg(radians):
	"""
	Converts radians to degrees. 
	"""
	return radians*180.0/pi

	
def dB2gain(db, unit='db20'):
	"""
	Converts gain magnitude in decibels to gain as a factor. 
	
	*unit* is the type of decibels to convert from:  
	
	* ``db`` and ``db20`` are equivalent and should be used when the conversion 
	  of voltage/current gain decibels is required (20dB = gain factor of 10.0). 
	* ``db10`` should be used for power gain decibels conversion 
	  (10dB = gain factor of 10.0)
	"""
	if unit=='db':
		return 10.0**(db/20.0)
	elif unit=='db20':
		return 10.0**(db/20.0)
	elif unit=='db10':
		return 10.0**(db/10.0)
	else:
		raise Exception, "Bad magnitude unit."

		
def gain2dB(x, unit='db20'):
	"""
	Converts gain as a factor to magnitude in decibels. 
	
	*unit* is the type of decibels to convert to:  
	
	* ``db`` and ``db20`` are equivalent and should be used when the conversion 
	  to voltage/current gain decibels is required (gain factor of 10.0 = 20dB). 
	* ``db10`` should be used for conversion to power gain decibels conversion 
	  (gain factor of 10.0 = 10dB)
	"""
	if unit=='db':
		return 20.0*log10(x)
	elif unit=='db20':
		return 20.0*log10(x)
	elif unit=='db10':
		return 10.0*log10(x)
	else:
		raise Exception, "Bad magnitude unit."

		
#------------------------------------------------------------------------------
# Fractional indexing and cursors

def XatI(x, i):
	"""
	Returns the value in 1-dimensional array *x* corresponding to fractional 
	index *i*. This operation is equivalent to a table lookup with linear 
	interpolation where the first column of the table represents the index (*i) 
	and the second column holds the components of *x*. 
	
	If *i* is a 1-dimensional array the return value is an array of the same 
	shape holding the results of table lookups corresponding to fractional 
	indices in array *i*. 
	
	*i* must satisfy 0 <= *i* <= x.size-1. 
	"""
	xLen=x.size
	
	xa=array(x)
	ia=array(i)
	
	if ia.size==0:
		return array([])
	
	if array(i<0).any() or array(i>xLen-1).any():
		raise Exception, "Index out of range."
	
	# Interpolate
	i1=floor(ia).astype(int)
	i2=ceil(ia).astype(int)
	frac=ia-i1
	xa1=xa[i1]
	xa2=xa[i2]
	
	return xa1+(xa2-xa1)*frac


def IatXval(x, val, slope='any'):
	"""
	Returns a 1-dimensional array of fractional indices corresponding places in 
	vector *x* where the linearly interpolated value is equal to *val*. These 
	are the crossings of function f(i)=x(i) with g(i)=val. *slope* specifies 
	what kind of crossings are returned:
	
	* ``any`` - return all crossings
	* ``rising`` - return only crossings where the slope of f(i) is 
	  positive or zero
	* ``falling`` - return only crossings where the slope of f(i) is 
	  negative or zero
	
	This function corresponds to a reverse table lookup with linear 
	interpolation where the first folumn of the table contains the index and 
	the second column contains the corresponding values of *x*. The reverse 
	lookup finds the fractional indices coresponding to *val* in the second 
	column of the table. 
	
	There can be no crossings (empty return array) or more than one crossing 
	(return array size>1). 
	
	The fractional indices are returned in an increasing sequence. 
	"""
	xa=array(x)
	val=array(val)
	
	if val.size!=1:
		raise Exception, "Value must be a scalar."
	
	if (val<xa.min()) or (val>xa.max()):
		return array([], float)
	
	# Detect level corssing
	belowOrEqual=(xa<=val)
	aboveOrEqual=(xa>=val)
	
	# Detect edges
	risingEdge=belowOrEqual[:-1] & aboveOrEqual[1:]
	fallingEdge=aboveOrEqual[:-1] & belowOrEqual[1:]
	anyEdge=risingEdge | fallingEdge
	
	if slope=='rising':
		edge=risingEdge
	elif slope=='falling':
		edge=fallingEdge
	elif slope=='any':
		edge=anyEdge
	else:
		raise Exception, "Bad edge type."
	
	# Get candidate intervals
	candidates=where(edge)[0]
	
	# Prepare interval edges and deltas
	i1=candidates
	i2=candidates+array(1)
	x1=xa[i1]
	x2=xa[i2]
	dx=x2-x1
	
	# Zero delta interval indices
	zeroDeltaI=where(dx==0)[0]
	nonzeroDeltaI=where(dx!=0)[0]
	
	# Handle zero delta intervals
	ii=hstack((i1[zeroDeltaI], i2[zeroDeltaI]))
	
	# Handle nonzero delta intervals
	ii=hstack((ii, 1.0/dx[nonzeroDeltaI]*(val-x1[nonzeroDeltaI])+i1[nonzeroDeltaI]))
	
	return unique(ii)
	

def filterI(i, direction='right', start=None, includeStart=False):
	"""
	Returns a 1-dimensional array of fractional indices obtained by processing 
	fractional indices given in *i*. 
	
	If *direction* is ``right`` *i* is traversed from lower to higher indices. 
	Only the indices from *i* that are greater than *start* are included in the 
	return value.
	
	If *direction* is ``left`` *i* is traversed from higher to lower indices. 
	Only the indices from *i* that are less than *start* are included in the 
	return value. 
	
	The filtered indices are returned in the same order as they appear in *i*. 
	
	If *includeStart* is ``True`` the *greater than* and *less than* comparison 
	operators are replaced by *greater than or equal to* and *less than or 
	equal to*. This includes indices which are equal to *start* in the return 
	value. 
	
	If *start* is not given, it defaults to ``i[0]`` if *direction* is 
	``right`` and ``i[-1]`` if *direction* is ``left``. 
	"""
	if start is None:
		if direction=='right':
			if i.size>0:
				start=i[0]
			else:
				start=0
		elif direction=='left':
			if i.size>0:
				start=i[-1]
			else:
				start=0
		else:
			raise Exception, "Bad direction."
	
	if direction=='right':
		if includeStart:
			selector=i>=start
		else:
			selector=i>start
	elif direction=='left':
		if includeStart:
			selector=i<=start
		else:
			selector=i<start
	else:
		raise Exception, "Bad direction."
	
	return i[where(selector)[0]]


def XatIrange(x, i1, i2=None):
	"""
	Returns a subvector (1-dimensional array) of a vector given by 
	1-dimensional array *x*. The endpoints of the subvector correspond to 
	fractional indices *i1* and *i2*. 
	
	If *i2* is not given the return value is the same as the return value of 
	``XatI(x, i1)``. 
	
	*i1* and *i2* must satisfy 
	
	* 0 <= *i1* <= x.size-1
	* 0 <= *i2* <= x.size-1
	* *i1* <= *i2*
	
	If the endpoints do not correspond to integer indices the subvector 
	endpoints are obtained with linear interpolation (see :func:`XatI` 
	function). 
	"""
	xLen=x.size
	
	if (i1<0) or (i1>xLen-1):
		raise Exception, "Bad fractional index (i1)."
	
	if i2 is None:
		return XatI(x, i1)
	
	if (i2<i1) or (i2<0) or (i2>xLen-1):
		raise Exception, "Bad fractional index range."
	
	if i1==i2:
		return XatI(x, i1)
	
	# Get integer subrange
	ilo=ceil(i1).astype(int)
	ihi=floor(i2).astype(int)
	
	# Get core vector for the integer subrange
	if ilo<=ihi:
		coreVector=x[ilo:(ihi+1)]
	else:
		coreVector=[]
	
	# Construct result
	if i1!=ilo:
		retval=XatI(x, i1)
	else:
		retval=[]
	
	retval=hstack((retval, coreVector))
	if i2!=ihi:
		retval=hstack((retval, XatI(x, i2)))
	
	# Done
	return retval


#------------------------------------------------------------------------------
# Calculus

# Derivative wrt integer index
def dYdI(y):
	"""
	Returns the derivative of 1-dimensional vector *y* with respect to its 
	index. 

	Uses 2nd order polynomial interpolation before the actual derivative is 
	calculated. 
	"""
	# Interpolating polynomial of 2nd order
	#   indices used	for
	#   0, 1, 2 		0 and 1
	#   1, 2, 3 		2
	#   2, 3, 4 		3
	#   ...				...
	#   n-4, n-3, n-2	n-3
	#   n-3, n-2, n-1 	n-2 and n-1
	
	# y=a x^2 + b x + c
	# dy/dx = 2 a x + b
	
	# There are n-2 interpolating polynomials, get their coefficients
	yminus=array(y[:-2])
	y0=array(y[1:-1])
	yplus=array(y[2:])
	
	c=y0
	b=(yplus-yminus)/2.0
	a=(yplus+yminus-2.0*y0)/2.0
	
	# Now differentiate polynomial
	c=b
	b=2.0*a
	a=0
	
	# Generate edge points
	# For i=0 (effective index x=-1)
	dylo=-b[0]+c[0]
	# For i=n-1 (effective index x=1)
	dyhi=b[-1]+c[-1]
	# For everything else (effective index x=0)
	coreVector=c
	
	# Done
	return hstack((dylo, coreVector, dyhi))


def dYdX(y, x):
	"""
	Derivative of a 1-dimensional vector *y* with respect to the 1-dimensional 
	vector *x*. The arrays representing *y* and *x* must be of the same size. 
	"""
	return dYdI(y)/dYdI(x)


# Integrate vector wrt scale
# Returns an array of values. Each value is an integral from the beginning to belonging x component.
def integYdX(y, x):
	"""
	Integral of a 1-dimensional vector *x* with respect to its scale given by a 
	1-dimensional vector *x*. The arrays representing *y* and *x* must be of 
	the same size. 
	
	Uses 2nd order polynomial interpolation before the actual integral is 
	calculated. 
	
	The lower limit for integration is ``x[0]`` while the pints in *x* define 
	the upper limits. This means that the first point of the result (the one 
	corresponding to ``x[0]``) is 0. 
	"""
	# Interpolating polynomial of 2nd order
	#   indices used	for
	#   0, 1, 2 		0 and 1
	#   1, 2, 3 		2
	#   2, 3, 4 		3
	#   ...				...
	#   n-4, n-3, n-2	n-3
	#   n-3, n-2, n-1 	n-2 and n-1
	
	# y=a x^2 + b x + c
	# dy/dx = 2 a x + b
	
	# There are n-2 interpolating polynomials, get their coefficients
	hminus=array(x[:-2]-x[1:-1])
	hplus=array(x[2:]-x[1:-1])
	
	yminus=array(y[:-2])
	y0=array(y[1:-1])
	yplus=array(y[2:])
	
	c=y0
	a=(yminus*hplus-yplus*hminus-c*(hplus-hminus))/(hplus*hminus*(hminus-hplus))
	b=(yminus-c-a*hminus*hminus)/hminus
	
	# Integrate polynomial (resulting in a x^3 + b x^2 + c x + C)
	# Constant C is ignored.
	a=a/3.0
	b=b/2.0
	
	# Calculate integral for last interval based on last interpolation
	#   (corresponding to 0..hplus)
	ydxLast=a[-1]*(hplus[-1]**3)+b[-1]*(hplus[-1]**2)+c[-1]*hplus[-1]
	
	# Calculate integral for first interval based on first interpolation
	#   (corresponding to hminus..0)
	ydxFirst=-(a[0]*(hminus[0]**3)+b[0]*(hminus[0]**2)+c[0]*hminus[0])
	
	# Calculate core integral - leading part
	#   values of integral for i..i+1 (corresponding to hminus..0)
	coreVectorLeading=-(a*(hminus**3)+b*(hminus**2)+c*hminus)
	
	# Calculate core integral - trailing part
	#   values of integral for i..i+1 (corresponding to 0..hplus)
	coreVectorTrailing=a*(hplus**3)+b*(hplus**2)+c*hplus
	
	# With zero, leading core vector, and ydxLast do a cumulative sum
	integLeading=hstack((array(0.0), coreVectorLeading, ydxLast)).cumsum()
	
	# With zero, ydxFirst, and trailing core vector do a cumulative sum
	integTrailing=hstack((array(0.0), ydxFirst, coreVectorTrailing)).cumsum()
	
	# Done
	return (integLeading+integTrailing)/2.0


#------------------------------------------------------------------------------	
# DC measurements

def DCgain(output, input):
	"""
	Returns the maximal gain (slope) of a nonlinear transfer function 
	*output(input*). 
	
	*output* and *input* are 1-dimensional arrays of the same size. 
	"""
	# Get gain
	A=dYdX(output, input)
	
	# Return maximum
	return A.max()


def DCswingAtGain(output, input, relLevel, type='out'):
	"""
	Returns the *input* or *output* interval corresponding to the range where 
	the gain (slope) of *output(input)* is above *relLevel* times maximal 
	slope. Only *rellevel* < 1 makes sense in this measurement. 
	
	*type* specifies what to return
	
	* ``out`` - return the *output* values interval 
	* ``in`` - return the *input* values interval
	
	*relLevel* must satisfy 0 <= *relLevel* <= 1. 
	"""
	# Check
	if (relLevel<=0) or (relLevel>=1):
		raise Exception, "Bad relative level."
	
	# Get gain (absolute)
	A=abs(dYdX(output, input))
	
	# Get maximum and level
	Amax=A.max()
	Alev=Amax*relLevel
	
	# Find maximum
	maxI=IatXval(A, Amax)
	
	# Find crossings
	crossI=IatXval(A, Alev)
	
	if crossI.size<=0:
		raise Exception, "No crossings with specified level found."
	
	# Extract crossings to left and to right
	Ileft=filterI(crossI, 'left', maxI.min())
	Iright=filterI(crossI, 'right', maxI.max())
	
	if Ileft.size<=0:
		raise Exception, "No crossing to the left from the maximum found."
	
	if Iright.size<=0:
		raise Exception, "No crossing to the right from the maximum found."
	
	# max(), min() will raise an exception if no crossing is found
	i1=Ileft.max()
	i2=Iright.min()
	
	# Get corresponding range
	if type=='out':
		vec=output
	elif type=='in':
		vec=input
	else:
		raise Exception, "Bad output type."

	return abs(XatI(vec, i2)-XatI(vec, i1))


#------------------------------------------------------------------------------
# AC measurements

def ACcircle(unit='deg'):
	"""
	Returns the full circle in units specified by *unit*
	
	* ``deg`` - return 360
	* ``rad`` - return 2*``pi``
	"""
	if unit=='deg':
		return 360
	elif unit=='rad':
		return 2*pi
	else:
		raise Exception, "Bad angle unit."

		
def ACtf(output, input):
	"""
	Return the transfer function *output/input* where *output* and *input* are 
	complex vectors of the same size representing the systems response at 
	various frequencies. 
	"""
	return array(output)/array(input)

	
def ACmag(tf, unit='db'):
	"""
	Return the magnitude in desired *unit* of a small signal tranfer function 
	*tf*. 
	
	* ``db`` and ``db20`` stand for voltage/current gain decibels where 
	  20dB = gain factor of 10.0 
	* ``db10`` stands for voltage/current gain decibels where 
	  10dB = gain factor of 10.0 
	* ``abs`` stands for gain factor
	"""
	mag=abs(array(tf))
	if (unit=='db') or (unit=='db20'):
		return 20*log10(mag)
	elif unit=='db10':
		return 10*log10(mag)
	elif unit=='abs':
		return mag
	else:
		raise Exception, "Bad magnitude unit."


def ACphase(tf, unit='deg', unwrapTol=0.5):
	"""
	Return the phase in desired *unit* of a transfer function *tf*
	
	* ``deg`` stands for degrees
	* ``rad`` stands for radians
	
	The phase is unwrapped (discontinuities are stiched together to make it 
	continuous). The tolerance of the unwrapping (in radians) is 
	*unwrapTol* times ``pi``. 
	"""
	# Get argument
	ph=angle(tf)
	
	# Unwrap if requested
	if (unwrapTol>0) and (unwrapTol<1):
		ph=unwrap(ph, unwrapTol*pi)
	
	# Convert to requested unit
	if unit=='deg':
		return ph/pi*180.0
	elif unit=='rad':
		return ph
	else:
		raise Exception, "Bad phase unit."

		
def ACgain(tf, unit='db'):
	"""
	Returns the maximal gain magnitude of a transfer function in units given 
	by *unit*
	
	* ``db`` and ``db20`` stand for voltage/current gain decibels where 
	  20dB = gain factor of 10.0
	* ``db10`` stands for power gain decibels where 
	  10dB = gain factor of 10.0
	* ``abs`` stands for gain factor
	"""
	mag=ACmag(tf, unit)
	return mag.max()


def ACbandwidth(tf, scl, filter='lp', levelType='db', level=-3.0):
	"""
	Return the bandwidth of a transfer function *tf* on frequency scale *scl*. 
	*tf* and *scl* must be 1-dimensional arrays of the same size. 
	
	The type of the transfer function is given by *filter* where 
	
	* ``lp`` stands for low-pass (return frequency at *level*)
	* ``hp`` stands for high-pass (return frequency at *level*)
	* ``bp`` stands for band-pass (return bandwidth at *level*)
	
	*levelType* gives the units for the *level* argument. Allowed values for 
	*levelType* are 
	
	* ``db`` and ``db20`` stand for voltage/current gain decibels where 
	  20dB = gain factor of 10.0
	* ``db10`` stands for power gain decibels where 
	  10dB = gain factor of 10.0
	* ``abs`` stands for gain factor

	*level* specifies the level at which the bandwidth should be measured. For 
	``db``, ``db10``, and ``db20`` *levelType* the level is relative to the 
	maximal gain and is added to the maximal gain. For ``abs`` *levelType* the 
	level is a factor with which the maximal gain factor must be multiplied to 
	obtain the gain factor level at which the bandwidth should be measured. 
	"""
	# Magnitude
	mag=ACmag(tf, levelType)
	
	# Reference level
	ref=mag.max()
	
	# Crossing level
	if levelType=='abs':
		cross=ref*level
	else:
		cross=ref+level
	
	# Find crossings
	crossI=IatXval(mag, cross)
	
	# Find reference position
	crossMaxI=IatXval(mag, ref).min()
	
	if crossI.size<=0:
		raise Exception, "No crossings with specified level found."
		
	# Make scale real
	scl=abs(scl)
	
	# Handle filter type
	if filter=='lp':
		# Get first crossing to the right of the reference position
		# min() will raise an exception if no crossing is found
		bwI=filterI(crossI, 'right', crossMaxI)
		
		if bwI.size<=0:
			raise Exception, "No crossing to the right from the maximum found."
		
		bwI=bwI.min()
		
		bw=XatI(scl, bwI)
	elif filter=='hp':
		# Get first crossing to the left of the reference position
		# max() will raise an exception if no crossing is found
		bwI=filterI(crossI, 'left', crossMaxI).max()
		
		if bwI.size<=0:
			raise Exception, "No crossing to the left from the maximum found."
		
		bwI=bwI.max()
			
		bw=XatI(scl, bwI)
	elif filter=='bp':
		# Get first crossing to the left and the right of the reference position
		# max(), min() will raise an exception if no crossing is found
		bwI1=filterI(crossI, 'left', crossMaxI).max()
		bwI2=filterI(crossI, 'right', crossMaxI).min()
		
		if bwI1.size<=0:
			raise Exception, "No crossing to the left from the maximum found."
		
		if bwI2.size<=0:
			raise Exception, "No crossing to the right from the maximum found."
		
		bwI1=bwI1.max()
		bwI2=bwI2.man()
		
		bw=XatI(scl, bwI2)-XatI(scl, bwI1)
	else:
		raise Exception, "Bad filter type."
	
	return bw


def ACugbw(tf, scl):
	"""
	Returns the uniti-gain bandwidth of a transfer function *tf* on frequency 
	scale *scl*. 1-dimensional arrays *tf* and *scl* must be of the same size. 
	
	The return value is the frequency at which the transfer function 
	reaches 1.0 (0dB). 
	"""
	# Magnitude
	mag=ACmag(tf, 'db')
	
	# Make scale real
	scl=abs(scl)
	
	# Find 0dB magnitude
	# min() will raise an exception if no crossing is found
	crossI=IatXval(mag, 0.0)
	
	if crossI.size<=0:
		raise Exception, "No crossing with 0dB level found."
		
	crossI=crossI.min()
	
	# Calculate ugbw
	ugbw=XatI(scl, crossI)
	
	return ugbw


def ACphaseMargin(tf, unit='deg', unwrapTol=0.5):
	"""
	Returns the phase margin of a transfer function given by 1-dimensional 
	array *tf*. Uses *unwrapTol* as the unwrap tolerance for phase 
	(see :func:`ACphase`). The phase margin is returned in units given by 
	*unit* where
	
	* ``deg`` stands for degrees
	* ``rad`` stands for radians
	
	The phase margin (in degrees) is the amount the phase at the point where 
	the transfer function magnitude reaches 0dB should be decreased to become 
	equal to -180. 
	
	For stable systems the phase margin is >0. 
	"""
	# Magnitude
	mag=ACmag(tf, 'db')
	
	# Phase
	ph=ACphase(tf, unit, unwrapTol)
	
	# Find 0dB magnitude
	crossI=IatXval(mag, 0.0)
	
	if crossI.size<=0:
		raise Exception, "No crossing with 0dB level found."
		
	crossI=crossI.min()
	
	# Calculate phase at 0dB
	ph0=XatI(ph, crossI)
	
	# Return phase margin
	pm=ph0+ACcircle(unit)/2
	
	return pm

# Gain margin of a tf
def ACgainMargin(tf, unit='db', unwrapTol=0.5):
	"""
	Returns the gain margin of a transfer function given by 1-dimensional array 
	*tf*. Uses *unwrapTol* as the unwrap tolerance for phase 
	(see :func:`ACphase`). The gain margin is returned in units given by *unit* 
	where
	
	* ``db`` and ``db20`` stand for voltage/current gain decibels where 
	  20dB = gain factor of 10.0
	* ``db10`` stands for power gain decibels where 
	  10dB = gain factor of 10.0
	* ``abs`` stands for gain factor
	
	The phase margin (in voltage/current gain decibels) is the amount the gain 
	at the point where phase reaches -180 degrees should be increased to become 
	equal to 0. 
	
	For stable systems the gain margin is >0. 
	"""
	# Magnitude
	mag=ACmag(tf, 'abs')
	
	# Phase
	ph=ACphase(tf, 'deg', unwrapTol)
	
	# Find -180deg in phase
	crossI=IatXval(ph, -180.0)
	
	if crossI.size<=0:
		raise Exception, "No crossing with -180 degrees level found."
	
	crossI=crossI.min()
	
	# Get gain at -180 degrees
	mag180=XatI(mag, crossI)
	
	# Gain margin in absolute units
	gm=1.0/mag180
	
	# Return selected units
	return ACmag(gm, unit)

	
#------------------------------------------------------------------------------
# Transient measurements

def _refLevel(sig, scl, t1=None, t2=None):
	"""
	In signal *sig* with scale *scl* looks up the points where scale is equal 
	to *t1* and *t2*. The default values of *t1* and *t2* are the first and the 
	last value in *scl. 
	
	Returns a tuple (i1, s1, i2, s2) where i1 and i2 represent the fractional 
	indices of the two points in signal (or scale) while s1 and s2 represent 
	the values of the signal at those two points. 
	"""
	# Get interesting part in terms of indices
	if t1 is None:
		i1=0
	else:
		i1=IatXval(scl, t1, 'rising')
		if i1.size<=0:
			raise Exception, "Start point not found."
		i1=i1[0]
	
	if t2 is None:
		i2=scl.size-1
	else:
		i2=IatXval(scl, t2, 'rising')
		if i2.size<=0:
			raise Exception, "End point not found."
		i2=i2[0]
			
	if i1>=i2:
		raise Exception, "Start point after end point."
	
	# Get reference levels
	s1=XatI(sig, i1)
	s2=XatI(sig, i2)
	
	return (i1, s1, i2, s2)


def Tdelay(sig1, sig2, scl, 
			lev1type='rel', lev1=0.5, edge1='any', skip1=0, 
			lev2type='rel', lev2=0.5, edge2='any', skip2=0, 
			t1=None, t2=None):
	"""
	Calculates the delay of signal *sig2* with respect to signal *sig1*. Both 
	signals share a common scale *scl*. The delay is the difference in scale 
	between the point where *sig2* reaches level *lev2*. *edge2* defines the 
	type of crossing between *sig2* and *lev2* 
	
	* ``rising`` - the slope of *sig2* is positive or zero at the crossing
	* ``falling`` - the slope of *sig2* is negative or zero at the crossing
	* ``any`` - the slope of *sig2* does not matter
	
	*skip2* specifies how many crossings since the beginning of *sig2* are 
	skipped before the crossing that is used as the point in *sig2* is reached. 
	0 means that the first crossing is used as the point in *sig2*. 
	
	Similarly the point in *sig1* is defined with *lev1*, *edge1*, and *skip1*. 
	
	*t1* and *t2* are the points on the scale defining the beginning and the 
	end of the part of *sig1* and *sig2* which is used in the calculation of 
	the delay. *skip1* and *skip2* are counted from point *t1* on the scale. 
	The default values of *t1* and *t2* are the first and the last value in 
	*scl*. 
	
	If *lev1type* is ``abs`` *lev1* specifies the value of the signal at the 
	crossing. If *lev1type* is ``rel`` *lev1* specifies the relative value of 
	the signal (between 0.0 and 1.0) where the 0.0 level is defined as the 
	*sig1* level at point *t1* on the scale while the 1.0 level is defined as 
	the *sig1* level at point *t2* on the scale. If *t1* and *t2* are not given 
	the 0.0 and 1.0 relative levels are taken at the beginning and the end of 
	*sig2*. 
	
	Similarly *lev2type* defines the meaning of *lev2* with respect to *sig2*, 
	*t1*, and *t2*. 
	"""
	# Get reference levels of sig1
	(i1, s11, i2, s12)=_refLevel(sig1, scl, t1, t2)
	
	# Get reference levels of sig2
	s21=XatI(sig2, i1)
	s22=XatI(sig2, i2)
	
	# Extract interesting part
	partsig1=XatIrange(sig1, i1, i2)
	partsig2=XatIrange(sig2, i1, i2)
	partscl=XatIrange(scl, i1, i2)
	
	# Get level crossing for signal 1
	if lev1type=='abs':
		crossI1=IatXval(partsig1, lev1, edge1)
	elif lev1type=='rel':
		crossI1=IatXval(partsig1, s11+(s12-s11)*lev1, edge1)
	else:
		raise Exception, "Bad level type for first signal."
	
	if skip1>=crossI1.size:
		raise Exception, "No such crossing for first signal."
	
	# Get level crossing for signal 2
	if lev2type=='abs':
		crossI2=IatXval(partsig2, lev2, edge2)
	elif lev2type=='rel':
		crossI2=IatXval(partsig2, s21+(s22-s21)*lev2, edge2)
	else:
		raise Exception, "Bad level type for first signal."
	
	if skip2>=crossI2.size:
		raise Exception, "No such crossing for second signal."
	
	crossI1=crossI1[skip1]
	crossI2=crossI2[skip2]
	
	delay=XatI(partscl, crossI2)-XatI(partscl, crossI1)
	
	return delay


def Tshoot(measureType, sig, scl, 
			t1=None, t2=None, outputType='rel'): 
	"""
	Gets the overshoot or the undershoot of signal *sig* with scale *scl*. The 
	over/undershoot is measured on the scale interval between *t1* and *t2*. If 
	*t1* and *t2* are not given the whole signal *sig1* is used in the 
	measurement. 
	
	The 0.0 and 1.0 relative levels in the signal are defined as the values of 
	*sig* at points *t1* and *t2* on the scale. The default values of *t1* and 
	*t2* are the first and the last value in *scl*. 
	
	Overshoot is the amount the signal rises above the 1.0 relative level on 
	the observed scale interval defined by *t1* and *t2*. Undershoot is the 
	amount the signal falls below the 0.0 relative level on the observed scale 
	interval. 
	
	If *measureType* is set to ``over``, overshoot is measured and the function 
	expects the signal level at *t1* to be lower than the signal level at *t2*. 
	If *measureType* is ``under`` the opposite must hold. 
	
	Over/undershoot can be measured as relative (when *outputType* is ``rel``) 
	or absolute (when *outputType* is ``abs``). Abolute values reflect actual 
	signal values while relative values are measured with respect to the 0.0 
	and 1.0 relative signal level. 
	"""
	# Get reference levels
	(i1, s1, i2, s2)=_refLevel(sig, scl, t1, t2)
	
	# Extract interesting part
	partsig=XatIrange(sig, i1, i2)
	partscl=XatIrange(scl, i1, i2)
	
	if measureType=='over':
		# Overshoot
		if s1<=s2:
			delta=partsig.max()-s2
		else:
			delta=0
	elif measureType=='under':
		# Undershoot
		if s1>=s2:
			delta=s2-partsig.min()
		else:
			delta=0
	else:
		raise Exception, "Bad measurement type."
	
	if outputType=='abs':
		return delta
	elif outputType=='rel':
		span=abs(s2-s1)
		if span==0.0:
			raise Exception, "Can't get relative value on flat signal."
		return delta/span
	else:
		raise Exception, "Bad output type."

		
def Tovershoot(sig, scl, 
				t1=None, t2=None, outputType='rel'):
	"""
	An alias for :func:`Tshoot` with *measureType* set to ``over``. 
	"""
	return Tshoot('over', sig, scl, t1, t2, outputType);

	
def Tundershoot(sig, scl, 
				t1=None, t2=None, outputType='rel'):
	"""
	An alias for :func:`Tshoot` with *measureType* set to ``under``.  
	"""
	return Tshoot('under', sig, scl, t1, t2, outputType);


def TedgeTime(edgeType, sig, scl, 
				lev1type='rel', lev1=0.1, 
				lev2type='rel', lev2=0.9, 
				t1=None, t2=None): 
	"""
	Measures rise or fall time (scale interval) of signal *sig* on scale *scl*. 
	The value of the *edgeType* parameter determines the type of the 
	measurement 
	
	* ``rising`` - measures rise time
	* ``falling`` - measures fall time
	
	*t1* and *t2* specify the scale interval on which the measurement takes 
	place. Their default values correspond to the first and the last value in 
	*scl. The values of the signal at *t1* and *t2* define the 0.0 and the 1.0 
	relative signal value. 
	
	*lev1type* and *lev* specify the point at which the signal rise/fall 
	begins. If *lev1type* is ``abs`` the level specified by *lev1* is the 
	actual signal value. If *lev1type* is ``rel`` the value given by *lev1* is 
	a relative signal value. 
	
	Similarly *lev2type* and *lev2* apply to the point at which the signal 
	rise/fall ends. 
	
	*lev1type*, *lev1*, *lev2type*, and *lev2* are by default set to measure 
	the 10%..90% rise/fall time. 
	"""	
	# Get reference levels
	(i1, s1, i2, s2)=_refLevel(sig, scl, t1, t2)

	# Extract interesting part
	partsig=XatIrange(sig, i1, i2)
	partscl=XatIrange(scl, i1, i2)
	
	# Get crossing levels
	if lev1type=='abs':
		sc1=lev1
	elif lev1type=='rel':
		sc1=s1+(s2-s1)*lev1
	else:
		raise Exception, "Bad level type for first point."
	
	if lev2type=='abs':
		sc2=lev2
	elif lev1type=='rel':
		sc2=s1+(s2-s1)*lev2
	else:
		raise Exception, "Bad level type for second point."
	
	# Get level crossings
	crossI1=IatXval(partsig, sc1, edgeType)
	
	if crossI1.size<=0:
		raise Exception, "First point not found."
	
	# Use first crossing
	crossI1=crossI1.min()
	
	crossI2=IatXval(partsig, sc2, edgeType)
	# Expect second point after first point
	crossI2=filterI(crossI2, 'right', crossI1, includeStart=True)

	if crossI2.size<=0:
		raise Exception, "Second point not found."
	
	# Use first crossing that remains unfiltered
	crossI2=crossI2.min()
	
	# Get crossing times
	delta=XatI(partscl, crossI2)-XatI(partscl, crossI1)
	
	return delta


def TriseTime(sig, scl, 
				lev1type='rel', lev1=0.1, 
				lev2type='rel', lev2=0.9, 
				t1=None, t2=None): 
	"""
	An alias for :func:`TedgeTime` with *edgeType* set to ``rising``. 
	"""
	return TedgeTime('rising', sig, scl, lev1type, lev1, lev2type, lev2, t1, t2)


def TfallTime(sig, scl, 
				lev1type='rel', lev1=0.1, 
				lev2type='rel', lev2=0.9, 
				t1=None, t2=None): 
	"""
	An alias for :func:`TedgeTime` with *edgeType* set to ``falling``. 
	"""
	return TedgeTime('falling', sig, scl, lev1type, lev1, lev2type, lev2, t1, t2)

	
def TslewRate(edgeType, sig, scl, 
				lev1type='rel', lev1=0.1, 
				lev2type='rel', lev2=0.9, 
				t1=None, t2=None): 
	""" 
	Measures the slew rate of a signal. The slew rate is defined as the 
	quotient dx/dt where dx denotes the signal difference between the beginning 
	and the end of signal's rise/fall, while dt denotes the rise/fall time. 
	Slew rate is always positive.
	
	See :func:`TedgeTime` for the explanation of the function's parameters. 
	"""
	# Get reference levels
	(i1, s1, i2, s2)=_refLevel(sig, scl, t1, t2)
	if lev1type=='abs':
		sl1=lev1
	else:
		sl1=s1+(s2-s1)*lev1
	if lev2type=='abs':
		sl2=lev2
	else:
		sl2=s1+(s2-s1)*lev2
	sigDelta=sl2-sl1
	
	# Get edge time
	dt=TedgeTime(edgeType, sig, scl, lev1type, lev1, lev2type, lev2, t1, t2)
	
	if dt==0:
		raise Exception, "Can't evaluate slew rate if edge time is zero." 
	
	# Get slew rate
	return abs(sigDelta)/dt


def TsettlingTime(sig, scl, 
					tolType='rel', tol=0.05, 
					t1=None, t2=None):
	"""
	Measures the time (scale interval on scale *scl*) in which signal *sig* 
	settles within some prescribed tolerance of its final value. 
	
	*t1* and *t2* define the scale interval within which the settling time is 
	measured. The default values of *t1* and *t2* are the first and the last 
	value of *scl*. The final signal value if the value of the signal 
	corresponding to point *t2* on the scale. 
	
	The 0.0 and the 1.0 relative signal levels are defined as signal levels at 
	points *t1* and *t2* on the scale. 
	
	If *tolType* is ``abs`` the settling time is measured from *t1* to the 
	point at which the signal remains within *tol* of its final value at *t2*. 
	
	If *tolType* is ``rel`` the settling tolerance is defined as *tol* times 
	the difference between the signal levels corresponding to the 0.0 and 1.0 
	relative signal level. 
	"""
	# Get reference levels
	(i1, s1, i2, s2)=_refLevel(sig, scl, t1, t2)
	sigDelta=s2-s1
	
	# Extract interesting part
	partsig=XatIrange(sig, i1, i2)
	partscl=XatIrange(scl, i1, i2)
	
	# Get tolerance level
	if tolType=='abs':
		tolInt=tol
	elif tolType=='rel':
		tolInt=tol*abs(sigDelta)
	else:
		raise Exception, "Bad tolerance type."
	
	# Get absolute deviation from settled value
	sigdev=abs(partsig-s2)
	
	# Find crossing of absolute deviation with tolerance level
	crossI=IatXval(sigdev, tolInt, 'any')
	
	if crossI.size<=0:
		raise Exception, "No crossing with tolerance level found."
	
	# Take last crossing
	cross=crossI[-1]
	
	# Get settling time
	delta=XatI(partscl, cross)-partscl[0]
	
	if delta<0:
		raise Exception, "This is weird. Negative settling time."
	
	return delta

	
#------------------------------------------------------------------------------
# Overdrive calculation (generic)
#   e.g. Vgs-Vth

class Poverdrive:
	"""
	Calculates the difference between the values obtained from two driver 
	functions. 
	
	Objects of this class are callable. The calling convention is 
	``object(name)``. When called it returns the difference between the values 
	returned by a call to *driver1* with arguments (name, p1) and the value 
	returned by a call to *driver2* with arguments (name, p2). The difference 
	is returned as an array. If the size of the array is 1, it is returned as a 
	scalar (0-dimensional array). 
	
	:class:`Poverdrive` can be used for calculating the Vgs-Vth difference of 
	one or more MOS transistors by defining the measurement script in the 
	following way:: 
	
		obj=m.Poverdrive(p, 'vgs', p, 'vth')
		retval=map(obj, ['mn2', 'mn3', 'mn9'])
		__result=np.array(retval)
		
	
	The :func:`map` Python builtin function calls the :class:`Poverdrive` 
	object ``obj`` 3 times, once for every member of the list 
	``['mn2', 'mn3', 'mn9']`` and collects the return values in 
	a list which is then returned by ``map`` and stored in ``retval``. 
	
	A call to :class:`Poverdrive` object ``obj`` with argument ``mn2`` returns 
	the result of::
		
		p('mn2', 'vgs')-p('mn2', 'vth')
	
	which is actually the difference between the Vgs and the threshold voltage 
	of MOS transistor ``mn1``. So ``retval`` is a list holding the values of 
	the difference between Vgs and the threshold voltage of transistors listed 
	in ``['mn2', 'mn3', 'mn9']``. 
	
	Finally the list is converted to an array because the 
	:class:`~pyopus.evaluator.performance.PerformanceEvaluator` object can's 
	handle lists. 
	
	The previous measurement script could also be written as a measurement 
	expression::
	
		np.array(map(m.Poverdrive(p, 'vgs', p, 'vth'), ['mn2', 'mn3', 'mn9']))
		
	Note that during measurement evaluation 
	(when a :class:`~pyopus.evaluator.performance.PerformanceEvaluator` object 
	is called) the function :func:`p` accesses device properties calculated by 
	the simulator while the :mod:`pyopus.evaluator.measure` and :mod:`numpy` 
	modules are available as :mod:`m` and :mod:`np`. 
	"""
	def __init__(self, driver1, p1, driver2, p2): 
		self.driver1=driver1
		self.p1=p1
		self.driver2=driver2
		self.p2=p2
	
	def __call__(self, instance):
		if type(self.p1) is str:
			v1=self.driver1(instance, self.p1)
		else:
			v1=self.driver1(instance, *self.p1)
		
		if type(self.p2) is str:
			v2=self.driver2(instance, self.p2)
		else:
			v2=self.driver2(instance, *self.p2)
		
		diff=array(v1-v2)
		
		# Scalarize if diff is 1 long
		if diff.size==1:
			diff=diff[0]
		
		return diff

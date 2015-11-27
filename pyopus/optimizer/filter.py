"""
.. inheritance-diagram:: pyopus.optimizer.filter
    :parts: 1

**Filter support for constrained optimization
(PyOPUS subsystem name: FILT)**
"""

from ..misc.debug import DbgMsgOut, DbgMsg

__all__ = [ 'Filter' ]

class Filter(object):
	"""
	All points are stored in a dictionary with keys of 
	the form (f,h). 
	
	The value stored alongside (f,h) can be anything. 
	
	A point (f,h) dominates (f0,h0) if 
	
	* h<h0 and f<=f0 or 
	* h<=h0 and f<f0 
	
	(f,h) is dominated by the filter if h>hmax.
	
	No point in the filter dominates any other point in the filter. 
	
	hmax=0 results in extreme barrier behavior. 
	"""
	def __init__(self, hmax=0.0, debug=0):
		
		self.debug=debug
		self.reset(hmax)
		
	def reset(self, hmax=None):
		"""
		Resets the filter. 
		
		Points are stored in a dictionary with h as key. 
		Dictionary values are tuples of the form (f, misc). 
		There can be only one point for every h0 value. 
		"""
		self.points={}
		if hmax is not None:
			self.hmax=hmax
			
	def updateHmax(self, hmax):
		"""
		Updates hmax and purges points with h>hmax. 
		"""
		self.hmax=hmax
		toDelete=[]
		for h0,value in self.points.iteritems():
			if h0>self.hmax:
				toDelete.append(h0)
		
		for h0 in toDelete:
			del self.points[h0]
	
	def orderedHlist(self):
		"""
		Returns h values in increasing order. 
		"""
		hlist=self.points.keys()
		hlist.sort()
		return hlist
	
	def orderedPoints(self):
		"""
		Returns the f and h values ordered by increasing h. 
		"""
		hlist=self.points.keys()
		hlist.sort()
		flist=[]
		for h in hlist:
			f,dummy=self.points[h]
			flist.append(f)
		
		return flist, hlist
	
	def bestFeasible(self):
		"""
		Returns (f,h,misc) of the feasible point. 
		
		Returns (None, None, None) if no such point exists. 
		"""
		hlist=self.orderedHlist()
		if len(hlist)==0 or hlist[0]>0.0:
			return (None, None, None)
		else:
			(f0, misc0)=self.points[hlist[0]]
			return(f0, hlist[0], misc0)
	
	def leastInfeasible(self):
		"""
		Returns (f,h,misc) of the infeasible point with lowest h. 
		
		Returns (None, None, None) if no such point exists. 
		"""
		hlist=self.orderedHlist()
		if len(hlist)==0:
			# Empty filter
			return (None, None, None)
		elif hlist[0]>0.0:
			# Filter with only infeasible points
			(f0, misc0)=self.points[hlist[0]]
			return(f0, hlist[0], misc0)
		elif len(hlist)==1:
			# Only feasible point
			return (None, None, None)
		else:
			# Feasible point and infesible points
			(f0, misc0)=self.points[hlist[1]]
			return(f0, hlist[1], misc0)
		
	def mostInfeasible(self):
		"""
		Returns (f,h,misc) of the infeasible point with highest h. 
		
		Returns (None, None, None) if no such point exists. 
		"""
		hlist=self.orderedHlist()
		if len(hlist)==0:
			# Empty filter
			return (None, None, None)
		elif hlist[-1]==0.0:
			# Filter with only feasible point
			return (None, None, None)
		else:
			(f0, misc0)=self.points[hlist[-1]]
			return(f0, hlist[-1], misc0)
		
	def position(self, f, h):
		"""
		Returns the position of h in the ordered list of h values. 
		
		0 ... feasible point
		1 ... best infeasible point
		2 ... second infeasible point 
		...
		
		Returns ``None`` if the point is not in the filter. 
		"""
		hlist=self.orderedHlist()
		try:
			ii=hlist.index(h)
		except:
			# Empty filter or not found
			return None
		
		(f0, misc0)=self.points[h]
		if f!=f0:
			# Function value does not match
			return None
		
		if hlist[0]==0.0:
			# Have feasible point
			return ii
		else:
			# No feasible point in filter
			return ii+1
		
	def accept(self, f, h, misc):
		"""
		Checks a point against the filter. 
		
		If h>hmax the point does not dominate nor is dominated. 
		Such a point is rejected. 
		
		If filter is empty, a point is accepted. Such a point is 
		considered to dominate filter. 
		
		If point dominates any filter point it is accepted. 
		Dominated filter points are deleted. 
		
		If a point is dominated by any filter point it is rejected. 
		If a point does not dominate nor is dominated it is accepted. 
		
		Returns boolean tuple (dominates, dominated, accepted). 
		"""
		
		if h>self.hmax:
			if self.debug>0:
				DbgMsgOut("FILT", "h>hmax, point rejected")
			# does not dominate, not dominated, rejected
			return (False, False, False)
		
		if len(self.points)==0:
			if self.debug>0:
				DbgMsgOut("FILT", "empty filter, point accepted")
			self.points[float(h)]=(f,misc)
			# dominates, not dominated, accepted
			return (True, False, True)
		
		# Go through all filter points, check if they are dominated by (f,h) 
		# Make a list of dominated points
		dominated=[]
		for h0, value in self.points.iteritems():
			(f0, misc0)=value
			if self.dominates(f, h, f0, h0):
				dominated.append(h0)
		
		# Purge dominated points
		for h0 in dominated:
			del self.points[h0]
		
		# If at least one filter point was dominated, accept point
		if len(dominated)>0:
			self.points[float(h)]=(f,misc)
			if self.debug>0:
				DbgMsgOut("FILT", "dominates %d points, point accepted" % len(dominated))
			# dominates, not dominated, accepted
			return (True, False, True)
		
		# Check if (f,h) is dominated by any filter point
		# If a point is equal to filter point, it should not be accepted. 
		# Such a point is considered to be dominated by a filter point. 
		for h0, value in self.points.iteritems():
			(f0, misc0)=value
			if self.dominates(f0, h0, f, h) or (f==f0 and h==h0):
				if self.debug>0:
					DbgMsgOut("FILT", "dominated by filter, point rejected")
				# does not dominate, dominated, rejected
				return (False, True, False)
			
		# Got to this point so (f,h) can be accepted
		self.points[float(h)]=(f,misc)
		if self.debug>0:
			DbgMsgOut("FILT", "does not dominate nor dominated by filter, point accepted")
		# does not dominate, not dominated, accepted
		return (False, False, True)
	
	@staticmethod
	def dominates(f, h, f0, h0):
		"""
		Returns ``True`` if (f,h) dominates (f0,h0). 
		"""
		if (h<h0 and f<=f0) or (h<=h0 and f<f0):
			return True
		else:
			return False

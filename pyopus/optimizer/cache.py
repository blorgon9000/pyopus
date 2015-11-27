"""
.. inheritance-diagram:: pyopus.optimizer.cache
    :parts: 1
	
**Point caching (PyOPUS subsystem name: CACHE)**
"""

from hashlib import sha1
import numpy as np

__all__ = [ 'Cache', 'HashableWrapper' ]

class HashableWrapper(object):
	"""
	Constructs a hashable wrapper arounf a numpy array. If *copy* is ``True`` 
	a copy of the array is stored and returned. 
	"""
	def __init__(self, arr, copy=True):
		self.copy=copy
		self.arr=np.array(arr) if copy else arr
		self.hashValue=int(sha1(arr.view(np.uint8)).hexdigest(), 16)
		
	def __eq__(self, other):
		"""
		Compares two objects. 
		"""
		return (self.arr == other.arr).all()

	def __hash__(self):
		"""
		Returns a precomputed hash value.
		"""
		return self.hashValue
	
	def array(self):
		"""
		Returns the original array. 
		Returns a copy if *copy* is ``True``.
		"""
		if self.copy:
			return np.array(self.arr)
		else:
			return self.arr

class Cache(object):
	"""
	Cache for storing points. The point is the key. 
	Every point has an object associated with it. 
	"""
	def __init__(self):
		self.reset()
	
	def reset(self):
		"""
		Clears cache and resets hit count. 
		"""
		self.hits=0
		self.storage={}
	
	def getHits(self):
		"""
		Returns cache hit count. 
		"""
		return self.hits
	
	def lookup(self, point):
		"""
		Looks up a *point* and returns the associated object. 
		
		Returns ``None`` if the *point* is not in the cache. 
		
		Increases hit count if the *point* is found in the cache. 
		"""
		hw=HashableWrapper(point)
		if hw in self.storage:
			self.hits+=1
			return self.storage[hw]
		else:
			return None
	
	def insert(self, point, value):
		"""
		Inserts an object *value* for point *point*. 
		
		Returns ``True`` if a point in the cache is replaced. 
		"""
		hw=HashableWrapper(point)
		if hw in self.storage:
			replaced=True
		else:
			replaced=False
		self.storage[hw]=value
		return replaced
	
	def remove(self, point):
		"""
		Removes a *point* from the cache. 
		
		Returns ``True`` if a point is found and removed. 
		"""
		hw=HashableWrapper(point)
		if hw in self.storage:
			del self.storage[hw]
			return True
		return False
	

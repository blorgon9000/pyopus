"""
.. inheritance-diagram:: pyopus.misc.sobol
    :parts: 1

**Sobol sequence generator** 

Details can be found in

[joekuo1] S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's 
          quasirandom sequence generator, ACM Trans. Math. Softw. 29, 49-57 (2003). 

[joekuo2] S. Joe and F. Y. Kuo, Constructing Sobol sequences with better two-dimensional 
          projections, SIAM J. Sci. Comput. 30, 2635-2654 (2008). 

The code is a modification (efficiency reasons) of the code published at 
http://web.maths.unsw.edu.au/~fkuo/sobol/
"""

import _sobol
import numpy as np
import copy

__all__ = [ 'Sobol' ] 

class Sobol(object):
	"""
	Constructs a Sobol sequence generator with dimension *dim*. 
	The sequence members are in graycode order. 
	"""
	def __init__(self, dim):
		self.dim=dim
		
		self.V=_sobol.precompute(self.dim, 32)
		
		self.reset()
		
	def reset(self):
		"""
		Resets the generator. 
		"""
		self.index=np.zeros(1, dtype=np.uint32)
		self.X=np.zeros(self.dim, dtype=np.uint32)
		
	def skip(self, n):
		"""
		Skips *n* values. 
		"""
		_sobol.generate(self.dim, 32, self.V, self.index, self.X, n, 0)
	
	def get(self, n):
		"""
		Returns *n* values as rows of a matrix with *dim* columns. 
		"""
		return _sobol.generate(self.dim, 32, self.V, self.index, self.X, n, 1)
	
	def clone(self):
		"""
		Returns a clone of self. 
		"""
		return copy.deepcopy(self)
		
	def get_state(self):
		"""
		Returns the state of the generator. 
		"""
		return (self.dim, self.V.copy(), self.index.copy(), self.X.copy())
		
	def set_state(self, state):
		"""
		Sets the state of the generator. 
		"""
		self.dim=state[0]
		self.V=state[1].copy()
		self.index=state[2].copy()
		self.X=state[3].copy()
		

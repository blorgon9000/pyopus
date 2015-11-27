# -*- coding: UTF-8 -*-

"""
.. inheritance-diagram:: pyopus.problems.glbc
    :parts: 1

**Global optimization bound constrained test problems 
(PyOPUS subsystem name: GLBC)** 

Implemented by Árpád Bűrmen and Jernej Olenšek. 

All test functions in this module are maps from :math:`R` to :math:`R^n`. 

Gradient is not implemented and is in some cases even impossible to implement 
(e.g. Quartic noisy function). 

The functions can be wrapped into :class:`~pyopus.optimizer.base.RandomDelay` 
objects to introduce a random delay in function evaluation. 

The Yao et. al. set and the Hedar set have some functions in common. 
The "Hump" function from the Hedar set is named "SixHump" here. Hedar's 
version of the Rosenbrock problem is obtained by setting *hedar* to ``True``. 

The Yang set of test functions extends the Easom's function to n dimensions. 
The Zakharov test function is extended beyond K=2. 
Yang's version of problems is obtained by setting *yang* to ``True``. 
Equality constrained function is omitted (nonlinear equality constraint). 
Both stochastic functions are also omitted. 

The functions were taken from [yao]_, [hedar]_, and [yang]_. 

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. It depends only on the cpi module. 

.. [yao] Yao X., Liu Y., Lin G.: Evolutionary programming made faster. 
         IEEE Transactions on Evolutionary Computation, vol. 3, pp. 82-102, 
         1999. 

.. [hedar] Hedar A.: Global optimization test problems. 
         http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm

.. [yang] Yang X.-S.: Test Problems in Optimization.
         arXiv preprint at http://arxiv.org/abs/1008.0549, 2010. 
"""

import numpy as np
from numpy import array, zeros, ones, sin, cos, exp, pi, sum, prod, floor, arange, sqrt	
from numpy.random import rand
from cpi import CPI, MemberWrapper

try:
	from ..misc.debug import DbgMsg
except:
	def DbgMsg(x, y):
		return x+": "+y

__all__ = [ 
	'GlobalProblem', 
	'GlobalBCsuite', 
	'Quadratic', 
	'SchwefelA', 
	'SchwefelB', 
	'SchwefelC', 
	'Rosenbrock', 
	'Step',
	'QuarticNoisy', 
	'SchwefelD', 
	'Rastrigin', 
	'Ackley', 
	'Griewank', 
	'Penalty1', 
	'Penalty2', 
	'ShekelFoxholes', 
	'Kowalik', 
	'SixHump', 
	'Branin', 
	'GoldsteinPrice', 
	'Hartman',  
	'Shekel', 
	# Extra functions from Hedar
	'Beale', 
	'Bohachevsky', 
	'Booth', 
	'Colville', 
	'DixonPrice', 
	'Easom', 
	'Levy', 
	'Matyas', 
	'Michalewicz', 
	'Perm', 
	'Perm0', 
	'Powell', 
	'PowerSum', 
	'Schwefel', 
	'Shubert', 
	'Sphere', 
	'SumSquares', 
	'Trid', 
	'Zakharov', 
	# Extra functions from Yang
	'DifferentPowerSum', 
	'Yang1', 
	'Yang2', 
	'Yang3', 
]

class GlobalProblem(CPI):
	"""
	Base class for global optimization test functions 
	
	The full name of the problem is in the :attr:`name` member. The lower and 
	the upper bounds are in the :attr:`xl` and :attr:`xh` member. 
	
	The position and the function value for the best known solution are given 
	by :attr:`xmin` and :attr:`fmin`. 
	
	Objects of this class are callable. The calling convention is 
	
	``object(x)``
	
	where *x* is the input values vector. The function value at *x* is 
	returned.  
	
	Most functions are variably dimensional (n can be specified as an argument 
	to the constructor). 
	
	Example: create an instance of the Schwefel C function with n=40 and 
	evaluate it at the origin:: 
	  
	  from pyopus.optimizer.glbc import SchwefelC
	  from numpy import zeros
	  
	  sc=SchwefelC(n=40)
	  
	  # Evaluate the function at the origin
	  f=sc(zeros(40))
	"""
	name=None
	
	xl=None
	xh=None
	
	xmin=None
	fmin=None
	
	def __init__(self, n):
		self.n=n
	
	def __call__(self, x):
		return None
		
	def cpi(self):
		"""
		Returns the common problem interface. 
		
		Initial point and gradient function are not available. 
		
		See the :class:`CPI` class for more information. 
		"""
		itf=self.prepareCPI(self.n, m=0)
		itf['name']=self.name
		itf['fmin']=self.fmin
		itf['xmin']=self.xmin
		
		if 'xl' in self.__dict__:
			itf['xlo']=self.xl
			
		if 'xh' in self.__dict__:
			itf['xhi']=self.xh
			
		itf['f']=self
		
		return self.fixBounds(itf)
	

# 1		
class Quadratic(GlobalProblem):
	"""
	Quadratic function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Quadratic"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*100.0
		self.xh=ones(self.n)*100.0
		
		self.xmin=zeros(self.n)*0.0
		self.fmin=0.0
	
	def __call__(self, x):
		return sum(x**2)

# 2
class SchwefelA(GlobalProblem):
	"""
	Schwefel 2.22 function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Schwefel 2.22"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=zeros(self.n)*0.0
		self.fmin=0.0
		
	def __call__(self, x):
		return sum(abs(x)) + prod(abs(x))
		
# 3
class SchwefelB(GlobalProblem):
	"""
	Schwefel 1.2 function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Schwefel 1.2"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*100.0
		self.xh=ones(self.n)*100.0
		
		self.xmin=zeros(self.n)*0.0
		self.fmin=0.0
		
	def __call__(self, x):
		fi=0.0
		for i in range(1,self.n+1):
			fi+=sum(x[0:i])**2
		return fi
# 4
class SchwefelC(GlobalProblem):
	"""
	Schwefel 2.21 function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Schwefel 2.21"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*100.0
		self.xh=ones(self.n)*100.0
		
		self.xmin=zeros(self.n)*0.0
		self.fmin=0.0
		
	def __call__(self, x):
		return max(abs(x))
		
# 5
class Rosenbrock(GlobalProblem):
	"""
	Generalized Rosenbrock function (n>=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Generalized Rosenbrock"

	def __init__(self, n=30, yang=False, hedar=False):
		GlobalProblem.__init__(self, n)
		if self.n<2:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		if yang:
			self.xl=-ones(self.n)*5.0
			self.xh=ones(self.n)*5.0
		elif hedar:
			self.xl=-ones(self.n)*5.0
			self.xh=ones(self.n)*10.0
		else:
			self.xl=-ones(self.n)*30.0
			self.xh=ones(self.n)*30.0
		
		self.xmin=ones(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return sum(100*(x[1:self.n] - x[0:self.n-1]**2)**2 + (x[0:self.n-1]-1)**2)

# 6
class Step(GlobalProblem):
	"""
	Step function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Step"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*100.0
		self.xh=ones(self.n)*100.0
		
		self.xmin=zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return sum(floor(x+0.5)**2)

# 7
class QuarticNoisy(GlobalProblem):
	"""
	Quartic noisy function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Quartic noisy"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*1.28
		self.xh=ones(self.n)*1.28
		
		self.xmin=zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return sum((arange(len(x))+1)*x**4) + rand(1)[0]

# 8
class SchwefelD(GlobalProblem):
	"""
	Schwefel 2.26 function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Generalized Schwefel 2.26"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*500.0
		self.xh=ones(self.n)*500.0
		
		if self.n==30:
			self.xmin=420.9687*ones(self.n)
			self.fmin=-12569.5
		
	def __call__(self, x):
		return sum(-x*sin(sqrt(abs(x))))
		
# 9
class Rastrigin(GlobalProblem):
	"""
	Generalized Rastrigin function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Generalized Rastrigin"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*5.12
		self.xh=ones(self.n)*5.12
		
		self.xmin=zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return sum(x**2-10*cos(2*pi*x)+10)
		
# 10
class Ackley(GlobalProblem):
	"""
	Ackley function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Ackley"

	def __init__(self, n=30, yang=False):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		if yang:
			self.xl=-ones(self.n)*32.768
			self.xh=ones(self.n)*32.768
		else:
			self.xl=-ones(self.n)*32.0
			self.xh=ones(self.n)*32.0
		
		self.xmin=zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return -20.0*exp(-0.2*sqrt(1.0/self.n*sum(x**2))) - exp(1.0/self.n*sum(cos(2*pi*x))) + 20.0 + exp(1.0)

# 11
class Griewank(GlobalProblem):
	"""
	Generalized Griewank function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Generalized Griewank"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*600.0
		self.xh=ones(self.n)*600.0
		
		self.xmin=zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return 1.0/4000*sum(x**2) - prod(cos(x * 1.0/sqrt(arange(self.n)+1))) + 1.0

# 12
class Penalty1(GlobalProblem):
	"""
	Generalized penalty function 1 (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Generalized penalty function 1"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*50.0
		self.xh=ones(self.n)*50.0
		
		self.xmin=-ones(self.n)*1.0
		self.fmin=0.0
	
	def u(self,x,a,k,m):
		return (1.0*k*(x>a)*(x-a)**m + 1.0*k*(x<-a)*(-x-a)**m)
		
	def __call__(self, x):
		y=1+1.0/4*(x+1)
		return pi/self.n*(
					10.0*sin(pi*y[0])**2+
					sum((y[0:self.n-1]-1)**2 *(1.0+10.0*sin(pi*y[1:self.n])**2))+
					(y[-1]-1)**2
				) + sum(self.u(x,10,100,4))

# 13				
class Penalty2(GlobalProblem):
	"""
	Generalized penalty function 2 (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Generalized penalty function 2"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*50.0
		self.xh=ones(self.n)*50.0
		
		self.xmin=ones(self.n)*1.0
		self.fmin=0.0
	
	def u(self,x,a,k,m):
		return (1.0*k*(x>a)*(x-a)**m + 1.0*k*(x<-a)*(-x-a)**m)
		
	def __call__(self, x):
		y=1+1.0/4*(x+1)
		return 0.1*(
					sin(3*pi*x[0])**2+
					sum(((x[0:self.n-1]-1)**2)*(1+sin(3*pi*x[1:self.n])**2))+
					((x[-1]-1)**2)*(1+sin(2*pi*x[-1])**2)
				) + sum(self.u(x,5,100,4))

# 14
class ShekelFoxholes(GlobalProblem):
	"""
	Shekel foxholes function (n=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Shekel's foxholes"

	def __init__(self, n=2):
		GlobalProblem.__init__(self, n)
		if self.n!=2:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.a=array([
				[-32,-16,  0, 16, 32,-32,-16,  0, 16, 32,-32,-16, 0,16,32,-32,-16, 0,16,32,-32,-16, 0,16,32],
				[-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,  0,  0, 0, 0, 0, 16, 16,16,16,16, 32, 32,32,32,32]
			])
			
		self.xl=-ones(self.n)*65.536
		self.xh=ones(self.n)*65.536
		
		self.xmin=array([-32.0, -32.0])
		self.fmin=1.0

	def __call__(self, x):
		fi=1.0/500
		for k in range(25):
			fi+=1.0/((k+1)+sum((x-self.a[:,k])**6))
		return 1.0/fi

# 15
class Kowalik(GlobalProblem):
	"""
	Kowalik function (n=4). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Kowalik"

	def __init__(self, n=4):
		GlobalProblem.__init__(self, n)
		if self.n!=4:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.a=array([0.1957,0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]);
		self.b=1.0/array([0.25,0.5,1.0,2.0, 4.0,6.0,8.0,10.0,12.0,14.0,16.0])
			
		self.xl=-ones(self.n)*5.0
		self.xh=ones(self.n)*5.0
		
		self.xmin=array([0.1928, 0.1908, 0.1231, 0.1358])
		self.fmin=3.075e-4

	def __call__(self, x):
		fi=0
		for i in range(11):
			fi+=(self.a[i] - x[0]*(self.b[i]**2 + self.b[i]*x[1])/(self.b[i]**2 + self.b[i]*x[2] + x[3]))**2
		return fi
# 16
class SixHump(GlobalProblem):
	"""
	Six-hump camel-back function (n=2). 
	
	This function is named "Hump" in Hedar's set of test problems.
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Six-hump camel-back"

	def __init__(self, n=2, yang=False):
		GlobalProblem.__init__(self, n)
		if self.n!=2:
			raise Exception, DbgMsg("GLBC", "Bad n.")
			
		if yang:
			self.xl=np.array([-3.0,-2.0])
			self.xh=np.array([3.0,2.0])
		else:
			self.xl=-ones(self.n)*5.0
			self.xh=ones(self.n)*5.0
		
		# Also (-0.08983, 0.7126)
		self.xmin=array([0.09883, -0.7126])
		self.fmin=-1.0316285

	def __call__(self, x):
		return 4*x[0]**2 - 2.1*x[0]**4 + 1.0/3*x[0]**6 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4
				
# 17
class Branin(GlobalProblem):
	"""
	Branin function (n=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Branin"

	def __init__(self, n=2):
		GlobalProblem.__init__(self, n)
		if self.n!=2:
			raise Exception, DbgMsg("GLBC", "Bad n.")
			
		self.xl=array([-5.0, 0])
		self.xh=array([10.0, 15.0])
		
		# Also  (-pi , 12.275), (pi , 2.275), (9.42478, 2.475)
		self.xmin=array([-pi, 12.275])
		self.fmin=0.397887

	def __call__(self, x):
		return (x[1]-5.1/(4*pi**2)*x[0]**2 + 5.0/pi*x[0] - 6)**2+10*(1-1.0/(8*pi))*cos(x[0])+10
		
# 18
class GoldsteinPrice(GlobalProblem):
	"""
	Goldstein-Price function (n=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Goldstein-Price"

	def __init__(self, n=2):
		GlobalProblem.__init__(self, n)
		if self.n!=2:
			raise Exception, DbgMsg("GLBC", "Bad n.")
			
		self.xl=array([-2.0, -2.0])
		self.xh=array([2.0, 2.0])
		
		self.xmin=array([0, -1])
		self.fmin=3.0

	def __call__(self, x):
		return (
				1.0 + (x[0]+x[1]+1)**2 * (19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)
			)*( 
				30.0 + (2*x[0]-3*x[1])**2 * (18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)
			)

# 19
class Hartman(GlobalProblem):
	"""
	Hartman function (n=3 or n=6). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Hartman"

	def __init__(self, n=3):
		GlobalProblem.__init__(self, n)
		if self.n!=3 and n!=6:
			raise Exception, DbgMsg("GLBC", "Bad n.")
			
		self.xl=zeros(self.n)
		self.xh=ones(self.n)
		
		if self.n==3:
			self.c=array([1,1.2,3,3.2])
			self.a=array([
					[3,10,30],
					[0.1,10,35],
					[3,10,30],
					[0.1,10,35]
				])
			self.p=array([
					[0.3689,0.1170,0.2673],
					[0.4699,0.4387,0.7470],
					[0.1091,0.8732,0.5547],
					[0.038150,0.5743,0.8828]
				])
			
			self.xmin=array([0.114614, 0.555649, 0.852547])
			self.fmin=-3.86278
		else:
			self.c=array([1,1.2,3,3.2])
			self.a = array([
					[10, 3, 17, 3.5, 1.7, 8], 
					[0.05, 10, 17, 0.1, 8, 14], 
					[3, 3.5, 1.7, 10, 17, 8], 
					[17, 8, 0.05, 10, 0.1, 14]
				])
			self.p=array([
					[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], 
					[0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], 
					[0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], 
					[0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
				])
		
			self.xmin=array([0.20169,0.150011, 0.476874,0.275332,0.311652,0.6573])
			self.fmin=-3.32237

	def __call__(self, x):
		fi=0.0
		for i in range(4):
			fi-=self.c[i]*exp(-sum(self.a[i,:]*(x-self.p[i,:])**2))
		return fi

# 20
class Shekel(GlobalProblem):
	"""
	Shekel function (n=4, m=5, 7, or 10). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Shekel"

	def __init__(self, n=4, m=5):
		GlobalProblem.__init__(self, n)
		self.m=m
		
		if self.n!=4:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		if self.m!=5 and self.m!=7 and self.m!=10:
			raise Exception, DbgMsg("GLBC", "Bad m.")
		
		self.xl=zeros(self.n)
		self.xh=ones(self.n)*10.0
		
		self.a=array([
				[4, 4, 4, 4],
				[1, 1, 1, 1],
				[8, 8, 8, 8],
				[6, 6, 6, 6],
				[3, 7, 3, 7],
				[2, 9, 2, 9], 
				[5, 5, 3, 3], 
				[8, 1, 8, 1], 
				[6, 2, 6, 2], 
				[7, 3.6, 7, 3.6]
			])
		self.c=array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
		
		# Solutions approximately a[i,:]
		if self.m==5:
			self.xmin=array([4, 4, 4, 4])
			self.fmin=-10.1532
		elif self.m==7:
			self.xmin=array([4, 4, 4, 4])
			self.fmin=-10.4029
		else:
			self.xmin=array([4, 4, 4, 4])
			self.fmin=-10.5364

	def __call__(self, x):
		fi=0.0
		for i in range(self.m):
			fi-=sum(1.0/(sum((x-self.a[i,:])**2) + self.c[i]))
		return fi
		
class Beale(GlobalProblem):
	"""
	Beale function (n=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Beale"

	def __init__(self):
		GlobalProblem.__init__(self, 2)
		
		self.xl=-ones(self.n)*4.5
		self.xh=ones(self.n)*4.5
		
		self.xmin=np.array([3.0,0.5])
		self.fmin=0.0
		
	def __call__(self, x):
		return (1.5-x[0]*(1.0-x[1]))**2+(2.25-x[0]*(1.0-x[1]**2))**2+(2.625-x[0]*(1-x[1]**3))**2

class Bohachevsky(GlobalProblem):
	"""
	Bohachevsky functions (n=2, j=1,2,3). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	def __init__(self, j=1):
		GlobalProblem.__init__(self, 2)
		
		if j!=1 and j!=2 and j!=3:
			raise Exception, DbgMsg("GLBC", "Bad j.")
		
		self.j=j
		self.name="Bohachevsky_%d" % (j)
		
		self.xl=-ones(self.n)*100.0
		self.xh=ones(self.n)*100.0
		
		self.xmin=np.zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		if self.j==1:
			return x[0]**2+2.0*x[1]**2-0.3*np.cos(3*np.pi*x[0])-0.4*np.cos(4*np.pi*x[1])+0.7
		elif self.j==2:
			return x[0]**2+2.0*x[1]**2-0.3*np.cos(3*np.pi*x[0])*np.cos(4*np.pi*x[1])+0.3
		elif self.j==3:
			return x[0]**2+2.0*x[1]**2-0.3*np.cos(3*np.pi*x[0]+4*np.pi*x[1])+0.3

class Booth(GlobalProblem):
	"""
	Booth function (n=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Booth"
	
	def __init__(self):
		GlobalProblem.__init__(self, 2)
		
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=np.array([1.0, 3.0])
		self.fmin=0.0
		
	def __call__(self, x):
		return (x[0]+2*x[1]-7.0)**2+(2*x[0]+x[1]-5.0)**2
		
class Colville(GlobalProblem):
	"""
	Colville function (n=4). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Colville"
	
	def __init__(self):
		GlobalProblem.__init__(self, 4)
		
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=np.ones(4)
		self.fmin=0.0
		
	def __call__(self, x):
		return (
			100*(x[0]**2-x[1])**2+(x[0]-1.0)**2+(x[2]-1.0)**2+90*(x[2]**2-x[3])**2
			+10.1*((x[1]-1.0)**2+(x[3]-1.0)**2)+19.8*(x[1]**-1)*(x[3]-1.0)
		)
		
class DixonPrice(GlobalProblem):
	"""
	Dixon and Price function (n>=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="DixonPrice"
	
	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		
		if self.n<2:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=None
		self.fmin=0.0
		
	def __call__(self, x):
		return (
			(x[0]-1.0)**2+
			(np.arange(2,self.n+1)*(2*x[1:]**2-x[0:-1])**2).sum()
		)
		
class Easom(GlobalProblem):
	"""
	Easom function (n=2). The generalization for n>2 was given by Yang. 
	n>2 assumes Yang's version is requested. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Easom"
	
	def __init__(self, n=2, yang=False):
		GlobalProblem.__init__(self, n)
		self.yang=yang
		if n>2:
			self.yang=True
		
		if yang:
			self.xl=-ones(self.n)*2*np.pi
			self.xh=ones(self.n)*2*np.pi
		else:
			self.xl=-ones(self.n)*100.0
			self.xh=ones(self.n)*100.0
		
		self.xmin=np.ones(self.n)*np.pi
		self.fmin=-1.0
		
	def __call__(self, x):
		if self.yang:
			return -(-1)**self.n*(np.cos(x)**2).prod()*np.exp(-((x-np.pi)**2).sum())
		else:
			return (
				-np.cos(x[0])*np.cos(x[1])*np.exp(-(x[0]-np.pi)**2-(x[1]-np.pi)**2)
			)		
		
class Levy(GlobalProblem):
	"""
	Levy function (n>=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Levy"
	
	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		
		if self.n<2:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=np.ones(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		y=1.0+(x-1.0)/4
		return (
			np.sin(np.pi*y[0])**2
			+((y[:-1]-1.0)**2*(1.0+10*(np.sin(np.pi*y[:-1]+1.0))**2)).sum()
			+(y[-1]-1.0)**2*(1.0+np.sin(2*np.pi*y[-1])**2)
		)

class Matyas(GlobalProblem):
	"""
+	Matyas function (n=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Matyas"
	
	def __init__(self):
		GlobalProblem.__init__(self, 2)
		
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=np.zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return (
			0.26*(x**2).sum()-0.48*x.prod()
		)		

class Michalewicz(GlobalProblem):
	"""
	Michalewicz function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Michalewicz"
	
	def __init__(self, n=10):
		GlobalProblem.__init__(self, n)
		
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=np.zeros(self.n)
		self.xh=np.ones(self.n)*np.pi
		
		self.xmin=None
		if self.n==2:
			self.fmin=-1.8013
		elif self.n==5:	
			self.fmin=-4.687658
		elif self.n==10:
			self.fmin=-9.66015
		else:
			self.fmin=None
		
	def __call__(self, x):
		return (
			-(np.sin(x)*(np.sin(np.arange(1,self.n+1)*x**2/np.pi))**(2*10)).sum()
		)

class Perm(GlobalProblem):
	"""
	Perm function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Perm"
	
	def __init__(self, n=20, beta=0.5):
		GlobalProblem.__init__(self, n)
		
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-np.ones(self.n)*n
		self.xh=np.ones(self.n)*n
		
		self.beta=beta
		
		self.xmin=np.arange(1,self.n+1)
		self.fmin=0.0
		
	def __call__(self, x):
		sum=0.0
		for k in range(1,self.n+1):
			sum+=(((1.0*np.arange(1,self.n+1))**k+self.beta)*(x/np.arange(1,self.n+1))**k-1).sum()
		return sum

class Perm0(GlobalProblem):
	"""
	Perm function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Perm0"
	
	def __init__(self, n=30, beta=10, yang=False):
		GlobalProblem.__init__(self, n)
		
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		if yang:
			self.xl=-np.ones(self.n)
			self.xh=np.ones(self.n)
		else:
			self.xl=-np.ones(self.n)*n
			self.xh=np.ones(self.n)*n
		
		self.beta=beta 
		
		self.xmin=1.0/np.arange(1,self.n+1)
		self.fmin=0.0
		
	def __call__(self, x):
		sum=0.0
		for k in range(1,self.n+1):
			sum+=(((np.arange(1,self.n+1)+self.beta)*(x**k-1.0/(1.0*np.arange(1,self.n+1))**k))**2).sum()
		return sum
		
class Powell(GlobalProblem):
	"""
	Powell function (n=4k, k>0). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Powell"
	
	def __init__(self, n=32, beta=0.5):
		GlobalProblem.__init__(self, n)
		
		if self.n<4 or self.n%4!=0:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-np.ones(self.n)*4
		self.xh=np.ones(self.n)*5
		
		self.xmin=np.zeros(self.n)
		self.xmin[::4]=3.0
		self.xmin[1::4]=-1.0
		self.xmin[3::4]=1.0
		self.fmin=0.0
		
	def __call__(self, x):
		return (
			(( x[::4]+10*(x[1::4]-2.0) )**2).sum()
			+(( 5.0**0.5*(x[2::4]-x[3::4]) )**2).sum()
			+(( (x[1::4]-2*x[3::4])**2 )**2).sum()
			+(( 10.0**0.5*(x[::4]-x[3::4])**2 )**2).sum()
		)
		
class PowerSum(GlobalProblem):
	"""
	Power sum function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="PowerSum"
	
	def __init__(self, n=4, b=None):
		GlobalProblem.__init__(self, n)
		
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		if b is None:
			if self.n==4:
				self.b=np.array([8.0,18,44,114])
			else:
				raise Exception, DbgMsg("GLBC", "Need vector b of length n.")
		elif len(b.shape)!=1 or b.shape[0]!=n:
			raise Exception, DbgMsg("GLBC", "Vector b must have n components.")
		else:
			self.b=b
		
		self.xl=np.zeros(self.n)
		self.xh=np.ones(self.n)*self.n
		
		if self.n==4:
			self.xmin=np.array([1.0,2,3,4])
			self.fmin=0.0
		else:
			self.xmin=None
			self.fmin=None
		
	def __call__(self, x):
		sum=0.0
		for k in range(1, self.n+1):
			sum+=((x**k).sum()-self.b[k-1])**2
		return sum
		
class Schwefel(GlobalProblem):
	"""
	Schwefel function (n>=1), slightly modified SchwefelD with a general 
	global minimum valid for arbitrary n. 
	
	Yang's version is obtained with *yang* set to ``True``. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Schwefel"

	def __init__(self, n=30, yang=False):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		self.yang=yang
		
		self.xl=-ones(self.n)*500.0
		self.xh=ones(self.n)*500.0
		
		self.xmin=np.ones(self.n)
		if yang:
			self.fmin=-self.n*418.9829
		else:
			self.fmin=0.0
		
	def __call__(self, x):
		if self.yang:
			return sum(-x*sin(sqrt(abs(x))))
		else:
			return 418.9829*self.n+sum(-x*sin(sqrt(abs(x))))
		
class Shubert(GlobalProblem):
	"""
	Shubert function (n=2). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Schubert"

	def __init__(self, m=5):
		GlobalProblem.__init__(self, 2)
		self.m=m
		
		if m<1:
			raise Exception, DbgMsg("GLBC", "Bad m.")
		
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=None
		if self.m==5:
			self.fmin=-186.7309
		else:
			self.fmin=None
		
	def __call__(self, x):
		ii=np.arange(1,self.m+1)
		return (ii*np.cos((ii+1)*x[0]+ii)).sum()*(ii*np.cos((ii+1)*x[1]+ii)).sum()
		
class Sphere(GlobalProblem):
	"""
	Sphere function (n>=1). Also known as DeJong's sphere function. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Sphere"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*5.12
		self.xh=ones(self.n)*5.12
		
		self.xmin=zeros(self.n)*0.0
		self.fmin=0.0
	
	def __call__(self, x):
		return sum(x**2)	

class SumSquares(GlobalProblem):
	"""
	Sphere function (n>=1). Also known as DeJong's weighted sphere function. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="SumSquares"

	def __init__(self, n=30, yang=False):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		if yang:
			self.xl=-ones(self.n)*5.12
			self.xh=ones(self.n)*5.12
		else:
			self.xl=-ones(self.n)*10.0
			self.xh=ones(self.n)*10.0
		
		self.xmin=zeros(self.n)*0.0
		self.fmin=0.0
	
	def __call__(self, x):
		return sum(np.arange(1,self.n+1)*x**2)

class DifferentPowerSum(GlobalProblem):
	"""
	Sum of different powers function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="DifferentPowerSum"

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*1.0
		self.xh=ones(self.n)*1.0
		
		self.xmin=zeros(self.n)*0.0
		self.fmin=0.0
	
	def __call__(self, x):
		return sum(np.abs(x)**np.arange(2,self.n+2))
		
class Trid(GlobalProblem):
	"""
	Trid function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Trid"

	def __init__(self, n=10):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*self.n**2
		self.xh=ones(self.n)*self.n**2
		
		self.xmin=None
		if self.n==6:
			self.fmin=-50.0
		elif self.n==10:
			self.fmin=-200.0
		else:
			self.fmin=None
	
	def __call__(self, x):
		return ((x-1.0)**2).sum()-(x[1:]*x[:-1]).sum()

class Zakharov(GlobalProblem):
	"""
	Zakharov function (n>=1). 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Zakharov"

	def __init__(self, n=10, K=2):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		self.K=K
		
		if self.K<2:
			raise Exception, DbgMsg("GLBC", "Bad K.")
		
		self.xl=-ones(self.n)*5.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=np.zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		tmp=(x**2).sum()
		tmp1=((0.5*np.arange(1,self.n+1)*x).sum())
		return tmp+tmp1**2+tmp1**4

class Yang1(GlobalProblem):
	"""
	Yang1 function (n>=1). See (20) in the corresponding paper. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Yang1"

	def __init__(self, n=10):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*2*np.pi
		self.xh=ones(self.n)*2*np.pi
		
		self.xmin=np.zeros(self.n)
		self.fmin=0.0
		
	def __call__(self, x):
		return np.abs(x).sum()*np.exp(-np.sin(x**2).sum())
		
class Yang1(GlobalProblem):
	"""
	Yang2 function (n>=1). See (21) in the corresponding paper. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Yang1"

	def __init__(self, n=10):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*2*np.pi
		self.xh=ones(self.n)*2*np.pi
		
		self.xmin=None
		self.fmin=-1.0/np.e**0.5
		
	def __call__(self, x):
		return -np.abs(x).sum()*np.exp(-(x**2).sum())
		
class Yang2(GlobalProblem):
	"""
	Yang2 function (n>=1). See (21) in the corresponding paper. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Yang2"

	def __init__(self, n=10):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
		
		self.xl=-ones(self.n)*2*np.pi
		self.xh=ones(self.n)*2*np.pi
		
		self.xmin=None
		self.fmin=-1.0/np.e**0.5
		
	def __call__(self, x):
		return -np.abs(x).sum()*np.exp(-(x**2).sum())
	
class Yang3(GlobalProblem):
	"""
	Yang2 function (n>=1). See (24) in the corresponding paper. 
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	name="Yang3"

	def __init__(self, n=10):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLBC", "Bad n.")
			
		self.xl=-ones(self.n)*10.0
		self.xh=ones(self.n)*10.0
		
		self.xmin=np.zeros(self.n)
		self.fmin=-1.0
		
	def __call__(self, x):
		return ((np.sin(x)**2).sum()-np.exp(-(x**2).sum()))*np.exp(-(np.sin(np.abs(x)**0.5)**2).sum())
	
	
	
GlobalBCsuite=[
	Quadratic, 
	SchwefelA, 
	SchwefelB, 
	SchwefelC, 
	Rosenbrock, 
	Step,
	QuarticNoisy, 
	SchwefelD, 
	Rastrigin, 
	Ackley, 
	Griewank, 
	Penalty1, 
	Penalty2, 
	ShekelFoxholes, 
	Kowalik, 
	SixHump, 
	Branin, 
	GoldsteinPrice, 
	Hartman,  
	Shekel, 
	# Extra functions from Hedar
	Beale, 
	Bohachevsky, 
	Booth, 
	Colville, 
	DixonPrice, 
	Easom, 
	Levy, 
	Matyas, 
	Michalewicz, 
	Perm, 
	Perm0, 
	Powell, 
	PowerSum, 
	Schwefel, 
	Shubert, 
	Sphere, 
	SumSquares, 
	Trid, 
	Zakharov, 
	# Extra functions from Yang
	DifferentPowerSum, 
	Yang1, 
	Yang2, 
	Yang3, 
]
"""
A list holding references to all function classes in this module. 
"""

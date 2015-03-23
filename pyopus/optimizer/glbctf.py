# -*- coding: cp1250 -*-

"""
.. inheritance-diagram:: pyopus.optimizer.glbctf
    :parts: 1

**Global optimization test functions (PyOPUS subsystem name: GLTF)** 

Implemented by Árpád Bűrmen and Jernej Olenšek. 

All test functions in this module are maps from :math:`R` to :math:`R^n`. 

Gradient is not implemented and is in some cases even impossible to implement 
(e.g. Quartic noisy function). 

The functions can be wrapped into :class:`~pyopus.optimizer.mgh.RandomDelay` 
objects to introduce a random delay in function evaluation. 

The functions were taken from  

Yao X., Liu Y., Lin G.: Evolutionary programming made faster. 
IEEE Transactions on Evolutionary Computation, vol. 3, pp. 82-102, 1999. 

This module is independent of PyOPUS, meaning that it can be taken as is 
and used as a module in some other package. 
"""

from numpy import array, zeros, ones, sin, cos, exp, pi, sum, prod, floor, arange, sqrt	
from numpy.random import rand

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
	'Shekel' 
]

class GlobalProblem(object):
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
	  
	  from pyopus.optimizer.glbctf import SchwefelC
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<2:
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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

	def __init__(self, n=30):
		GlobalProblem.__init__(self, n)
		if self.n<1:
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
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
	
	See the :class:`GlobalProblem` class for more information. 
	"""
	
	name="Six-hump camel-back"

	def __init__(self, n=2):
		GlobalProblem.__init__(self, n)
		if self.n!=2:
			raise Exception, DbgMsg("GLTF", "Bad n.")
			
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
			
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
			
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
			raise Exception, DbgMsg("GLTF", "Bad n.")
			
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
			
			self.xmin=array([0.114, 0.556, 0.852])
			self.fmin=-3.86
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
		
			self.xmin=array([0.201,0.150, 0.477,0.275,0.311,0.657])
			self.fmin=-3.322

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
			raise Exception, DbgMsg("GLTF", "Bad n.")
		
		if self.m!=5 and self.m!=7 and self.m!=10:
			raise Exception, DbgMsg("GLTF", "Bad m.")
		
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
			self.fmin=-10.153
		elif self.m==7:
			self.xmin=array([4, 4, 4, 4])
			self.fmin=-10.403
		else:
			self.xmin=array([4, 4, 4, 4])
			self.fmin=-10.536

	def __call__(self, x):
		fi=0.0
		for i in range(self.m):
			fi-=sum(1.0/(sum((x-self.a[i,:])**2) + self.c[i]))
		return fi
		
		
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
]
"""
A list holding references to all function classes in this module. 
"""

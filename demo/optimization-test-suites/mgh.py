# More-Garbow-Hillstrom test suite

from pyopus.optimizer.mgh import MGHsuite
from numpy import where, zeros, sqrt

def numGradCentral(f, x, tol=1e-6, abstol=1e-9):
	deltaRel=x*tol
	delta=where(deltaRel>abstol, deltaRel, abstol)
	
	n=x.size
	
	g=zeros(n)
	for i in range(0, n):
		newx1=x.copy()
		newx2=x.copy()
		newx1[i]-=delta[i]
		newx2[i]+=delta[i]
		
		f2=f(newx2)
		f1=f(newx1)

		g[i]=(f2-f1)/(2*delta[i])
	
	return g
	
def numGradForward(f, x, tol=1e-6, abstol=1e-9):
	deltaRel=x*tol
	delta=where(deltaRel>abstol, deltaRel, abstol)
	
	n=x.size
	
	g=zeros(n)
	for i in range(0, n):
		newx1=x.copy()
		newx2=x.copy()
		newx2[i]+=delta[i]
		
		f2=f(newx2)
		f1=f(newx1)

		g[i]=(f2-f1)/delta[i]
	
	return g

if __name__=='__main__':
	print("")
	print("Function value, analytical and numerical gradient \nrelative difference at initial point\n")
	print("Syntax / gradient implementation test")
	print("No exceptions expected")
	print("Error should be around 1e-7 or less")
	print("McKinnon has large error due to")
	print("  central difference across discontinuity in second derivative")
	print("--------------------------------------------------------------")
	for cls in MGHsuite:
		obj=cls()
		
		xini=obj.initial
		fini=obj.f(xini)
		gini=obj.g(xini)
		gini=obj.g(xini)
		gapprox=numGradCentral(obj.f, xini, tol=1e-6, abstol=1e-6)
		gerr=sqrt(((gini-gapprox)**2).sum())/sqrt((gini**2).sum())
		
		print("%45s (%02d): f=%16.8e  gerr=%11.3e " % (obj.name, obj.n, fini, gerr))

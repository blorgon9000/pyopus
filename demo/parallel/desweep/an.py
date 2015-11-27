# Load results summary, plot function value history for given function, 
# population size, and run. 
#  python an.py

from numpy import array, zeros, arange
from cPickle import dump, load
import pyopus.wxmplplot as pyopl

if __name__=='__main__':
	f=open("fsummary.pck", "rb")
	summary=load(f)
	f.close()
	
	print summary
	
	findex=0
	popsize=10
	runindex=0
	
	f=open("fhist_f%d_p%d_r%d.pck" % (findex, popsize, runindex), "rb")
	fhist=load(f)
	f.close()
	
	pyopl.init()
	pyopl.close()
	
	f1=pyopl.figure()
	
	pyopl.lock(True)
	
	if pyopl.alive(f1):
		ax=f1.add_subplot(1,1,1)
		ax.semilogy(arange(len(fhist))/popsize, fhist)
		ax.set_xlabel('generation')
		ax.set_ylabel('f')
		ax.set_title('Progress of differential evolution')
		ax.grid()
		
	pyopl.lock(False)
	
	pyopl.join()
	
	

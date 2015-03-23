# Optimize Rosenbrock function with differential evolution
# Collect cost function and plot progress

from pyopus.optimizer.de import DifferentialEvolution
from pyopus.optimizer import glbctf
from pyopus.optimizer.base import Reporter
from pyopus.evaluator.cost import CostCollector
import pyopus.wxmplplot as pyopl
from numpy import array, zeros, arange

if __name__=='__main__':
	ndim=30
	popSize=50
	
	prob=glbctf.Rosenbrock(n=ndim)
	
	opt=DifferentialEvolution(
		prob, prob.xl, prob.xh, debug=0, 
		maxGen=1500, populationSize=popSize, w=0.5, pc=0.3
	)
	cc=CostCollector()
	opt.installPlugin(cc)
	opt.installPlugin(Reporter(onIterStep=1000))
	opt.reset()
	opt.run()
	cc.finalize()
	
	pyopl.init()
	pyopl.close()
	f1=pyopl.figure()
	pyopl.lock(True)
	if pyopl.alive(f1):
		ax=f1.add_subplot(1,1,1)
		ax.semilogy(arange(len(cc.fval))/popSize, cc.fval)
		ax.set_xlabel('generations')
		ax.set_ylabel('f')
		ax.set_title('Progress of differential evolution')
		ax.grid()
	pyopl.lock(False)
	
	print("x=%s f=%e" % (str(opt.x), opt.f))
	
	pyopl.join()
	

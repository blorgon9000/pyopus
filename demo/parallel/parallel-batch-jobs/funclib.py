# A function thar executes a run of given function with given population size. 
# Must be in a module so it can be pickled and sent to a task. 

from pyopus.optimizer.de import DifferentialEvolution
from pyopus.evaluator.cost import CostCollector
from numpy import array, zeros, random

def deRun(prob, popSize, maxiter=75000, maxGen=1500, w=0.5, pc=0.3):
	random.seed()
	opt=DifferentialEvolution(
		prob, prob.xl, prob.xh, debug=0, maxiter=maxiter, 
		maxGen=maxGen, populationSize=popSize, w=w, pc=pc
	)
	cc=CostCollector()
	opt.installPlugin(cc)
	opt.reset(zeros(len(prob.xl)))
	opt.run()
	cc.finalize()
	
	return (opt.f, cc.fval)
	
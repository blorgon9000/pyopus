# Optimize STYRENE function with QPMADS
# Collect cost function and plot progress

from pyopus.optimizer.qpmads import QPMADS
from pyopus.problems.madsprob import STYRENE
from pyopus.optimizer.base import Reporter, CostCollector
import pyopus.wxmplplot as pyopl
import numpy as np
from numpy import array, zeros, arange
from numpy.random import seed


if __name__=='__main__':
	seed(0)
	
	prob=STYRENE()
	
	opt=QPMADS(
		function=None, 
		fc=prob.fc, # f and c are evaluated simultaneously
		xlo=prob.xl, xhi=prob.xh, 
		clo=prob.cl, chi=prob.ch, 
		debug=0, maxiter=1000
	)
	cc=CostCollector()
	opt.installPlugin(cc)
	opt.installPlugin(Reporter(onIterStep=100))
	opt.reset(prob.initial)
	opt.run()
	cc.finalize()
	
	pyopl.init()
	pyopl.close()
	f1=pyopl.figure()
	pyopl.lock(True)
	
	# If constraints are violated, no point is plotted
	fval=np.where(cc.hval>0, np.nan, cc.fval)
	
	if pyopl.alive(f1):
		ax=f1.add_subplot(1,1,1)
		ax.plot(arange(len(fval)), fval, '-o')
		ax.set_xlabel('evaluations')
		ax.set_ylabel('f')
		ax.set_title('Progress of QPMADS')
		ax.grid()
	pyopl.lock(False)
	
	print("x=%s f=%e" % (str(opt.x), opt.f))
	
	pyopl.join()

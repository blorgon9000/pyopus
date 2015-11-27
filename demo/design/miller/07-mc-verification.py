# Monte Carlo verification of yiled

from definitions import *
from pyopus.design.mc import MonteCarlo
from pyopus.design.wc import WorstCase
from pyopus.design.wcd import WorstCaseDistance
from pyopus.evaluator.aggregate import formatParameters
# If MPI is imported an application not using MPI will behave correctly
# (i.e. only slot 0 will run the program) even when started with mpirun
from pyopus.parallel.mpi import MPI
from pyopus.parallel.cooperative import cOS


if __name__=='__main__':
	# Result of yield targeting
	atDesign={
		'c_out':    5.878921e-12,
		'dif_l':    3.303722e-06,
		'dif_w':    4.213081e-05,
		'load_l':   2.738099e-06,
		'load_w':   5.381707e-05,
		'mirr_l':   2.688886e-06,
		'mirr_ld':  3.504204e-06,
		'mirr_w':   6.747047e-05,
		'mirr_wd':  1.131046e-05,
		'mirr_wo':  8.230304e-05,
		'out_l':    2.409226e-07,
		'out_w':    6.642604e-05,
		'r_out':    2.663909e+04,
	}
	
	# Add 'wcd' module to all analyses (we have no corners)
	for name,an in analyses.iteritems():
		an['modules'].append('wcd')
		
	# Worst case performances (skip area, transistor op conditions, and auxiliary measures)
	wcList=measures.keys()
	wcList.sort()
	wcList.remove('area')
	wcList.remove('vgs_drv')
	wcList.remove('vds_drv')
	wcList.remove('gain_com')
	wcList.remove('gain_vdd')
	wcList.remove('gain_vss')
	
	# Prepare parallel environment
	cOS.setVM(MPI(mirrorMap={'*':'.'}))
	
	# Verify yield with Monte-Carlo 
	mc=MonteCarlo(
		heads, analyses, measures, 
		statParams, opParams, atDesign, 
		variables=variables, debug=2, 
		wcOptions={ 'debug': 0 }, 
		nSamples=10000, 
		spawnerLevel=1
	)
	results, anCount = mc(wcList)
	print mc.formatResults() 
	print anCount
	
	# Verify yield with worst case distances
	#wcd=WorstCaseDistance(
	#	heads, analyses, measures, statParams, opParams, variables=variables, 
	#	fixedParams=atDesign, 
	#	debug=1, 
	#	spawnerLevel=1
	#)
	#res, anCount = wcd(wcList)
	#print wcd.formatResults()
	#print anCount
	
	# Verify yield with worst case analysis
	#wc=WorstCase(
	#	heads, analyses, measures, statParams, opParams, variables=variables, 
	#	fixedParams=atDesign, beta=3.0, 
	#	debug=1, 
	#	spawnerLevel=1
	#)
	#res, anCount = wc(wcList)
	#print wc.formatResults()
	#print anCount
	
	# Finalize cOS parallel environment
	cOS.finalize()
	
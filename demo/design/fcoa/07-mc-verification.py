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
		'dif_l':   1.212027e-06,
		'dif_w':   1.000000e-06,
		'nl_l':    2.779390e-06,
		'nl_w':    7.947918e-05,
		'nm_l':    3.884062e-06,
		'nm_w':    2.249913e-05,
		'pm_l':    1.522706e-06,
		'pm_w0':   4.430858e-05,
		'pm_w1':   2.973440e-05,
		'pm_w2':   7.761160e-05,
		'pm_w3':   2.786359e-05
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
	#mc=MonteCarlo(
	#	heads, analyses, measures, 
	#	statParams, opParams, atDesign, 
	#	variables=variables, debug=2, 
	#	wcOptions={ 'debug': 0 }, 
	#	nSamples=10000, 
	#	spawnerLevel=2
	#)
	#results, anCount = mc(wcList)
	#print mc.formatResults() 
	#print anCount
	
	# Verify yield with worst case distances
	wcd=WorstCaseDistance(
		heads, analyses, measures, statParams, opParams, variables=variables, 
		fixedParams=atDesign, 
		debug=1, 
		spawnerLevel=1
	)
	res, anCount = wcd(wcList)
	print wcd.formatResults()
	print anCount
	
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
	
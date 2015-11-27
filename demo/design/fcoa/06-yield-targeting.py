# Yield targeting

from definitions import *
from pyopus.design.yt import YieldTargeting
from pyopus.evaluator.aggregate import formatParameters
# If MPI is imported an application not using MPI will behave correctly
# (i.e. only slot 0 will run the program) even when started with mpirun
from pyopus.parallel.mpi import MPI
from pyopus.parallel.cooperative import cOS


if __name__=='__main__':
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
	
	# Initial point, ibias=5u
	atDesign={
		'dif_l':    6.816424e-07,
		'dif_w':    3.332037e-06,
		'nl_l':    2.655088e-06,
		'nl_w':    4.977226e-05,
		'nm_l':    3.665018e-06,
		'nm_w':    7.507191e-05,
		'pm_l':    1.487570e-06,
		'pm_w0':    2.871096e-05,
		'pm_w1':    2.871096e-05,
		'pm_w2':    6.389441e-05,
		'pm_w3':    8.310102e-05,
	}
	
	# Prepare parallel environment
	cOS.setVM(MPI(mirrorMap={'*':'.'}))
	
	# 3-sigma target yield
	yt=YieldTargeting(
		designParams, statParams, opParams, 
		heads, analyses, measures, variables=variables, 
		beta=3.0, wcSpecs=wcList, 
		# Comment out to use default initial point (lo+hi)/2
		# initial=atDesign, 
		initialNominalDesign=True, 
		# Norms for measures with zero goal
		norms={ 'area': 100e-12, 'vgs_drv': 1e-3, 'vds_drv':1e-3 }, 
		tradeoffs=1e-6, # Tradeoff optimization weight, can be overridden in *CbdOptions
		stopWhenAllSatisfied=True, 
		# Initial nominal optimization
		initialCbdOptions={ 
			'debug': 1, 'method': 'local', 'stepTol': 1e-5, 
		}, 
		# Main optimization
		cbdOptions={ 
			'debug': 1, 'method': 'local', 'stepTol': 1e-5, 
		}, wcOptions={ 'debug': 0 }, 
		debug=2, spawnerLevel=1
	)
	atDesign, agg, wc, anCount = yt()
	print(formatParameters(atDesign))
	print(wc.formatResults())
	print(agg.formatResults())
	print(anCount)
	
	# Finalize cOS parallel environment
	cOS.finalize()
	
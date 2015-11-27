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
	
	# Result of sizing across corners
	atDesign={
		'c_out':    6.737942e-13,
		'dif_l':    2.170269e-06,
		'dif_w':    4.396577e-06,
		'load_l':    2.800742e-06,
		'load_w':    6.228898e-05,
		'mirr_l':    3.378882e-07,
		'mirr_ld':    2.072842e-06,
		'mirr_w':    5.311666e-05,
		'mirr_wd':    1.979695e-06,
		'mirr_wo':    4.983195e-05,
		'out_l':    6.257517e-07,
		'out_w':    3.441845e-05,
		'r_out':    1.402169e+05
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
			'debug': 1, 'method': 'global', 'stepTol': 1e-5, 
		}, 
		# Main optimization
		cbdOptions={ 
			'debug': 1, 'method': 'global', 'stepTol': 1e-5, 
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
	
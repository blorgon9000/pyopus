# Sizing across corners

from definitions import *
from pyopus.evaluator.auxfunc import listParamDesc, paramDict
from pyopus.design.cbd import CornerBasedDesign, generateCorners
from pyopus.evaluator.aggregate import formatParameters
# If MPI is imported an application not using MPI will behave correctly
# (i.e. only slot 0 will run the program) even when started with mpirun
from pyopus.parallel.mpi import MPI
from pyopus.parallel.cooperative import cOS
import numpy as np


if __name__=='__main__':
	# Prepare statistical parameters dictionary with nominal values (0)
	nominalStat=paramDict(np.zeros(len(statParams)), statParams.keys())
	
	# Prepare nominal design parameters dictionaries
	names=designParams.keys()
	nominalDesign=paramDict(listParamDesc(designParams, names, "init"), names)
	loDesign=paramDict(listParamDesc(designParams, names, "lo"), names)
	
	# Prepare one corner, module 'tm', nominal op parameters
	corners=generateCorners(
		paramSpec={
			'vdd': [1.7, 1.8, 2.0], 
			'temperature': [0.0, 25, 100.0]
		}, 
		modelSpec={
			'mos': ['tm', 'wp', 'ws', 'wo', 'wz']
		}
	)
	# Add nominal corner
	nominalCorner={
		'nom': {
			'params': {
				'vdd': 1.8, 
				'temperature': 25
			}, 
			'modules': [ 'tm' ]
		}
	}
	corners.update(nominalCorner)
	
	# Are should be evauated only in corner 'nom'
	measures['area']['corners']=[ 'nom' ]
	
	# Prepare parallel environment
	cOS.setVM(MPI(mirrorMap={'*':'.'}))
	
	# Design it, nominal statistical parameters are treated as fixed parameters
	cbd=CornerBasedDesign(
		designParams, heads, analyses, measures, corners, 
		fixedParams=[nominalStat], variables=variables, norms={ 'area': 100e-12 }, 
		# tradeoff=1e-6, 
		# stepTol=1e-4, 
		initial=loDesign, 
		method='global', 
		evaluatorOptions={'debug': 0}, 
		debug=1
	)
	atDesign, aggregator, analysisCount = cbd()
	print(formatParameters(atDesign))
	print(aggregator.formatResults())
	
	# Finalize cOS parallel environment
	cOS.finalize()
	
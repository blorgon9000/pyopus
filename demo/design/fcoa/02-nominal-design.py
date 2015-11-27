# Nominal design

from definitions import *
from pyopus.evaluator.auxfunc import listParamDesc, paramDict
from pyopus.design.cbd import CornerBasedDesign
from pyopus.evaluator.aggregate import formatParameters
# If MPI is imported an application not using MPI will behave correctly
# (i.e. only slot 0 will run the program) even when started with mpirun
from pyopus.parallel.mpi import MPI
from pyopus.parallel.cooperative import cOS
import numpy as np


if __name__=='__main__':
	# Prepare statistical parameters dictionary with nominal values (0)
	nominalStat=paramDict(np.zeros(len(statParams)), statParams.keys())
	
	# Prepare operating parameters dictionary with nominal values
	names=opParams.keys()
	nominalOp=paramDict(listParamDesc(opParams, names, "init"), names)
	
	# Prepare nominal design parameters dictionaries
	names=designParams.keys()
	nominalDesign=paramDict(listParamDesc(designParams, names, "init"), names)
	
	# Prepare one corner, module 'tm', nominal op parameters
	corners={
		'nom': {
			'params': nominalOp, 
			'modules': ['tm']
		}
	}
	
	# Prepare parallel environment
	cOS.setVM(MPI(mirrorMap={'*':'.'}))
	
	# Design it, nominal statistical parameters are treated as fixed parameters
	cbd=CornerBasedDesign(
		designParams, heads, analyses, measures, corners, 
		fixedParams=[nominalStat], variables=variables, norms={ 'area': 100e-12 }, 
		# tradeoff=1e-6, 
		# stepTol=1e-4, 
		# initial=nominalDesign, 
		method='global', 
		evaluatorOptions={'debug': 0}, 
		debug=1
	)
	atDesign, aggregator, analysisCount = cbd()
	print(formatParameters(atDesign))
	print(aggregator.formatResults())
	
	# Finalize cOS parallel environment
	cOS.finalize()
	
# Test performance evaluator

from definitions import *
from pyopus.evaluator.auxfunc import listParamDesc, paramDict
from pyopus.evaluator.performance import PerformanceEvaluator
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
	
	# Measures have no corners listed - they are evaluated across all specified corners
	pe=PerformanceEvaluator(heads, analyses, measures, corners, variables=variables, debug=2)
	results, anCount = pe([nominalDesign, nominalStat])
	print(pe.formatResults(nMeasureName=10, nCornerName=15))

	# Cleanup temporary files
	pe.finalize()
	
	# Finalize cOS parallel environment
	cOS.finalize()
	
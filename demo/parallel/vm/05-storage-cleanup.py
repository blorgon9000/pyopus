# Demonstrates mirrored storage cleanup

import sys
from pyopus.parallel.mpi import MPI as VM
	
import funclib
import os

if __name__=='__main__':
	vm=VM(debug=2)
	
	# Clean up local storage on all machines in the virtual machine. 
	print("\nCleaning up.")
	vm.clearLocalStorage(timeout=6.0)

	vm.finalize()
	
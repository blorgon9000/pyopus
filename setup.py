#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc, PREFIX
from platform import system
import os

# Detect platform, set up include directories and preprocessor macros 
if system()=='Windows':
	define_macros=[('WINDOWS', None)]
	include_dirs=[os.path.join(PREFIX, 'Lib', 'site-packages', 'numpy', 'core', 'include', 'numpy')]
else:
	define_macros=[('LINUX', None)]
	include_dirs=[os.path.join(get_python_inc(plat_specific=1), 'numpy')]
	
# Extensions
ext_modules=[
	Extension(
		'pyopus.simulator._rawfile', 
		['src/rawfile/rawfile.c'], 
		include_dirs=include_dirs, 
		define_macros=define_macros
	), 
	Extension(
		'pyopus.simulator._hspice_read', 
		['src/hspicefile/hspice_read.c'], 
		include_dirs=include_dirs, 
		define_macros=define_macros
	) 
]

# If not building under Windows, add the pypvm extension. 
# Version 2.6.x does not have the 'optional' option for extension so we have to 
# do this the old fashined way (if platform()!='WINDOWS'). 
if system()!='Windows':
	ext_modules.append(
		Extension(
			'pyopus.parallel._pypvm_core', 
			['src/pypvm/pypvm_coremodule.c'], 
			include_dirs=include_dirs, 
			define_macros=define_macros, 
			libraries=['pvm3', 'fpvm3', 'gpvm3', 'pvmtrc'], 
			# optional=True # Not supported in 2.6.x
		)
	)

# Settings
setup(name='PyOPUS',
	version='0.7',
	description='A simulation-based design optimization library',
	long_description=\
"""
PyOpus is a library for simulation-based circuit optimization. It provides 
several optimization algorithms (Coordinate Search, Hooke-Jeeves, 
Nelder-Mead Simplex, Successive Approximation Simplex, PSADE (global), ...). 
Optimization algorithms can be fitted with plugins that are triggered at 
every function evaluation and have full access to the internals of the 
optimization algorithm. 

PyOpus currently supports SpiceOpus and HSPICE (supports OP, DC, AC, TRAN, 
and NOISE analyses, supports collecting of device properties like Vdsat). 
The simulator interface is simple and can be easily extended to support any 
simulator. 

PyOpus provides an extensible library of postprocessing functions which 
enable you to easily extract performance measures like gain, bandwidth, rise 
time, slew-rate, etc. from simulation results. 

The collected performance measures can be further post-processed to obtain
a user-defined cost function which can be used for guiding the optimization
algorithms toward better circuits. 

Parallel computing is supported through the use of the PVM library. The 
cluster of computers is represented by a VirtualMachine object which 
provides a simple interface to the underlying parallel programming library. 

An event-driven master-slave model of computation is implemented in the 
EvtDrvMS class and enables you to quicly and easily implement parallel 
algorithms by means of messages and message handlers. EvtDrvMS is capable of 
handling host and task failures, as well as on-the-fly addition of new hosts 
to the parallel computing environment provided that the underlying 
virtual machine library supports it (e.g. PVM). 

Currently the Parallel Point Evaluator (for Monte-Carlo and sweeps) and the 
PSADE (global optimization) method support parallel runs across multiple 
computers. 

PyOpus provides a plotting mechanism based on Matplotlib and wxPython which 
has an interface and capabilities similar to those available in MATLAB. 
The plots are handled by a separate thread so you can write your programs 
just like you were used to in MATLAB. The plotting capability is used in 
the visual module that enables the programmer to quickly visualize the 
simulation results after or even during the optimization run. 
""", 
	author='Arpad Buermen',
	author_email='arpadb@fides.fe.uni-lj.si',
	url='http://www.pyopus.si/',
	platforms='Linux, Windows', 
	license='LGPL V3', 
	packages=[
		'pyopus', 
		'pyopus.misc', 
		'pyopus.wxmplplot',
		'pyopus.visual', 
		'pyopus.optimizer', 
		'pyopus.parallel', 
		'pyopus.simulator', 
		'pyopus.evaluator'
	],
	ext_modules=ext_modules
)

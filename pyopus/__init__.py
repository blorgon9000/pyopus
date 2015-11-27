# -*- coding: UTF-8 -*-
"""
PyOPUS is a library for simulation-based optimization of arbitrary systems. 
It was developed with circuit optimization in mind. PyOPUS provides
several optimization algorithms (Coordinate Search, Hooke-Jeeves,
Nelder-Mead Simplex, Successive Approximation Simplex, PSADE (global), MADS, ...).
Optimization algorithms can be fitted with plugins that are triggered at
every function evaluation and have full access to the internals of the
optimization algorithm. 

PyOPUS has a large library of optimization test functions that can be used for 
optimization algorithm development. The functions include benchmark sets by 
Moré-Garbow-Hillstrom, Lukšan-Vlček (nonsmooth problems), Karmitsa (nonsmooth 
problems), Moré-Wild, global optimization problems by Yao, Hedar, and Yang, 
problems used in the developement of MADS algorithms, and an interface to 
thousands of problems in the CUTEr/CUTEst collection. Benchmark results can 
be converted to data profiles that visualize the relative performance of 
optimization algorithms. 

The ``pyopus.simulator`` module currently supports SPICE OPUS, HSPICE, and 
SPECTRE (supports OP, DC, AC, TRAN, and NOISE analyses, as well as, collecting 
device properties like Vdsat). The interface is simple can be easily extended to 
support any simulator.

PyOPUS provides an extensible library of postprocessing functions which
enable you to easily extract performance measures like gain, bandwidth, rise
time, slew-rate, etc. from simulation results.

The collected performance measures can be further post-processed to obtain
a user-defined cost function which can be used for guiding the optimization
algorithms toward better circuits.

PyOPUS provides sensitivity analysis, parameter screening, worst case 
performance analysis, worst case distance analysis (deterministic approximation 
of parametric yield), and Monte Carlo analysis (statistical approximation of 
parametric yield). Designs can be sized efficiently across a large number of 
corners. Finally, automated design for achieving a parametric yield target is 
also available. Most of these procedures can take advantage of parallel 
computing which significantly speeds up the process. 

Parallel computing is supported through the use of the MPI library. A 
cluster of computers is represented by a VirtualMachine object which
provides a simple interface to the underlying MPI library. Parallel programs 
can be written with the help of a simple cooperative multitasking OS. This 
OS can outsource function evaluations to computing nodes, but it can also 
perform all evaluations on a single processor. 

Writing parallel programs follows the UNIX philosophy. A function can be run 
remotely with the ``Spawn`` OS call. One or more remote functions can be 
waited on with the ``Join`` OS call. The OS is capable of running a parallel 
program on a single computing node using cooperative multitasking or on a set 
of multiple computing nodes using a VirtualMachine object. Parallelism can be 
introduced on multiple levels of the program (i.e. parallel performance 
evaluation across multiple corners, parallel optimization algorithms, evaluation 
of multiple worst case performances in parallel, ...). 

PyOPUS provides a plotting mechanism based on MatPlotLib and wxPython with 
an interface and capabilities similar to those available in MATLAB.
The plots are handled by a separate thread so you can write your programs
just like you are were used to in MATLAB. Professional quality plots can be 
easily exported to a large number of raster and vector formats for inclusion 
in your documents. The plotting capability is used in the ``pyopus.visual`` module 
that enables the programmer to visualize the simulation results after an 
optimization run or even during an optimization run. 
"""

from misc.identify import revision, locationID 
# print("PyOpus library R%d @ %s, (c)2008-2015 Arpad Buermen" % (revision, locationID()))

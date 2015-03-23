"""
PyOPUS is a library for simulation-based circuit optimization. It provides
several optimization algorithms (Coordinate Search, Hooke-Jeeves,
Nelder-Mead Simplex, Successive Approximation Simplex, PSADE (global), ...).
Optimization algorithms can be fitted with plugins that are triggered at
every function evaluation and have full access to the internals of the
optimization algorithm.

PyOPUS currently supports SpiceOpus and HSPICE (supports OP, DC, AC, TRAN,
and NOISE analyses, supports collecting of device properties like Vdsat).
The simulator interface is simple can be easily extended to support any
simulator.

PyOPUS provides an extensible library of postprocessing functions which
enable you to easily extract performance measures like gain, bandwidth, rise
time, slew-rate, etc. from simulation results.

The collected performance measures can be further post-processed to obtain
a user-defined cost function which can be used for guiding the optimization
algorithms toward better circuits.

Parallel computing is supported through the use of the PVM library. The
cluster of computers is represented by a VirtualMachine object which
provides a simple interface to the underlying PVM library. An event-driven
master-slave model of computation is implemented in the EvtDrvMS class and
enables you to quicly and easily implement parallel algorithms by means of
messages and message handlers. EvtDrvMS is capable of handling host and task
failures, as well as on-the-fly addition of new hosts to the parallel
computing environment provided that the underlying virtual machine library
supports it (e.g. PVM).

Currently only the Parallel Point Evaluator (e.g. Monte-Carlo) and the PSADE
(global optimization) method support parallel runs across multiple computers.

PyOPUS provides a plotting mechanism based on MatPlotLib and wxPython which
has an interface and capabilities similar to those available in Matlab.
The plots are handled by a separate thread so you can write your programs
just like you are were used to in Matlab. The plotting capability is used in
the visual module that enables the programmer to visualize the simulation
results after or even during the optimization run. 
"""

from misc.id import revision, locationID 
print("pyOpus library R%d @ %s, (c)2008-2011 Arpad Buermen" % (revision, locationID))

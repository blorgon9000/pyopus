"""
**CUTEr problem manager**

Currently works only under Linux.

CUTEr is a collection of problems and a systef for interfacing to these 
problems. Every problem is described in a .sif file. This file is converted 
to FORTRAN code (evaluators) that evaluate the function's terms and their 
derivatives. CUTEr provides fortran subroutines (CUTEr tools) that call 
evaluators and return things like the function's value or the value of the 
function's gradient at a given point, etc. CUTEr tools are available in a 
static library libcuter.a. 

Every CUTEr problem is first compiled with a tool named sifdecode that produces 
the evaluators in FORTRAN. The evaluators are compiled and linked with the 
CUTEr tools and the Python interface to produce a loadable binary module for 
Python. The evaluators must be compiled with the same compiler as CUTEr tools. 
The loadable Python module is stored in a cache so that it does not have to be 
recompiled every time one wants to use the CUTEr problem. 

**Installing CUTEr**

Before you use this module, you must build sifdecode and the CUTEr tools library. 

Unpack the SIF decoder, build it by running install_sifdec (answers to the first 
five questions should be: PC, Linux, gfortran, double precision, large). Set 
the SIFDEC enavironmental variable so that it contains teh path to the folder 
where you unpacked the sifdecode source (where install_sifdec is located). 
After sifdecode is built a subfolder named SifDec.large.pc.lnx.gfo is created 
in $SIFDEC. Set the MYSIFDEC environmental variable so that it contains the 
path to this folder. Add $MYSIFDEC/bin to PATH so sifdecode will be accessible 
from anywhere. Also add $SIFDEC/common/man to the MANPATH environmental 
variable. 

Unpack CUTEr. The tools library (libcuter.a) must be built with -fPIC (as 
position independent code). To achieve this, the file config/linux.cf in CUTEr 
installation must be edited and the -fPIC option added to lines that begin with::
  
#define FortranFlags ...
#define CFlags ...

Build the CUTEr library by running install_cuter (answers to the first five 
questions should be: PC, Linux, gfortran, GNU gcc, double precision, large). 
Set the CUTER environmental variable so that it contains the path to the 
folder where you unpacked CUTEr (the one where install_cuter is located). 
After CUTEr is compiled (i.e. libcuter.a is built) a subfolder named 
CUTEr.large.pc.lnx.gfo is created. Set the MYCUTER environmental variable so 
that it contains the path to this folder. Add $CUTER/common/man to MANPATH  
$CUTER/common/include to C_INCLUDE_PATH, $MYCUTER/double/lib to LIBPATH, and 
$MYCUTER/bin to PATH. 

On 64-bit systems use the standard version of CUTEr (not the 64-bit one). 

Download SIF files describing the available problems from 
http://www.cuter.rl.ac.uk/ and create a SIF repository (create a directory, 
unpack the problems, and set the MASTSIF environmental variable to point to 
the directory holding the SIF files). 

**The Python interface to CUTEr**

Create a folder for the CUTEr cache. Set the ``PYCUTER_CACHE`` environmental 
variable to contain the path to the cache folder. Add ``$PYCUTER_CACHE`` to the 
``PYTHONPATH`` envorinmental variable. The last step will make sure you can 
access the cached problems from Python. 

If ``PYCUTER_CACHE`` is not set the current directory (.) is used for caching. 
Problems are cached in ``$PYCUTER_CACHE/pycuter``. 

The interface depends on gfortran and lapack. Make sure they are installed. 

Once a problem is built only the following files in 
``$PYCUTER_CACHE/pycuter/PROBLEM_NAME`` are needed for the problem to work: 

* ``_pycuteritf.so``    -- CUTEr problem interface module
* ``__init__.py``       -- module initialization and wrappers
* ``OUTSDIF.d``         -- problem information

A cached problem can be imported with ``from pycuter import PROBLEM_NAME``. One 
can also use :func:`importProblem` which returns a reference to the problem module. 

Available functions

* :func:`clearCache`     -- remove Python interface to problem from cache
* :func:`prepareProblem` -- decode problem and build Python interface
* :func:`importProblem`  -- import problem interface module from cache
  (prepareProblem must be called first)
* :func:`isCached`       -- returns ``True`` if a problem is in cache 

**CUTEr problem structure**

CUTEr provides constrained and unconstrained test problems. The objective 
function is a map from Rn to R:

.. math::
  y = f(x)
  
where :math:`y` is a real scalar (member of :math:`R`) and :math:`x` is a 
n-dimensional real vector (member of :math:`R^n`). The i-th component of 
:math:`x` is denoted by :math:`x_i` and is also referred to as the i-th 
problem variable. Problem variables can be continuous (take any value from 
:math:`R`), integer (take only integer values) or boolean (only 0 and 1 
are allowed as variable's value).

The optimization problem can be subject to simple bounds of the form

.. math::
  b_{li} \le x_i \le b_{ui}
  
where :math:`b_{li}` and :math:`b_{ui}` denote the lower and the upper bound 
on the i-th component of vector x. If there is no lower (or upper) bound on 
:math:`x_i` CUTEr reports :math:`b_{li}=-1e20` (or :math:`b_{ui}=1e20`).

Beside simple bounds on problem variables some CUTEr problems also have more 
sophisticated constraints of the form

.. math::
  c_{li} \leq c_i(x) \leq c_{ui}
  
  c_i(x) = 0   
  
where :math:`c_i` is the i-th constraint function. The former is an inequality 
constraint while the latter is an equality constraint. CUTEr problems have 
generally m such constraints. In :math:`c_{li}` and :math:`c_{ui}` the values 
-1e20 and 1e20 stand for :math:`-\infty` and :math:`+\infty`. For equality 
constraints :math:`c_{li}` and :math:`c_{ui}` are both equal to 0. All 
constraint functions :math:`c_i(x)` are joined in a single vector-valued 
function (map from :math:`R^n` to :math:`R^m`) named :math:`c(x)`.

CUTEr can order the constraints in such manner that equality constraints 
appear before inequality constraints. It is also possible to place linear 
constraints before nonlinear constraints. This of course reorders the 
components of c(x). Similarly variables (components of x) can also be 
reordered in such manner that nonlinear variables appear before linear ones. 

**The Lagrangian function, the Jacobian matrix, and the Hessian matrix**

The Lagrangian function is defined as

.. math::
  L(x, v) = f(x) + v_1 c_1(x) + v_2 c_2(x) + ... + v_m c_m(x)
  
Vector v is m-dimensional. Its components are the Lagrange multipliers.

The Jacobian matrix (:math:`J`) is the matrix of constraint gradients. 
One row corresponds to one constraint function :math:`c_i(x)`. The matrix 
has n columns. The element in the i-th row and j-th column is the derivative 
of the i-th constraint function with respect to the j-th problem variable. 

The Hessian matrix (:math:`H`) is the matrix of second derivatives. The 
element in i-th row and j-th column corresponds to the second derivative with 
respect to the i-th and j-th problem variable. The Hessian is a symmetric 
matrix so it is sufficient to know its diagonal and its upper triangle.

Beside Hessian of the objective CUTEr can also calculate the Hessian of the 
Lagrangian and the Hessians of the constraint functions.

The gradient (of objective, Lagrangian, or constraint functions) is always 
taken with respect to the problem's variables. Therefore it always has n 
components. 

**What does the CUTEr interface of a problem offer**

All compiled test problems are stored in a cache. The location of the cache 
can be set by defining the ``PYCUTER_CACHE`` environmental variable. If no 
cache location is defined the current working directory is used for caching 
the compiled test problems. The CUTEr interface to Python has a manager 
module named ``cutermgr``.

The manager module (``cutermgr``) offers the following functions:

* :func:`clearCache` -- remove a compiled problem from cache
* :func:`prepareProblem` -- compile a problem, place it in the cache, 
  and return a reference to the imported problem module
* :func:`importProblem` -- import an already compiled problem from cache 
  and return a reference to its module

Every problem module has several functions that access the corresponding problem's CUTEr tools:

* :func:`getinfo` -- get problem description
* :func:`varnames` -- get names of problem's variables
* :func:`connames` -- get names of problem's constraints
* :func:`objcons` -- objective and constraints
* :func:`obj` -- objective and objective gradient
* :func:`cons` -- constraints and constraints gradients/Jacobian
* :func:`lagjac` -- gradient of objective/Lagrangian and constraints Jacobian
* :func:`jprod` -- product of constraints Jacobian with a vector
* :func:`hess` -- Hessian of objective/Lagrangian
* :func:`ihess` -- Hessian of objective/constraint
* :func:`hprod` -- product of Hessian of objective/Lagrangian with a vector
* :func:`gradhess` -- gradient and Hessian of objective (if m=0) or
  gradient of objective/Lagrangian, Jacobian, and Hessian of Lagrangian (if m > 0)
* :func:`scons` -- constraints and sparse Jacobian of constraints
* :func:`slagjac` -- gradient of objective/Lagrangian and sparse Jacobian
* :func:`sphess` -- sparse Hessian of objective/Lagrangian
* :func:`isphess` -- sparse Hessian of objective/constraint
* :func:`gradsphess` -- gradient and sparse Hessian of objective (if m=0) or
  gradient of objective/Lagrangian, sparse Jacobian, and sparse Hessian of Lagrangian (if m > 0)
* :func:`report` -- get usage statistics

All sparse matrices are returned as scipy.sparse.coo_matrix objects.

**How to use cutermgr**

First you have to import the cutermgr module::

  from pyopus.problems import cutermgr

If you want to remove the ``HS71`` problem from the cache, type::

  cutermgr.clearCache('HS71')

To prepare the ``ROSENBR`` problem, type::

  cutermgr.prepareProblem('ROSENBR')

This removes the existing ``ROSENBR`` entry from the cache before rebuilding the 
problem interface. The compiled problem is stored as ``ROSENBR`` in cache.

Importing a prepared problem can be done with cutermgr::

  rb=cutermgr.importProblem('ROSENBR')

Now you can use the problem. Let's get the information about the imported 
problem and extract the initial point::

  info=rb.getinfo()
  x0=info['x']

To evaluate the objective function's value at the extracted initial point, 
type::

  f=rb.obj(x0)
  print "f(x0)=", f

To get help on all interface functions of the previously imported problem, 
type::

  help(rb)

You can also get help on individual functions of a problem interface::

  help(rb.obj)

The cutermgr module has also builtin help::

  help(cutermgr)
  help(cutermgr.importProblem)

**Storing compiled problems in cache under arbitrary names**

A problem can be stored in cache using a different name than the original 
CUTEr problem name (the name of the corresponding SIF file) by specifying the 
destination parameter to :func:`prepareProblem`. For instance to prepare 
the ``ROSENBR`` problem (the one defined by ``ROSENBR.SIF``) and store it in 
cache as ``rbentry``, type::

  cutermgr.prepareProblem('ROSENBR', destination='rbentry')

Importing the compiled problem interface and its removal from the cache must 
now use rbentry instead of ``ROSENBR``::

  # Use cutermgr.importProblem()
  rb=cutermgr.importProblem('rbentry')  

  # Remove the compiled problem from cache
  cutermgr.clearCache('rbentry')  

To check if a problem is in cache under the name ``rbentry`` without trying to 
import the actual module, use::

  if cutermgr.isCached('rbentry'):
  ...

**Specifying problem parameters and sifdecode command line options**

Some CUTEr problems have parameters on which the problem itself depends. Often 
the dimension of the problem depends on some parameter. Such parameters must 
be passed to sifdecode with the ``-param`` option. The CUTEr interface handles 
such parameters with the ``sifParams`` argument to :func:`prepareProblem`. 
Parameters are passed in the form of a Python dictionary, where the key 
specifies the name of a parameter. The value of a parameter is converted 
using str() to a string and passed to sifdecode's command line as 
``-param key=value``::

  # Prepare the LUBRIFC problem, pass NN=10 to sifdecode
  cutermgr.prepareProblem("LUBRIFC", sifParams={'NN': 10})

Arbitrary command line options can be passed to sifdecode by specifying them 
in form of a list of strings and passing the list to :func:`prepareProblem` 
as ``sifOptions``. The following is the equivalent of the last example::

  # Prepare the LUBRIFC problem, pass NN=10 to sifdecode
  cutermgr.prepareProblem("LUBRIFC", sifOptions=['-param', 'NN=10'])

**Specifying variable and constraint ordering**

To put nonlinear variables before linear variables set the ``nvfirst`` parameter 
to ``True`` and pass it to func:`prepareProblem`::

  cutermgr.prepareProblem("SOMEPROBLEM", nvfirst=True)

If ``nvfirst`` is not specified it defaults to ``False``. In that case no 
particular variable ordering is imposed. The variable ordering will be 
reflected in the order of variable names returned by the :func:`varnames` problem 
interface function.

To put equality constraints before inequality constraints set the ``efirst`` 
parameter to ``True``::

  pycutermgr.prepareProblem("SOMEPROBLEM", efirst=True)

Similarly linear constraints can be placed before nonlinear ones by setting 
``lfirst`` to ``True``::

  pycutermgr.prepareProblem("SOMEPROBLEM", lfirst=True)

Parameters ``efirst`` and ``lfirst`` default to ``False`` meaning that no particular 
constraint ordering is imposed. The constraint ordering will be reflected in 
the order of constraint names returned by the :func:`connames` problem interface 
function.

If both ``efirst`` and ``lfirst`` are set to ``True``, the ordering is a follows: 
linear equality constraints followed by linear inequality constraints, 
nonlinear equality constraints, and finally nonlinear inequality constraints. 

**Problem information**

The problem information dictionary is returned by the :func:`getinfo` problem 
interface function. The dictionary has the following entries

* ``name`` -- problem name
* ``n`` -- number of variables
* ``m`` -- number of constraints (excluding bounds)
* ``x`` -- initial point (1D array of length n)
* ``bl`` -- 1D array of length n with lower bounds on variables
* ``bu`` -- 1D array of length n with upper bounds on variables
* ``nnzh`` -- number of nonzero elements in the diagonal and upper triangle of sparse 
  Hessian
* ``vartype`` -- 1D integer array of length n storing variable type
  0=real, 1=boolean (0 or 1), 2=integer
* ``nvfirst`` -- boolean flag indicating that nonlinear variables were placed before 
  linear variables 
* ``sifparams`` -- parameters passed to sifdecode with the sifParams argument to 
  :func:`prepareProblem`. ``None`` if no parameters were given
* ``sifoptions`` -- additional options passed to sifdecode with the sifOptions 
  argument to :func:`prepareProblem`. ``None`` if no additional options were given.

Additional entries are available if the problem has constraints (m>0):

* ``nnzj`` -- number of nonzero elements in sparse Jacobian of constraints
* ``v`` -- 1D array of length m with initial values of Lagrange multipliers
* ``cl`` -- 1D array of length m with lower bounds on constraint functions
* ``cu`` -- 1D array of length m with upper bounds on constraint functions
* ``equatn`` -- 1D boolean array of length m indicating whether a constraint is an 
  equality constraint
* ``linear`` -- 1D boolean array of length m indicating whether a constraint is a 
  linear constraint
* ``efirst`` -- boolean flag indicating that equality constraints were places 
  before inequality constraints
* ``lfirst`` -- boolean flag indicating that linear constraints were placed before 
  nonlinear constraints

The names of variables and constraints are returned by the :func:`varnames` and 
:func:`connames` problem interface functions.

**Usage statistics**

The usage statistics dictionary is returned by the report() problem interface 
function. The dictionary has the following entries

* ``f`` -- number of objective evaluations
* ``g`` -- number of objective gradient evaluations
* ``H`` -- number of objective Hessian evaluations
* ``Hprod`` -- number of Hessian multiplications with a vector
* ``tsetup`` -- CPU time used in setup
* ``trun`` -- CPU time used in run

For constrained problems the following additional members are available

* ``c`` -- number of constraint evaluations
* ``cg`` -- number of constraint gradient evaluations
* ``cH`` -- number of constraint Hessian evaluations 
  
**Problem preparation and internal cache organization**

The cache (``$PYCUTER_CACHE``) has one single subdirectory named pycuter holding 
all compiled problem interafaces. This way problem interface modules are 
accessible as ``pycuter.NAME`` because ``$PYCUTER_CACHE`` is also listed in 
``PYTHONPATH``.

``$PYCUTER_CACHE/pycuter`` has a dummy ``__init__.py`` file generated by 
:func:`prepareProblem` which specifies that ``$PYCUTER_CACHE/pycuter`` 
is a Python module. Every problem has its own subdirectory in 
``$PYCUTER_CACHE/pycuter``. In that subdirectory problem decoding (with sifdecode) 
and compilation (with gfortran and Python distutils) take place. 
:func:`prepareProblem` also generates an ``__init__.py`` file for every problem 
which takes care of initialization when the problem interface is imported.

The actual binary interaface is in ``_pycuteritf.so``. The ``__init__.py`` script 
requires the presence of the ``OUTSDIF.d`` file where the problem description is 
stored. Everything else is neded at compile time only.

Some functions in the ``_pycuteritf.so`` module are private (their name starts 
with an underscore. These functions are called by wrappers defined in 
problem's ``__init__.py``. An example for this are the sparse CUTEr tools 
like :func:`scons`. :func:`scons` is actually a wrapper defined in ``__init__.py``. 
It calls the :func:`_scons` function from the problem's ``_pycuteritf.so`` 
binary interface module and converts its return values to a 
:class:`~scipy.sparse.coo_matrix` object. 
:func:`scons` returns the :class:`~scipy.sparse.coo_matrix` object for J instead 
of a NumPy array object. The problem's ``_pycuteritf`` binary module is also 
accessible. If the interface module is imported as ``rb`` then the :func:`_scons` 
interface function can be accessed as ``rb._pycuteritf._scons``. 

This module does not depend on PyOPUS. It depends only on the 
:mod:`~pyopus.problems.cuteritf` module. 
"""

import os, shutil, sys, platform, itertools, re
from os import environ, getcwd
import subprocess
from glob import glob
from cuteritf import itf_c_source
from glob import iglob

__all__ = [ 'clearCache', 'prepareProblem', 'importProblem', 'isCached', 
'updateClassifications', 'problemProperties', 'findProblems'  ]

# 
# Verify if the CUTEr setup is sane
#
if platform.system()!='Linux':
	raise Exception, "Only Linux is supported at this time."
	
if not 'SIFDEC' in os.environ:
	raise Exception, "The SIFDEC environmental variable is not set."

if not 'MYSIFDEC' in os.environ:
	raise Exception, "The MYSIFDEC environmental variable is not set."

if not 'CUTER' in os.environ:
	raise Exception, "The CUTER environmental variable is not set."

if not 'MYCUTER' in os.environ:
	raise Exception, "The MYCUTER environmental variable is not set."

if not 'MASTSIF' in os.environ:
	raise Exception, "The MASTSIF environmental variable is not set."

if not 'PYCUTER_CACHE' in os.environ:
	print("Warning: the PYCUTER_CACHE environmental variable is not set.\nCurrent folder will be used for caching.")

if not os.path.isfile(os.path.join(os.environ['MYCUTER'], 'double', 'lib', 'libcuter.a')):
	raise Exception, "libcuter.a is not available. Is CUTEr installed?"

if not os.environ['PYCUTER_CACHE'] in sys.path:
	raise Exception, "$PYCUTER_CACHE is not in PYTHONPATH."
	
	
#
# The sutup.py script with a placeholder for platform-dependent part. 
#
setupScript="""#!/usr/bin/env python
# (C)2011 Arpad Buermen
# Licensed under LGPL V2.1

#
# Do not edit. This is a computer-generated file. 
#

from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc
import os
from subprocess import call
from glob import glob

#
# OS specific
#

%s

#
# End of OS specific
#

# Settings
setup(name='PyCuter automatic test function interface builder',
	version='1.0',
	description='Builds a CUTEr test function interface for Python.',
	long_description='Builds a CUTEr test function interface for Python.', 
	author='Arpad Buermen',
	author_email='arpadb@fides.fe.uni-lj.si',
	url='',
	platforms='Linux', 
	license='LGPL V2.1', 
	packages=[],
	ext_modules=[
		Extension(
			'_pycuteritf', 
			['cuteritf.c'], 
			include_dirs=include_dirs, 
			define_macros=define_macros, 
			extra_objects=objFileList, 
			libraries=libraries
		)
	]
)
"""

#
# Linux-specific part of setup.py
#
setupScriptLinux="""
define_macros=[('LINUX', None)]
include_dirs=[os.path.join(get_python_inc(plat_specific=1), 'numpy')]
objFileList=glob('*.o')
objFileList.append(os.path.join(os.environ['MYCUTER'], 'double', 'lib', 'libcuter.a'))
libraries=['gfortran', 'lapack', 'blas']
"""

#
# Problem interface module initialization file with placeholders for 
# efirst, lfirst, and nvfirst. This also defines the wrapper functions. 
# A placeholder is included for problem name and ordering. 
#
initScript="""# PyCUTEr problem interface module intialization file
# (C)2011 Arpad Buermen
# Licensed under LGPL V2.1

\"\"\"Interface module for CUTEr problem %s with ordering 
  efirst=%s, lfirst=%s, nvfirst=%s
sifdecode parameters : %s
sifdecode options    : %s

Available functions
getinfo    -- get problem information
varnames   -- get names of problem's variables
connames   -- get names of problem's constraints
objcons    -- objective and constraints
obj        -- objective and objective gradient
cons       -- constraints and constraints gradients/Jacobian
lagjac     -- gradient of objective/Lagrangian and constraints Jacobian 
jprod      -- product of constraints Jacobian with a vector
hess       -- Hessian of objective/Lagrangian
ihess      -- Hessian of objective/constraint
hprod      -- product of Hessian of objective/Lagrangian with a vector
gradhess   -- gradient and Hessian of objective (unconstrained problems) or
              gradient of objective/Lagrangian, Jacobian of constraints and 
              Hessian of Lagrangian (constrained problems)
scons      -- constraints and sparse Jacobian of constraints
slagjac    -- gradient of objective/Lagrangian and sparse Jacobian
sphess     -- sparse Hessian of objective/Lagrangian
isphess    -- sparse Hessian of objective/constraint
gradsphess -- gradient and sparse Hessian of objective (unconstrained probl.) 
              or gradient of objective/Lagrangian, sparse Jacobian of 
              constraints and sparse Hessian of Lagrangian (constrained probl.)
report     -- get usage statistics
\"\"\"

from _pycuteritf import *
import _pycuteritf
import os
from scipy.sparse import coo_matrix
from numpy import zeros

# Get the directory where the bunary module (and OUTSDIF.d) are found. 
(_directory, _module)=os.path.split(_pycuteritf.__file__)

# Problem info structure and dimension
info=None
n=None
m=None

# Constraints and variable ordering
efirst=%s
lfirst=%s
nvfirst=%s

# Remember current directory and go to module directory where OUTSDIF.d is located
fromDir=os.getcwd()
os.chdir(_directory)
	
# Get problem dimension
(n, m)=_pycuteritf._dims()
	
# Set up the problem and get basic information
info=_pycuteritf._setup(efirst, lfirst, nvfirst)
	
# Store constraint and variable ordering information 
if m>0:
	info['efirst']=efirst
	info['lfirst']=lfirst
info['nvfirst']=nvfirst 

# Store sifdecode parameters and options
info['sifparams']=%s
info['sifoptions']=%s

# Go back to initial directory
os.chdir(fromDir)

# Return problem info
def getinfo():
	\"\"\"
	Return the problem info dictionary.
	
	info=geinfo()
	
	Output 
	info -- dictionary with the summary of test function's properties
	
	The dictionary has the following members:
	name       -- problem name
	n          -- number of variables
	m          -- number of constraints (excluding bounds)
	x          -- initial point (1D array of length n)
	bl         -- 1D array of length n with lower bounds on variables 
	bu         -- 1D array of length n with upper bounds on variables
	nnzh       -- number of nonzero elements in the diagonal and upper triangle of
	              sparse Hessian
	vartype    -- 1D integer array of length n storing variable type
	              0=real,  1=boolean (0 or 1), 2=integer
	nvfirst    -- boolean flag indicating that nonlinear variables were placed
	              before linear variables
	sifparams  -- parameters passed to sifdecode with the -param option 
	              None if no parameters were given
	sifoptions -- additional options passed to sifdecode
	              None if no additional options were given. 
			   
	For constrained problems the following additional members are available
	nnzj    -- number of nonzero elements in sparse Jacobian of constraints
	v       -- 1D array of length m with initial values of Lagrange multipliers
	cl      -- 1D array of length m with lower bounds on constraint functions
	cu      -- 1D array of length m with upper bounds on constraint functions
	equatn  -- 1D boolean array of length m indicating whether a constraint
	           is an equation constraint
	linear  -- 1D boolean array of length m indicating whether a constraint
	           is a linear constraint
	efirst  -- boolean flag indicating that equation constraints were places
	           before inequation constraints
	lfirst  -- boolean flag indicating that linear constraints were placed 
	           before nonlinear constraints
	\"\"\"
	return info

def varnames():
	\"\"\"
	Return the names of problem's variables.
	
	nameList=varnames()
	
	nameList -- a list of strings representing the names of problem's variables.
	            The variabels are ordered according to nvfirst flag. 
	\"\"\"
	return _pycuteritf._varnames()
	
def connames():
	\"\"\"
	Return the names of problem's constraints.
	
	nameList=connames()
	
	nameList -- a list of strings representing the names of problem constraints. 
	            The constraints are ordered according to efirst and lfirst flags. 
	\"\"\"
	return _pycuteritf._connames()
	
# Sparse tool wrappers (return scipy.sparse.coo_matrix matrices)
# _scons() wrapper
def scons(x, i=None):
	\"\"\"Returns the value of constraints and 
	the sparse Jacobian of constraints at x.
	
	(c, J)=_scons(x)      -- Jacobian of constraints
	(ci, gi)=_scons(x, i) -- i-th constraint and its gradient
	
	Input
	x -- 1D array of length n with the values of variables
	i -- integer index of constraint (between 0 and m-1)
	
	Output
	c  -- 1D array of length m holding the values of constraints at x
	J  -- a scipy.sparse.coo_matrix of size m-by-n holding the Jacobian at x
	ci -- 1D array of length 1 holding the value of i-th constraint at x
	gi -- a scipy.sparse.coo_matrix of size 1-by-n holding the gradient of i-th constraint at x
	
	This function is a wrapper for _scons().
	\"\"\"
	
	if i is None:
		(c, Ji, Jif, Jv)=_pycuteritf._scons(x)
		return (c, coo_matrix((Jv, (Jif, Ji)), shape=(m, n)))
	else:
		(c, gi, gv)=_pycuteritf._scons(x, i)
		return (c, coo_matrix((gv, (zeros(n), gi)), shape=(1, n)))

# _slagjac() wrapper
def slagjac(x, v=None):
	\"\"\"Returns the sparse gradient of objective at x or Lagrangian at (x, v),
	and the sparse Jacobian of constraints at x.
	
	(g, J)=_slagjac(x)    -- objective gradient and Jacobian
	(g, J)=_slagjac(x, v) -- Lagrangian gradient and Jacobian
	
	Input
	x -- 1D array of length n with the values of variables
	v -- 1D array of length m with the values of Lagrange multipliers
	
	Output
	g -- a scipy.sparse.coo_matrix of size 1-by-n holding the gradient of objective at x or
	     the gradient of Lagrangian at (x, v)
	J -- a scipy.sparse.coo_matrix of size m-by-n holding the sparse Jacobian
	     of constraints at x
	
	This function is a wrapper for _slagjac().
	\"\"\"
	
	if v is None:
		(gi, gv, Ji, Jfi, Jv)=_pycuteritf._slagjac(x)
	else:
		(gi, gv, Ji, Jfi, Jv)=_pycuteritf._slagjac(x, v)
	return (
		coo_matrix((gv, (zeros(n), gi)), shape=(1, n)), 
		coo_matrix((Jv, (Jfi, Ji)), shape=(m, n))
	)

# _sphess() wrapper
def sphess(x, v=None):
	\"\"\"Returns the sparse Hessian of the objective at x (unconstrained problems) 
	or the sparse Hessian of the Lagrangian (constrained problems) at (x, v).
	
	H=_sphess(x)    -- Hessian of objective (unconstrained problems)
	H=_sphess(x, v) -- Hessian of Lagrangian (constrained problems)
	
	Input
	x -- 1D array of length n with the values of variables
	v -- 1D array of length m with the values of Lagrange multipliers
	
	Output
	H -- a scipy.sparse.coo_matrix of size n-by-n holding the sparse Hessian
	     of objective at x or the sparse Hessian of the Lagrangian at (x, v)
	
	This function is a wrapper for _sphess().
	\"\"\"
	
	if v is None:
		(Hi, Hj, Hv)=_pycuteritf._sphess(x)
	else:
		(Hi, Hj, Hv)=_pycuteritf._sphess(x, v)
	return coo_matrix((Hv, (Hi, Hj)), shape=(n, n))

# _isphess() wrapper
def isphess(x, i=None):
	\"\"\"Returns the sparse Hessian of the objective or the sparse Hessian of i-th
	constraint at x.
	
	H=_isphess(x)    -- Hessian of objective 
	H=_isphess(x, i) -- Hessian of i-th constraint
	
	Input
	x -- 1D array of length n with the values of variables
	i -- integer holding the index of constraint (between 0 and m-1)
	
	Output
	H -- a scipy.sparse.coo_matrix of size n-by-n holding the sparse Hessian
	     of objective or the sparse Hessian i-th constraint at x 
	
	This function is a wrapper for _isphess().
	\"\"\"
	
	if i is None:
		(Hi, Hj, Hv)=_pycuteritf._isphess(x)
	else:
		(Hi, Hj, Hv)=_pycuteritf._isphess(x, i)
	return coo_matrix((Hv, (Hi, Hj)), shape=(n, n))

# _gradsphess() wrapper
def gradsphess(x, v=None, lagrFlag=False):
	\"\"\"Returns the sparse Hessian of the Lagrangian, the sparse Jacobian of
	constraints, and the gradient of the objective or Lagrangian.
	
	(g, H)=gradsphess(x)              -- unconstrained problems
	(g, J, H)=gradsphess(x, v, gradl) -- constrained problems
	
	Input
	x     -- 1D array of length n with the values of variables
	v     -- 1D array of length m with the values of Lagrange multipliers
	gradl -- boolean flag. If False the gradient of the objective is returned, 
	         if True the gradient of the Lagrangian is returned. 
	         Default is False
			 
	Output
	g -- a scipy.sparse.coo_matrix of size 1-by-n holding the gradient of objective at x or 
	     the gradient of Lagrangian at (x, v)
	J -- a scipy.sparse.coo_matrix of size m-by-n holding the sparse Jacobian
	     of constraints at x
	H -- a scipy.sparse.coo_matrix of size n-by-n holding the sparse Hessian
	     of objective at x or the sparse Hessian of the Lagrangian at (x, v) 
	
	This function is a wrapper for _gradsphess().
	\"\"\"
	
	if v is None:
		(g, Hi, Hj, Hv)=_pycuteritf._gradsphess(x)
		return (coo_matrix(g), coo_matrix((Hv, (Hi, Hj)), shape=(n, n)))
	else:
		(gi, gv, Ji, Jfi, Jv, Hi, Hj, Hv)=_pycuteritf._gradsphess(x, v, lagrFlag)
		return (
			coo_matrix((gv, (zeros(n), gi)), shape=(1, n)), 
			coo_matrix((Jv, (Jfi, Ji)), shape=(m, n)), 
			coo_matrix((Hv, (Hi, Hj)), shape=(n, n))
		)

# Clean up
del os, fromDir, efirst, lfirst, nvfirst
""" 

def _cachePath():
	"""Return the path to PyCUTEr cache (``PYCUTER_CACHE`` environmental variable). 
	If ``PYCUTER_CACHE`` is not set, return the full path to current work directory. 
	"""
	
	if 'PYCUTER_CACHE' in environ:
		return environ['PYCUTER_CACHE']
	else:
		return os.getcwd()

def isCached(cachedName):
	"""
	Return ``True`` if a problem is in cache.
	
	Keyword arguments:
	
	* *cachedName* -- cache entry name
	"""
	
	# The problem's cache entry 
	problemDir=os.path.join(cachePath, 'pycuter', cachedName)
	
	# See if a directory with problem's name exists
	return os.path.isdir(problemDir)
	
def clearCache(cachedName):
	"""
	Removes a cache entry from cache. 
	
	Keyword arguments:
	
	* *cachedName* -- cache entry name
	""" 
	
	# The problem's cache entry 
	problemDir=os.path.join(cachePath, 'pycuter', cachedName)
	
	# See if a directory with problem's name exists
	if os.path.isdir(problemDir):
		# It exists, delete it. 
		shutil.rmtree(problemDir, True)
	elif os.path.isfile(problemDir):
		# It is a file, delete it. 
		os.remove(problemDir)
	
def prepareCache(cachedName):
	"""
	Prepares a cache entry. 
	If an entry already exists it is deleted first. 
	
	Keyword arguments:
	
	* *cachedName* -- cache entry name
	"""
	
	# The directory with test function entries
	pycuterDir=os.path.join(cachePath, 'pycuter')
	
	# The problem's cache entry 
	problemDir=os.path.join(pycuterDir, cachedName)
	
	# See if a folder named pycuter exists in the cache path. 
	if not os.path.isdir(pycuterDir):
		# Create it. If this fails, give up. The user should delete manualy the
		# offending file which prevents the creation of a directory. 
		os.mkdir(pycuterDir)
	
	# See in pycuterDir if there is an __init__.py file. 
	initfile=os.path.join(pycuterDir, '__init__.py')
	if not os.path.isfile(initfile):
		# Create it
		f=open(initfile, 'w+')
		f.write("#PyCUTEr cache initialization file\n")
		f.close()
	
	# Remove old entry
	clearCache(cachedName)
	
	# Create folder with problem's name
	os.mkdir(problemDir)
	
def decodeAndCompileProblem(problemName, destination=None, sifParams=None, sifOptions=None, quiet=True):
	"""
	Call sifdecode on given problem and compile the resulting .f files. 
	Use gfortran with ``-fPIC`` option for compiling. 
	Collect the resulting object file names and return them. 
	This function is OS dependent. Currently works only for Linux. 
	
	Keyword arguments:
	
	* *problemName* -- CUTEr problem name
	* *destination* -- the name under which the compiled problem interface is stored in the cache
	  If not given problemName is used. 
	* *sifParams* -- parameters passed to sifdecode using the ``-param`` command line option 
	  given in the form of a dictionary with parameter name as key. Values 
	  are converted to strings using :func:`str` and every parameter contributes
	  ``-param key=str(value)`` to the sifdecode's command line options. 
	* *sifOptions* -- additional options passed to sifdecode given in the form of a list of strings. 
	* *quiet* -- supress output (default ``True``)
	
	*destination* must not contain dots because it is a part of a Python module name. 
	"""
	
	# Default destination
	if destination is None:
		destination=problemName
	
	# The problem's cache entry 
	problemDir=os.path.join(cachePath, 'pycuter', destination)
	
	# Remember current work directory and go to cache
	fromDir=os.getcwd()
	os.chdir(problemDir)
	
	# Additional args
	args=[]
	
	# Handle params
	if sifParams is not None:
		for (key, value) in sifParams.iteritems():
			if type(key) is not str:
				raise Exception, "sifParams keys must be strings"
			args+=['-param', key+"="+str(value)]
	
	# Handle options
	if sifOptions is not None: 
		for opt in sifOptions:
			if type(opt) is not str:
				raise Exception, "sifOptions must consist of strings"
			args+=[str(opt)]
	
	# Call sifdecode (assume it is in PATH)
	spawnOK=True
	p=None
	try:
		# Start sifdecode
		p=subprocess.Popen(
				['sifdecode']+args+[problemName], 
				universal_newlines=True, 
				stdout=subprocess.PIPE, stderr=subprocess.STDOUT
			)
		
		# Collect output
		messages=p.stdout.read()
		
		# Now wait for the process to finish. If we don't wait p might get garbage-collected before the
		# actual process finishes which can result in a crash of the interpreter. 
		retcode=p.wait()
		
		# Check return code. Nonzero return code means that something has gone bad. 
		if retcode!=0:
			spawnOK=False
	except:
		spawnOK=False
	
	if not spawnOK or not quiet:
		print messages
		
	# Collect all .f files
	filelist=glob('*.f')
	
	# Compile FORTRAN files 
	for filename in filelist:
		cmd=['gfortran', '-fPIC', '-c', filename]
		if not quiet:
			for s in cmd:
				print s,  
			print
		if subprocess.call(cmd)!=0:
			raise Exception, "gfortran call failed for "+filename
		
	# Collect list of all object files (.o)
	objFileList=glob('*.o')
	
	# Go back to original work directory
	os.chdir(fromDir)
	
	return objFileList
	
def compileAndInstallInterface(problemName, objFileList, destination=None, sifParams=None, sifOptions=None, 
								efirst=False, lfirst=False, nvfirst=False, quiet=True):
	"""
	Compiles and installs the binary interface module. 
	Uses distutils to achieve this. 
	Assumes :func:`decodeAndCompile` successfully completed its task. 
	This function is OS dependent. Currently works only for Linux. 
	
	Keyword arguments:
	
	* *problemName* -- CUTEr problem name
	* *destination* -- the name under which the compiled problem interface is stored in the cache
	  If not given problemName is used. 
	* *objFileList* -- list of object files that were generated using gfortran
	* *sifParams* -- parameters passed to sifdecode using the ``-param`` command line option 
	  given in the form of a dictionary with parameter name as key. Values 
	  are converted to strings using :func:`str` and every parameter contributes::
	  ``-param key=str(value)`` to the sifdecode's command line options. 
	* *sifOptions* -- additional options passed to sifdecode given in the form of a list of strings. 
	* *efirst* -- order equation constraints first (default ``True``)
	* *lfirst* -- order linear constraints first (default ``True``)
	* *nvfirst* -- order nonlinear variables before linear variables 
          (default ``False``)
	* *quiet* -- supress output (default ``True``)
	
	*destination* must not contain dots because it is a part of a Python module name. 
	"""
	
	# Default destination
	if destination is None:
		destination=problemName
	
	# The problem's cache entry 
	problemDir=os.path.join(cachePath, 'pycuter', destination)
	
	# Remember current work directory and go to cache
	fromDir=os.getcwd()
	os.chdir(problemDir)
	
	# Prepare C source of the interface
	modulePath=os.path.split(__file__)[0]
	f=open('cuteritf.c', 'w')
	f.write(itf_c_source)
	f.close()
	
	# Prepare a setup script file 
	f=open('setup.py', 'w+')
	f.write(setupScript % (setupScriptLinux))
	f.close()
	
	# Convert sifParams to a string
	sifParamsStr=""
	if sifParams is not None:
		for (key, value) in sifParams.iteritems():
			sifParamsStr+="%s=%s " % (str(key), str(value))
	
	# Convert sifOptions to a string
	sifOptionsStr=""
	if sifOptions is not None:
		for opt in sifOptions:
			sifOptionsStr+=str(opt)+" "
			
	# Prepare -q option for setup.py
	if quiet:
		quietopt=['-q']
	else:
		quietopt=[]
	
	# Call 'python setup.py build'
	if subprocess.call([sys.executable, 'setup.py']+quietopt+['build'])!=0:
		raise Exception, "Failed to build the Python interface module"
	
	# Call 'python setup.py install --install-lib .'
	if subprocess.call([sys.executable, 'setup.py']+quietopt+['install', '--install-lib', '.'])!=0:
		raise Exception, "Failed to install the Python interface module"
		
	# Create __init__.py
	f=open('__init__.py', 'w+')
	f.write(initScript % (
			problemName, 
			str(bool(efirst)), str(bool(lfirst)), str(bool(nvfirst)), 
			sifParamsStr, sifOptionsStr, 
			str(bool(efirst)), str(bool(lfirst)), str(bool(nvfirst)), 
			str(sifParams), str(sifOptions)
		)
	)
	f.close()
	
	# Go back to original work directory
	os.chdir(fromDir)

def prepareProblem(problemName, destination=None, sifParams=None, sifOptions=None, 
					efirst=False, lfirst=False, nvfirst=False, quiet=True):
	"""
	Prepares a problem interface module, imports and initializes it, 
	and returns a reference to the imported module. 
	
	Keyword arguments:
	
	* *problemName* -- CUTEr problem name
	* *destination* -- the name under which the compiled problem interface is stored in the cache
	  If not given problemName is used. 
	* *sifParams* -- parameters passed to sifdecode using the -param command line option 
	  given in the form of a dictionary with parameter name as key. Values 
	  are converted to strings using :func:`str` and every parameter contributes::
	  ``-param key=str(value)`` to the sifdecode's command line options. 
	* *sifOptions* -- additional options passed to sifdecode given in the form of a list of strings. 
	* *efirst* -- order equation constraints first (default ``True``)
	* *lfirst* -- order linear constraints first (default ``True``)
	* *nvfirst* -- order nonlinear variables before linear variables (default ``False``)
	* *quiet* -- supress output (default ``True``)
	
	*destination* must not contain dots because it is a part of a Python module name. 
	"""
	
	# Default destination
	if destination is None:
		destination=problemName
	
	# Build it
	prepareCache(destination)
	objList=decodeAndCompileProblem(problemName, destination, sifParams, sifOptions, quiet)
	compileAndInstallInterface(problemName, objList, destination, sifParams, sifOptions, efirst, lfirst, nvfirst, quiet)
	
	# Import interface module. Initialization is done by __init__.py. 
	return importProblem(destination)
	
def importProblem(cachedName):
	"""
	Imports and initializes a problem module with CUTEr interface functions. 
	The module must be available in cache (see :func:`prepareProblem`). 
	
	Keyword arguments:
	
	* *cachedName* -- name under which the problem is stored in cache
	"""
	
	# Import interface module. Initialization is done by __init__.py. 
	return __import__('pycuter.'+cachedName, globals(), locals(), [cachedName]) 
	
	
#
# CUTEr problem classification management
#
	
# Problem classifications
classification=None

def updateClassifications(verbose=False):
	"""
	Updates the list of problem classifications from SIF files. 
	Collects the CUTEr problem classification strings. 
	
	* *verbose* -- if set to ``True``, prints output as files are scanned
	
	Every SIF file contains a line of the form 
	  ``-something- classification -code-``
	
	Code has the following format
	  ``OCRr-GI-N-M``
	
	*O* (single letter) - type of objective
	
	* ``N`` .. no objective function defined
	* ``C`` .. constant objective function
	* ``L`` .. linear objective function
	* ``Q`` .. quadratic objective function
	* ``S`` .. objective function is a sum of squares
	* ``O`` .. none of the above
	
	*C* (single letter) - type of constraints
	
	* ``U`` .. unconstrained
	* ``X`` .. equality constraints on variables
	* ``B`` .. bounds on variables
	* ``N`` .. constraints represent the adjacency matrix of a (linear) network
	* ``L`` .. linear constraints
	* ``Q`` .. quadratic constraints
	* ``O`` .. more general than any of the above
	  
	*R* (single letter) - problem regularity
	
	* ``R`` .. regular - first and second derivatives exist and are continuous
	* ``I`` .. irregular problem
	
	*r* (integer) - degree of the highest derivatives provided analytically 
        within the problem description, can be 0, 1, or 2
	
	*G* (single letter) - origin of the problem
	
	* ``A`` .. academic (created for testing algorithms)
	* ``M`` .. modelling exercise (actual value not used in practical application)
	* ``R`` .. real-world problem
	
	*I* (single letter) - problem contains explicit internal variables
	
	* ``Y`` .. yes
	* ``N`` .. no
	
	*N* (integer or ``V``) - number of variables, ``V`` = can be set by user
	
	*M* (integer or ``V``) - number of constraints, ``V`` = can be set by user
	"""
	global classification
	
	classification={}
	# Get a list of files in the MASTSIF folder
	if platform.system()!='Windows':
		# Case-dependent filenames
		it=itertools.chain(
			iglob(os.path.join(os.environ['MASTSIF'], '*.SIF')), 
			iglob(os.path.join(os.environ['MASTSIF'], '*.sif'))
		)
	else:
		# Case-independent filenames
		it=iglob(os.path.join(os.environ['MASTSIF'], '*.sif'))
	
	p=re.compile('\\s*\\*\\s*classification\\s*', re.IGNORECASE)
	for fileName in it:
		# Extract problem name
		head, problemName=os.path.split(fileName)
		problemName=problemName[:-4]
		
		# Open and scan
		fh=open(fileName, 'r')
		
		while True:
			line=fh.readline()
			if not line:
				break
			# Match
			m=p.match(line)
			if m:
				# Found a match
				cf=line[m.end():].strip()
				
				# Report
				if verbose:
					print("%8s: %s" % (problemName, cf))
					
				# Process
				classification[problemName]=cf
				
				# Done with file
				break
		# Close file
		fh.close()
	
def problemProperties(problemName):
	"""
	Returns problem properties (uses the CUTEr problem classification string). 
	
	*problemName* -- problem name
	
	Returns a dictionary with the following members:
	
	* ``objective`` -- objective type code
	* ``constraints`` -- constraints type code
	* ``regular`` -- ``True`` if problem is regular
	* ``degree`` -- highest degree of analytically available derivative
	* ``origin`` -- problem origin code
	* ``internal`` -- ``True`` if problem has internal variables
	* ``n`` -- number of variables (``None`` = can be set by the user)
	* ``m`` -- number of constraints (``None`` = can be set by the user)
	"""
	cfString=classification[problemName]
	
	data={
		'objective': cfString[0].upper(), 
		'constraints': cfString[1].upper(), 
		'regular': cfString[2] in "Rr", 
		'degree': int(cfString[3]), 
		'origin': cfString[5].upper(), 
		'internal': cfString[6] in "Yy", 
	}
	
	parts=cfString.split("-")
	
	if parts[2] in "Vv": 
		data['n']=None
	else:
		data['n']=int(parts[2])
	
	if parts[3] in "Vv": 
		data['m']=None
	else:
		data['m']=int(parts[3])
	
	return data
	
def findProblems(objective=None, constraints=None, regular=None, 
		degree=None, origin=None, internal=None, 
		n=None, userN=None, m=None, userM=None): 
	"""
	Returns the problem names of problems that match the given requirements. 
	The search is based on the CUTEr problem classification string. 
	
	* *objective* -- a string containg one or more letters (NCLQSO) specifying
	  the type of the objective function
	* *constraints* -- a string containg one or more letters (UXBNLQO)
	  the type of the constraints
	* *regular* -- a boolean, ``True`` if the problem must be regular, 
	  ``False`` if it must be irregular
	* *degree* -- list of the form [min, max] specifying the minimum and the 
	  maximum number of analytically available derivatives
	* *origin* -- a string containg one or more letters (AMR) specifying
	  the origin of the problem
	* *internal* -- a boolean, ``True`` if the problem must have internal 
	  variables, ``False`` if internal variables are not allowed
	* *n* -- a list of the form [min, max] specifying the lowest and 
	  the highest allowed number of variables
	* *userN* -- ``True`` of the problems must have user settable number 
          of variables, ``False`` if the number must be hardcoded
	* *m* -- a list of the form [min, max] specifying the lowest and 
	  the highest allowed number of constraints
	* *userM* -- ``True`` of the problems must have user settable number 
          of variables, ``False`` if the number must be hardcoded
	
	Problems with a user-settable number of variables/constraints match any 
	given *n* / *m*. 
	
	Returns the problem names of problems that matched the given requirements. 
	
	If a requirement is not given, it is not applied. 
	
	See :func:`updateClassifications` for details on the letters used in the 
	requirements. 
	"""
	# Prepare classifications
	if classification is None:
		updateClassifications()
		
	# Prepare name list
	nameList=[]
	
	# Go through all problems
	for name in classification.keys():
		# Extract data
		data=problemProperties(name)
		
		# Match
		if objective is not None and data['objective'] not in objective:
			continue
		if constraints is not None and data['constraints'] not in constraints:
			continue
		if regular is not None and data['regular']!=regular:
			continue
		if degree is not None and (data['degree']<degree[0] or data['degree']>degree[1]):
			continue
		if origin is not None and data['origin'] not in origin:
			continue
		if internal is not None and data['internal']!=internal:
			continue
		if n is not None and data['n'] is not None and (data['n']<n[0] or data['n']>n[1]):
			continue
		if  userN is not None:
			if userN and data['n'] is not None:
				continue
			if not userN and data['n'] is None:
				continue
		if m is not None and data['m'] is not None and (data['m']<m[0] or data['m']>m[1]):
			continue
		if userM is not None:
			if userM and data['m'] is not None:
				continue
			if not userM and data['m'] is None:
				continue
		
		# Problem matches, append it to the list
		nameList.append(name)
		
	return nameList
	
# Initialization (performed at first import)
# Save full path to PyCUTEr cache in cachePath. 
cachePath=_cachePath()






































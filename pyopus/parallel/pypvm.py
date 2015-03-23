"""
**The pypvm PVM library wrapper**

Written by W. Michael Petullo and Greg Baker. 

http://pypvm.sourceforge.net

Functions :func:`pkbytestr` and :func:`upkbytestr` added by Arpad Buermen. 

Currently supported under Linux only. 
"""

#!/usr/bin/env python

#   FILE: pypvm.py -- A wrapper to provide additional functionality to the
#         pypvm module.  Import this instead of importing pypvm directly.
# AUTHOR: W. Michael Petullo, wp0002@drake.edu
#   DATE: 16 MAR 1998
#
# Copyright (c) 1999 W. Michael Petullo
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

from _pypvm_core import *
from cPickle import dumps, loads, HIGHEST_PROTOCOL

__all__ = [ 'pk', 'upk' ]

# Add public members of the binary module to __all__ so everything gets documented by Sphinx. 
import _pypvm_core as tmp
pypvm_all=dir(tmp)
del tmp

for name in pypvm_all: 
	if not name.startswith('_'):
		__all__.append(name)

# ============================ pk () ==========================================
def pk(obj):
	"""
	Packs a Python object *obj* into the default pvm buffer by picking it and 
	calling the :func:`pkbytestr` function. 
	"""
#  PRE: obj is assigned a Python object
# POST: obj has been packed into PVM send buffer
	pkbytestr(dumps(obj, HIGHEST_PROTOCOL))

# ============================ upk () =========================================
def upk ():
	"""
	Unpacks a Python object *obj* from the default pvm buffer by first calling 
	the :func:`upkbytestr` function and then unpickles the Python object which 
	is returned to the caller. 
	"""
# POST: FCTVAL == a Python object read from PVM receive buffer
	return (loads(upkbytestr()))

def pk_dump(obj):
	"""
	Pickles a Python object and returns the corresponding byte string. 
	"""
	return dumps(obj, HIGHEST_PROTOCOL)
	
	
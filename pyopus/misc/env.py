"""
**Login shell environment access module**

This module portably handles environmental varibales. On some systems (Windows) 
the environment is the same for all applications. In Linux the environment 
depends on the type of the shell (login or non-login). 

Usually only the login shell provides the full environment. This module imports 
the environmental variables by spawaning user's login shell and collecting the 
environment from it. 

Under Windows this is a trivial module. After being imported under Linux this 
module spawns a login shell, runs ``python env.py`` and collects the 
environment on standard output. 

If this module is run (with ``python env.py``) it dumps the environment in 
form of a hex dumped (:func:`binascii.b2a_hex`) and pickled dictionary 
(:func:`cPickle.dumps`) to standard output and exits. 
"""

import sys
import os
from binascii import a2b_hex, b2a_hex
from platform import system
from subprocess import Popen, PIPE
from cPickle import dumps, loads

__all__ = [ 'environ' ]

environ={}	
"""
A dictionary with variable name for key  holding the login shell's environment.

Not automatically imported to the main PyOPUS module. 
"""

# Internal stuff
prefix="---- Environment ----"
suffix="---- End of environemnt ----"
	
# Dump hex environment to stdout, start with prefix, end with suffix
def dumpEnv():
	"""
	Dumps the string from ``prefix``, the :func:`binascii.b2a_hex` encoded and 
	pickled (:func:`cPickle.dumps`) dictionary :attr:`os.environ`, and the 
	string from ``suffix``. 
	"""
	print prefix
	print b2a_hex(dumps(os.environ))
	print suffix

# Extract environment from string
def extractEnv(txt):
	"""
	Extracts environment from string. The environment is found between 
	``prefix`` and ``suffix``. The environment is decoded using 
	:func:`binascii.a2b_hex` and unpickled with :func:``cPickle.loads`. 
	"""
	# Find prefix and suffix
	start=txt.rfind(prefix)+len(prefix)
	end=txt.rfind(suffix)
	return loads(a2b_hex(txt[start+1:end-1]))

# Collect environment (Unix/Windows)
def init():
	"""
	Initializes the module. Under windows simply copies :attr:`os.environ` to 
	:attr:`environ`. Under Linux it spawns a login shell, dumps its 
	environment, and puts it in :attr:`environ`. 
	"""
	if system()=='Windows':
		# Windows
		# This is easy. Just get the environment. It is always there. 
		# No need to start a login shell. 
		environ.update(os.environ)
	else:
		# Unix
		# Get user's login shell. Default to /bin/sh. 
		from pwd import getpwuid
		pwent=getpwuid(os.getuid())
		home=pwent[5]
		shell=pwent[6]
		if len(shell)==0:
			shell='/bin/sh'
		
		# Run login shell
		p=Popen([shell, '-l'], bufsize=-1, stdin=PIPE, stdout=PIPE, universal_newlines=True)
		# Run this module as a script, use sys.executable because the run happens on the same machine. 
		p.stdin.write(sys.executable+' '+__file__)
		p.stdin.close()
		out=p.stdout.read()
		p.wait()
		
		# Extract hex dump and update environment
		environ.update(extractEnv(out))
		
		# Get PYTHONPATH from environ and update the module search path
		# This is needed if Python was not started from a login shell
		# e.g. when a task is invoked in PVM. 
		if 'PYTHONPATH' in environ:
			for entry in environ['PYTHONPATH'].split(":"):
				# Is the entry already present? 
				if entry not in sys.path:
					# No, append at the end of sys.path
					sys.path.append(entry)
						

if __name__=='__main__':
	# This file is being run, dump environment
	dumpEnv()
else:
	# Imported as module, initialize
	init()

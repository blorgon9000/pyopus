"""
**A unique identifier generator module** 

Also provides the PyOPUS revision number. 

This module is imported automatically when PyOPUS is imported.  
"""

from os import getpid
from socket import gethostname

__all__ = [ 'locationID', 'revision' ] 

revision_str="$Rev: 405 $" 
"""
Temporary storage for revision number where Subversion writes the 
revision number.
"""

revision=int(revision_str.split(' ')[1])
"PyOPUS revision number."

# Unique location fingerprint for debug output
# Get host ID (IP and hostname), works only for IPv4
# (myName, myAliases, myIPs)=gethostbyname_ex(gethostname())
myName=gethostname()
"Hostname where this instance of PyOPUS is running."

# Task id
pid=getpid()
"Task ID of the process under which this instance of PyOPUS is running."

tid=0
"Microthread ID. Set by module :mod:`pyopus.parallel.cooperative`"

# Fingerprint: hostname_pid_microthread
def locationID():
	"""
	Generates a unique identifier for every microthread. 
	
	The identifier has the form ``hostname_processID_microthreadID``. 
	"""
	return "%s_%x_%x" % (myName, pid, tid)

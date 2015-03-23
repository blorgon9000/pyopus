"""
**A unique identifier generator module** 

Also provides the PyOPUS revision number. 

This module is imported automatically when PyOPUS is imported.  
"""

from os import getpid
from socket import gethostname

__all__ = [ 'locationID', 'revision' ] 

revision_str="$Rev: 240 $"
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

# Fingerprint: hostname_pid
locationID=("%s_%x" % (myName, getpid()))
"""
Unique identifier for every Python process on every host.

The identifier has the form ``hostname_processIdentifier``. 
"""

"""
**Debug message generation module** 

Every PyOPUS module generates debug messages of the form 

``locationID subsystem: bodyText``. 

*locationID* uniquely identifies a Python process on a host. 
See :mod:`pyopus.misc.id` for details. 

*subsystem* is a string identifying the PyOPUS subsystem that generated the 
message. 
"""
# Debug message output

from id import locationID

__all__ = [ 'DbgMsg', 'DbgMsgOut' ]

# Format a debug message. 
# Text can be a multiline text. Prefix every line with locationID and subsystem. 
def DbgMsg(subsystem, text):
	"""
	Generates a debug message with *text* in its body. The message originates 
	from the given PyOPUS *subsystem*.
	"""
	prefix="%s %s: " % (locationID, subsystem)
	rows=text.split("\n");
	out=[]
	for row in rows:
		out.append(prefix+row)
	return out
	
# Format and print debug message. 
def DbgMsgOut(subsystem, text):
	"""
	Generates and prints on stdout a debug message with *text* in its body. 
	The message originates from the given PyOPUS *subsystem*.
	"""
	prefix="%s %s: " % (locationID, subsystem)
	rows=text.split("\n");
	for row in rows:
		print(prefix+row)

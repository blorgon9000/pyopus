from pyopus.parallel.base import Msg
import os, sys, time

def hello(vm):
	# Print some output
	print "Worker "+str(vm.taskID())+" at "+str(vm.hostID())+"."
	sys.stdout.flush()
	
	# Return a message with my task ID and host
	return "Hello, I am worker "+str(vm.taskID())+" on "+str(vm.hostID())+" in "+str(os.getcwd())+"."

def bounceBack(vm=None):
	# Enter a loop, receive messages and send them back. 
	# If a None is received, exit. 
	
	# Loop
	while True:
		recv=vm.receiveMessage()
		
		# Not an error and not timeout
		if recv is not None and len(recv)==2:
			# Get source and message
			(sourceID, msg)=recv
			
			# Check if we must exit
			if msg is None:
				break
			# Send back
			vm.sendMessage(sourceID, msg)

def helloLs(vm=None):
	print hello(vm)
	
	# Print current directory contents. 
	contents=os.listdir('.')
	dirs=[]
	files=[]
	for entry in contents:
		if os.path.isdir(entry):
			dirs.append(entry)
		else:
			files.append(entry)
	print("Dirs      : "+str(dirs))
	print("Files     : "+str(files))
	
	sys.stdout.flush()

def pyEvaluator(vm=None, expr=None):
	# Evaluate expression and return the result. Return None if error occurs. 
	try:
		result=eval(expr)
	except:
		result=None
	
	return result
	
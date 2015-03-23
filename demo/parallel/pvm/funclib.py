from pyopus.parallel.base import Msg
import os, sys, time

def idList2str(idList):
	txt=""
	for id in idList:
		txt+=str(id)+" "
	return txt
	
def receiveMessages(vm, timeout):
	mark=time.time()
	while True:
		delta=time.time()-mark
		if delta>timeout:
			break
		msg=vm.receiveMessage(timeout-delta)
		print("Message: "+str(msg))
	
def hello(vm=None):
	print("Hello world. I am worker.")
	
	# Update worker info
	vm.updateWorkerInfo()
	
	# Print info
	print("Host ID   : "+str(vm.hostID()))
	print("Task ID   : "+str(vm.taskID()))
	print("Parent ID : "+str(vm.parentTaskID()))
	print("Work dir  : "+os.getcwd())
	
	sys.stdout.flush()

def helloLs(vm=None):
	hello(vm)
	
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
	hello(vm)
	
	# Evaluate expression and return the result. Return None if error occurs. 
	try:
		result=eval(expr)
	except:
		result=None
	
	return result
	
def bounceBack(vm=None):
	hello(vm)
	
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

	
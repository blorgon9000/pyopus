# SPICE OPUS interactive simulator interface

# Benchmark result on opamp, OPUS, Windows XP 32bit AMD64 farm
#	131 iterations, best in 129, final cost -0.106015891203
#	33.576s/36.315s = 89.9% time spent in simulator

# Benchmark result on opamp, OPUS, Windows XP 32bit
#	131 iterations, best in 129, final cost -0.106015891203
#	45.205s/50.302s = 89.9% time spent in simulator

# Benchmark result on opamp, OPUS interactive, Windows XP 32bit
#	131 iterations, best in 129, final cost -0.104213303008
#	(23.776s+13.819s)/41.856s = 89.8% time spent in simulator

# Benchmark result on opamp, OPUS, Linux AMD64
#	131 iterations, best in 129, final cost -0.106015891203
#	34.175s/36.411s = 93.9% time spent in simulator

# Benchmark result on opamp, OPUS interactive, Linux AMD64 farm
#	131 iterations, best in 129, final cost -0.104213303008
#	32.550s/34.411s = 94.6% time spent in simulator


# Comparison of restart and interactive mode in python
# py  100iter		34.978 (with cost evaluation)
# pyi 100iter		25.140s (with cost evaluation)
#
# restart mode spends 9.838s more time (39% extra wrt interactive)
#
# Comparison of interactive mode in nutmeg script (.control) and in python
# pyi 100iter		25.140s (with cost evaluation)
# spi 100iter		22.422s (no cost evaluation)
# 
# python spent 2.718s for its own things (cost evaluation included)
# that is 12% of pure spice time
#
# profile shows that 4.24s per 100iter are spent for cost evaluation

import subprocess
from base import Simulator
from rawfile import raw_read
import os
import platform
from ..misc.env import environ
from ..misc.debug import DbgMsgOut, DbgMsg
from spiceopus import SpiceOpus

__all__ = [ 'SpiceOpusInteractive' ] 

import pdb
		
class SpiceOpusInteractive(SpiceOpus):
	def __init__(self, binary=None, args=[], restartPeriod=None, debug=0):
		SpiceOpus.__init__(self, binary, args, debug)
		
		self.restartPeriod=restartPeriod
		self.jobCounter=0
		self.process=None
		self.initialized=False
		self.terminator="---- - - ::"
		
		self.paramSet={}
		
		# self.f=open("scr.txt", "w")
	
	# Destructor - not allowed due to circular references
	# def __del__(self):
	# 	# Stop simulator
	# 	self.stopSimulator()
	# 	print "Deleting"
	
	#
	# Interactive mode test, startup, shutdown
	#
	
	def simulatorRunning(self):
		if self.process is None:
			if self.debug:
				print "SI: simulator not running, no process"
			return False
		elif self.process.poll() is not None:
			if self.debug:
				print "SI: simulator not running, process exit detected"
			return False
		else:
			return True
	
	def startSimulator(self):
		if self.simulatorRunning(): 
			return None
		else:
			self.process=subprocess.Popen(
					[self.binary, '-c']+self.cmdline, 
					bufsize=-1, 
					universal_newlines=True, 
					stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE
				)
			
			if self.process.poll() is not None:
				# Startup failed
				if self.debug:
					print "SI: simulator startup failed"
				self.process=None
				return None
			
			if self.debug:
				print "SI: simulator started"
				
			# Not initialized
			self.initialized=False
			
			# Set job counter to 0
			self.jobCounter=0
			
			# Set the prompt
			return self.sendScript('set prompt=\'\\> \'\necho * Commands are now accepted from a pipe...\n')
	
	def stopSimulator(self):
		if not self.simulatorRunning():
			return None
		else:
			txt=self.sendScript('set noaskquit\nquit\n')
			# self.process.stdin.close()
			self.process.wait()
			self.process=None
			if self.debug:
				print "SI: simulator stopped"
			return txt
			
	#
	# Interactive mode IO
	#
	
	def readUntilTerminator(self): 
		if not self.simulatorRunning():
			return None
		else:
			txt=''
			while True:
				line=self.process.stdout.readline()
				if line=='':
					# EOF
					break;
				elif line[-(len(self.terminator)+1):-1]==self.terminator:
					# Done
					break
				else:
					line=line.lstrip('> ')
					
					if self.debug>1:
						print "SI  : "+line, 
						
					txt+=line
				
			return txt
			
	def sendScript(self, script):
		if not self.simulatorRunning():
			return None
		else:
			if self.debug>2:
				print "SI  : SCRIPT\n"+script+"SI  : END SCRIPT\n"
			self.process.stdin.write(script+'\necho \''+self.terminator+'\'\n')
			self.process.stdin.flush()
			output=self.readUntilTerminator()
			# self.f.write(script)
			
			return output
	
	#
	# Job optimization
	#
	
	# Job grouping criterion
	# Two jobs can be merged (are compatible) if the following things are equal
	#	definition
	#	topology
	#	model
	def jobsCompatible(self, job1, job2):
		return (self.memberDictEqual(job1, job2, 'definition') and
				self.memberDictEqual(job1, job2, 'topology') and
				self.memberDictEqual(job1, job2, 'model'))
	
	# Called before grouping
	# Move members that can be modified by the simulator to analysis level. 
	def preGroup(self, jobList):
		for job in jobList:
			self.dictDown(job, 'options')
			self.dictDown(job, 'params')

	# Called after grouping
	# Move members that can be modified by the simulator, but need not be modified
	# because they remain unchanged, to job level
	def postGroup(self, jobList):
		for job in jobList:
			self.dictUp(job, 'options')
			self.dictUp(job, 'params')
		
		# Create a list of all parameters across all jobs
		paramset={}
		for job in jobList:
			if 'params' in jobList:
				paramset.update(jobList['params'])
			for an in job['analyses']:
				if 'params' in an:
					paramset.update(an['params'])
		
		# Remove 'temp' parameter because it is a simulator parameter and does not have to 
		# appear as a mparam
		if 'temp' in paramset:
			del paramset['temperature']
			
		# Store parameter set
		self.paramSet=paramset
		
		# Stop simulator if it is running because job optimization indicates that 
		# the set of jobs has changed and therefore the parameters might also have changed
		if self.simulatorRunning():
			self.stopSimulator()
		
		# Mark simulator as uninitialized
		self.initialized=False
	
	#
	# Simulator initialization
	#
	
	def initSimulator(self, jobList, inputParams):
		if self.debug:
			print "SI: initializing interactive simulator"
		
		# Build file
		fileName=self.simulatorID+"_interactive.cir"
		f=open(fileName, 'w')
		f.write('Interactive mode top-level file\n')
		
		f.write('\n')
		
		# Params (the list of all params is in paramSet member)
		params={}
		params.update(inputParams)
		params.update(self.paramSet)
		for (name, value) in params.iteritems():
			f.write('.mparam '+name+'='+str(value)+'\n')
		
		f.write('\n')
		
		# Definition (assumed to be equal for all jobs)
		if 'definition' in jobList[0]:
			defs=jobList[0]['definition']
			if 'section' in defs:
				f.write('.lib \''+str(defs['file'])+'\' '+str(defs['section'])+'\n')
			else:
				f.write('.include \''+str(defs['file'])+'\'\n')
		
		f.write('\n')
		
		# Model/topology netclasses
		for job in jobList:
			f.write('.netclass job '+job['name']+'\n')
			
			mod=job['model']
			if 'section' in mod:
				f.write('.lib \''+str(mod['file'])+'\' '+str(mod['section'])+'\n')
			else:
				f.write('.include \''+str(mod['file'])+'\'\n')
			
			topo=job['topology']
			if 'section' in topo:
				f.write('.lib \''+str(topo['file'])+'\' '+str(topo['section'])+'\n')
			else:
				f.write('.include \''+str(topo['file'])+'\'\n')
				
			f.write('.endn\n')
		
		f.write('\n')
		
		# Control block (set behavior)
		f.write('.control\n')
		f.write('set manualscktreparse\n')
		f.write('echo Initialization finished.\n')
		f.write('.endc\n')
		
		f.write('\n')
		
		# End
		f.write('.end\n')
		
		f.close()
		
		# Send it to the simulator
		return self.sendScript('source \''+fileName+'\'\n')
	
	#
	# Build a script for a job
	#
	
	def buildScript(self, inputParams, job):
		if self.debug:
			print "SI: building script for job '"+job['name']+"'"
		
		# Script starts here
		script='echo Preparing job: '+job['name']+'\n'
		
		# Select topology
		script+='netclass select job::'+job['name']+'\n'
		script+='netclass rebuild\n'
		
		# Prepare job parameters dictionary, update it with inputParams
		jobParams={}
		jobParams.update(inputParams)
		
		# Set job parameters, update job parameters dictionary with job params
		if 'params' in job:
			jobParams.update(job['params'])
		
		# Write job params
		for (name, value) in jobParams.iteritems():
			if name=='temperature':
				script+='set temp='+str(value)+'\n'
			else:
				script+='let @@topdef_['+name+']='+str(value)+'\n'
		
		# Set options
		if 'options' in job:
			for (option, value) in job['options'].iteritems():
				if value is True:
					script+='set '+str(option)+'\n'
				elif value is False:
					script+='unset '+str(option)+'\n'
				else:
					script+='set '+str(option)+'='+str(value)+'\n'
		
		# Handle analyses
		for an in job['analyses']:
			script+='echo Preparing: '+an['name']+'\n'
			
			# Prepare analysis params
			analysisParams={}
			analysisParams.update(jobParams)
			
			# Delete old stuff
			script+='destroy all\n'
			script+='delete all\n'
			
			# Analysis level parameters
			if 'params' in an:
				# Update analysis params with parameters at analysis level
				# These will be available to the user. 
				analysisParams.update(an['params'])
				
			# Set all parameters again. This is needed if a parameter does not appear in all
			# analyses and can be overriden by a job parameter. 
			for (name, value) in analysisParams.iteritems():
				if name=='temperature':
					script+='set temp='+str(value)+'\n'
				else:
					script+='let @@topdef_['+name+']='+str(value)+'\n'
			
			# Set analysis level options
			if 'options' in an:
				for (option, value) in an['options'].iteritems():
					if value is True:
						script+='set '+str(option)+'\n'
					elif value is False:
						script+='unset '+str(option)+'\n'
					else:
						script+='set '+str(option)+'='+str(value)+'\n'
		
			# Reparse top level circuit
			script+='scktreparse xtopinst_\n'
		
			# Set saves
			if 'saves' in an:
				if 'compiledSaves' not in an: 
					an['compiledSaves']=self.createSaves(an['saves'])
				
				count=0
				for save in an['compiledSaves']:
					if count == 0:
						script+='save '
					script+=save+' '
					count+=1
					if count == 10:
						count=0
						script+='\n'
				script+='\n'
					
			# Prepare parameters dictionary for local namespace
			self.analysisLocals['param'].clear()
			self.analysisLocals['param'].update(analysisParams)
			
			# Write analysis
			script+='echo Performing: '+str(an['name'])+'\n'
			script+=eval(an['command'], globals(), self.analysisLocals)+'\n'
			script+='if $(#plots) gt 1\n  set filetype=binary\n  write '+self.simulatorID+"_"+str(an['name'])+'.raw\nelse\n echo '+str(an['name'])+' analysis failed.\nend\n'

		# Script ends here
		script+='echo Finished: '+job['name']+'\n'
		
		return script
		
		
	#
	# Run jobs
	#
		
	def runJobList(self, params, jobList, indices=None):
		if self.restartPeriod is not None and self.jobCounter>=self.restartPeriod:
			self.stopSimulator()
			
		if not self.simulatorRunning():
			self.startSimulator(),
			
		if not self.initialized: 
			self.initSimulator(jobList, params),
			self.initialized=True
		
		if indices is None:
			for job in jobList:
				script=self.buildScript(params, job)
				self.sendScript(script)
				self.collectResults(True, job)
				self.jobCounter+=1
		else:
			for i in indices:
				job=jobList[i]
				script=self.buildScript(params, job)
				self.sendScript(script)
				self.collectResults(True, job)
				self.jobCounter+=1
	
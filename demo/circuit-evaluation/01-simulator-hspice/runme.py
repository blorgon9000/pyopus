# Test SpiceOpus simulator interface

from pyopus.simulator import simulatorClass

if __name__=='__main__':
	# Job list for simulator
	jobList=[
		{	# First job - op analysis
			'name': 'dcop', 
			'definitions': [
				{ 'file': 'hs-cmos180n.lib', 'section': 'tm' }, 
				{ 'file': 'hs-opamp.cir' }
			], 
			'params': {
				'vdd': 1.8, 
				'temperature': 25
			}, 
			'options': {
				'method': 'trap'
			}, 
			'saves': [
			], 
			'command': 'op()'
		},
		{	# Second job - op analysis with different temperature
			'name': 'dcop100', 
			'definitions': [
				{ 'file': 'hs-cmos180n.lib', 'section': 'tm' }, 
				{ 'file': 'hs-opamp.cir' }
			], 
			'params': {
				'vdd': 1.6, 
				'temperature': 100
			}, 
			'options': {
				'method': 'trap'
			}, 
			'saves': [
			], 
			'command': 'op()'
		},
		{	# Third job - op analysis with different supply voltage
			'name': 'dcopv33', 
			'definitions': [
				{ 'file': 'hs-cmos180n.lib', 'section': 'tm' }, 
				{ 'file': 'hs-opamp.cir' }
			], 
			'params': {
				'vdd': 2.0, 
				'temperature': 25
			}, 
			'options': {
				'method': 'trap'
			}, 
			'saves': [
			], 
			'command': 'op()'
		},
		{	# Fourth job - op analysis with different library
			'name': 'dcopff', 
			'definitions': [
				{ 'file': 'hs-cmos180n.lib', 'section': 'ws' }, 
				{ 'file': 'hs-opamp.cir' }
			], 
			'params': {
				'vdd': 2.0, 
				'temperature': 25
			}, 
			'options': {
				'method': 'trap'
			}, 
			'saves': [
			], 
			'command': 'op()'
		}, 
		{	# Fifth job - op analysis with different library
			'name': 'dcopff100', 
			'definitions': [
				{ 'file': 'hs-cmos180n.lib', 'section': 'ws' }, 
				{ 'file': 'hs-opamp.cir' }
			], 
			'params': {
				'vdd': 2.0, 
				'temperature': 100
			}, 
			'options': {
				'method': 'trap'
			}, 
			'saves': [
			], 
			'command': 'op()'
		}
	]

	# Input parameters
	inParams={
		'mirr_w': 7.46e-005, 
		'mirr_l': 5.63e-007
	}

	# Create simulator, load HSpice class on demand. 
	sim=simulatorClass("HSpice")(debug=1)

	# Set job list and optimize it
	sim.setJobList(jobList)

	# Print optimized job groups - need to get them manually from simulator
	# because the simulator behaves officially as if there was only one job group. 
	jobSeq=sim.jobSequence
	print("\nJob Groups:")
	for i in range(len(jobSeq)):
		group=jobSeq[i]
		gstr=''
		for j in group:
			gstr+=" %d (%s), " % (j, jobList[j]['name'])
		print("  %d: %s" % (i, gstr))

	print("")
	# Set input parameters
	sim.setInputParameters(inParams)

	# Go through all job groups, write file, run it and collect results
	ngroups=sim.jobGroupCount()
	for i in range(ngroups):
		# Delete old loaded results (free memory). 
		sim.resetResults()
		
		# Run jobs in job group. 
		(jobIndices, status)=sim.runJobGroup(i)
		
		print("")
		for j in jobIndices:
			name=jobList[i]['name']
			sim.collectResults([j], status)
			if sim.activateResult(j) is not None:
				print("Job %d (%s): Vout=%e" % (j, name, sim.res_voltage("out")))
			else:
				print("Job %d (%s): no results" % (j, name))
		print("")

	sim.cleanup()

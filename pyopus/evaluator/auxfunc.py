# Miscellaneous tools for setting up problems - deprecated

from ..misc.debug import DbgMsgOut, DbgMsg
from numpy import array, ndarray

__all__ = [ 'reduceDictionary', 'reduceMeasures' ]

import pdb

# Returns reduced dictionary with references to original dictionary entries
def reduceDictionary(keys, d):
	dnew={}
	
	for key in keys:
		dnew[key]=d[key]
	
	return dnew

# Returns reduced measure list with fixed dependencies
def reduceMeasures(measureList, measures):
	# Empty result
	result={}
	# First copy all measures in measureList
	for measureName in measureList:
		result[measureName]=measures[measureName]
	
	# Now check dependencies
	for measureName in measureList:
		# Get measure
		measure=measures[measureName]
		
		# Measures with an analysis, ignore depends
		if 'analysis' in measure:
			if measure['analysis'] is not None:
				continue
		
		# Does it have a depends list
		if 'depends' in measure:
			# Yes, scan list
			for depend in measure['depends']:
				if type(depend) is str:
					# String dependency, depends on all corners of a dependent measure
					dependsOnMeasures=[depend]
					dependsOnCorners=measure['corners']
				else:
					# List or tuple dependency (pairs of the form (measures, corners))
					dependsOnMeasures=depend[0]
					dependsOnCorners=depend[1]
				
				# If dependendcy is a string, change it to a list
				if type(dependsOnMeasures) is str:
					dependsOnMeasures=[dependsOnMeasures]
				if type(dependsOnCorners) is str:
					dependsOnCorners=[dependsOnCorners]
				
				# Handle all (measure, corner) pairs
				for depMeasureName in dependsOnMeasures: 
					# Dependency may not be a measure with no analysis
					if 'analysis' not in measures[depMeasureName]:
						raise Exception, DbgMsgOut("MI", "Dependent measures can't depend on dependent measures.")
					if measures[depMeasureName]['analysis'] is None:
						raise Exception, DbgMsgOut("MI", "Dependent measures can't depend on dependent measures.")
						
					# Add dependency 
					if depMeasureName not in result:
						result[depMeasureName]=measures[depMeasureName]
					for depCornerName in dependsOnCorners:
						# Add corners
						if depCornerName not in result[depMeasureName]['corners']:
							result[depMeasureName]['corners'].append(depCornerName)

	return result
	

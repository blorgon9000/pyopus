"""
**Auxiliary functions for parameter management**

**Parameter descriptions in list form** consist of arrays or lists where 
every one of them specifies one propery for every parameter. 
An additional list is needed that holds the names of the parameters. 
An example::

  paramLow =[0.1, 0.2, 0.5, 0.2]
  paramHigh=[0.5, 0.9, 0.7, 0.5]
  paramName=['a', 'b', 'c', 'd']

The three lists specify the lower and the upper bounds, and the names of 
four parameters (a, b, c, and d). Every list specifies one aspect of all 
parameters. List form specifies an ordering for the parameters. 

An alternative is to organize this information in **dictionary form**. 
This way the parameters are organized in a dictionary with parameter 
name as key and parameter information as value. Parameter information 
is a dictionary with parameter aspect name as key and parameter aspect 
as value. Dictionary form does not imply parameter ordering, except for 
the one obtained by the :meth:`keys` method of the dictionary object. 
The above example could be organized in dictionary form as:: 

  params={
    'a': {
      'lo': 0.1, 
      'hi': 0.5
    }, 
    'b': {
      'lo': 0.2, 
      'hi': 0.9
    }, 
    'c': {
      'lo': 0.5, 
      'hi': 0.7
    }, 
    'd': {
      'lo': 0.2, 
      'hi': 0.5
    }
  }

Conevrsion between list form and dictionary form of parameter description 
is performed by the :func:`dictParamDesc` and the :func:`listParamDesc` 
function. The list form can be converted to the dictionary form using::

  params=dictParamDesc({'lo': paramLow, 'hi': paramHi}, paramName)
  
To extract only selected parameters from the list form, one can specify
the indices::
  
  # Extract only the first two parameters
  params=dictParamDesc({'lo': paramLow, 'hi': paramHi}, paramName, indices=[0,1])

Conversion in the opposite direction can be performend with::

  # paramName is the parameter name list specifying the parameter order
  paramLists=listParamDesc(params, paramName)
  # Extract the paramLow and the paramHigh list
  paramLow=paramLists['lo']
  paramHigh=paramLists['hi']
  
Selected parameters can be extracted by specifying indices::

  # Extract only the first two parameters specified in the nameList
  paramLists=listParamDesc(params, paramName, indices=[0,1])
  
or by specifying a shorter parameter list::
  
  # Extract parameters a and b
  paramLists=listParamDesc(params, ['a', 'b'])
  
Finally, one can extract only one property by specifying the *propName* parameter::

  paramLow=listParamDesc(params, paramName, propName='lo')

Parameter values can be given as lists or as dictionaries. In list form
the corresponding parameter names must be specified by a list of names. 
This list also implies the parameter ordering::

  paramValues=[0.1, 0.5, 0.7, 0.3]
  paramNames =['a', 'b', 'c', 'd']

In dictionary form parameter names are specified implicitly with keys, 
but no ordering is implied apart from that obtained with the :meth:`keys` 
method of the dictionary::

  paramDictionary={
    'a': 0.1, 
    'b': 0.5, 
    'c': 0.7, 
    'd': 0.3
  }

Conversion between the two forms of parameter values is performed using 
the :func:`paramList` and the :func:`paramDict` function:: 

  # To dictionary form
  paramDictionary=paramDict(paramValues, paramNames)
  
  # To list form, paramNames specifies the ordering for the list form
  paramValues=paramList(paramDictionary, paramNames)
  
Selected parameters can be extracted by specifying the *indices* parameter:: 

  # Extract only the first two parameters given in the paramValues list, 
  # return the parameters in dictionary form
  paramDictionary=paramDict(paramValues, paramNames, indices=[0,1])
  
  # Extract the parameters corresponding to the first two names in paramNames, 
  # return parameters in list form
  paramValues=paramList(paramDictionary, paramNames, indices=[0,1])
"""

from ..misc.debug import DbgMsgOut, DbgMsg
from numpy import array, ndarray

__all__ = [ 'dictParamDesc', 'listParamDesc', 'paramList', 'paramDict' ]

# Returns reduced dictionary with references to original dictionary entries
# Obsolete, TODO: remove
def reduceDictionary(keys, d):
	dnew={}
	
	for key in keys:
		dnew[key]=d[key]
	
	return dnew

# Returns reduced measure list with fixed dependencies
# Obsolete, TODO: remove
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
	
def listParamDesc(inputDict, nameList, propName=None, indices=None):
	"""
	Processes parameter definition dictionary and produces ordered 
	property value lists. 
	
	A parameter definition dictionary has entries with parameter 
	names for keys and property dictionaries for values. A property 
	dictionary has property names for keys and stores peroperty values.
	
	*nameList* is the list of parameter names used in vectorization. 
	
	Returns a dictionary with property names for keys and property 
	value lists for values. The order of values in the property 
	value vectors corresponds to the ordered parameter names. 
	
	If some property is not given for a parameter the corresponding 
	entry in the property value list is is set to ``None``. 
	
	If *propname* is given a single list is returned corresponding to 
	the specified property. 
	
	If *indices* is given, only the parameters with specified indices 
	are included in the output. 
	"""
	if indices is None:
		ndx=range(len(nameList))
	else:
		ndx=indices
		
	if propName is not None:
		output=[]
		for ii in ndx:
			name=nameList[ii]
			output.append(inputDict[name][propName])
		
		return output
	else:
		# Prepare set of property names
		propSet=set([])
		for key, value in inputDict.iteritems():
			propSet=propSet.union(set(value.keys()))
		
		# Prepare output dictionary
		output={}
		for key in propSet:
			output[key]=[]
			
		# Extract entries from dictionary
		for ii in ndx:
			name=nameList[ii]
			for prop in propSet:
				if prop in inputDict[name]:
					output[prop].append(inputDict[name][prop])
				else:
					output[prop].append(None)
		
		return output

def dictParamDesc(inputList, nameList, indices=None):
	"""
	Processes property value lists stored in a dictionary with property 
	name for key. Returns a parameter definition dictionary with parameter 
	name for key and a property dictionary corresponding to a parameter for 
	value. 
	
	*nameList* is a list of parameter names corresponding to individual 
	propeerty list positions. 
	
	If *indices* is given only the parameters with specified indices 
	are included in the output. 
	
	If property value is ``None`` it is not put into the corresponding 
	parameter's dictionary. 
	"""
	if indices is None:
		ndx=range(len(nameList))
	else:
		ndx=indices
		
	output={}
	for ii in ndx:
		name=nameList[ii]
		output[name]={}
		for prop,propList in inputList.iteritems():
			val=propList[ii]
			if val is not None:
				output[name][prop]=val 
	
	return output

def paramList(inputDict, nameList, indices=None):
	"""
	Coverts a dictionary of values into a vector of values. *nameList* 
	is a list of keys specifying the order of values. 
	
	If *indices* is given only the parameters with specified indices 
	are included in the output. 
	"""
	if indices is None:
		ndx=range(len(nameList))
	else:
		ndx=indices
		
	output=[]
	
	for ii in ndx:
		name=nameList[ii]
		output.append(inputDict[name])
	
	return output

def paramDict(inputList, nameList, indices=None):
	"""
	Converts a list of parameter values into a dictionary. *nameList* 
	specifies the keys corresponding to values in the list. 
	
	If *indices* is given only the parameters with specified indices 
	are included in the output. 
	"""
	if indices is None:
		ndx=range(len(nameList))
	else:
		ndx=indices
		
	output={}
	for ii in ndx: 
		name=nameList[ii]
		output[name]=inputList[ii]
	
	return output


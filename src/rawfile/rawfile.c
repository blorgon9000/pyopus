/* (c)2008 Arpad Buermen */
/* SPICE raw file import/export module */

/* Note that in Windows we do not use Debug compile because we don't have the debug version
   of Python libraries and interpreter. We use Release version instead where optimizations
   are disabled. Such a Release version can be debugged. 
 */

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include "rawfile.h"
#include <math.h>
#include <stdio.h>

/* Debug file */
#define df stdout

/* Methods table */
static PyMethodDef _rawfile_methods[] = {
	{"raw_write", raw_write, METH_VARARGS},
	{"raw_read", raw_read, METH_VARARGS},
	{NULL, NULL}     // Marks the end of this structure
};

/* Module initialization 
   Module name must be _rawfile in compile and link */
__declspec(dllexport) void init_rawfile()  {
	(void) Py_InitModule("_rawfile", _rawfile_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_rawfile_amd64()  {
	(void) Py_InitModule("_rawfile_amd64", _rawfile_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

__declspec(dllexport) void init_rawfile_i386()  {
	(void) Py_InitModule("_rawfile_i386", _rawfile_methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

/* Structure for fast vector access */
struct FastArray {
	npy_intp isComplex;
	char *data;
	char *pos;
	Py_ssize_t stride;
	Py_ssize_t length;
};

/* Check if key-value pair is valid, return keyString by reference, return 1 for valid, 0 otherwise */
__inline npy_intp validateKeyData(PyObject *key, PyObject *value, const char **keyString) {
	PyArrayObject *numarray;

	// Check if key is a string 
	if (!PyString_Check(key)) 
		// Skip it if key is not a string
		return 0;
	
// Check if key has zero length
	*keyString=PyString_AsString(key); 
	if (strlen(*keyString)<=0) 
		// Skip it if key has zero length
		return 0;
	
	// Check if value is an array
	if (!PyArray_Check(value)) 
		// Skip it if value is not an array
		return 0;
	
	// Check if it is double or double complex
	numarray=(PyArrayObject *)value; 
	// if (numarray->descr->type_num!=NPY_DOUBLE && numarray->descr->type_num!=NPY_CDOUBLE)
	if (PyArray_TYPE(numarray)!=NPY_DOUBLE && PyArray_TYPE(numarray)!=NPY_CDOUBLE) 
		// Skip it if it is not double or double complex
		return 0;

	return 1;
}

__inline void reverse(void *ptr, int chunksize, int nchunks) {
  int i;
  char *p, *p1, *p2, tmp;
  p=(char *)ptr;
  for(i=0;i<nchunks;i++,p+=chunksize) {
    for(p1=p, p2=p+chunksize-1; p1<p2; p1++, p2--) {
      tmp=*p2;
      *p2=*p1;
      *p1=tmp;
    }
  }
}

/* Write SPICE .raw file 
   Arguments:
     - dictionary where key is a vector name and the value is a NumPy array of type double or complex
	 - dictionary where key is vector name and value is the name of the corresponding scale vector
	 - string with the default vector name
	 - title string
	 - date string
	 - plotname string
	 - optional: binary file type (bool), 
	 - optional: append mode (bool), 
	 - optional: padding (bool), 
	 - optional: ascii format precision (int), default=15 when precision<1 is specified

   If default scale is not given, it is assumed to be the first vector in the dictionary.  
   Ignores dictionary members that are not arrays.
   Ignores scales members that are not strings.
   Writes nothing and returns 0 if no actual vectors are found. 
   If any key is not a string, nothing is written and retiurns 0.
   
   On all errors 0 is returned. 

   Returns 1 on success. 
 */
static PyObject *raw_write(PyObject *self, PyObject *args) {
	npy_intp status=0;
	PyObject *data, *scales;
	const char *fileName, *dflVecName, *title, *date, *plotName, *keyString, *scaleName;
	PyObject *binary, *append, *padding;
	npy_intp didDflVec;
	int precision, debugMode;
	PyObject *key, *value, *dflVecKey, *dflVecValue, *longestKey, *scaleObj;
	PyArrayObject *numarray;
	Py_ssize_t pos, longest, len, i, j;
	npy_intp count, *longestDims, longestNdims;
	npy_intp isComplex; 
	FILE *f=NULL;
	struct FastArray *faPtr, *faPos;
	double dd, di, zero=0.0;

	// Retrieve filename, vectors, default scale, scales, title, date, plotname, binary, append, padding, ascii precision
	if (!PyArg_ParseTuple(args, "sOsOsssO!O!O!ii", 
			&fileName, 
			&data, 
			&dflVecName, 
			&scales, 
			&title, 
			&date, 
			&plotName, 
			&PyBool_Type, &binary, 
			&PyBool_Type, &append, 
			&PyBool_Type, &padding, 
			&precision, 
			&debugMode)) {
		status=1;
	}

	if (debugMode) {
		fprintf(df, "raw_write: writing file %s\n", fileName);
	}

	// Fix precision to default
	if (precision<1) {
		precision=15;
	}

	// Verify if data is a dictionary
	if (!status) {
		if (!PyDict_Check(data)) {
			// Not a dictionary
			status=1;
			if (debugMode) {
				fprintf(df, "raw_write: argument 2 is not a dictionary.\n");
			}
		}
	}

	// Verify if scales is a dictionary
	if (!status) {
		if (!PyDict_Check(scales)) {
			// Not a dictionary
			status=1;
			if (debugMode) {
				fprintf(df, "raw_write: argument 4 is not a dictionary.\n");
			}
		}
	}

	// Scan all vectors, check
	//   - valid data must be a double or double complex array
	//   - valid key must be a sring with nonzero length
	//   - count valid key-data pairs
	//   - if there are any complex arrays, remember it
	//   - remember longest array's length and its shape (default shape)
	//   - check if any valid key-data pair is the default vector
	//   - remember first valid key-data pair as default vector if no default vector is found
	//   - if no valid key-data pairs are found an error is returned
	// Return value 0 is error, 1 is OK. 
	count=0;
	isComplex=0;
	longest=0;
	longestNdims=0;
	longestDims=NULL;
	dflVecValue=NULL;
	dflVecKey=NULL;
	longestKey=NULL;
	if (!status) {
		pos=0;
		while (PyDict_Next(data, &pos, &key, &value)) {
			if (!validateKeyData(key, value, &keyString)) {
				continue;
			}
			
			// Set complex flag
			numarray=(PyArrayObject *)value; 
			// if (numarray->descr->type_num==NPY_CDOUBLE)
			if (PyArray_TYPE(numarray)==NPY_CDOUBLE)
				isComplex=1;

			// Calculate length, store longest
			len=PyArray_Size(value);
			if (len>longest) {
				longest=len;
				// longestDims=numarray->dimensions;
				longestDims=PyArray_DIMS(numarray);
				// longestNdims=numarray->nd;
				longestNdims=PyArray_NDIM(numarray);
				longestKey=key;			
			}

			// Store first valid array as default vector
			if (count==0) {
				dflVecValue=value; 
				dflVecKey=key;
			}

			// Check if this vector is the default vector
			if (!strcmp(keyString, dflVecName)) {
				// Found default vector
				dflVecValue=value;
				dflVecKey=key;
			}

			// Increase count
			count++;
		}
		if (count<=0) {
			status=1;
			if (debugMode) {
				fprintf(df, "raw_write: no arrays to write.\n");
			}
		}
	}

	if (count>0) {
		// We have something to write
		// Open file
		if (append==Py_True) {
			f=fopen(fileName, "ab");
		} else {
			f=fopen(fileName, "wb");
		}
		if (!f) {
			status=1;
			if (debugMode) {
				fprintf(df, "raw_write: can't open file.\n");
			}
		}
	}

	if (!status) {
		// Write header
		fprintf(f, "Title: %s\n", title);
		fprintf(f, "Date: %s\n", date);
		fprintf(f, "Plotname: %s\n", plotName);
		fprintf(f, "Flags: %s%s\n", isComplex ? "complex" : "real", padding ? "" : " unpadded");
		fprintf(f, "No. Variables: %d\n", count);
		fprintf(f, "No. Points: %d\n", longest);
		if (longestNdims>1) {
			fprintf(f, "Dimensions: ");
			for(i=0;i<longestNdims;i++) {
				fprintf(f, "%d%s", longestDims[i], (i < longestNdims - 1) ? "," : "");
			}
		}
	}

	faPtr=NULL;
	if (!status) {
		// Initialize fast access structure
		faPtr=(struct FastArray *)PyMem_Malloc(sizeof(struct FastArray)*count);
		if (!faPtr) {
			// Out of memory
			status=1;
			if (debugMode) {
				fprintf(df, "raw_write: can't allocate fast access structure.\n");
			}
		}
	}

	if (!status) {
		// Write data headers
		fprintf(f, "Variables:\n");

		pos=0;
		i=0;
		// Dump default scale
		didDflVec=0;
		key=dflVecKey;
		value=dflVecValue;
		while (1) {
			// If we already dumped default vector, go to the next one
			if (didDflVec) {
				// Move to next vector, stop if there are no more vectors
				if (!PyDict_Next(data, &pos, &key, &value)) {
					break;
				}
			}
			
			// Validate it, skip it if it is not valid
			if (!validateKeyData(key, value, &keyString)) {
				continue;
			}
			
			// Vector is valid, check if we already dumped it (default vector)
			if (didDflVec && !strcmp(PyString_AsString(key), PyString_AsString(dflVecKey))) {
				continue;
			}
				
			// Dump it
			fprintf(f, "\t%d\t%s\t%s", i, PyString_AsString(key), "notype");
			
			// Check if this vector has a scale vector (scales dictionary)
			scaleObj=PyDict_GetItemString(scales, PyString_AsString(key));
			if (scaleObj && PyString_Check(scaleObj)) {
				scaleName=PyString_AsString(scaleObj);
				if (strlen(scaleName)>0) {
					// Dump scale vector name
					fprintf(f, " scale=%s", scaleName);
				}
			}

			// Get array object and dump dimensions
			numarray=(PyArrayObject *)value;
			// Verify if dimension differ from longest vector's dimensions
			// for(j=0;j<numarray->nd && j<longestNdims;j++) {
			for(j=0;j<PyArray_NDIM(numarray) && j<longestNdims;j++) {
				// if (numarray->dimensions[j]!=longestDims[j])
				if (PyArray_DIMS(numarray)[j]!=longestDims[j])
					break;
			}
			// if (j==numarray->nd && j==longestNdims) {
			if (j==PyArray_NDIM(numarray) && j==longestNdims) {
				// Nothing further to do, vector's dimensions match those of the longest vector
			} else {
				// Dump dimensions
				fprintf(f, " dims=");
				// for(j=0;j<numarray->nd;j++) {
				for(j=0;j<PyArray_NDIM(numarray);j++) {
					// fprintf(f, "%d%s", numarray->dimensions[j], (j < numarray->nd - 1) ? "," : "");
					fprintf(f, "%d%s", PyArray_DIMS(numarray)[j], (j < PyArray_NDIM(numarray) - 1) ? "," : "");
				}
			}

			// Initialize FastArray entry
			// if (numarray->descr->type_num==NPY_CDOUBLE)
			if (PyArray_TYPE(numarray)==NPY_CDOUBLE)
				faPtr[i].isComplex=1;
			else
				faPtr[i].isComplex=0;
			// faPtr[i].data=numarray->data;
			faPtr[i].data=PyArray_DATA(numarray);
			// faPtr[i].pos=numarray->data;
			faPtr[i].pos=PyArray_DATA(numarray);
			// faPtr[i].stride=numarray->strides[numarray->nd-1];
			faPtr[i].stride=PyArray_STRIDE(numarray, PyArray_NDIM(numarray)-1);
			// faPtr[i].length=PyArray_Size(value);
			faPtr[i].length=PyArray_SIZE(numarray);

			// End line
			putc('\n', f);
			
			// Default vector must be dumped at this point
			didDflVec=1;

			// Increase counter
			i++;
		}
	}

	// Dump data
	if (!status) {
		if (binary==Py_True) {
			// Binary dump
			fprintf(f, "Binary:\n");
			for(i=0;i<longest;i++) {
				for(faPos=faPtr, j=0; j<count; faPos++, j++) {
					if (i<faPos->length) {
						if (!isComplex) {
							// Dump a real vector in a real .raw file
							dd=(double)*((npy_double *)(faPos->pos));
							fwrite((char *) &dd, sizeof(double), 1, f);
						} else if (!faPos->isComplex) {
							// Dump a real vector in a complex .raw file
							dd=(double)*((npy_double *)(faPos->pos));
							fwrite((char *) &dd, sizeof(double), 1, f);
							fwrite((char *) &zero, sizeof(double), 1, f);
						} else {
							// Dump a complex vector in a complex .raw file
							dd=(double)(((npy_cdouble *)(faPos->pos))->real);
							fwrite((char *) &dd, sizeof(double), 1, f);
							di=(double)(((npy_cdouble *)(faPos->pos))->imag);
							fwrite((char *) &di, sizeof(double), 1, f);
						}
						faPos->pos+=faPos->stride;
					} else if (padding==Py_True) {
						if (!isComplex) {
							fwrite((char *) &zero, sizeof(double), 1, f);
						} else {
							fwrite((char *) &zero, sizeof(double), 1, f);
							fwrite((char *) &zero, sizeof(double), 1, f);
						}
					}
				}
			}
		} else {
			// Ascii dump
			fprintf(f, "Values:\n");
			for(i=0;i<longest;i++) {
				// Print index
				fprintf(f, " %d", i);
				for(faPos=faPtr, j=0; j<count; faPos++, j++) {
					if (i<faPos->length) {
						if (!isComplex) {
							// Dump a real vector in a real .raw file
							dd=(double)*((npy_double *)(faPos->pos));
							fprintf(f, "\t%.*e\n", precision, dd); 
						} else if (!faPos->isComplex) {
							// Dump a real vector in a complex .raw file
							dd=(double)*((npy_double *)(faPos->pos));
							fprintf(f, "\t%.*e,0.0\n", precision, dd); 
						} else {
							// Dump a complex vector in a complex .raw file
							dd=(double)(((npy_cdouble *)(faPos->pos))->real);
							di=(double)(((npy_cdouble *)(faPos->pos))->imag);
							fprintf(f, "\t%.*e,%.*e\n", precision, dd, precision, di);
						}
						faPos->pos+=faPos->stride;
					} else if (padding==Py_True) {
						if (!isComplex) {
							fprintf(f, "\t0.0\n"); 
						} else {
							fprintf(f, "\t0.0,0.0\n"); 
						}
					}
				}
			}
		}
	}
	
	// Close file
	if (f) {
		fclose(f);
	}

	// Free allocated memory
	if (faPtr) {
		PyMem_Free(faPtr);
	}

	// Form a Pythonm integer from status and return it
	return PyInt_FromLong(!status);
}

/* Reads a line from file and resizes the buffer if the line doesn't fit.
   The trailing newline is not stored in the string. 
   Returns 0 if outOfMem, 1 otherwise. 
   The number of bytes in buffer is stored in bytesInBuf. -1 means read error. 
 */
static npy_intp readLine(FILE *f, char **buf, npy_uintp *bufSize, npy_uintp *bytesInBuf) {
	npy_uintp tmpN; // temporary buffer size
	char *tmpBuf; // temporary buffer
	char *ptr, *p; // return value of fgets() and the position of null character

	// Allocate initial buffer
	if (*bufSize==0) {
		tmpN=*bufSize+256;
		tmpBuf=(char *)PyMem_Realloc(*buf, tmpN*sizeof(char));
		if (!tmpBuf) {
			// Out of memory
			return 0;
		}
		*buf=tmpBuf;
		*bufSize=tmpN;
	}

	// We have nothing yet
	*bytesInBuf=0;

	// Read until EOF or newline
	
	while (1) {
		// How much to read? Well, bytesInBuf out of *bufSize is already full.

		// Try to read
		ptr=fgets((*buf)+*bytesInBuf, (int)(*bufSize-*bytesInBuf), f);
		if (!ptr) {
			// fgets() failed, but not out of memory
			return 1;
		}
		// Find null character
		p=ptr;
		while (*p) 
			p++;
		(*bytesInBuf)+=(npy_uintp)(p-ptr);
		// Do we have a newline before null character?
		if (*(p-1)=='\n') {
			// Yes, we are finished. 
			// Remove newline
			*(p-1)=0;
			(*bytesInBuf)--;
			return 1;
		} else if (feof(f)) {
			// No newline, but we are at EOF. 
			// Ok, we let it go this time since the last line may not end in a newline. 
			return 1;
		} else {
			// Null character, no newline, no EOF. 
			// Either we ran out of buffer or
			// the file contains a null character. So much for ASCII files. 
			// Oh well, ignore the null character and read on. 
		}
		// There must be more. Check if the buffer is not full.
		if (*bufSize<=*bytesInBuf+1) {
			// Resize buffer
			tmpN=(*bufSize)*2;
			tmpBuf=(char *)PyMem_Realloc(*buf, tmpN*sizeof(char));
			if (!tmpBuf) {
				// Out of memory
				return 0;
			}
			*buf=tmpBuf;
			*bufSize=tmpN;
		}
	}
	// Finished.
	return 1;
}

/* Match a prefix in a string. 
   Advance string pointer beyond prefix if prefix is found. 
   Return 1 if match found, 0 otherwise. 
 */
npy_intp matchPrefix(char *prefix, char **string, npy_intp caseIndependent) {
	char *pp; 
	char *p;

	pp=prefix;
	p=*string;

	// Go until you hit a newline
	while(*p && *pp) {
		// Compare
		if (caseIndependent) {
			if (tolower(*p)!=tolower(*pp)) {
				return 0;
			}
		} else if (*p!=*pp) {
			return 0;
		}

		// Move on
		p++;
		pp++;
	}

	if (!*pp) {
		// Reached end of needle. We have a match.
		*string=p;
		return 1;
	} else {
		// No match
		return 0;
	}
}

/* Skips spaces and tabs. Updates string. 
 */
npy_intp skipSpaces(char **string) {
	char *p;

	p=*string;
	while (*p && (*p==' ' || *p=='\t'))
		p++;

	*string=p;

	return 1;
}

/* Skips all characters util a space or tab is found. 
 */
npy_intp skipToSpace(char **string) {
	char *p;

	p=*string;
	while (*p && *p!=' ' && *p!='\t')
		p++;

	*string=p;

	return 1;
}

/* Checks if a character is a digit. */
npy_intp isDigit(char c) {
	if (c>='0' && c<='9')
		return 1;
	else
		return 0;
}

/* Reads a positive integer. No leading '+' is allowed. 
   Stops when the first character is encountered that cannot 
   be interpreted as a part of an integer. 
   Updates string to the point where it stopped. 
   If nothing was read (no integer) ok is set to 0. 
 */
npy_intp matchInteger(char **string, npy_intp *ok) {
	char *pos;
	npy_intp n;

	// Check if this is really a number 
	if (!isDigit(**string)) {
		*ok=0;
		return 0;
	}

	// Read number
	n=strtol(*string, &pos, 10);

	// Check if everything is OK
	if (pos && !isDigit(*pos)) {
		*ok=1;
	} else {
		*ok=0;
		return 0;
	}

	// Return position where we stopped and the number
	*string=pos;
	return n;
}

/* Add dimension to dimensions array.
   Return 0 on out of memory, 1 otherwise.
 */
npy_intp addDimension(npy_intp **dim, npy_intp *dimSize, int *nDim, npy_intp n) {
	npy_intp *tmpDim, tmpSize;
	// Add dimension to dimensions array
	if (*nDim>=*dimSize) {
		// Need more space
		if (*dimSize==0)
			tmpSize=8;
		else
			tmpSize=(*dimSize)*2;
		tmpDim=(npy_intp *)PyMem_Realloc(*dim, tmpSize*sizeof(npy_intp));
		if (!tmpDim)
			return 0;
		*dim=tmpDim;
		*dimSize=tmpSize;
	}
	(*dim)[*nDim]=n;
	(*nDim)++;

	return 1;
}

/* Reading dimensions 
   1. skip spaces
      read optional leading [, increase bracket count
   2. skip spaces
      read number, if reading fails it is 0
	  add to dimensions array
   3. skip spaces
      match one of
	    - comma -> skip
		- ][ inside brackets -> skip
		- ] inside brackets -> decrease bracket count, skip, and done
		- digit
		- anything else outside brackets -> done
		- anything else inside brackets -> error
   4. go back to 2

   Returns 1 on OK, 0 on error or out of memory.
 */
npy_intp readDims(char **string, npy_intp **dim, npy_intp *dimSize, int *nDim) {
	npy_intp n, st;
	npy_intp bracket=0;

	// Skip whitespaces
	skipSpaces(string);

	// Try to match opening bracket
	if (**string=='[') {
		bracket++;
	}

	*nDim=0;
	while (1) {
		// Skip whitespace
		skipSpaces(string);

		// A number is expected
		n=matchInteger(string, &st);
		// If no number read, dimension is 0
		if (!st)
			n=0;
		// Add dimension to dimensions array
		if (!addDimension(dim, dimSize, nDim, n)) {
			// Out of memory
			return 0;
		}
		
		// Skip whitespace
		skipSpaces(string);

		// Check for separator
		if (**string==',') {
			// Comma, continue
			(*string++);
			skipSpaces(string);
		} else if (bracket && **string==']' && *((*string)+1)=='[') {
			// in brackets, ][, continue
			(*string)+=2;
			skipSpaces(string);
		} else if (bracket && **string==']') {
			// in brackets, ], done
			(*string)++;
			bracket--;
			break;
		} else if (isDigit(**string)) {
			// new number, continue
		} else if (bracket==0) {
			// unmatched, outside brackets, done
			break;
		} else {
			// unmatched, inside brackets, error
			return 0;
		}
	}

	return 1;
}


/* Returns a list of tuples (one per plot) with tuple members:
     - vector dictionary 
     - default vector name 
     - scales dictionary 
     - title 
     - date 
     - plot name 
   Section and property name matching is case-independent. 
   Vector names are case-dependent. 
   Ignores all sections except for 
     Title, Date, Plotname, Flags, No. Variables, No. Points, Variables, Values, Binary.
   Ignores unknown flags except for real, complex, unpadded, and padded. 
   Ignores vector type and all properties except for scale and dims. 
   Returns None on error. 
 */ 
static PyObject *raw_read(PyObject *self, PyObject *args) {
	const char *fileName;
	int debugMode, reverseBytes, noReadFirst;
	npy_intp status=0;
	FILE *f;
	char *p, *pp, *p1, *p2, *p3, *p4, tmpc, tmpc1;
	char *lineBuf=NULL;
	npy_uintp lineBufSize=0, bytesRead;
	char *lineBuf1=NULL;
	npy_uintp lineBufSize1=0, bytesRead1;
	char *firstTitle=NULL, *firstDate=NULL;
	PyObject *plotTitle=NULL, *plotDate=NULL, *plotName=NULL;
	PyObject *tmpStr=NULL, *tmpArray=NULL, *data=NULL, *scales=NULL, *dflScale=NULL;
	PyArrayObject *numarray=NULL;
	PyObject *tuple=NULL, *list=NULL;
	npy_intp padding, isComplex, count, longest, binary;
	npy_intp *dim=NULL, dimSize=0;
	npy_intp *dim1=NULL, dimSize1=0; 
	int nDim, nDim1;
	npy_intp dimProd, i, j, ok, hasOwnDims;
	struct FastArray *faPtr=NULL, *tmpFaPtr, *faPos;
	npy_intp haveFlags, haveCount, haveLength, haveDims, haveVariables, haveScale;
	npy_intp storeResult=1;
	double dd, di, cc[2];
	npy_intp ii;
			
	if (!PyArg_ParseTuple(args, "sii", &fileName, &debugMode, &reverseBytes)) {
		status=1;
	}

	if (debugMode) {
		fprintf(df, "raw_read: reading file %s.\n", fileName);
	}

	// We have no open file
	f=NULL;

	// Open file
	if (!status) {
		f=fopen(fileName, "rb");
		if (!f) {
			status=1;
			if (debugMode) {
				fprintf(df, "raw_read: can't open file.\n", fileName);
			}
		}
	}

	// Create empty list
	if (!status) {
		list=PyList_New(0);
		if (!list) {
			// Out of memory
			status=1;
			if (debugMode) {
				fprintf(df, "raw_read: can't allocate return list.\n");
			}
		}
	}

	// Read plot blocks
	while (!status) {
		// Must store result. Initially this means that we are resetting structures 
		// and starting with a new plot block. 
		if (storeResult) {
			tuple=NULL;
			plotTitle=NULL;
			plotDate=NULL;
			plotName=NULL;
			data=NULL;
			scales=NULL;
			dflScale=NULL;
			padding=1;

			// Assume complex plot. 
			isComplex=0;
			count=0;
			longest=0;
			binary=0;
			nDim=0;

			// We don't have plot flags yet
			haveFlags=0;
			haveCount=0;
			haveLength=0;
			haveDims=0;
			haveVariables=0;
			haveScale=0;

			// Now go to ordinary mode
			storeResult=0;
		}
		// Read first line of the plot block
		if (!status) {
			if (!readLine(f, &lineBuf, &lineBufSize, &bytesRead)) {
				// Out of memory
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: can't allocate line buffer.\n");
				}
				break;
			}
			if (feof(f)) {
				// EOF, we're done
				break;
			}
		}

		// Handle line
		p=lineBuf;
		if (matchPrefix("title:", &p, 1)) {
			// Plot title
			if (plotTitle) {
				// Multiple title: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple title lines.\n");
				}
				break;
			}
			skipSpaces(&p);
			plotTitle=PyString_FromString(p);
		} else if (matchPrefix("date:", &p, 1)) {
			// Plot date
			if (plotDate) {
				// Multiple date: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple date lines.\n");
				}
				break;
			}
			skipSpaces(&p);
			plotDate=PyString_FromString(p);
		} else if (matchPrefix("plotname:", &p, 1)) {
			// Plot name
			if (plotName) {
				// Multiple plotname: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple plotname lines.\n");
				}
				break;
			}
			skipSpaces(&p);
			plotName=PyString_FromString(p);
		} else if (matchPrefix("flags:", &p, 1)) {
			// Plot flags
			if (haveFlags) {
				// Multiple flags: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple flags lines.\n");
				}
				break;
			}
			while (*p) {
				skipSpaces(&p);
				if (matchPrefix("real", &p, 1)) {
					isComplex=0;
				} else if (matchPrefix("complex", &p, 1)) {
					isComplex=1;
				} else if (matchPrefix("unpadded", &p, 1)) {
					padding=0;
				} else if (matchPrefix("padded", &p, 1)) {
					padding=1;
				} else {
					// Ignore unknown flags
					skipToSpace(&p);
				}
			}
			haveFlags=1;
		} else if (matchPrefix("no. variables:", &p, 1)) {
			// Number of vectors
			if (haveCount) {
				// Multiple no. variables: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple no. variables lines.\n");
				}
				break;
			}
			skipSpaces(&p);
			count=matchInteger(&p, &ok);
			haveCount=1;
		} else if (matchPrefix("no. points:", &p, 1)) {
			// Number of points in longest vector
			if (haveLength) {
				// Multiple no. points: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple no. points lines.\n");
				}
				break;
			}
			skipSpaces(&p);
			longest=matchInteger(&p, &ok);
			haveLength=1;
		} else if (matchPrefix("dimensions:", &p, 1)) {
			// Shape of the longest vector
			if (haveDims) {
				// Multiple dimensions: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple dimensions lines.\n");
				}
				break;
			}
			// If we already have variables, the plot is not valid
			if (haveVariables) {
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: variables declared before dimensions.\n");
				}
				break;
			}
			skipSpaces(&p);
			if (!readDims(&p, &dim, &dimSize, &nDim)) {
				// Out of me				p1=lineBuf1;mory
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: can't allocate default dimensions.\n");
				}
				break;
			}
			haveDims=1;
		} else if (matchPrefix("variables:", &p, 1)) {
			// List of variables
			if (haveVariables) {
				// Multiple variables: lines, stop
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: multiple variable lines.\n");
				}
				break;
			}
			// If we don't have flags, count or length, the plot is not valid
			if (!haveFlags || !haveCount || !haveLength) {
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: variables declared before flags, no. variables, or no. points.\n");
				}
				break;
			}

			// Default dimensions and consistency check
			if (nDim==0) {
				if (!addDimension(&dim, &dimSize, &nDim, longest)) {
					// Out of memory, stop reading
					status=1;
					if (debugMode) {
						fprintf(df, "raw_read: can't allocate default dimensions for length.\n");
					}
					break;
				}
			} else  {
				// Check if dimensions is consistent with longest
				dimProd=1;
				for(i=0;i<nDim;i++)
					dimProd*=dim[i];
				if (dimProd!=longest) {
					// Not consistent, ignore default dimensions
					dim[0]=longest;
					nDim=1;
				}
			}

			// Prepare dictionaries]
			data=PyDict_New();
			if (!data) {
				// Out of memory
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: failed to allocate array dictionary.\n");
				}
				break;
			}
			scales=PyDict_New();
			if (!scales) {
				// Out of memory
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: failed to allocate scales dictionary.\n");
				}
				break;
			}

			// Prepare fast access structure
			tmpFaPtr=(struct FastArray *)PyMem_Realloc(faPtr, sizeof(struct FastArray)*count);
			if (!tmpFaPtr) {
				// Out of memory
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: can't allocate fast access structure.\n");
				}
				break; 
			}
			faPtr=tmpFaPtr;

			// Do not read line if there is anything left in the current line to read
			// For spectre output - 'Variables:' is followed by the first variable on the same line
			for(;*p==' '||*p=='\t';p++);
			noReadFirst=0;
			if (*p!='\n' && *p!='\r' && *p!=0) {
			  // Do not read line, use current one
			  noReadFirst=1;
			}
				
			// Read variable descriptions
			for(i=0;i<count;i++) {
				if (noReadFirst && i==0) {
					p1=p;
				} else {
					bytesRead1=0;
					
					if (!readLine(f, &lineBuf1, &lineBufSize1, &bytesRead1)) {
						// Out of memory
						status=1;
						if (debugMode) {
							fprintf(df, "raw_read: can't allocate variable line buffer.\n");
						}
						break;
					}
					p1=lineBuf1;
				}

				// Handle one variable entry
				// Skip variable number (integer)
				skipSpaces(&p1);
				matchInteger(&p1, &ok);
				// Find the end of variable name
				skipSpaces(&p1); // moves to the beginning of variable name
				p2=p1;
				skipToSpace(&p2); // moves beyond the end of variable name
				if (p2==p1) {
					// Oh-oh, empty variable name.
					status=1;
					if (debugMode) {
						fprintf(df, "raw_read: empty variable name found.\n");
					}
					break;
				}
				
				// Skip variable type (string)
				// Its position is in pp
				pp=p2;
				skipSpaces(&pp);
				skipToSpace(&pp);

				// Extract name, convert v(name) to name
				if (matchPrefix("v(", &p1, 1)) {
					// Move p2 on ')'
					p2--;
				}

				// Mark end of name
				tmpc=*p2;
				*p2=0;

				// p1 and p2 mark the beginning and the end of name

				// Assume dimensions are equal to default dimensions
				hasOwnDims=0;

				// Extract variable flags
				haveScale=0;
				while (*pp) {
					skipSpaces(&pp);
					if (matchPrefix("scale=", &pp, 1)) {
						// Get scale name
						if (haveScale) {
							// Multiple scale=: flags, stop
							status=1;
							if (debugMode) {
								fprintf(df, "raw_read: multiple scale flags.\n");
							}
							break;
						}
						skipSpaces(&pp);
						p3=pp;
						p4=p3;
						skipToSpace(&p4);
						// Convert scale name from v(name) to name
						if (matchPrefix("v(", &p3, 1)) {
							// Move p4 on ')'
							p4--;
						}
						// Mark end of scale name with a null character
						tmpc1=*p4;
						*p4=0;
						// p3 and p4 mark the beginning and the end of scale name
						// Create Python string with scale name
						tmpStr=PyString_FromString(p3);
						if (!tmpStr) {
							// Out of memory
							status=1;
							if (debugMode) {
								fprintf(df, "raw_read: failed to allocate scale name string.\n");
							}
							break;
						}
						// Insert it in scales dictionary
						if (PyDict_SetItemString(scales, p1, tmpStr)!=0) {
							// Failed to insert
							Py_DECREF(tmpStr);
							tmpStr=NULL;
							status=1;
							if (debugMode) {
								fprintf(df, "raw_read: failed to insert scale name in dictionary.\n");
							}
							break;
						}
						// Need to do this because refcount was increased by SetItem
						Py_DECREF(tmpStr);
						tmpStr=NULL;
						// Restore original string at end of scale name
						*p4=tmpc1;
						haveScale=1;
					} else if (matchPrefix("dims=", &pp, 1)) {
						// Get dimensions
						nDim1=0;
						skipSpaces(&pp);
						if (!readDims(&pp, &dim1, &dimSize1, &nDim1)) {
							// Out of memory
							status=1;
							if (debugMode) {
								fprintf(df, "raw_read: can;t allocate variable dimensions.\n");
							}
							break;
						}
						// This vector has own dimensionsin
						hasOwnDims=1;
					} else {
						// Ignore everything else
						skipToSpace(&pp);
					}
				} // Extract variable flags

				// An error occured while reading flags, stop
				if (status) {
					break;
				}

				// Create dictionary entry for variable
				// Allocate array
				if (hasOwnDims) {
					if (isComplex) {
						// tmpArray=PyArray_FromDims(nDim1, dim1, PyArray_CDOUBLE);
						// tmpArray=PyArray_SimpleNew(nDim1, dim1, PyArray_CDOUBLE);
						tmpArray=PyArray_SimpleNew(nDim1, dim1, NPY_CDOUBLE);
					} else {
						// tmpArray=PyArray_FromDims(nDim1, dim1, PyArray_DOUBLE);
						// tmpArray=PyArray_SimpleNew(nDim1, dim1, PyArray_DOUBLE);
						tmpArray=PyArray_SimpleNew(nDim1, dim1, NPY_DOUBLE);
					}
				} else {
					if (isComplex) {
						// tmpArray=PyArray_FromDims(nDim, dim, PyArray_CDOUBLE);
						// tmpArray=PyArray_SimpleNew(nDim, dim, PyArray_CDOUBLE);
						tmpArray=PyArray_SimpleNew(nDim, dim, NPY_CDOUBLE);
					} else {
						// tmpArray=PyArray_FromDims(nDim, dim, PyArray_DOUBLE);
						// tmpArray=PyArray_SimpleNew(nDim, dim, PyArray_DOUBLE);
						tmpArray=PyArray_SimpleNew(nDim, dim, NPY_DOUBLE);
					}
				}
				if (!tmpArray) {
					// Out of memory
					status=1;
					if (debugMode) {
						fprintf(df, "raw_read: failed to allocate array.\n");
					}
					break;
				}

				// Put it in data dictionary
				if (PyDict_SetItemString(data, p1, tmpArray)!=0) {
					// Failed to insert
					Py_DECREF(numarray);
					numarray=NULL;
					status=1;
					if (debugMode) {
						fprintf(df, "raw_read: failed to insert array in dictionary.\n");
					}
					break;
				}
				// Need to do this because refcount was increased by SetItem
				Py_DECREF(tmpArray);
				// tmpArray is now a borrowed reference

				// If this is the first vector, note that it is the default scale
				if (i==0) {
					dflScale=PyString_FromString(p1);
					if (!dflScale) {
						// Out of memory
						status=1;
						if (debugMode) {
							fprintf(df, "raw_read: failed to allocate default scale string.\n");
						}
						break;
					}
				}

				// Prepare fast access structure
				numarray=(PyArrayObject *)tmpArray;
				if (isComplex)
					faPtr[i].isComplex=1;
				else
					faPtr[i].isComplex=0;
				// faPtr[i].data=numarray->data;
				faPtr[i].data=PyArray_DATA(numarray);
				// faPtr[i].pos=numarray->data;
				faPtr[i].pos=PyArray_DATA(numarray);
				// faPtr[i].stride=numarray->strides[numarray->nd-1];
				faPtr[i].stride=PyArray_STRIDE(numarray, PyArray_NDIM(numarray)-1);
				// faPtr[i].length=PyArray_Size(tmpArray);
				faPtr[i].length=PyArray_SIZE(numarray);

				// Restore original string at end of name
				*p2=tmpc;

				haveVariables=1;
			} // Read variable descriptions

			if (status) {
				// We had an error in variables list, stop now
				break;
			}
		} else if (matchPrefix("values:", &p, 1)) {
			// Ascii data starts here
			binary=0;

			// Check if we have everything
			if (!haveFlags || !haveCount || !haveLength || !haveVariables) {
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: missing flags, no. variables, no. points, or variables lines before data.\n");
				}
				break;
			}

			// No error in data
			ok=1;

			// Read ascii data
			for(i=0;i<longest && ok;i++) {
				int dummyRetval;
				// Read index
				dummyRetval=fscanf(f, " %d", &ii);
				for(faPos=faPtr, j=0; j<count; faPos++, j++) {
					if (i<faPos->length) {
						if (!isComplex) {
							if (fscanf(f, " %lf", &dd) != 1) {
								ok=0;
								break;
							} 
							*((npy_double *)(faPos->pos))=dd;
						} else {
							if (fscanf(f, " %lf, %lf", &dd, &di) != 2) {
								ok=0;
								break;
							} 
							((npy_cdouble *)(faPos->pos))->real=dd;
							((npy_cdouble *)(faPos->pos))->imag=di;
						}
						faPos->pos+=faPos->stride;
					} else if (padding) {
						if (!isComplex) {
							if (fscanf(f, " %lf", &dd) != 1) {
								ok=0;
								break;
							} 
						} else {
							if (fscanf(f, " %lf, %lf", &dd, &di) != 2) {
								ok=0;
								break;
							} 
						}
					}
				}
			}

			// Check if reading went well
			if (!ok) {
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: reading ascii data failed.\n");
				}
				break;
			}

			// Flag for triggering the creation of a tuple 
			// and resetting temporary structure pointers
			storeResult=1;
		} else if (matchPrefix("binary:", &p, 1)) {
			// Binary data starts here
			binary=1;

			// Check if we have everything
			if (!haveFlags || !haveCount || !haveLength || !haveVariables) {
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: missing flags, no. variables, no. points, or variables lines before data.\n");
				}
				break;
			}

			// No error in data
			ok=1;

			// Read binary data
			for(i=0;i<longest && ok;i++) {
				for(faPos=faPtr, j=0; j<count; faPos++, j++) {
					if (i<faPos->length) {
						if (!isComplex) {
							if (fread((char *) &dd, sizeof(double), 1, f) != 1) {
								ok=0;
								break;
							} 
							if (reverseBytes) 
							  reverse(&dd, sizeof(double), 1);
							*((npy_double *)(faPos->pos))=dd;
						} else {
							if (fread((char *) cc, 2*sizeof(double), 1, f) != 1) {
								ok=0;
								break;
							} 
							if (reverseBytes) 
							  reverse(&cc, sizeof(double), 2);
							((npy_cdouble *)(faPos->pos))->real=*cc;
							((npy_cdouble *)(faPos->pos))->imag=*(cc+1);
						}
						faPos->pos+=faPos->stride;
					} else if (padding) {
						if (!isComplex) {
							if (fread((char *) &dd, sizeof(npy_double), 1, f) != 1) {
								ok=0;
								break;
							} 
						} else {
							if (fread((char *) &cc, 2*sizeof(double), 1, f) != 1) {
								ok=0;
								break;
							} 
						}
					}
				}
			}

			// Check if reading went well
			if (!ok) {
				status=1;
				if (debugMode) {
					fprintf(df, "raw_read: reading binary data failed.\n");
				}
				break;
			}
			
			// Reverse bytes

			// Flag for triggering the creation of a tuple 
			// and resetting temporary structure pointers
			storeResult=1;
		} else {
			// Ignore unknown section
		}

		if (storeResult) {
			// Done reading, create tuple
			if (!status) {
				// No title, use empty string
				if (!plotTitle)
					plotTitle=PyString_FromString("");
				if (!plotDate)
					plotDate=PyString_FromString("");
			  
				tuple=PyTuple_Pack(6, data, dflScale, scales, plotTitle, plotDate, plotName);
				if (!tuple) {
					// Out of memory
					status=1;
					if (debugMode) {
						fprintf(df, "raw_read: failed to allocate tuple.\n");
					}
				}
			}

			// Add tuple to the list
			if (!status) {
				if (PyList_Append(list, tuple)!=0) {
					status=1;
					if (debugMode) {
						fprintf(df, "raw_read: failed to append tuple to return list.\n");
					}
					// Delete tuple
					Py_XDECREF(tuple);
				}
			}

			// PyTuple_Pack() increases reference counts. Now decrease them.
			// Even if PyTuple_Pack() was not called due to an error or failed 
			// we must do this to get rid of dangling references. 
			// By this references become borrowed references.
			Py_XDECREF(data);
			Py_XDECREF(scales);
			Py_XDECREF(dflScale);
			Py_XDECREF(plotTitle);
			Py_XDECREF(plotDate);
			Py_XDECREF(plotName);
			Py_XDECREF(tuple);

			// Set pointers to NULL so that we know later these objects are already in some
			// higher level structure and will be freed along with that structure. 
			// By this we throw away borrowed references. 
			data=NULL;
			scales=NULL;
			dflScale=NULL;
			plotTitle=NULL;
			plotDate=NULL;
			plotName=NULL;
			tuple=NULL;
		}
	} // Read plot block

	// Have error
	if (status) {
		// Delete temporary objects (either not in list or NULL)
		Py_XDECREF(data);
		Py_XDECREF(scales);
		Py_XDECREF(dflScale);
		Py_XDECREF(plotTitle);
		Py_XDECREF(plotDate);
		Py_XDECREF(plotName);

		// Delete list (if there is any)
		Py_XDECREF(list);

		// Make return value None
		list=Py_None;
	}

	// Close file
	if (f) {
		fclose(f);
	}

	// Free default line buffer
	if (lineBuf)
		PyMem_Free(lineBuf);

	// Free variable line buffer
	if (lineBuf1)
		PyMem_Free(lineBuf1);

	// Free default dimensions 
	if (dim)
		PyMem_Free(dim);

	// Free array dimensions
	if (dim1)
		PyMem_Free(dim1);

	// Free fast access structure
	if (faPtr) 
		PyMem_Free(faPtr);

	return list;
}

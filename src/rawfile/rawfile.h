#include "Python.h"

/* Python callable functions */
static PyObject *raw_write(PyObject *self, PyObject *args);
static PyObject *raw_read(PyObject *self, PyObject *args);

#ifdef LINUX
#define __declspec(a) extern
#endif

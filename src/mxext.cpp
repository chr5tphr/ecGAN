#include "Python.h"
// #include "mxnet-cpp/MxNetCpp.h"

// using namespace mxnet::cpp;



static PyObject * mxext_ndlist2raw(PyObject *self, PyObject *args) {

  PyObject *ndmodule = PyImport_ImportModule("mxnet.ndarray");
  PyObject *nddict = PyModule_GetDict(ndmodule);
  PyObject *PyNDArray_Type = PyDict_GetItemString(nddict, "NDArray");

  PyObject *pyarr;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!", PyNDArray_Type, &pyarr)){
    return NULL;
  }

  PyObject *pyhandle = PyObject_GetAttrString(pyarr,"handle");
  PyObject *pyvalue = PyObject_GetAttrString(pyhandle,"value");

  void *handle = PyLong_AsVoidPtr(pyvalue);

  return PyLong_FromLong((long)handle);
  // Py_RETURN_NONE;
  // return ;
}

static PyMethodDef methods[] = {
  {"ndlist2raw", mxext_ndlist2raw, METH_VARARGS, "Serialize ndlist." },
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT,
  "mxext",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_mxext() {
  Py_Initialize();
  PyObject *m = PyModule_Create(&module_def);
  if (m == NULL){
    return NULL;
  }

  return m;
}

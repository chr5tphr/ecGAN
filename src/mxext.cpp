#include "Python.h"
#include "mxnet/ndarray.h"

//g++ -fPIC -shared -I/usr/include/python3.6m -I/home/chrstphr/libs/mxnet/{{.,nnvm,dmlc-core,dlpack}/include,mshadow} -o mxext.so mxext.cpp -DMSHADOW_USE_{MKL,CUDA}=0

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

  mxnet::NDArray *ndarr = static_cast<mxnet::NDArray*>(PyLong_AsVoidPtr(pyvalue));

  // std::vector<NDArray> data(num_args);
  // std::vector<std::string> names;
  // for (mx_uint i = 0; i < num_args; ++i) {
  //   data[i] = *static_cast<NDArray*>(args[i]);
  // }
  // if (keys != nullptr) {
  //   names.resize(num_args);
  //   for (mx_uint i = 0; i < num_args; ++i) {
  //     names[i] = keys[i];
  //   }
  // }
  // {
  //   dmlc::MemoryStringStream strm(&ret->ret_str);
  //   mxnet::NDArray::Save(strm, data, names);
  // }

  return PyLong_FromLong((long)(ndarr->dtype()));
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

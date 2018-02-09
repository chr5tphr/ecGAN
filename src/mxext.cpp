#include <Python.h>
#include <mxnet/ndarray.h>
#include <dmlc/memory_io.h>
#include <vector>

//g++ -fPIC -shared -I/usr/include/python3.6m -I/home/chrstphr/libs/mxnet/{{.,nnvm,dmlc-core,dlpack}/include,mshadow} -o mxext.so mxext.cpp -DMSHADOW_USE_{MKL,CUDA}=0

static PyObject * mxext_nddict2raw(PyObject *self, PyObject *args) {

  PyObject *ndmodule = PyImport_ImportModule("mxnet.ndarray");
  PyObject *nddict = PyModule_GetDict(ndmodule);
  PyObject *PyNDArray_Type = PyDict_GetItemString(nddict, "NDArray");

  PyObject *pydict;

  // parse arguments
  if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &pydict)){
    return NULL;
  }

  Py_ssize_t num_args = PyDict_Size(pydict);

  // return PyLong_FromLong(num_args);

  std::vector<mxnet::NDArray> data(num_args);
  std::vector<std::string> names(num_args);

  Py_ssize_t pos = 0;
  int offset = 0;
  PyObject *key,*val;

  while (PyDict_Next(pydict, &pos, &key, &val)){
    PyObject *pyhandle = PyObject_GetAttrString(val,"handle");
    PyObject *pylong = PyObject_GetAttrString(pyhandle,"value");
    data[offset] = *static_cast<mxnet::NDArray*>(PyLong_AsVoidPtr(pylong));
    Py_ssize_t strsize;
    char *cname = PyUnicode_AsUTF8AndSize(key,&strsize);
    std::string name(cname,strsize);
    names[offset] = name;
    offset++;
  }

  std::string *ret = dmlc::ThreadLocalStore<std::string>::Get();

  dmlc::MemoryStringStream strm(ret);
  mxnet::NDArray::Save(&strm, data, names);

  return PyBytes_FromStringAndSize(ret->c_str(), ret->length());
  // return PyLong_FromLong((long)(ndarr->dtype()));
  // Py_RETURN_NONE;
  // return ;
}

// static PyObject * mxext_raw2nddict(PyObject *self, PyObject *args) {
//
//   PyObject *ndmodule = PyImport_ImportModule("mxnet.ndarray");
//   PyObject *nddict = PyModule_GetDict(ndmodule);
//   PyObject *PyNDArray = PyDict_GetItemString(nddict, "NDArray");
//
//   PyObject *pybytes;
//
//   // parse arguments
//   if (!PyArg_ParseTuple(args, "O!", &PyBytes_Type, &pybytes)){
//     return NULL;
//   }
//
//   Py_ssize_t bsize = PyBytes_Size(pybytes);
//   char *buf = PyBytes_AsString(pybytes);
//
//   dmlc::MemoryFixedSizeStream strm((void*)buf, bsize);
//
//   std::vector<mxnet::NDArray> data;
//   std::vector<std::string> &names = *dmlc::ThreadLocalStore<std::vector<std::string> >::Get();
//
//   mxnet::NDArray::Load(&strm, &data, &names);
//
//   std::vector<void *> &handles = *dmlc::ThreadLocalStore<std::vector<void *> >::Get();
//   handles.resize(data.size());
//   std::vector<const char *> &keys = *dmlc::ThreadLocalStore<std::vector<const char *> >::Get();
//   keys.resize(data.size());
//
//   PyObject *pydict = PyDict_New();
//
//   for (size_t i = 0; i < data.size(); ++i) {
//     NDArray *handle = new NDArray();
//     *handle = data[i];
//     PyObject *pyndarr =
//
//     key = names[i].c_str();
//
//
//
//
//   }
//
//
//   for (size_t i = 0; i < names.size(); ++i) {
//   }
//
//
//
//   Py_ssize_t pos = 0;
//   int offset = 0;
//   PyObject *key,*val;
//
//   while (PyDict_Next(pydict, &pos, &key, &val)){
//     PyObject *pyhandle = PyObject_GetAttrString(val,"handle");
//     PyObject *pylong = PyObject_GetAttrString(pyhandle,"value");
//     data[offset] = *static_cast<mxnet::NDArray*>(PyLong_AsVoidPtr(pylong));
//     Py_ssize_t strsize;
//     char *cname = PyUnicode_AsUTF8AndSize(key,&strsize);
//     std::string name(cname,strsize);
//     names[offset] = name;
//     offset++;
//   }
//
//   std::string *ret = dmlc::ThreadLocalStore<std::string>::Get();
//
//   dmlc::MemoryStringStream strm(ret);
//   mxnet::NDArray::Save(&strm, data, names);
//
//   return PyBytes_FromStringAndSize(ret->c_str(), ret->length());
//   // return PyLong_FromLong((long)(ndarr->dtype()));
//   // Py_RETURN_NONE;
//   // return ;
// }

static PyMethodDef methods[] = {
  {"nddict2raw", mxext_nddict2raw, METH_VARARGS, "Serialize dict of NDArrays." },
  // {"raw2nddict", mxext_raw2nddict, METH_VARARGS, "Deserialize dict of NDArrays." },
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

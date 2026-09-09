// Embed the actual Python capture adapters behind the same native source ABI.
// Python object creation, GIL acquisition, copying and serialization are SUT cost.
#include <Python.h>
#include <cstdint>
static PyObject *offer_fn, *close_fn;
extern "C" int hf_open(const char* path) noexcept {
    Py_Initialize();
    PyObject* module = PyImport_ImportModule("benchmarks.capture.fair.python_endpoint");
    if (!module) { PyErr_Print(); return -1; }
    auto* init = PyObject_GetAttrString(module, "initialize");
    offer_fn = PyObject_GetAttrString(module, "offer");
    close_fn = PyObject_GetAttrString(module, "close");
    auto* result = PyObject_CallFunction(init, "s", path);
    if (!result) { PyErr_Print(); return -1; }
    Py_DECREF(result); Py_DECREF(init); Py_DECREF(module);
    PyEval_SaveThread();
    return 0;
}
extern "C" int hf_offer(const char* run, uint32_t p, uint64_t i, uint64_t ts, const char* body) noexcept {
    auto state = PyGILState_Ensure();
    auto* result = PyObject_CallFunction(offer_fn, "sIKKs", run, p,
        static_cast<unsigned long long>(i), static_cast<unsigned long long>(ts), body);
    int code = result ? int(PyLong_AsLong(result)) : 6;
    if (!result || PyErr_Occurred()) { PyErr_Print(); code = 6; }
    Py_XDECREF(result); PyGILState_Release(state); return code;
}
extern "C" int hf_close() noexcept {
    auto state = PyGILState_Ensure();
    auto* result = PyObject_CallNoArgs(close_fn);
    int code = result ? 0 : -1;
    if (!result) PyErr_Print();
    Py_XDECREF(result); PyGILState_Release(state); return code;
}

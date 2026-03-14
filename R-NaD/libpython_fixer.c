#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>
#include <stdio.h>

static PyObject* fix_libpython(PyObject* self, PyObject* args) {
    const char* libname;
    if (!PyArg_ParseTuple(args, "s", &libname)) {
        return NULL;
    }
    
    void* handle = dlopen(libname, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        return Py_BuildValue("s", dlerror());
    }
    return Py_BuildValue("s", "success");
}

static PyMethodDef myMethods[] = {
    {"fix_libpython", fix_libpython, METH_VARARGS, "Re-dlopen libpython with RTLD_GLOBAL"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "libpython_fixer", // name of module
    NULL,              // module documentation, may be NULL
    -1,                // size of per-interpreter state of the module
    myMethods
};

PyMODINIT_FUNC PyInit_libpython_fixer(void) {
    return PyModule_Create(&myModule);
}

//TODO: add credit
#include <Python.h>
#include <string>
#include "call_obj.h" // cython helper file
#include <iostream>
using namespace std;
class PyObjWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyObjWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyObjWrapper(const PyObjWrapper& rhs): PyObjWrapper(rhs.held) { // C++11 onwards only
    }

    PyObjWrapper(PyObjWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyObjWrapper(): PyObjWrapper(nullptr) {
    }

    ~PyObjWrapper() {
        Py_XDECREF(held);
    }

    PyObjWrapper& operator=(const PyObjWrapper& rhs) {
        PyObjWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyObjWrapper& operator=(PyObjWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    bool operator()(const double* x,  double const* const* p, double* residual, int numResiduals, int numParams) {
        bool flag = false;
        double* tmp_p = new double[numParams];
        // send tmp_p because cython can't receive const* const*.
        // This  doesn't matter because we don't change this values
        for (int i = 0; i < numParams; i++)
        {
            tmp_p[i] = p[0][i];
        }
        if (held) { // nullptr check
            flag = call_obj(held,x, tmp_p, residual, numResiduals, numParams); // note, no way of checking for errors until you return to Python
        }
        delete tmp_p;
        return flag;
    }

private:
    PyObject* held;
};
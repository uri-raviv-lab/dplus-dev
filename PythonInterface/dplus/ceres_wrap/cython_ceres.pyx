
# distutils: language = c++

import cython
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint8_t, uint16_t, int32_t, uint64_t
cimport cython.operator.dereference as drf


import numpy as np
cimport numpy as np
from enum import Enum
cimport ceres_static
# from cyres cimport *


############################

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

Ownership = enum("DO_NOT_TAKE_OWNERSHIP", "TAKE_OWNERSHIP")

MinimizerType = enum("LINE_SEARCH", "TRUST_REGION")

LinearSolverType = enum("DENSE_NORMAL_CHOLESKY", "DENSE_QR",
                        "SPARSE_NORMAL_CHOLESKY", "DENSE_SCHUR", "SPARSE_SCHUR",
                        "ITERATIVE_SCHUR", "CGNR")
PreconditionerType = enum("IDENTITY", "JACOBI", "SCHUR_JACOBI",
                          "CLUSTER_JACOBI", "CLUSTER_TRIDIAGONAL")
SparseLinearAlgebraLibraryType = enum("SUITE_SPARSE", "CX_SPARSE")
LinearSolverTerminationType = enum("TOLERANCE", "MAX_ITERATIONS", "STAGNATION",
                                   "FAILURE")
LoggingType = enum("SILENT", "PER_MINIMIZER_ITERATION")
LineSearchDirectionType = enum("STEEPEST_DESCENT",
                               "NONLINEAR_CONJUGATE_GRADIENT",
                               "LBFGS")
NonlinearConjugateGradientType = enum("FLETCHER_REEVES", "POLAK_RIBIRERE",
                                      "HESTENES_STIEFEL")
LineSearchType = enum("ARMIJO")
TrustRegionStrategyType = enum("LEVENBERG_MARQUARDT", "DOGLEG")
DoglegType = enum("TRADITIONAL_DOGLEG", "SUBSPACE_DOGLEG")
SolverTerminationType = enum("CONVERGENCE", "NO_CONVERGENCE", "FAILURE", "USER_SUCCESS", "USER_FAILURE")
CallbackReturnType = enum("SOLVER_CONTINUE", "SOLVER_ABORT", "SOLVER_TERMINATE_SUCCESSFULLY")
DumpFormatType = enum("CONSOLE", "PROTOBUF", "TEXTFILE")
DimensionType = enum(DYNAMIC=-1)
NumericDiffMethod = enum("CENTRAL", "FORWARD")

# This class exist because you can't just send python function to c++
# You will receive "bad argument to internal function" without it
class RunFunc:
    def __init__(self, func2run):
        self.func = func2run

    def in_func2run(self, params, num_residual):
        return self.func(params, num_residual)

cdef class PyResidual:
    cdef ceres_static.CostFunction* _residual_cost_function

    def __init__(self, x , y, num_params, num_residual, func2run, step_size, eps, best_param, best_eval):
        # const double *x, const double *y,
		# int numParams, int numResiduals, PyObjWrapper calcVector, double stepSize,
		# double eps, vector[double] pBestParams, double *pBestEval
        # np_y, num_params, num_residual, func2run, step_size, eps, best_param, best_eval
        cdef double* _x_ptr = NULL
        cdef np.ndarray[np.double_t, ndim=1] np_x
        np_x = x
        _x_ptr = <double*> np_x.data

        cdef double* _y_ptr = NULL
        cdef np.ndarray[np.double_t, ndim=1] np_y
        np_y = y
        _x_ptr = <double*> np_y.data


        cdef np.ndarray[np.double_t, ndim=1] array
        array = best_param
        cdef long size = array.size
        cdef vector[double] best_param_vec
        cdef long i
        for i in range(size):
            best_param_vec.push_back(array[i])
        in_class = RunFunc(func2run)
        cdef ceres_static.PyObjWrapper func2run_wrapper = ceres_static.PyObjWrapper(in_class.in_func2run)

        cdef double* _best_v_ptr = NULL
        cdef np.ndarray[np.double_t, ndim=1] best_v
        best_v = best_eval
        _best_v_ptr = <double*> best_v.data

        self._residual_cost_function = ceres_static.Residual.GetCeresCostFunction(_x_ptr, _y_ptr, num_params, num_residual,
                                                                            func2run_wrapper, step_size, eps,
                                                                             best_param_vec, _best_v_ptr)



cdef class PyCostFunction:
    cdef ceres_static.CostFunction* _cost_function

    cpdef parameter_block_sizes(self):
        block_sizes = []
        cdef vector[int32_t] _parameter_block_sizes = self._cost_function.parameter_block_sizes()
        for i in range(_parameter_block_sizes.size()):
            block_sizes.append(_parameter_block_sizes[i])
        return block_sizes

    cpdef num_residuals(self):
        return self._cost_function.num_residuals()

    def evaluate(self, *param_blocks, **kwargs):

        include_jacobians = kwargs.get("include_jacobians", False)

        cdef double** _params_ptr = NULL
        cdef double* _residuals_ptr = NULL
        cdef double** _jacobians_ptr = NULL

        block_sizes = self.parameter_block_sizes()

        _params_ptr = <double**> malloc(sizeof(double*)*len(block_sizes))

        cdef np.ndarray[np.double_t, ndim=1] _param_block

        for i, param_block in enumerate(param_blocks):
            if block_sizes[i] != len(param_block):
                raise Exception("Expected param block of size %d, got %d" % (block_sizes[i], len(param_block)))
            _param_block = param_block
            _params_ptr[i] = <double*> _param_block.data

        cdef np.ndarray[np.double_t, ndim=1] residuals

        residuals = np.empty((self.num_residuals()), dtype=np.double)
        _residuals_ptr = <double*> residuals.data

        cdef np.ndarray[np.double_t, ndim=2] _jacobian
        if include_jacobians:
            # jacobians is an array of size CostFunction::parameter_block_sizes_
            # containing pointers to storage for Jacobian matrices corresponding
            # to each parameter block. The Jacobian matrices are in the same
            # order as CostFunction::parameter_block_sizes_. jacobians[i] is an
            # array that contains CostFunction::num_residuals_ x
            # CostFunction::parameter_block_sizes_[i] elements. Each Jacobian
            # matrix is stored in row-major order, i.e., jacobians[i][r *
            # parameter_block_size_[i] + c]
            jacobians = []
            _jacobians_ptr = <double**> malloc(sizeof(double*)*len(block_sizes))
            for i, block_size in enumerate(block_sizes):
                jacobian = np.empty((self.num_residuals(), block_size), dtype=np.double)
                jacobians.append(jacobian)
                _jacobian = jacobian
                _jacobians_ptr[i] = <double*> _jacobian.data

        self._cost_function.Evaluate(_params_ptr, _residuals_ptr, _jacobians_ptr)

        free(_params_ptr)

        if include_jacobians:
            free(_jacobians_ptr)
            return residuals, jacobians
        else:
            return residuals


cdef class PyLossFunction:
    cdef ceres_static.LossFunction* _loss_function

    def __cinit__(self):
        pass

cdef class PyTrivialLoss(PyLossFunction):
    cdef ceres_static.TrivialLoss* _loss
    def __init__(self):
        self._loss_function =  self._loss

cdef class PyHuberLoss(PyLossFunction):
    cdef ceres_static.HuberLoss* _loss
    def __init__(self, double _a):
        self._loss_function =  self._loss


cdef class PySoftLOneLoss(PyLossFunction):
    def __init__(self, double _a):
        self._loss_function =  new ceres_static.SoftLOneLoss(_a)

cdef class PyCauchyLoss(PyLossFunction):
    def __init__(self, double _a):
        self._loss_function =  new ceres_static.CauchyLoss(_a)

cdef class PyArctanLoss(PyLossFunction):
    def __init__(self, double _a):
        """
        Loss that is capped beyond a certain level using the arc-tangent
        function. The scaling parameter 'a' determines the level where falloff
        occurs. For costs much smaller than 'a', the loss function is linear
        and behaves like TrivialLoss, and for values much larger than 'a' the
        value asymptotically approaches the constant value of a * PI / 2.

          rho(s) = a atan(s / a).

        At s = 0: rho = [0, 1, 0].
        """
        self._loss_function =  new ceres_static.ArctanLoss(_a)

cdef class PyTolerantLoss(PyLossFunction):
    """
    Loss function that maps to approximately zero cost in a range around the
    origin, and reverts to linear in error (quadratic in cost) beyond this
    range. The tolerance parameter 'a' sets the nominal point at which the
    transition occurs, and the transition size parameter 'b' sets the nominal
    distance over which most of the transition occurs. Both a and b must be
    greater than zero, and typically b will be set to a fraction of a. The
    slope rho'[s] varies smoothly from about 0 at s <= a - b to about 1 at s >=
    a + b.

    The term is computed as:

      rho(s) = b log(1 + exp((s - a) / b)) - c0.

    where c0 is chosen so that rho(0) == 0

      c0 = b log(1 + exp(-a / b)

    This has the following useful properties:

      rho(s) == 0               for s = 0
      rho'(s) ~= 0              for s << a - b
      rho'(s) ~= 1              for s >> a + b
      rho''(s) > 0              for all s

    In addition, all derivatives are continuous, and the curvature is
    concentrated in the range a - b to a + b.

    At s = 0: rho = [0, ~0, ~0].
    """
    def __init__(self, double _a, double _b):
        self._loss_function =  new ceres_static.TolerantLoss(_a, _b)

cdef class PyComposedLoss(PyLossFunction):

    def __init__(self, PyLossFunction f, PyLossFunction g):
        self._loss_function =  new ceres_static.ComposedLoss(f._loss_function,
                                                     ceres_static.DO_NOT_TAKE_OWNERSHIP,
                                                     g._loss_function,
                                                     ceres_static.DO_NOT_TAKE_OWNERSHIP)

cdef class PyScaledLoss(PyLossFunction):

    def __cinit__(self, PyLossFunction loss_function, double _a):
        self._loss_function =  new ceres_static.ScaledLoss(loss_function._loss_function,
                                                   _a,
                                                   ceres_static.DO_NOT_TAKE_OWNERSHIP)
    def __dealloc__(self):
        del self._loss_function

cdef class PySolverSummary:
    cdef ceres_static.SolverSummary _summary

    def briefReport(self):
        return self._summary.BriefReport()

    def fullReport(self):
        return self._summary.FullReport()

    # def get_ter_type(self):
    #     return self._summary.termination_type

    property termination_type:
        def __get__(self):
            return self._summary.termination_type
        def __set__(self, value):
            self._summary.termination_type = value

    property final_cost:
        def __get__(self):
            return self._summary.final_cost

cdef class PyProblem:
    cdef ceres_static.Problem _problem

    def __cinit__(self):
        pass

    # loss_function=NULL yields squared loss
    cpdef add_residual_block(self,
                             PyResidual cost_function,
                             PyLossFunction loss_function,
                             parameter_blocks=[]):

        cdef np.ndarray _tmp_array
        cdef vector[double*] _parameter_blocks
        cdef double f

        cdef ceres_static.ResidualBlockId _block_id

        for parameter_block in parameter_blocks:
            _tmp_array = np.ascontiguousarray(parameter_block, dtype=np.double)
            _parameter_blocks.push_back(<double*> _tmp_array.data)
        _block_id = self._problem.AddResidualBlock(cost_function._residual_cost_function,
                                                   loss_function._loss_function,
                                                   _parameter_blocks)
        block_id = PyResidualBlockId()
        block_id._block_id = _block_id
        return block_id

    cpdef evaluate(self, residual_blocks, apply_loss_function=True):

        cdef double cost

        options = PyEvaluateOptions()
        options.apply_loss_function = apply_loss_function
        options.residual_blocks = residual_blocks

        self._problem.Evaluate(options._options, &cost, NULL, NULL, NULL)
        return cost

    cpdef set_parameter_block_constant(self, block):
        cdef np.ndarray _tmp_array = np.ascontiguousarray(block, dtype=np.double)
        cdef double* _values = <double*> _tmp_array.data
        self._problem.SetParameterBlockConstant(_values)

    cpdef set_parameter_block_variable(self, block):
        cdef np.ndarray _tmp_array = np.ascontiguousarray(block, dtype=np.double)
        cdef double* _values = <double*> _tmp_array.data
        self._problem.SetParameterBlockVariable(_values)

    cpdef set_parameter_lower_bound(self, values, index, lower_bound):
        cdef np.ndarray _tmp_array = np.ascontiguousarray(values, dtype=np.double)
        cdef double* _values = <double*> _tmp_array.data
        self._problem.SetParameterLowerBound(_values, index, lower_bound)

    cpdef set_parameter_upper_bound(self, values, index, upper_bound):
        cdef np.ndarray _tmp_array = np.ascontiguousarray(values, dtype=np.double)
        cdef double* _values = <double*> _tmp_array.data
        self._problem.SetParameterUpperBound(_values, index, upper_bound)



def solve(PySolverOptions options, PyProblem problem, PySolverSummary summary):
    ceres_static.Solve(drf(options._options), &problem._problem, &summary._summary)

cdef class PyResidualBlockId:
    cdef ceres_static.ResidualBlockId _block_id

cdef class PyEvaluateOptions:
    cdef ceres_static.EvaluateOptions _options

    def __cinit__(self):
        pass

    def __init__(self):
        self._options = ceres_static.EvaluateOptions()

    property residual_blocks:
        def __get__(self):
            blocks = []
            cdef int i
            for i in range(self._options.residual_blocks.size()):
                block = PyResidualBlockId()
                block._block_id = self._options.residual_blocks[i]
                blocks.append(block)
            return blocks
        def __set__(self, blocks):
            self._options.residual_blocks.clear()
            cdef PyResidualBlockId block
            for block in blocks:
                self._options.residual_blocks.push_back(block._block_id)

    property apply_loss_function:
        def __get__(self):
            return self._options.apply_loss_function
        def __set__(self, value):
            self._options.apply_loss_function = value

cdef class PySolverOptions:
    cdef ceres_static.SolverOptions* _options

    def __cinit__(self):
        pass

    def __init__(self):
        self._options = new ceres_static.SolverOptions()

    property max_num_iterations:
        def __get__(self):
            return self._options.max_num_iterations

        def __set__(self, value):
            self._options.max_num_iterations = value

    property minimizer_progress_to_stdout:
        def __get__(self):
            return self._options.minimizer_progress_to_stdout

        def __set__(self, value):
            self._options.minimizer_progress_to_stdout = value

    property linear_solver_type:
        def __get__(self):
            return self._options.linear_solver_type

        def __set__(self, value):
            self._options.linear_solver_type = value


    property trust_region_strategy_type:

        def __get__(self):
            return self._options.trust_region_strategy_type

        def __set__(self, value):
            new_val = str_to_enum_str(value)
            int_val = 0
            if new_val not in TrustRegionStrategyType.reverse_mapping.values():
                raise Exception("no such trust region strategy type")

            for key, str_val in TrustRegionStrategyType.reverse_mapping.items():
                if str_val == new_val:
                    int_val = key
            self._options.trust_region_strategy_type = int_val

    property dogleg_type:
        def __get__(self):
            return self._options.dogleg_type

        def __set__(self, value):
            self._options.dogleg_type = value

    property preconditioner_type:
        def __get__(self):
            return self._options.preconditioner_type

        def __set__(self, value):
            self._options.preconditioner_type = value

    property num_threads:
        def __get__(self):
            return self._options.num_threads

        def __set__(self, value):
            self._options.num_threads = value


    property use_nonmonotonic_steps:
        def __get__(self):
            return self._options.use_nonmonotonic_steps

        def __set__(self, value):
            self._options.use_nonmonotonic_steps = value

    property function_tolerance:
        def __get__(self):
            return self._options.function_tolerance

        def __set__(self, value):
            self._options.function_tolerance = value

    property update_state_every_iteration:
        def __get__(self):
            return self._options.update_state_every_iteration

        def __set__(self, value):
            self._options.update_state_every_iteration = value

    property gradient_tolerance:
        def __get__(self):
            return self._options.gradient_tolerance

        def __set__(self, value):
            self._options.gradient_tolerance = value

    property use_inner_iterations:
        def __get__(self):
            return self._options.use_inner_iterations

        def __set__(self, value):
            self._options.use_inner_iterations = value

    property minimizer_type:
        def __get__(self):
            return self._options.minimizer_type

        def __set__(self, value):
            new_val = str_to_enum_str(value)
            int_val = 0
            if new_val not in MinimizerType.reverse_mapping.values():
                raise Exception("no such minimizer type")

            for key, str_val in MinimizerType.reverse_mapping.items():
                if str_val == new_val:
                    int_val = key
            self._options.minimizer_type = int_val

    property line_search_direction_type:
        def __get__(self):
            return self._options.line_search_direction_type

        def __set__(self, value):
            new_val = str_to_enum_str(value)
            int_val = 0
            if new_val not in LineSearchDirectionType.reverse_mapping.values():
                raise Exception("line search direction type")

            for key, str_val in LineSearchDirectionType.reverse_mapping.items():
                if str_val == new_val:
                    int_val = key
            self._options.line_search_direction_type = int_val

    property nonlinear_conjugate_gradient_type:
        def __get__(self):
            return self._options.nonlinear_conjugate_gradient_type

        def __set__(self, value):
            new_val = str_to_enum_str(value)
            int_val = 0
            if new_val not in NonlinearConjugateGradientType.reverse_mapping.values():
                raise Exception("no such nonlinear conjugate gradient type")

            for key, str_val in NonlinearConjugateGradientType.reverse_mapping.items():
                if str_val == new_val:
                    int_val = key
            self._options.nonlinear_conjugate_gradient_type = int_val

    property line_search_type:
        def __get__(self):
            return self._options.line_search_type

        def __set__(self, value):
            new_val = str_to_enum_str(value)
            int_val = 0
            if new_val not in LineSearchType.reverse_mapping.values():
                raise Exception("no such line search type")

            for key, str_val in LineSearchType.reverse_mapping.items():
                if str_val == new_val:
                    int_val = key
            self._options.line_search_type = int_val

#
def str_to_enum_str(value):
    val_up = value.upper()
    val_rep1 = val_up.replace("-", "_")
    val_rep2 = val_rep1.replace(" ", "_")
    return val_rep2


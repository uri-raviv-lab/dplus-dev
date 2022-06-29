import dplus_ceres as ceres
from dplus.Residuals import XRayResiduals, XRayLogResiduals, XRayRatioResiduals
import numpy as np
import os
import json
from dplus.FileReaders import _handle_infinity_for_json, NumpyHandlingEncoder

class PyCeresOptimizer:

    def __init__(self, calc_input, calc_runner=None, save_amp=False):

        self.calc_input = calc_input
        self.calc_runner = calc_runner
        if not self.calc_runner:
            from dplus.CalculationRunner import LocalRunner
            self.calc_runner = LocalRunner()

        self.problem = ceres.PyProblem()
        self.options = ceres.PySolverOptions()
        self.options.linear_solver_type=1
        self.best_params = None
        self.best_val = np.array([np.inf])
        self.bConverged = False
        self._best_results = None
        self.save_amp = save_amp
        self.init_problem()

    def init_problem(self):
        # Adapted from D+'s CeresOptimizer::InitProblem
        self.bConverged = False
        fit_pref = self.calc_input.FittingPreferences
        # This is the convergence that was writen in dplus
        conv = fit_pref.convergence * np.mean(self.calc_input.y) * 0.001
        self.options.function_tolerance = conv
        self.options.update_state_every_iteration = True
        self.options.gradient_tolerance = 1e-4 * self.options.function_tolerance
        mut_param = self.calc_input.get_mutable_params_array()

        paramdata = np.zeros(shape=(1,len(mut_param))) # D+'s curParams map to the same place
        if len(paramdata[0]) < 1 :
            raise Exception("There must be at least one mutable parameter in order to fit. Mark at least one parameter mutable and try again.",
						"No mutable parameters selected")
        for i in range (len(paramdata[0])):
            paramdata[0][i] = mut_param[i].value

        mut_param_values = self.calc_input.get_mutable_parameter_values()
        self.best_params = np.asanyarray(mut_param_values, dtype=np.double)
        self.best_val = np.array([np.inf])

        cost_class = None
        if fit_pref.x_ray_residuals_type == "Normal Residuals":
            cost_class = XRayResiduals(self.calc_input, self.calc_runner, self.best_params, self.best_val, self.save_amp)
        elif fit_pref.x_ray_residuals_type == "Ratio Residuals":
            cost_class = XRayRatioResiduals(self.calc_input, self.calc_runner, self.best_params, self.best_val, self.save_amp)
        elif fit_pref.x_ray_residuals_type == "Log Residuals":
            cost_class = XRayLogResiduals(self.calc_input, self.calc_runner, self.best_params, self.best_val, self.save_amp)
        else:
            raise Exception("residual type does not exists")

        y = np.asarray(self.calc_input.y)
        self.cost_function = ceres.PyResidual(np.asarray(self.calc_input.x), np.asarray(self.calc_input.y),
                                        len(mut_param_values), len(y), cost_class.run_generate, fit_pref.step_size, fit_pref.der_eps,
                                        self.best_params, self.best_val)
        self.params_value = paramdata[0]

        loss_type = fit_pref.loss_function
        # ["Trivial Loss", "Huber Loss", "Soft L One Loss",
        #                              "Cauchy Loss", "Arctan Loss", "Tolerant Loss"]
        if loss_type == "Trivial Loss":
            self.problem.add_residual_block(self.cost_function, ceres.PyTrivialLoss(), paramdata)
        elif loss_type == "Huber Loss":
            self.problem.add_residual_block(self.cost_function, ceres.PyHuberLoss(fit_pref.loss_func_param_one), paramdata)
        elif loss_type == "Soft L One Loss":
            self.problem.add_residual_block(self.cost_function, ceres.PySoftLOneLoss(fit_pref.loss_func_param_one), paramdata)
        elif loss_type == "Cauchy Loss":
            self.problem.add_residual_block(self.cost_function, ceres.PyCauchyLoss(fit_pref.loss_func_param_one), paramdata)
        elif loss_type == "Arctan Loss":
            self.problem.add_residual_block(self.cost_function, ceres.PyArctanLoss(fit_pref.loss_func_param_one), paramdata)
        elif loss_type == "Tolerant Loss":
            self.problem.add_residual_block(self.cost_function,
                                            ceres.PyTolerantLoss(fit_pref.loss_func_param_one, fit_pref.loss_func_param_two),
                                            paramdata)

        if fit_pref.minimizer_type == "Trust Region":
            start = 0
            for n in range (len(paramdata)):
                for i in range (start, int(start+ len(paramdata[n]))):
                    self.problem.set_parameter_lower_bound(paramdata[n], i - start, mut_param[i].constraints.min_value)
                    self.problem.set_parameter_upper_bound(paramdata[n], i - start, mut_param[i].constraints.max_value)
                start += len(paramdata[n])

        self.options.max_num_iterations = fit_pref.fitting_iterations
        self.options.minimizer_progress_to_stdout = True
        self.options.use_nonmonotonic_steps = True
        if fit_pref.minimizer_type == "Line Search":
            self.options.minimizer_type = fit_pref.minimizer_type
            self.options.line_search_direction_type = fit_pref.line_search_direction_type
            self.options.nonlinear_conjugate_gradient_type = fit_pref.nonlinear_conjugate_gradient_type
            self.options.line_search_type = fit_pref.line_search_type
        else:
            self.options.trust_region_strategy_type = fit_pref.trust_region_strategy_type
            self.options.use_inner_iterations = False

    def solve(self): #this function is called "CeresOptimizer::Iterate" in D+
        self._best_results = None
        summary = ceres.PySolverSummary()
        ceres.solve(self.options, self.problem, summary)

        print(summary.fullReport().decode("utf-8"))
        print("\n")
        cur_eval = summary.final_cost
        self.bConverged = (summary.termination_type == ceres.SolverTerminationType.CONVERGENCE)
        flag_valid_constraint = True
        mut_param = self.calc_input.get_mutable_params_array()
        for i in range(len(self.best_params)):
            if self.best_params[i] >= mut_param[i].constraints.min_value \
                    and self.best_params[i] <= mut_param[i].constraints.max_value:
                continue
            flag_valid_constraint = False
        if (flag_valid_constraint):
            cur_eval = self.best_val
            self.calc_input.set_mutable_parameter_values(self.best_params)
        return cur_eval

    @property
    def best_results(self):
        if not self._best_results:
            self._best_results = self.calc_runner.generate(self.calc_input)
        return self._best_results


    @staticmethod
    def fit(calc_input, calc_runner=None, save_amp=False):
        # Adapted from D+'s PerformModelFitting
        if not calc_runner:
            from dplus.CalculationRunner import LocalRunner
            calc_runner = LocalRunner()
        session_dir = calc_runner.session_directory
        PyCeresOptimizer.save_status(session_dir, error=False, is_running=True)
        try:
            optimizer = PyCeresOptimizer(calc_input, calc_runner, save_amp)
            gof = optimizer.solve()
            print("Iteration GoF = %f\n", gof)

            best_results = optimizer.best_results
            data_path = os.path.join(session_dir, "data.json")
            PyCeresOptimizer.save_dplus_arrays(best_results, data_path)
            PyCeresOptimizer.save_status(session_dir, error=False, is_running=False, progress=1.0, code=0, message="OK")
            return best_results
        except Exception as e:
            PyCeresOptimizer.save_status(session_dir, error=False, code=24, message=str(e), is_running=False, progress=0)
            raise e

    @staticmethod
    def save_status(session_dir, error,is_running=False, progress=0.0, code=0, message="OK"):
        if not error:
            status_dict = {"isRunning": is_running, "progress": progress, "code": code,
                           "message": str(message)}
        else:
            status_dict = {"error": {"code": code, "message": str(message)}}
        with open(os.path.join(session_dir, "fit_job.json"), 'w') as file:
            json.dump(status_dict, file)

    @staticmethod
    def save_dplus_arrays(best_results, outfile=None):
        '''
        a function for saving fit results in the bizarre special format D+ expects
        :param outfile:
        :return:
        '''
        param_tree = best_results._calc_data._get_dplus_fit_results_json()
        result_dict = {
            "ParameterTree": param_tree,
            "Graph": list(best_results.y)
        }
        if outfile:
            with open(outfile, 'w') as file:
                json.dump(_handle_infinity_for_json(result_dict), file, cls=NumpyHandlingEncoder)

        return result_dict

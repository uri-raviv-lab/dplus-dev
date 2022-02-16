import json
import math
import numpy as np
import os
from scipy import optimize
from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import LocalRunner
from dplus.FileReaders import _handle_infinity_for_json, NumpyHandlingEncoder
import codecs
import multiprocessing
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)

TRIVIAL = "Trivial Loss"
HUBER = "Huber Loss"
SOFT = "Soft L One Loss"
CAUCHY =  "Cauchy Loss"
ARCTAN = "Arctan Loss"
TOLERANT = "Tolerant Loss"


class LossFunction:
    def __init__(self, loss_name="Trivial Loss", param1=0.5, param2=0.5):
        losses_list = [HUBER, SOFT, CAUCHY, ARCTAN, TOLERANT, TRIVIAL]
        loss_str = ', '.join(losses_list)
        if loss_name not in losses_list:
            raise Exception("{} is invalid loss name, please choose loss from the list: {}".format(loss_name, loss_str))
        self.loss_name = loss_name
        if loss_name == TRIVIAL:
            self.loss_func = self.trivial
        if loss_name == HUBER:
            self.loss_func = self.huber
        if loss_name == SOFT:
            self.loss_func = self.soft_l1
        if loss_name == CAUCHY:
            self.loss_func = self.cauchy
        if loss_name == ARCTAN:
            self.loss_func = self.arctan
        if loss_name == TOLERANT:
            self.loss_func = self.tolerant

        self.param1 = param1
        self.param2 = param2

    # dplus loss functions
    def trivial(self, z):
        rho = np.empty((3, len(z)))
        rho[0] = z
        rho[1] = 1
        rho[2] = 0
        return rho

    def huber(self, z):
        rho = np.empty((3, len(z)))
        b = np.pow(self.param1, 2)
        if z > b:
            r = np.sqrt(z)
            rho[0] = 2 * self.param1 * r - b
            rho[1] = self.param1 / r
            rho[2] = - rho[1] / (2 * z)
        else:
            rho[0] = z
            rho[1] = 1
            rho[2] = 0
        return rho

    def soft_l1(self, z):
        rho = np.empty((3, len(z)))
        b = np.pow(self.param1)
        c = 1 / b

        sum_ = 1 + z * c
        tmp = np.sqrt(sum_)
        rho[0] = 2 * b * (tmp - 1)
        rho[1] = 1 / tmp
        rho[2] = -1 * (c * rho[1]) / (2 * sum_)
        return rho

    def cauchy(self, z):
        rho = np.empty((3, len(z)))
        b = np.pow(self.param1)
        c = 1 / b

        sum_ = 1 + z * c
        tmp = 1 / sum_
        rho[0] = b * np.log(sum_)
        rho[1] = tmp
        rho[2] = -c * (tmp * tmp)
        return rho

    def arctan(self, z):
        rho = np.empty((3, len(z)))
        b = 1 / np.pow(self.param1, 2)
        sum_ = 1 + z * z * b
        inv = 1 / sum_

        rho[0] = self.param1 * np.arctan(z, self.param1)
        rho[1] = inv
        rho[2] = -2 * z * b * (inv * inv)
        return rho

    def tolerant(self, z):
        rho = np.empty((3, len(z)))
        c = self.param2 * np.log(1.0 + np.exp((-self.param1 / self.param2)))

        tmp = (z - self.param1) / self.param2
        kLog2Pow53 = 36.7
        if tmp > kLog2Pow53:
            rho[0] = z - self.param1 - c
            rho[1] = 1.0
            rho[2] = 0.0
        else:
            e_x = np.exp(tmp)
            rho[0] = self.param2 * np.log(1.0 + e_x) - c
            rho[1] = e_x / (1.0 + e_x)
            rho[2] = 0.5 / (self.param2 * (1.0 + np.cosh(tmp)))
        return rho


class GenerateWrapper:
    def __init__(self, calc_input, dplus_runner=None, save_amp=False):
        self.input = calc_input
        self.dplus_runner = dplus_runner
        self.save_amp = save_amp
        if not self.dplus_runner:
            self.dplus_runner = LocalRunner()

    def run_generate(self, xdata, *params):
        '''
        scipy's optimization algorithms require a function that receives an x array and an array of parameters, and
        returns a y array.
        this function will be called repeatedly, until scipy's optimization has completed.
        '''
        self.input.set_mutable_parameter_values(
            params)  # we take the parameters given by scipy and place them inside our parameter tree
        generate_results = self.dplus_runner.generate(self.input, save_amp=self.save_amp)  # call generate
        return np.array(generate_results.y)  # return the results of the generate call


class MinimizeFitter:
    '''
    placeholder class. it should be possible to greatly increase list of optimizers we can run if we use scipy's minimize, not just scipy's curve_fit
    '''

    def __init__(self, calc_input):
        self.input = calc_input
        self.generate_runner = GenerateWrapper(calc_input)
        self.kwargs = {}

    def run_generate(self, xdata, *params):
        return self.generate_runner.run_generate(xdata, *params)

    def fnc2min(self, xdata, *params):
        return self.run_generate(xdata, *params) - self.input.y


class CurveFitter:
    '''
    a class for running fitting using scipy's optimize.curve_fit.
    it can be customized with any of the arguments scipy's curve_Fit can take, documented on the scipy website, by adding the
    values for those arguments to self,kwargs (eg, method="lm")

    some of the options allow writing and passing references to your own code.
    specifically:
    jac and loss for non-lm, Dfun for lm
    '''

    def __init__(self, calc_input, calc_run=None, save_amp=False):
        self.input = calc_input
        self.generate_runner = GenerateWrapper(calc_input, calc_run, save_amp=save_amp)
        self.session_dir = self.generate_runner.dplus_runner.session_directory
        self.kwargs = {}
        self.save_status(error=False)

    def run_generate(self, xdata, *params):
        # self.generate_count += 1
        return self.generate_runner.run_generate(xdata, *params)

    def get_args(self, y):
        # a work in progress function that translates as many of the existing fitting parameters as possible into arguments to curve_fit
        # based on:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        fit_prefs = self.input.FittingPreferences
        if fit_prefs.minimizer_type == "Trust Region" and fit_prefs.trust_region_strategy_type == "Levenberg-Marquardt":
            self.kwargs['method'] = 'lm'
            self.kwargs['ftol'] = fit_prefs.convergence * np.mean(y) * 1e-3
            self.kwargs['gtol'] = self.kwargs['ftol'] * 1e-4
            return  # levenberg marquardt can't accept most of the remaining arguments

        if fit_prefs.minimizer_type == "Trust Region" and fit_prefs.trust_region_strategy_type == "Dogleg":
            self.kwargs['method'] = 'dogbox'  # I think?
        else:
            self.kwargs[
                'method'] = 'trf'  # this is what scipy has to offer and hence worth testing, NOT a translation of D+'s other methods

        sigma, bounds = self.input.get_mutable_parameter_options()
        # the sigma given in parameters does not match the sigma scipy expects, which is for points of y
        # bounds is only allowed with methods other than lm
        self.kwargs['bounds'] = bounds
        loss = LossFunction(fit_prefs.loss_function, fit_prefs.loss_func_param_one, fit_prefs.loss_func_param_two)
        self.kwargs['loss'] = loss.loss_func
        # self.kwargs['max_nfev'] =  this is the max value of opetations not the max value of iteration
        # (each iteration can have few operations)
        self.kwargs['diff_step'] = fit_prefs.step_size
        self.kwargs['ftol'] = fit_prefs.convergence * np.mean(y) * 1e-3
        self.kwargs['gtol'] = self.kwargs['ftol'] * 1e-4
        self.kwargs['verbose'] = 1

        pass

    def run_fit(self):
        # all fittings needs x, y, and initial params
        self.save_status(error=False, is_running=True)
        try:
            x_data = self.input.x
            y_data = self.input.y
            p0 = self.input.get_mutable_parameter_values()

            # load optional additional parameters
            self.get_args(y_data)

            # call the specific fitting you want
            popt, pcov = optimize.curve_fit(self.run_generate, x_data, y_data, p0=p0, **self.kwargs)
            # popt is the optimized set of parameters from those we have indicated as mutable
            # we can insert them back into our CalculationInput and create the optimized parameter tree
            self.input.set_mutable_parameter_values(popt)

            # we can re-run generate to get the results of generate with them
            self.best_results = self.generate_runner.dplus_runner.generate(self.input)
            data_path = os.path.join(self.generate_runner.dplus_runner.session_directory, "data.json")
            self.save_dplus_arrays(data_path)
        except Exception as e:
            self.save_status(error=False, code=24, message=str(e), is_running=False, progress=0)
            return
        self.save_status(error=False, is_running=False, progress=1.0, code=0, message="OK")

    def save_dplus_arrays(self, outfile):
        '''
        a function for saving fit results in the bizarre special format D+ expects
        :param outfile:
        :return:
        '''
        param_tree = self.best_results._calc_data._get_dplus_fit_results_json()
        result_dict = {
            "ParameterTree": param_tree,
            "Graph": list(self.best_results.y)
        }
        with open(outfile, 'w') as file:
            json.dump(_handle_infinity_for_json(result_dict), file, cls=NumpyHandlingEncoder)
        # self.best_results.save_to_out_file(r"C:\Users\yael\Sources\temp\python_out_6.out")

    def save_status(self, error,is_running=False, progress=0.0, code=0, message="OK"):
        if not error:
            status_dict = {"isRunning": is_running, "progress": progress, "code": code,
                           "message": str(message)}
        else:
            status_dict = {"error": {"code": code, "message": str(message)}}
        with open(os.path.join(self.session_dir, "fit_job.json"), 'w') as file:
            json.dump(status_dict, file)

    def get_status(self):
        filename = os.path.join(self.session_dir, "fit_job.json")
        with codecs.open(filename, 'r', encoding='utf8') as f:
            result = f.read()
        try:
            result = json.loads(result)
            keys = ["isRunning", "progress", "code", "message"]
            if all(k in result for k in keys):
                return result
        except json.JSONDecodeError:
            return {"error": {"code": 22, "message": "json error in job status"}}
        except Exception as e:
            return {"error": {"code": 24, "message": str(e)}}



def main(infile, outfile):
    input = CalculationInput.load_from_state_file(infile)
    fitter = CurveFitter(input)
    fitter.run_fit()
    fitter.save_dplus_arrays(outfile)

# main(r"C:\Users\yael\Sources\dplus\PythonInterface\tests\reviewer_tests\files_for_tests\fit\gpu\short\Cylinder_3EDFit\Cylinder_3EDFit_fixed.state",
# main(r"C:\Users\yael\Sources\temp\sphere dogleg.state",
#      r"C:\Users\yael\Sources\temp\testfitoutput.json")
main(r"C:\Users\yael\Sources\dplus\Example Files\Examples\GPU\Very_Slow_GPU\7_Hydration_Correction\MT_With_Hydration_Correction.state",
     r"C:\Users\yael\Sources\temp\example_6.json")

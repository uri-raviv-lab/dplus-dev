
import numpy as np
from dplus.CalculationInput import CalculationInput
import math
import time

class Residual:
    def __init__(self, calc_input, calc_runner, best_params, best_eval, save_amp=False):

        self.calc_input = CalculationInput.copy_from_state(calc_input)
        self.calc_input.use_gpu = calc_input.use_gpu

        self.calc_runner = calc_runner
        self.np_y = np.asanyarray(self.calc_input.y, dtype=np.double)
        self._best_params = best_params
        self._best_eval = best_eval
        self.save_amp = save_amp

    def run_generate(self, params, num_residual):
        #comment out the timer
        start_time = time.time()
        self.calc_input.set_mutable_parameter_values(params)
        calc_result = self.calc_runner.generate(self.calc_input, self.save_amp)
        end_time = time.time()
        #print("generate time: {}".format(end_time - start_time))
        residual = np.asanyarray(calc_result.y, dtype=np.double)
        res = self.calc_residual(residual, num_residual) #note that residual changes within this function
        
        # with open("d:\\temp\\dplus\\py_fit_stages.txt", "a") as fp:
        #     print(params, res, file=fp)

        if res < self.best_eval:
            self.best_eval = res
            if self.best_params.size != 0:
                self.best_params = params
        #print("p: {}".format(params))
        #print("cost: {}".format(res))
        return residual

    @property
    def best_eval(self):
        """
        double array. we works with array in order to mimic double* (pointer) behavior

        :return: double value
        """
        return self._best_eval[0]

    @best_eval.setter
    def best_eval(self, best_val):
        self._best_eval[0] = best_val

    @property
    def best_params(self):
        """
        double array. we change the values inside the array and to the array itself because we want to
         mimic pointers behavior

        :return: double array
        """
        return self._best_params

    @best_params.setter
    def best_params(self, new_best_params):
        for i in range(len(new_best_params)):
            self._best_params[i] = new_best_params[i]

    def calc_residual(self, residual, num_residual):
        raise NotImplementedError

class XRayResiduals(Residual):
    def calc_residual(self, residual, num_residual):
        res = np.double(0)
        for i in range(num_residual):
            residual[i] = math.fabs(self.np_y[i] - residual[i])
            res += residual[i] * residual[i]

        res /= 2.0
        return res


class XRayLogResiduals(XRayResiduals):

    def calc_residual(self, residual, num_residual):
        res = np.double(0)
        for i in range(num_residual):
            residual[i] = (math.log10(residual[i]) - math.log10(self.np_y[i]))
            res += residual[i] * residual[i]

        return res


class XRayRatioResiduals(XRayResiduals):

    def calc_residual(self, residual, num_residual):
        # res = np.double(0)
        # for i in range(num_residual):
        #     rat = residual[i] / self.np_y[i]
        #     rat = 1. - rat
        #     residual[i] = rat
        #     res += rat * rat

        rat = 1. - residual / self.np_y
        residual[:] = rat
        res = np.sum(residual * residual, dtype=np.double)
        res /= 2.0
        return res
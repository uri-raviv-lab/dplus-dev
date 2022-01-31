import os
import threading
import time
from dplus.CalculationRunner import EmbeddedLocalRunner
from dplus.PyCeresOptimizer import PyCeresOptimizer
from dplus.CalculationResult import FitResult

class InterfaceFitRunner(object):
    def __init__(self):
        self.status_process = 0

    def fit(self, calc_data):
        raise NotImplementedError()        

    def fit_async(self, calc_data):
        raise NotImplementedError() 

    def get_status(self):
        raise NotImplementedError() 

    def get_result(self):
        raise NotImplementedError() 

class FitRunner():
    def __init__(self):
        self.python_fit = None
        self.calc_runner = EmbeddedLocalRunner()
        self.calc_data = None
        self.thread = None

    def fit(self, calc_data):
        self.calc_data = calc_data
        self.python_fit = PyCeresOptimizer(calc_data, self.calc_runner)
        self.python_fit.solve()
        return self.get_result()

    def get_result(self):
        fit_result = self.python_fit.save_dplus_arrays(self.python_fit.best_results)
        calc_result = FitResult(self.calc_data, fit_result)
        return calc_result

    def inner_fit_async(self):
        self.python_fit = PyCeresOptimizer(self.calc_data, self.calc_runner)
        self.python_fit.solve()

    def fit_async(self, calc_data):
        self.calc_data = calc_data
        self.cur_thread = threading.Thread(target = self.inner_fit_async, args = ())
        self.cur_thread.stop = False
        self.cur_thread.start()

    def get_status(self):
        try:
            if self.cur_thread:
                is_alive = self.cur_thread.is_alive()
                if is_alive:
                    result = {"isRunning": True, "progress": 0.0, "code": -1, "message": ""}
                else:
                    self.cur_thread.join()
                    result = {"isRunning": False, "progress": 100.0, "code": 0, "message": ""}
            else:
                raise
        except Exception as ex:
            return {"error": {"code": 22, "message": str(ex)}}
        return result

    def stop(self):
        # https://stackoverflow.com/a/36499538/10787867
        self.cur_thread.stop = True
        
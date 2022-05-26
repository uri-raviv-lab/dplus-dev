import os
import threading
import time
import ctypes

from dplus.CalculationRunner import EmbeddedLocalRunner
from dplus.PyCeresOptimizer import PyCeresOptimizer, stop_flag
from dplus.CalculationResult import FitResult


class FitRunner():
    def __init__(self):
        self.python_fit = None
        self.calc_runner = EmbeddedLocalRunner()
        self.calc_data = None
        self.cur_thread = None
        self.gof = None

    def fit(self, calc_data):
        stop_flag["stop"] = 0
        self.calc_data = calc_data
        self.python_fit = PyCeresOptimizer(calc_data, self.calc_runner)
        self.python_fit.solve()
        return self.get_result()

    def get_result(self):
        fit_result = self.python_fit.save_dplus_arrays(self.python_fit.best_results)
        calc_result = FitResult(self.calc_data, fit_result)
        return calc_result

    def _inner_fit_async(self):
        self.python_fit = PyCeresOptimizer(self.calc_data, self.calc_runner)
        self.gof = self.python_fit.solve()

    def fit_async(self, calc_data):
        stop_flag["stop"] = 0
        self.calc_data = calc_data
        self.cur_thread = threading.Thread(target = self._inner_fit_async, args = ())
        self.cur_thread.start()

    def get_status(self):
        try:
            if self.cur_thread: 
                is_alive = self.cur_thread.is_alive()
                if is_alive:
                    result = {"isRunning": True, "progress": 0.0, "code": -1, "message": ""}
                else:
                    if stop_flag["stop"] == 1: # TODO if did the TODO below - take it out from the if cur_thread
                        result = {"error": {"code": 22, "message": "job stop run"}}
                    else:
                        result = {"isRunning": False, "progress": 100.0, "code": 0, "message": ""}
                    self.cur_thread.join()
            else:
                result = {"isRunning": False, "progress": 0.0, "code": -1, "message": ""}
        except Exception as ex:
            return {"error": {"code": 22, "message": str(ex)}}
        return result

    def stop(self):
        stop_flag["stop"] = 1
        # self.cur_thread = None
        self.cur_thread.join() # TODO try remove this line and maybe set the cur_thread to None

import os
import threading
import time
import ctypes

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


class thread_with_exception(threading.Thread):
    def __init__(self, target):
        threading.Thread.__init__(self, target=target)
        self.target = target
        self._stopped = False
          
    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    @property
    def is_stopped(self):
        return self._stopped

    def raise_exception(self):
        self._stopped = True
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
              ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')
      

class FitRunner():
    def __init__(self):
        self.python_fit = None
        self.calc_runner = EmbeddedLocalRunner()
        self.calc_data = None
        self.cur_thread = None

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
        self.cur_thread = thread_with_exception(target = self.inner_fit_async)
        self.cur_thread.start()
        # self.cur_thread = threading.Thread(target = self.inner_fit_async, args = ())
        # self.cur_thread.stop_thread = False
        # self.cur_thread.start()

    def get_status(self):
        try:
            if self.cur_thread: 
                is_alive = self.cur_thread.is_alive()
                stopped = self.cur_thread.is_stopped
                if is_alive and not stopped: # and not :
                    result = {"isRunning": True, "progress": 0.0, "code": -1, "message": ""}
                else:
                    if stopped:
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
        # https://stackoverflow.com/a/36499538/10787867
        self.cur_thread.raise_exception()

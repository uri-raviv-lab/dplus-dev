from __future__ import absolute_import

import json
import platform
import signal
import subprocess
import time
import uuid

import codecs
import hashlib
import os
import psutil as psutil
import requests
import shutil
import tempfile
import uuid

from dplus.FileReaders import _handle_infinity_for_json, NumpyHandlingEncoder
from dplus.CalculationResult import GenerateResult, FitResult

from dplus.PyCeresOptimizer import PyCeresOptimizer
from dplus.wrappers import BackendWrapper

# from .Fit import Fitter

def check_capabilities(check_tdr=True):
    wrapper = BackendWrapper()
    return wrapper.check_capabilities(check_tdr)

class JobRunningException(Exception):
    def __init__(self):
        self.value = "there is a job already running in this session"

    def __str__(self):
        return repr(self.value)


_is_windows = platform.system() == 'Windows'


class Runner(object):
    """abstract class that presents the interface for dplus calls."""

    def generate(self, calc_data, save_amp=True):
        '''
        run sync dplus generate.

        :param calc_data: an instance of a CalculationInput class
        :param save_amp: bool , should the system save amp and pdb at the end of calculation
        :rtype: an instance of a GenerateResult class
        '''
        raise NotImplemented

    def generate_async(self, calc_data, save_amp=True):
        '''
        run async dplus generate.

        :param calc_data: an instance of a CalculationInput class
        :param save_amp: bool , should the system save amp and pdb at the end of calculation
        :rtype: an instance of a RunningJob class
        '''
        raise NotImplemented

    def fit(self, calc_data, save_amp=True):
        '''
        run sync dplus fit.

        :param calc_data: an instance of a CalculationInput class
        :param save_amp: bool , should the system save amp and pdb at the end of calculation
        :rtype: an instance of a FitResult class
        '''
        raise NotImplemented

    def fit_async(self, calc_data, save_amp=True):
        '''
        run async dplus fit.

        :param calc_data: an instance of a CalculationInput class
        :param save_amp: bool , should the system save amp and pdb at the end of calculation
        :rtype: an instance of a RunningJob class
        '''
        raise NotImplemented


class RunningJob(object):
    """abstract class for working with dplus jobs."""

    def __init__(self, calculation_type):
        self._calculation_type = calculation_type

    def _start(self, calc_data):
        '''
        :param calc_data:
        :return:
        '''
        raise NotImplemented

    def get_status(self):
        '''
        The function get_status returns a json dictionary reporting the job's current status.

        :rtype: a json dictionary.
        '''
        raise NotImplemented

    def _get_result(self):
        '''
        :return:
        '''
        raise NotImplemented

    def abort(self):
        '''
        ends the current job.
        '''
        raise NotImplemented

    def _get_pdb(self, model_ptr, destination_folder=None):
        '''
        :param model_ptr:
        :return:
        '''
        raise NotImplemented

    def _get_amp(self, model_ptr, destination_folder=None):
        '''
        :param model_ptr:
        :return:
        '''
        raise NotImplemented

    def get_result(self, calc_data):
        """
        The method get_results requires a copy of the CalculationInput used to create the job. /
        Should only be called when job is completed. It is the user's responsibility to verify job completion with get_status before calling.
        :param  calc_data: an instance of a CalculationInput class
        :rtype: an instance of a GenerateResult or FitResult class"""
        if self._calculation_type == "generate":
            return GenerateResult(calc_data, self._get_result(), self)
        else:
            return FitResult(calc_data, self._get_result(), self)


def as_job(job_dict):
    '''
    this is the function used as an object hook for json deserialization of a dictionary to a job object.
    all changes made to __init__ of RunningJob need to be reflected here.
    '''
    job = LocalRunner.RunningJob(job_dict["session_directory"], job_dict["pid"])
    return job


class LocalRunner(Runner):
    """The LocalRunner is intended for users who have the D+ executable files installed on their system. It takes two optional initialization arguments:

        * `exe_directory` is the folder location of the D+ executables. By default, its value is None- on Windows, \
        this will lead to the python interface searching the registry for an installed D+ on its own, \
        but on linux the executable directory *must* be specified.
        * `session_directory` is the folder where the arguments for the calculation are stored, as well as the output results,\
        amplitude files, and pdb files, from the c++ executable. By default, its value is None, and an automatically generated temporary folder will be used."""

    class RunningJob(RunningJob):
        """implementation of RunningJob for localRunner class"""

        def __init__(self, session_directory, pid=-1, calculation_type="generate"):
            super().__init__(calculation_type)
            self.session_directory = os.path.abspath(session_directory)
            self.pid = pid

        def _check_running(self):
            try:
                if self.pid != -1:
                    process = psutil.Process(self.pid)
                    if process.status() == psutil.STATUS_ZOMBIE:
                        process.wait()
                    else:
                        raise JobRunningException
            except psutil.NoSuchProcess:
                pass

        def _clean_folder(self):
            filename = os.path.join(self.session_directory, "data.json")
            if os.path.exists(filename):
                os.remove(filename)

        def _start(self, calc_data):
            # 1: check that no job is already running in this session folder
            self._check_running()
            # clean out the session folder
            self._clean_folder()

            # save the calc_data as args
            filename = os.path.join(self.session_directory, "args.json")

            with open(filename, 'w') as outfile:
                json.dump(_handle_infinity_for_json(calc_data.args), outfile, cls=NumpyHandlingEncoder)

            # save an initial job status file:
            filename = os.path.join(self.session_directory, "job.json")
            jobstat = {"isRunning": True, "progress": 0.0, "code": 0}
            with open(filename, 'w') as outfile:
                json.dump(jobstat, outfile, cls=NumpyHandlingEncoder)

        def get_status(self):
            '''
               The function get_status returns a json dictionary reporting the job's current status.

               :rtype: a json dictionary.
            '''
            filename = os.path.join(self.session_directory, "job.json")
            for x in range(4):  # try 3 times
                try:
                    with codecs.open(filename, 'r', encoding='utf8') as f:
                        result = f.read()
                        assert result
                        break
                except (AssertionError, FileNotFoundError, BlockingIOError) as e:
                    if x == 4:
                        return {"error": {"code": 22, "message": "failed to read job status"}}
                    time.sleep(0.1)
            try:
                result = json.loads(result)
                keys = ["isRunning", "progress", "code", "message"]
                if all(k in result for k in keys):
                    return result
                try:
                    # the dict of job status isn't full, so we are checking if the job is running.
                    # if so - just return result that everything is ok
                    # else - if there is no process, there program stop run for some reason- raise an error
                    # if access denied - should wait
                    if self.pid != -1:
                        process = psutil.Process(self.pid)
                        result = {"isRunning": True, "progress": 0.0, "code": -1, "message": ""}
                        return result
                except psutil.NoSuchProcess:
                    result = {"isRunning": False, "progress": 0.0, "code": 3 , "message": "job stop run with an error"}
                    return result
                except psutil.AccessDenied:
                    result = {"isRunning": True, "progress": 0.0, "code": -1, "message": ""}
                    return result
            except json.JSONDecodeError:
                return {"error": {"code": "22", "message": "json error in job status"}}

        def _get_file_status(self):
            filename = os.path.join(self.session_directory, "notrunning.txt")
            with open(filename, 'r') as f:
                status = f.readline()
            return status == "True"

        def _get_result(self):
            filename = os.path.join(self.session_directory, "data.json")
            with codecs.open(filename, 'r', encoding='utf8') as f:
                result = json.load(f)
            if type(result) is dict:
                if 'error' in result.keys():
                    error_message_dict = result['error']
                    raise Exception(error_message_dict['message'], error_message_dict['code'])
            return result

        def abort(self):
            '''
             The function abort ends the current localRunner job.
             '''
            try:
                os.kill(self.pid, signal.SIGTERM)
            finally:
                filename = os.path.join(self.session_directory, "notrunning.txt")
                os.remove(filename)

        def _get_pdb(self, model_ptr, destination_folder=None):
            '''
            :param model_ptr:
            :return:
            '''

            ptr_string = '%08d.pdb' % (int(model_ptr))
            file_location = os.path.join(self.session_directory, 'pdb', ptr_string)
            if os.path.isfile(file_location):
                if destination_folder:
                    shutil.copy2(file_location, destination_folder)
                    return os.path.join(destination_folder, ptr_string)
                return file_location
            raise FileNotFoundError

        def _get_amp(self, model_ptr, model_name, destination_folder=None):
            '''
            :param model_ptr:
            :return:
            '''

            ptr_string_amp = '%08d.amp' % (int(model_ptr))
            file_location_amp = os.path.join(self.session_directory, 'cache', ptr_string_amp)

            ptr_string_ampj = '%08d.ampj' % (int(model_ptr))
            file_location_ampj = os.path.join(self.session_directory, 'cache', ptr_string_ampj)

            if os.path.isfile(file_location_amp):
                if destination_folder:
                    shutil.copy2(file_location_amp, destination_folder)
                    if model_name == '':
                        new_name = '%s.amp' %(model_name)
                        os.replace(os.path.join(destination_folder, ptr_string_amp), os.path.join(destination_folder, new_name))
                        return os.path.join(destination_folder, new_name)
                    return os.path.join(destination_folder, ptr_string_amp)
                return file_location_amp
            elif os.path.isfile(file_location_ampj):
                if destination_folder:
                    shutil.copy2(file_location_ampj, destination_folder)
                    if model_name != '':
                        new_name = '%s.ampj' %(model_name)
                        os.replace(os.path.join(destination_folder, ptr_string_ampj), os.path.join(destination_folder, new_name))
                        return os.path.join(destination_folder, new_name)
                    return os.path.join(destination_folder, ptr_string_ampj)
                return file_location_ampj
            raise FileNotFoundError

    def __init__(self, exe_directory=None, session_directory=None):
        """initialization function,  """
        if not exe_directory:
            try:
                exe_directory = self._find_dplus()
            except Exception as e:
                raise ValueError("Can't locate D+, please specify the D+ backend directory manually", e)
        if not session_directory:
            tmp_python_dir = os.path.join(tempfile.gettempdir(),"dplus_python")
            if not os.path.exists(tmp_python_dir):
                os.mkdir(tmp_python_dir)
            session_name = str(uuid.uuid4())
            session_directory = os.path.join(tmp_python_dir,session_name)
            os.mkdir(session_directory)

        self._exe_directory = os.path.abspath(exe_directory)
        self._check_exe_directory()

        self._session_directory = os.path.abspath(session_directory)
        self._make_file_directories(session_directory)

    @property
    def session_directory(self):
        return self._session_directory

    @session_directory.setter
    def session_directory(self, session_dir):
        self._session_directory = os.path.abspath(session_dir)
        self._make_file_directories(session_dir)

    def _find_dplus(self):
        if not _is_windows:
            raise ValueError("Can't find D+ on non Windows machine")

        other_view_flag = None
        import winreg
        try:
            RawKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\HUJI')
        except FileNotFoundError:

            import platform
            # was added to support cases when python architecture different than computer architecture (32bit/64bit)
            bitness = platform.architecture()[0]
            if bitness == '32bit':
                other_view_flag = winreg.KEY_WOW64_64KEY
            elif bitness == '64bit':
                other_view_flag = winreg.KEY_WOW64_32KEY

            RawKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\HUJI',
                                    access=winreg.KEY_READ | other_view_flag)

        d_plus = ""
        try:
            i = 0
            while 1:  # will fail when there are no more keys in  "SOFTWARE\HUJI"
                tmp_name = winreg.EnumKey(RawKey, i)
                if "D+" in tmp_name:
                    d_plus = tmp_name
                i += 1
        except WindowsError:
            pass
        if d_plus == "":
            raise Exception("D+ does not exists")

        dplus_key = os.path.join("SOFTWARE\HUJI", d_plus)
        if other_view_flag:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, dplus_key, access=winreg.KEY_READ | other_view_flag) as key:
                exe_dir = winreg.QueryValueEx(key, "binDir")[0]
                return exe_dir
        else:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, dplus_key) as key:
                exe_dir = winreg.QueryValueEx(key, "binDir")[0]
                return exe_dir

    def _get_program_path(self, calculation_type):
        program_path = os.path.join(self._exe_directory, calculation_type)
        if _is_windows:
            program_path += ".exe"
        return program_path

    def _check_exe_directory(self):
        if not os.path.isdir(self._exe_directory):
            raise NotADirectoryError("%s is not a directory" % self._exe_directory)

        programs = ['generate']#, 'getallmetadata', 'checkCapabilities'] #neither of those are needed...
        paths = [self._get_program_path(program) for program in programs]
        valid = [os.path.isfile(path) for path in paths]

        if not all(valid):
            raise ValueError(
                "The directory %s does not seem to contain the D+ backend executables" % self._exe_directory)

    def _make_file_directories(self, directory):
        os.makedirs(os.path.join(directory, 'pdb'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'amp'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'cache'), exist_ok=True)

    def generate(self, calc_data, save_amp=True):
        """
        The method generate waits until dplus has returned a result.

        :param  calc_data: an instance of a CalculationInput class
        :rtype: an instance of a CalculationResult class"""
        job = self._run(calc_data, save_amp=save_amp)
        calc_result = GenerateResult(calc_data, job._get_result(), job)
        return calc_result

    def generate_async(self, calc_data, save_amp=True):
        """
        The method generate_async allows dplus calculations to be run in the background.

        :param  calc_data: an instance of a CalculationInput class
        :rtype: an instance of a RunningJob class"""
        job = self._run(calc_data, is_async=True, save_amp=save_amp)
        return job

    def fit(self, calc_data, save_amp=True):
        """
        The method fit waits until dplus has returned a result.

        :param  calc_data: an instance of a CalculationInput class
        :rtype: an instance of a CalculationResult class"""
        job = self._run(calc_data, calculation_type="fit", save_amp=save_amp)
        python_fit = PyCeresOptimizer(calc_data, self)
        python_fit.solve()
        python_fit.save_dplus_arrays(python_fit.best_results, os.path.join(self.session_directory, "data.json"))
        calc_result = FitResult(calc_data, job._get_result(), job)
        return calc_result

    def fit_async(self, calc_data, save_amp=True):
        """
        The method fit_async allows dplus calculations to be run in the background.

        :param  calc_data: an instance of a CalculationInput class
        :rtype: an instance of a RunningJob class"""
        raise NotImplementedError("async for fit has not yet been implemented")

    def check_capabilities(self, check_tdr=True):
        """
        The method check_capabilities runs the exe CheckCapabilities
        raises an error if 'check_capabilities.json' has error with code different than 0

         :param  check_tdr: boolean value, determine if the system should check tdr or not)
         """
        err_file = open(os.path.join(self._session_directory, "error.txt"), 'w')
        out_file = open(os.path.join(self._session_directory, "output.txt"), 'w')
        program_path = self._get_program_path("CheckCapabilities")
        if not os.path.isfile(program_path):
            raise FileNotFoundError
        process = subprocess.Popen([program_path, self._session_directory], cwd=self._exe_directory, stdout=out_file,
                                   stderr=err_file)
        process.communicate()
        filename = os.path.join(self.session_directory, "check_capabilities.json")
        with codecs.open(filename, 'r', encoding='utf8') as f:
            result = json.load(f)
        if type(result) is dict:
            if 'error' in result.keys():
                error_message_dict = result['error']
                if error_message_dict['code'] != 0:
                    raise Exception(error_message_dict['message'], error_message_dict['code'])
        else:
            raise Exception("problem in file check_capabilities")

    def _run(self, calc_data, calculation_type="generate", is_async=False, save_amp=True):
        # 1 start the job
        job = LocalRunner.RunningJob(self._session_directory, calculation_type=calculation_type)
        job._start(calc_data)
        # 2 call the executable process
        process = self._call_exe(calculation_type, save_amp=save_amp)
        job.pid = process.pid
        # 3 return, based on async
        if is_async:
            return job

        process.communicate()
        return job

    def _call_exe(self, calculation_type, save_amp=True):
        err_file = open(os.path.join(self._session_directory, "error.txt"), 'w')
        out_file = open(os.path.join(self._session_directory, "output.txt"), 'w')
        program_path = self._get_program_path(calculation_type)
        if not os.path.isfile(program_path):
            raise FileNotFoundError
        if save_amp:
            process = subprocess.Popen([program_path, self._session_directory], cwd=self._exe_directory, stdout=out_file,
                                       stderr=err_file)
        else:
            cache_dir = os.path.join(self._session_directory, "cache")
            process = subprocess.Popen([program_path, self._session_directory, cache_dir, "False"], cwd=self._exe_directory,
                                       stdout=out_file,
                                       stderr=err_file)
        return process

    @staticmethod
    def get_running_job(session):
        return LocalRunner.RunningJob(session)


class WebRunner(Runner):
    """The WebRunner is intended for users accessing the D+ server. It takes two required initialization arguments, with no default values:

        * `url` is the address of the server.
        * `token` is the authentication token granting access to the server.
        """

    class RunningJob(RunningJob):
        """implementation of RunningJob for WebRunner class"""

        def __init__(self, base_url, token, session, calculation_type="generate"):
            super().__init__(calculation_type)
            self._url = base_url
            self._header = {'Authorization': "Token " + str(token)}
            self._session_string = "?ID=" + session

        def _start(self, calc_data):
            # check if files
            self._check_files(calc_data)

        def get_status(self):
            '''
               The function get_status returns a json dictionary reporting the job's current status.

               :rtype: a json dictionary.
               '''
            jobstatus = requests.get(url=self._url + 'job' + self._session_string, headers=self._header)
            if jobstatus.status_code == 200:
                result = json.loads(jobstatus.text)
                return result
            raise Exception("Error getting job status:" + jobstatus.reason)

        def _get_result(self, calculation_type="generate"):
            response = requests.get(url=self._url + calculation_type + self._session_string, headers=self._header)
            result = json.loads(response.text)
            if type(result) is dict:
                # web always returns an error field, so need to check if that field is OK or an error
                if result["error"]["code"] != 0:
                    error_message_dict = result['error']
                    raise Exception(error_message_dict['message'])
            return result

        def _check_files(self, calc_data):
            def calc_file_hash(filepath):  # taken from stackexchange 1869885/calculating-sha1-of-a-file
                sha = hashlib.sha1()
                with open(filepath, 'rb') as f:
                    while True:
                        block = f.read(2 ** 10)  # Magic number: one-megabyte blocks.
                        if not block: break
                        sha.update(block)
                    return sha.hexdigest().upper()

            filedict = {'files': []}
            filenames = calc_data._filenames
            for filename in filenames:
                dict = {}
                dict['filename'] = filename
                dict['size'] = os.stat(filename).st_size
                dict['hash'] = calc_file_hash(filename)
                filedict['files'].append(dict)
            data = json.dumps(filedict)
            response = requests.post(url=self._url + 'files', data=data, headers=self._header)
            if response.status_code != 200:
                raise Exception("Error checking file status" + response.reason)
            statuses = json.loads(response.text)

            files_missing = []
            for filename in filenames:
                if statuses[filename]['status'] == 'MISSING':
                    files_missing.append((filename, statuses[filename]['id']))

            self._upload_files(files_missing)

        def _upload_files(self, filenames):
            for filename, id in filenames:
                url = self._url + 'files/' + str(id)
                files = {'file': open(filename, 'rb')}
                response = requests.post(url=url, headers=self._header, files=files)
                if response.status_code not in [200, 204]:
                    raise Exception("Error uploading files" + response.reason)

        def _get_pdb(self, model_ptr, destination_folder=None):
            '''
            :param model_ptr:
            :return: file address
            '''
            if not destination_folder:
                destination_folder = tempfile.mkdtemp()
            ptr_string = '%08d.pdb' % (int(model_ptr))
            destination_file = os.path.join(destination_folder, ptr_string)
            # code used from: https://stackoverflow.com/a/16696317/5961793
            response = requests.get(url=self._url + "pdb/" + str(model_ptr) + self._session_string,
                                    headers=self._header, stream=True)
            if response.status_code != 200:
                raise Exception("Error downloading pdb" + response.reason)
            with open(destination_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            return destination_file

        def _get_amp(self, model_ptr, destination_folder=None):
            '''
            :param model_ptr:
            :return: file address
            '''
            if not destination_folder:
                destination_folder = tempfile.mkdtemp()
            # ptr_string_amp = '%08d.amp' % (int(model_ptr))
            ptr_string_ampj = '%08d.ampj' % (int(model_ptr))
            destination_file = os.path.join(destination_folder, ptr_string_ampj)
            # code used from: https://stackoverflow.com/questions/13137817/how-to-download-image-using-requests
            response = requests.get(url=self._url + "amplitude/" + str(model_ptr) + self._session_string,
                                    headers=self._header, stream=True)
            if response.status_code != 200:
                raise Exception("Error downloading amp" + response.reason)
            with open(destination_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            return destination_file

    def __init__(self, base_url, token):
        self._url = base_url + r'api/v1/'
        self._header = {'Authorization': "Token " + str(token)}
        self._session = str(uuid.uuid4())
        self._session_string = "?ID=" + self._session
        self._token = token

    def generate(self, calc_data):
        """
         The method generate waits until dplus has returned a result.

         :param  calc_data: an instance of a CalculationInput class
         :rtype: an instance of a CalculationResult class"""
        job = self._run(calc_data)
        result = job._get_result(calculation_type="generate")
        calc_result = GenerateResult(calc_data, result['result'], job)
        return calc_result

    def generate_async(self, calc_data):
        """
        The method generate_async allows dplus calculations to be run in the background.

        :param  calc_data: an instance of a CalculationInput class
        :rtype: an instance of a RunningJob class"""
        return self._run(calc_data, is_async=True)

    def fit(self, calc_data):
        """
                The method fit waits until dplus has returned a result.

                :param  calc_data: an instance of a CalculationInput class
                :rtype: an instance of a CalculationResult class"""
        job = self._run(calc_data, calculation_type="fit")
        result = job._get_result(calculation_type="fit")
        calc_result = FitResult(calc_data, result['result'], job)
        return calc_result

    def fit_async(self, calc_data):
        """
               The method fit_async allows dplus calculations to be run in the background.

               :param  calc_data: an instance of a CalculationInput class
               :rtype: an instance of a RunningJob class"""
        return self._run(calc_data, calculation_type="fit", is_async=True)

    def _run(self, calc_data, calculation_type="generate", is_async=False):
        # 1: start the job (checks files)
        job = WebRunner.RunningJob(self._url, self._token, self._session, calculation_type=calculation_type)
        job._start(calc_data)
        # 2: make the necessary call
        data = json.dumps(calc_data.args['args'])
        test = requests.put(url=self._url + calculation_type + self._session_string, data=data, headers=self._header)
        if test.status_code in [400, 500]:
            result = json.loads(test.text)
            raise Exception(result["error"]["message"])
        # 3 return depending on sync or async
        if is_async:
            return job

        while True:
            res = job.get_status()
            if res['result']['isRunning'] == False:
                break
            time.sleep(1)

        return job

    @staticmethod
    def get_running_job(url, token, session):
        return WebRunner.RunningJob(url, token, session)

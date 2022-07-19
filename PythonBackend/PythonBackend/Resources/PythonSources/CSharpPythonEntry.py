import json
import subprocess
from dplus.metadata import program_metadata
import signal
cur_meta = json.dumps(program_metadata)
from dplus.CalculationRunner import LocalRunner, EmbeddedLocalRunner
from dplus.CalculationInput import CalculationInput
from dplus.FitRunner import FitRunner
from dplus.FileReaders import _handle_infinity_for_json, NumpyHandlingEncoder
import os
import time
from dplus.Backend import Backend, BackendError

class LocalCSharpPython:
    def __init__(self, exe_dir, session_dir):
        self.exe_dir = exe_dir
        self.session_dir = session_dir
        self.cur_job = None
        self.calc_runner = LocalRunner(self.exe_dir, self.session_dir)
        # create all outputs file
        self.all_outs_filename = os.path.join(self.session_dir, "all_calculations_outputs.txt")
        f = open(self.all_outs_filename, 'w')
        f.close()
        
        
        self.run_fit = False
        self.cur_calc_input = None
        self.cur_results = None

    def perform_call(self, call):
        try:
            json2run = json.loads(call)
            result = ""
            if "GetAllModelMetadata" in json2run["function"]:
                metadeta = {"result": json.loads(cur_meta)}
                result = self.process_result(metadeta)

            elif "GetJobStatus" in json2run["function"]:
                if False:
                    if self.cur_job is not None:
                        status = self.cur_job.get_status()
                else:
                    if self.run_fit:
                        status = self.fit_runner.get_status()
                    else:
                        status = self.calc_runner.get_job_status()
                result = self.process_result({"result": status})

            elif "StartGenerate" in json2run["function"]:
                self.run_fit = False
                self.cur_job = None
                self.cur_calc_input = None
                self.cur_results = None
                calc_input = CalculationInput()
                calc_input.load_from_dictionary(json2run["args"]["state"])
                calc_input.use_gpu = bool(json2run["options"]["useGPU"])
                self.cur_calc_input = calc_input
                self.cur_job = self.calc_runner.generate_async(calc_input, save_amp=True)
                result = self.process_result({"result": self.cur_job.get_status()})


            elif "GetGenerateResults" in json2run["function"]:
                self.cur_results = self.cur_job.get_result(self.cur_calc_input)
                state = self.cur_results.processed_result
                result = self.process_result({"result": state})
    
                # self.cur_job.abort() # must abort the job otherwise- continue running (even if generate was finished)
                self.add_output()

            elif "StartFit" in json2run["function"]:
                print("startFit")
                self.run_fit = True
                self.cur_calc_input = None
                self.cur_job = None
                self.cur_results = None
                calc_input = CalculationInput._web_load(json2run["args"])
                calc_input.use_gpu = bool(json2run["options"]["useGPU"])

                self.cur_calc_input = calc_input
                self.cur_job = self.calc_runner.fit_async(calc_input, save_amp=True)
                status = self.cur_job.get_status()
                result = self.process_result({"result": status})

            elif "GetFitResults" in json2run["function"]:
                print("GetFitResults")
                self.cur_results = self.cur_job.get_result(self.cur_calc_input)
                result = self.process_result({"result": self.cur_results._raw_result})
                self.add_output()
                # result = self.process_result({"result": self.async_fit.get_results()})
                # self.add_python_fit_output()

            elif "Stop" in json2run["function"]:
                if not self.async_fit:
                    self.cur_job.abort()
                    result = self.process_result()
                    self.cur_calc_input = None
                    self.cur_job = None
                    self.cur_results = None
                else:
                    self.async_fit.abort()
                    result = self.async_fit.get_status()
                    self.async_fit = None
                self.add_output()

            elif "GetAmplitude" in json2run["function"]:
                args = json2run["args"]
                if self.cur_results is not None:
                    try:
                        pre_process_results = self.cur_results.get_amp(args["model"], args["filepath"])
                        result = self.process_result({"result": pre_process_results})
                    except FileNotFoundError:
                        raise Exception("The model was not found within the container or job", 14)
                else:
                    raise Exception("The model was not found within the container or job", 8)

            elif "GetPDB" in json2run["function"]:
                args = json2run["args"]
                if self.cur_results is not None:
                    try:
                            pre_process_results = self.cur_results.get_pdb(args["model"], args["filepath"])
                            result = self.process_result({"result": pre_process_results})
                    except FileNotFoundError:
                        raise Exception("The model was not found within the container or job", 14)
                    except Exception as ex:
                        raise Exception("Error GetPDB failed " + str(ex))
                else:
                    raise Exception("The model was not found within the container or job", 8)

            elif "CheckCapabilities" in json2run["function"]:
                self.cur_calc_input = None
                self.cur_job = None
                self.cur_results = None
                use_gpu = bool(json2run["options"]["useGPU"])
                backend = Backend()
                backend.check_capabilities(use_gpu)
                # self.calc_runner.check_capabilities(use_gpu)
                result = self.process_result()
        except BackendError as be:
            response_json = {"error": {"code": be.error_code, "message": str(be)}}
            result = self.process_result(response_json)
        except Exception as e:
            if len(e.args) > 1:
                if isinstance(e.args[0], int):  # python exceptions
                    response_json = {"error": {"code": 24, "message": e.args[1]}}
                else:  # dplus exceptions
                    response_json = {"error": {"code": e.args[1], "message": e.args[0]}}
            else:
                response_json = {"error": {"code": 24, "message": str(e)}}
            result = self.process_result(response_json)
        return result

    def process_result(self, response_json={'result': ''}):
        # print("process_result")
        return_json = {
            "error": {
                "code": 0,
                "message": "OK"
            },
            "client-data": "",
        }
        try:
            return_json["error"] = response_json["error"]
        except KeyError:
            try:
                return_json["result"] = response_json["result"]
                if "message" in return_json["result"]:
                    return_json["result"]["message"] = return_json["result"]["message"].replace("'", "")
            except KeyError:
                return_json["result"] = response_json
            for item in return_json["result"]:
                if 'Headers' in item:
                    for idx in range(len(return_json["result"][item])):
                        header_str = return_json["result"][item][idx]["Header"]
                        header_str = header_str.replace('"', '&"')
                        header_str = header_str.replace('= ', '=')
                        header_str = header_str.replace("'", "")
                        return_json["result"][item][idx]["Header"] = header_str
        return_json["error"]["message"] = return_json["error"]["message"].replace("'", "")
        # replace ' with ", False with false and True with true
        str_json = str(return_json)
        str_json = str_json.replace("'", "\"")
        str_json = str_json.replace('&"', '/"')
        str_json = str_json.replace("True", "true")
        str_json = str_json.replace("False", "false")
        return str_json

    def add_output(self):
        current_job_output_name = os.path.join(self.session_dir, "output.txt")
        with open(current_job_output_name, "r") as f:
            current_job_output_str = f.read() + "\n"
            with open(self.all_outs_filename, "a") as all_outputs:
                all_outputs.write(current_job_output_str)

    def add_python_fit_output(self):
        fit_output = self.async_fit.get_output()
        with open(self.all_outs_filename, "a") as all_outputs:
            all_outputs.write(fit_output)


class EmbeddedCSharpPython:
    def __init__(self):
        self.calc_runner = EmbeddedLocalRunner()
        self.fit_runner = FitRunner()
        self.run_fit = False
        self.cur_calc_input = None
        self.cur_results = None

    def start_generate(self, json2run):
        self.run_fit = False
        self.cur_calc_input = None
        self.cur_results = None
        calc_input = CalculationInput()
        calc_input.load_from_dictionary(json2run["args"]["state"])
        calc_input.use_gpu = bool(json2run["options"]["useGPU"])
        self.cur_calc_input = calc_input

        self.calc_runner.generate_async(calc_input)
        result = self.process_result({"result": self.calc_runner.get_job_status()})
        return result

    def start_fit(self, json2run):
        print("startFit")
        self.run_fit = True
        self.cur_calc_input = None
        self.cur_job = None
        self.cur_results = None
        calc_input = CalculationInput._web_load(json2run["args"])
        calc_input.use_gpu = bool(json2run["options"]["useGPU"])

        self.cur_calc_input = calc_input
        self.fit_runner.fit_async(calc_input)
        status = self.fit_runner.get_status()
        result = self.process_result({"result": status})
        return result

    def get_pdb(self, json2run):
        args = json2run["args"]
        if self.cur_results is not None:
            try:
                pdb_str = self.calc_runner.get_pdb(args["model"])
                with open(args["filepath"], 'w', encoding='utf8') as file_pdb_out:
                    file_pdb_out.write(pdb_str)
                result = self.process_result()  # {"result": pre_process_results}
                return result
            except FileNotFoundError:
                raise Exception("The model was not found within the container or job", 14)
            except Exception as ex:
                raise Exception("Error GetPDB failed " + str(ex))
        else:
            raise Exception("The model was not found within the container or job", 8)

    def perform_call(self, call):
        try:
            json2run = json.loads(call)
            result = ""
            if "GetAllModelMetadata" in json2run["function"]:
                metadeta = {"result": json.loads(cur_meta)}
                result = self.process_result(metadeta)

            elif "GetJobStatus" in json2run["function"]:
                if self.run_fit:
                    status = self.fit_runner.get_status()
                else:
                    status = self.calc_runner.get_job_status()
                result = self.process_result({"result": status})
                
            elif "StartGenerate" in json2run["function"]:
                result=self.start_generate(json2run)

               
            elif "GetGenerateResults" in json2run["function"]:
                self.cur_results = self.calc_runner.get_generate_results(self.cur_calc_input)
                state = self.cur_results.processed_result
                result = self.process_result({"result": state})


            elif "StartFit" in json2run["function"]:
                result = self.start_fit(json2run)


            elif "GetFitResults" in json2run["function"]:
                print("GetFitResults")
                self.cur_results = self.fit_runner.get_result()
                result = self.process_result({"result": self.cur_results._raw_result})
                state = self.cur_results.processed_result
                # result = self.process_result({"result": self.async_fit.get_results()})
                # self.add_python_fit_output()

            elif "Stop" in json2run["function"]:
                    self.cur_results = None
                    if not self.run_fit:
                        self.calc_runner.stop_generate()
                        result = self.process_result()
                    else:
                        self.fit_runner.stop()
                        result = self.process_result() # result=self.process_result(self.fit_runner.get_status())
                        self.run_fit=False

            elif "GetAmplitude" in json2run["function"]:
                args = json2run["args"]
                if self.cur_results is not None:
                    try:
                            self.calc_runner.save_amp(args["model"], args["filepath"])
                            result = self.process_result()
                    except FileNotFoundError:
                        raise Exception("The model was not found within the container or job", 14)
                else:
                    raise Exception("The model was not found within the container or job", 8)

            elif "GetPDB" in json2run["function"]:
                result=self.get_pdb(json2run)

            elif "CheckCapabilities" in json2run["function"]:
                self.cur_calc_input = None
                self.cur_job = None
                self.cur_results = None
                use_gpu = bool(json2run["options"]["useGPU"])
                backend = Backend()
                backend.check_capabilities(use_gpu)
                # self.calc_runner.check_capabilities(use_gpu)
                result = self.process_result()
        except BackendError as be:
            response_json = {"error": {"code": be.error_code, "message": str(be) }}
            result = self.process_result(response_json)
        except Exception as e:
            if len(e.args) > 1:
                if isinstance(e.args[0], int): # python exceptions
                    response_json = {"error": {"code": 24, "message": e.args[1]}}
                else: # dplus exceptions
                    response_json = {"error": {"code": e.args[1], "message": e.args[0]}}
            else:
                response_json = {"error": {"code": 24, "message": str(e)}}
            result = self.process_result(response_json)
        return result

    def process_result(self, response_json={'result': ''}):
        # print("process_result")
        return_json = {
            "error": {
                "code": 0,
                "message": "OK"
            },
            "client-data": "",
        }
        try:
            return_json["error"] = response_json["error"]
        except KeyError:
            try:
                return_json["result"] = response_json["result"]
                if "message" in return_json["result"]:
                    return_json["result"]["message"] = return_json["result"]["message"].replace("'", "")
            except KeyError:
                return_json["result"] = response_json
            for item in return_json["result"]:
                if 'Headers' in item:
                    for idx in range(len(return_json["result"][item])):
                        header_str = return_json["result"][item][idx]["Header"]
                        header_str = header_str.replace('"', '&"')
                        header_str = header_str.replace('= ', '=')
                        header_str = header_str.replace("'", "")
                        return_json["result"][item][idx]["Header"] = header_str
        return_json["error"]["message"] = return_json["error"]["message"].replace("'", "")
        # replace ' with ", False with false and True with true
        str_json = str(return_json)
        str_json = str_json.replace("'", "\"")
        str_json = str_json.replace('&"', '/"')
        str_json = str_json.replace("True", "true")
        str_json = str_json.replace("False", "false")
        return str_json

    def add_output(self):
        current_job_output_name = os.path.join(self.session_dir,"output.txt")
        with open(current_job_output_name, "r") as f:
            current_job_output_str = f.read() + "\n"
            with open(self.all_outs_filename, "a") as all_outputs:
                all_outputs.write(current_job_output_str)

    def add_python_fit_output(self):
        fit_output = self.async_fit.get_output()
        with open(self.all_outs_filename, "a") as all_outputs:
            all_outputs.write(fit_output)


def get_csharp_python_entry(exe_dir="", session_dir=""):
    embedded=True
    if embedded:
        return EmbeddedCSharpPython()
    else:
        return LocalCSharpPython(exe_dir, session_dir)


if __name__=="__main__":
    example_call_strings={
        "getJobStatus":'{ "client-id": "", "client-data": {}, "function": "GetJobStatus", "args": {}, "options": {} }',
        "stop":'{ "client-id": "", "client-data": {}, "function": "Stop", "args": {}, "options": {} }',
        "getFitResults": '{ "client-id": "", "client-data": {}, "function": "GetFitResults", "args": {}, "options": {} }'

    }
    t = get_csharp_python_entry()
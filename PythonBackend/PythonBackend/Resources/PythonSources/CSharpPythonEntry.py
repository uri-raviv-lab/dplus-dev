import json
import subprocess
from dplus.metadata import program_metadata
import signal
cur_meta = json.dumps(program_metadata)
from dplus.CalculationRunner import LocalRunner, EmbeddedLocalRunner
from dplus.CalculationInput import CalculationInput
from dplus.FileReaders import _handle_infinity_for_json, NumpyHandlingEncoder
import os
import time
from dplus.Backend import Backend, BackendError

class AsyncFit:
    def __init__(self, exe_dir, session_dir, python_dir,calc_input):
        # new fit
        self.session_dir = os.path.join(session_dir, "curve_fit")
        os.makedirs(self.session_dir, exist_ok=True)
        self.exe_dir = exe_dir
        self.python_dir = python_dir
        # self.program_path = os.path.join(os.getcwd(), "async_fit.py")
        self.program_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "async_fit.py")
        self.args_filename = os.path.join(self.session_dir, "args.json")
        with open(self.args_filename, 'w') as outfile:
            json.dump(_handle_infinity_for_json(calc_input.args), outfile, cls=NumpyHandlingEncoder)

        self.process = self.run()

    def run(self):

        try:
            self.save_status(error=False, code=0, is_running=True)

            python_path = os.path.join(self.python_dir, "python")
            print("paths in the running python:")
            print([python_path, self.program_path, self.exe_dir, self.session_dir])

            self.err_filename = os.path.join(self.session_dir, "fit_error.txt")
            err_file = open(self.err_filename, 'w')
            out_file = open(os.path.join(self.session_dir, "fit_output.txt"), 'w')

            process = subprocess.Popen([python_path,
                                        self.program_path, self.exe_dir, self.session_dir],stdout=out_file,
                                       stderr=err_file)
        except Exception as e:
            raise Exception(str(e))

        return process

    def check_process_error(self):
        if os.stat(self.err_filename).st_size > 0:
            with open(self.err_filename, 'r') as f:
                message = f.read()
                self.save_status(error=False, code=24, message=str(message), is_running=False, progress=0)
                self.save_status(error=True,code=24, message=str(message), filename="data.json")

    def get_status(self):

        self.check_process_error()
        filename = os.path.join(self.session_dir, "fit_job.json")
        final_results = {}
        for x in range(4):  # try 3 times
            try:
                with open(filename, 'r', encoding='utf8') as f:
                    result = f.read()
                    assert result
                    break
            except (AssertionError, FileNotFoundError, BlockingIOError) as e:
                if x == 4:
                    return {"error": {"code": 22, "message": "failed to read job status"}}
                time.sleep(0.1)
        with open(filename, 'r', encoding='utf8') as f:
            result = f.read()
        try:
            result = json.loads(result)
            keys = ["isRunning", "progress", "code", "message"]
            if all(k in result for k in keys):
                final_results = result
            else:
                final_results = result


        except json.JSONDecodeError:
            final_results = {"error": {"code": 22, "message": "json error in job status"}}
        except Exception as e:
            final_results = {"error": {"code": 24, "message": str(e)}}

        if final_results["code"] != 0:
            mess = final_results["message"] if "message" in final_results else ""
            self.save_status(error=True, code=final_results["code"], message=mess, filename="data.json")

        return final_results

    def abort(self):
        try:
            os.kill(self.process.pid, signal.SIGTERM)
        finally:
            filename = os.path.join(self.session_directory, "notrunning.txt")
            os.remove(filename)
        self.save_status(error=True, code= 1, message=str("The job was manually stopped"))

    def save_status(self, error,is_running=False, progress=0.0, code=0, message="OK", filename="fit_job.json"):
        if not error:
            status_dict = {"isRunning": is_running, "progress": progress, "code": code,
                           "message": str(message)}
        else:
            status_dict = {"error": {"code": code, "message": str(message)}}
        with open(os.path.join(self.session_dir, filename), 'w') as file:
            json.dump(status_dict, file)

    def get_results(self):
        filename = os.path.join(self.session_dir, "data.json")
        with open(filename, 'r', encoding='utf8') as f:
            result = json.load(f)
        if type(result) is dict:

            if 'error' in result.keys():
                error_message_dict = result['error']
                raise Exception(error_message_dict['message'], error_message_dict['code'])
        return result

    def get_output(self):
        fit_output = os.path.join(self.session_dir,"fit_output.txt")
        current_job_output_str = ""
        with open(fit_output, "r") as f:
            current_job_output_str += f.read() + "\n"

        last_generate_output = os.path.join(self.session_dir, "output.txt")
        with open(last_generate_output, "r") as f:
            current_job_output_str += f.read() + "\n"

        return current_job_output_str


class CSharpPython:
    def __init__(self, exe_dir, session_dir, python_dir):
        self.exe_dir = exe_dir
        self.session_dir = session_dir
        self.cur_job = None
        self.python_dir = python_dir
        self.calc_runner = LocalRunner(self.exe_dir, self.session_dir) # TODO delete?
        self.calc_runner_embedded = EmbeddedLocalRunner()
        self.cur_calc_input = None
        self.cur_job = None
        self.cur_results = None
        self.async_fit = None

    def perform_call(self, call):

        try:
            json2run = json.loads(call)
            result = ""
            if "GetAllModelMetadata" in json2run["function"]:
                metadeta = {"result": json.loads(cur_meta)}
                result = self.process_result(metadeta)

            elif "GetJobStatus" in json2run["function"]:
                if self.calc_runner_embedded is not None:
                    status = self.calc_runner_embedded.get_job_status()
                    result = self.process_result({"result": status})
                elif self.async_fit:
                    status = self.async_fit.get_status()
                    result = self.process_result({"result": status})


            elif "StartGenerate" in json2run["function"]:
                self.cur_calc_input = None
                # self.cur_job = None
                self.cur_results = None
                calc_input = CalculationInput()
                calc_input.load_from_dictionary(json2run["args"]["state"])
                calc_input.use_gpu = bool(json2run["options"]["useGPU"])
                
                self.calc_runner_embedded.generate(calc_input, save_amp=True) # generate_async
                self.cur_calc_input = calc_input
                result = self.process_result({"result": self.calc_runner_embedded.get_job_status()})

            elif "GetGenerateResults" in json2run["function"]:
                self.cur_results = self.calc_runner_embedded.get_generate_results(self.cur_calc_input)
                state = self.cur_results.processed_result
                result = self.process_result({"result": state})

            elif "StartFit" in json2run["function"]:
                self.cur_calc_input = None
                self.cur_job = None
                self.cur_results = None
                self.async_fit = None
                calc_input = CalculationInput._web_load(json2run["args"])
                calc_input.use_gpu = bool(json2run["options"]["useGPU"])

                self.cur_job = self.calc_runner.fit_async(calc_input, save_amp=True)
                self.cur_calc_input = calc_input
                result = self.process_result({"result": self.cur_job.get_status()})
                # self.async_fit = AsyncFit(self.exe_dir, self.session_dir, self.python_dir, calc_input)
                # result = self.process_result({"result": self.async_fit.get_status()})

            elif "GetFitResults" in json2run["function"]:
                # print("GetFitResults")
                # old fit
                if not self.async_fit:
                    self.cur_results = self.cur_job.get_result(self.cur_calc_input)
                    result = self.process_result({"result": self.cur_results._raw_result})
                    # self.add_output()

                else:
                    result = self.process_result({"result": self.async_fit.get_results()})
                    self.add_python_fit_output()



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
                print('GetAmplitude', args)
                if self.cur_results is not None:
                    try:
                        pre_process_results = self.calc_runner_embedded.save_amp(args["model"], args["filepath"])
                        result = self.process_result()
                    except FileNotFoundError:
                        raise Exception("The model was not found within the container or job", 14)
                else:
                    raise Exception("The model was not found within the container or job", 8)

            elif "GetPDB" in json2run["function"]:
                args = json2run["args"]
                if self.cur_results is not None:
                    try:
                        pdb_str = self.calc_runner_embedded.get_pdb(args["model"])
                        print("GetPDB CSharpPython")
                        print(len(pdb_str))
                        print("args['filepath']", args['filepath'])
                        with open(args["filepath"], 'w', encoding='utf8') as file_pdb_out:
                            file_pdb_out.write(pdb_str)
                        result = self.process_result() # {"result": pre_process_results}
                    except FileNotFoundError:
                        raise Exception("The model was not found within the container or job", 14)
                    except Exception as ex:
                        print(ex)
                        raise Exception("Error GetPDB failed " + str(ex))
                else:
                    raise Exception("The model was not found within the container or job", 8)

            elif "CheckCapabilities" in json2run["function"]:
                self.cur_calc_input = None
                self.cur_job = None
                self.cur_results = None
                use_gpu = bool(json2run["options"]["useGPU"])
                # backend = Backend()
                # backend.check_capabilities(use_gpu)
                self.calc_runner_embedded.check_capabilities(use_gpu)
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
        str_json = str_json.replace('&"', '\\"')
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



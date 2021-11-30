import json

from django.http import JsonResponse
import dplus
import pkg_resources
py_api_version= pkg_resources.get_distribution("dplus-api").version
web_debug_version='0.1'
from dplus.CalculationRunner import LocalRunner
from dplus.CalculationInput import CalculationInput
from django.conf import settings
import time

#TODO:
#start_new_job, V
# check_file_status, Vish
# delete_job, Vish
# get_exe_results, V
# create_metadata_file, Vish
# get_job_status V

class JobRunningException(Exception):
    def __init__(self):
        self.value = "there is a job already running in this session"

    def __str__(self):
        return repr(self.value)


def create_metadata_file(folder_path):
    '''
    currently no reason for this not to be called from jobs, but in the future perhaps metadata creation will be via python?
    for now, just gonna call the original
    '''
    from raw_backend.oldjobs import create_metadata_file as cmf
    cmf(folder_path)

def start_new_job(type, Dsession, args):
    local = LocalRunner(exe_directory=settings.EXE_DIR, session_directory= Dsession.directory)
    try:
        calc_data = CalculationInput._web_load(args)
        if type=='Fit':
            calc_data.use_gpu=settings.USE_GPU and calc_data.DomainPreferences.use_grid
            job= local.fit_async(calc_data)
            Dsession.running_job=job
        if type=='Generate':
            calc_data.use_gpu = settings.USE_GPU and calc_data.DomainPreferences.use_grid
            job = local.generate_async(calc_data)
            Dsession.running_job=job
        return get_job_status(Dsession)
    except JobRunningException:
        res_dict = {"error":{"code": 2, "message": "A job is already running"}}
        return modify_return_json(res_dict)


def get_exe_results(Dsession):
    '''
    previously: load result files from folder
    now: ??? either continue to load result files from folder, or get the result from API?
    '''

    running_job = Dsession.running_job
    #if running_job.OK:
    return modify_return_json(running_job._get_result())
    #else:
    #return #TODO: some kind of error handling

def get_job_status(Dsession):
    '''
    get the job status json
    '''
    running_job = Dsession.running_job
    #if running_job.OK:
    status=running_job.get_status()
    return modify_return_json(status)
    #else:
    #    return #TODO: some kind of error handling


def delete_job(Dsession):
    '''
    end any currently running job on this session, freeing the session to be used for a new job
    '''
    running_job = Dsession.running_job
    running_job.abort()
    res_dict = {"error": {"code": 1, "message": "The job was manually stopped"}}
    return modify_return_json(res_dict)
    #if running_job.OK:
    #    running_job.abort()
    #    res_dict = {"error": {"code": 1, "message": "The job was manually stopped"}}
    #    return modify_return_json(res_dict)
    #else:
    #    return #TODO: some kind of error handling

def check_file_status(Dsession):
    '''
    check whether current job has finished saving amp and pdb files
    The abpove desription not accurate it busywaits until files are ready
    should possible be handled by LocalAPI
    should it be folded into get_job_status, somehow?
    '''
    running_job = Dsession.running_job
    #if running_job.OK:
    status=False
    while not status:
        status=running_job.get_file_status()
        time.sleep(1)



def get_pdb(Dsession, pointer):
    '''
    this function was handled in views, not in jobs, but it should perhaps be handled differently with a LocalAPI
    '''
    pass

def get_amp(Dsession, pointer):
    '''
    this function was handled in views, not in jobs, but it should perhaps be handled differently with a LocalAPI
    '''
    pass



set_http_code = {0: 200, 1: 200, 2: 500, 3: 404, 4: 500, 5: 500, 6: 404, 7: 400, 8: 500, 9: 400, 10: 404, 11: 400,
                     12: 400, 13: 400, 14: 500, 15: 400, 16: 400, 17: 500, 18: 500, 19: 500, 20: 500}


def modify_return_json(response_json={'result': ''}):
    #			String ^addedPretendErrorField = pyrapidJsonResponse->Replace("\"result\"", "\"error\": {\"code\": 0, \"message\": \"OK\"}, \"client-data\":\"\", \"result\"");
        return_json={
            "error":{
                "code":0,
                "message":"OK"
            },
            "client-data":"",
            "debug_info": "pyInt version: "+ str(py_api_version) +" web version" + web_debug_version
        }
        try:
            return_json["error"]=response_json["error"]
            jr=JsonResponse(return_json)
            jr.status_code = set_http_code[response_json['error']['code']]
            jr.reason_phrase = response_json['error']['message']
            return jr
        except KeyError:
            try:
                return_json["result"]=response_json["result"]
            except KeyError:
                return_json["result"]=response_json
            jr = JsonResponse(return_json)
            return jr

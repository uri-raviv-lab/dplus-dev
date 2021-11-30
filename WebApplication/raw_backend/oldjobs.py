import glob
import platform
from time import sleep

import psutil as psutil

import os
import json

from django.http import JsonResponse
from django.utils.datetime_safe import datetime
from database_items.models import DplusSession
from django.conf import settings
import subprocess
import logging
import signal

from .jobs import modify_return_json, JobRunningException

logger = logging.getLogger('dplus.jobs')



def start_new_job(type, Dsession, args):
    '''
    This function handles all the necessary steps of starting a job:
    1. creating a new calculation
    2. loading the args
    3. deleting existing job files
    4. performing the actual exe call
    :param type: 'Generate' or 'Fit'
    :param Dsession: the current DPlus session
    :return: result dictionary from function call
    '''
    logging.info("Starting job %s (%s) for session %s" % (type, Dsession, args))
    if check_job_running(Dsession):
        res_dict = {"error":{"code": 2, "message": "A job is already running"}}
        return modify_return_json(res_dict)
        raise JobRunningException
    try:
        logger.debug("Deleting old job files")
        delete_job_files(Dsession)
        logger.debug("Saving arguments")
        exe_save_args(Dsession, args)
        rc = perform_exe_call(type, Dsession)
        logger.info("Job started")
        return rc
    except:
        #if job wasn't successfully started, make sure any progress that may have been spawned is killed
        delete_job(Dsession)

def check_job_running(Dsession):
    return False #ToDO fix this
    is_running=False
    try:
        pid=Dsession.get_pid()
        if pid != -1:
            process=psutil.Process(Dsession.get_pid())
            if process.status() == psutil.STATUS_ZOMBIE:
                process.wait()
                pass
            else:
                Dsession.switch_running(True)
                is_running=True
    except psutil.NoSuchProcess:
        pass
    return is_running

def delete_job(Dsession):
    try:
        os.kill(Dsession.get_pid(), signal.SIGTERM)
    finally:
        #Dsession.switch_running(False)
        res_dict = {"error": {"code": 1, "message": "The job was manually stopped"}}
        return modify_return_json(res_dict)


def check_file_status(Dsession):
    return
    '''
    :param Dsession: current sesson
    :return: check whether process has finished running (and hence finished saving amp files)
    '''
    try:
        process=psutil.Process(Dsession.get_pid())
        process.communicate()
    except psutil.NoSuchProcess:
        pass
    finally:
        return

def get_job_status(Dsession):
    filename = os.path.join(Dsession.directory, 'job.json')
    res_dict = {'result': {'isRunning': True, 'code': -1, 'progress': 0}}
    try:
        with open(filename, 'r', encoding='utf8') as f:
            data = json.load(f)
            res_dict = {'result': data}
    except:
        pass
    finally:
        return modify_return_json(res_dict)


def create_metadata_file(folder_path):
    program_path = os.path.join(settings.EXE_DIR, "getallmetadata")
    if platform.system() == 'Windows':
        program_path += ".exe"
    process = subprocess.Popen([program_path, folder_path], cwd=settings.EXE_DIR)
    stdoutdata, stderrdata = process.communicate()





def delete_job_files(Dsession):
    searchstring=Dsession.directory+"/*.*"
    filelist = glob.glob(searchstring)
    for f in filelist:
        os.remove(f)

def perform_exe_call(function, Dsession):
    err_file=open(os.path.join(Dsession.directory, "error.txt"), 'w')
    out_file=open(os.path.join(Dsession.directory, "output.txt"), 'w')

    Dsession.switch_running(True)
    program_path = os.path.join(settings.EXE_DIR, function.lower())
    if platform.system()=='Windows':
        program_path+=".exe"
    logger.debug("Running %s %s" % (program_path, Dsession.directory))
    curr_process = subprocess.Popen([program_path, Dsession.directory], cwd=settings.EXE_DIR, stdout=out_file, stderr=err_file)
    Dsession.update_process(curr_process.pid)
    return modify_return_json()

def get_exe_results(Dsession):
    if Dsession.get_job_running():
        filename = Dsession.directory + r"/data.json"
        with open(filename, 'r', encoding='utf8') as f:
            data = json.load(f)
        return modify_return_json(data)
    return JsonResponse


def exe_save_args(Dsession, args):
    filename = os.path.join(Dsession.directory, "args.json")
    data = {'args': args}
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)



import json

import math
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from file_management.syncing_names import fix_filenames
from sendfile import sendfile
import os
from rest_framework.decorators import api_view
from database_items.models import DplusSession
from raw_backend.jobs import start_new_job, check_file_status, get_exe_results, delete_job, create_metadata_file, \
    get_job_status, modify_return_json, py_api_version, web_debug_version

__author__ = "DevoraW"

class MyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "inf" in obj.values() or "-inf" in obj.values():
            for key, value in obj.items():
                if value=="inf":
                    obj[key]=math.inf
                elif value=="-inf":
                    obj[key]=-math.inf
        return obj

def get_body_json(body_bytes):
    body_str = body_bytes.decode('utf-8')
    body_json = json.loads(body_str, cls=MyDecoder)
    return body_json


def metadata(request):
    folder_path = settings.MEDIA_ROOT
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    try:
        filename = folder_path + r"/metadata.json"
        f = open(filename, 'r', encoding='utf8')
    except FileNotFoundError:
        create_metadata_file(folder_path)
        f = open(filename, 'r', encoding='utf8')
    data = json.load(f)
    f.close()
    res_dict = {'result': data}
    return modify_return_json(res_dict)

def handle_error(e):
    error_dict={
        "error":
            {
                "message":str(e),
                "debug_info": "pyInt version: "+ str(py_api_version) +" web version" + web_debug_version
            }
    }
    if isinstance(e, ValueError):
        error_dict["error"]["code"]=19
        jr = JsonResponse(error_dict)
        jr.status_code = 400
    else:
        error_dict["error"]["code"] = 5
        jr = JsonResponse(error_dict)
        jr.status_code = 500
    return jr

@api_view(['GET', 'PUT'])
@csrf_exempt
def generate(request):
    Dsession = DplusSession.get_or_create_session(request)
    try:
        if request.method == 'PUT':
            body_json = get_body_json(request.body)
            args = fix_filenames(body_json, request.user)
            return start_new_job('Generate', Dsession, args)
        if request.method == 'GET':
            return get_exe_results(Dsession)
    except Exception as e:
        return handle_error(e)

@api_view(['GET', 'PUT'])
@csrf_exempt
def fit(request):
    Dsession = DplusSession.get_or_create_session(request)
    try:
        if request.method == 'PUT':
            body_json = get_body_json(request.body)
            args = fix_filenames(body_json, request.user)
            return start_new_job('Fit', Dsession, args)
        if request.method == 'GET':
            return get_exe_results(Dsession)
    except Exception as e:
        return handle_error(e)


@api_view(['GET', 'DELETE'])
@csrf_exempt
def job(request):
    Dsession = DplusSession.get_or_create_session(request)
    if request.method == 'DELETE':
        return delete_job(Dsession)
    if request.method == 'GET':
        return get_job_status(Dsession)



@api_view(['GET'])
@csrf_exempt
def pdb(request, modelptr):
    Dsession = DplusSession.get_or_create_session(request)
    #check_file_status(Dsession) #TODO
    ptr_string = '%08d.pdb' % (int(modelptr))
    filepath = os.path.join(Dsession.directory, 'pdb', ptr_string)
    if not os.path.isfile(filepath):
        return JsonResponse({"error":{"code":6, "message":"could not find pdb for model "+modelptr}}, status=404, reason="file not found")
    # TODO: Check if something bad happened
    return sendfile(request, filepath, attachment=True, attachment_filename='returnedfile.pdb',
                    mimetype='application/octet-stream')


@api_view(['GET'])
@csrf_exempt
def amplitude(request, modelptr):
    Dsession = DplusSession.get_or_create_session(request)
    #check_file_status(Dsession) #TODO
    ptr_string = '%08d.amp' % (int(modelptr))
    filepath = os.path.join(Dsession.directory, 'cache', ptr_string)
    if not os.path.isfile(filepath):
        return JsonResponse({"error":{"code":6, "message":"could not find amplitude for model "+modelptr}}, status=404, reason="file not found")
    # TODO: Check if something bad happened
    return sendfile(request, filepath, attachment=True, attachment_filename='returnedfile.amp',
                    mimetype='application/octet-stream')


import json

import hashlib
from chelem.ajax.utils import request_to_json
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from file_management.models import InputFile
import os.path
from project import settings
from rest_framework.decorators import api_view


def check_one_file(file_dict, user):
    try:
        file_obj = InputFile.objects.get(original_filename=file_dict['filename'], user=user)
        if file_obj.sha1_hash == file_dict['hash'] and file_obj.size == file_dict['size']:
            abs_path = os.path.join(settings.MEDIA_ROOT, file_obj.file.name)
            if os.path.isfile(abs_path):
                return dict(id=file_obj.id, status='OK')
        file_obj.delete()
    except InputFile.DoesNotExist:
        pass
    file_obj = InputFile.objects.create(original_filename=file_dict['filename'],
                                        sha1_hash=file_dict['hash'],
                                        size=file_dict['size'],
                                        user=user)
    return dict(id=file_obj.id, status='MISSING')


@api_view(['POST'])
@csrf_exempt
def check_files(request):
    data = request_to_json(request)

    results = dict()
    for file_dict in data['files']:
        results[file_dict['filename']] = check_one_file(file_dict, request.user)

    return JsonResponse(results)


def calc_file_hash(uploaded_file):
    sha1 = hashlib.sha1()
    for chunk in uploaded_file.chunks():
       sha1.update(chunk)
    return sha1.hexdigest().upper()


@api_view(['POST'])
@csrf_exempt
def upload_file(request, id):
    if len(request.FILES) != 1:
        return JsonResponse({'error': 'NO FILE'}, status=400)  # Bad request - no file
    file_obj = get_object_or_404(InputFile, id=id)
    uploaded_file = request.FILES['file']
    if file_obj.size != uploaded_file.size:
        return JsonResponse({'error': 'SIZE MISMATCH'}, status=409)
    hash = calc_file_hash(uploaded_file)
    if hash != file_obj.sha1_hash:
        return JsonResponse({'error': 'HASH MISMATCH'}, status=409)
    file_obj.file = uploaded_file
    file_obj.save()

    return JsonResponse({}, status=200)
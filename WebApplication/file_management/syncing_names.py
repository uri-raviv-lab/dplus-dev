import os
from file_management.models import InputFile
from project import settings

_filename_re = r'"Filename":\s*"(?P<filename>.*)"'


def fix_filenames(obj, user):
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = fix_filenames(obj[i], user)
    elif isinstance(obj, dict):
        for (key, value) in obj.items():
            if 'file' in str.lower(key) and len(value) > 0:
                obj[key] = get_server_filename(value, user)
            else:
                obj[key] = fix_filenames(value, user)

    return obj


def get_server_filename(filename, user):
    try:
        file_obj = InputFile.objects.get(original_filename=filename, user=user)
        name = file_obj.file.name
        abs_path = os.path.join(settings.MEDIA_ROOT, name)
        return abs_path
    except:
        raise ValueError("Failed to find file object")
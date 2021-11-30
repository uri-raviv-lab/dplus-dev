from django.contrib.auth.models import User
from django.db import models
from django.core.files.storage import FileSystemStorage
import os.path

fs = FileSystemStorage(location='/media/users/uploaded_files')

def get_upload_path(instance, filename):
    return os.path.join('users',instance.user.username,'uploaded_files',filename)

class InputFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    original_filename = models.CharField(max_length=2048)
    size = models.IntegerField()
    sha1_hash = models.CharField(max_length=40)
    file = models.FileField(upload_to=get_upload_path)


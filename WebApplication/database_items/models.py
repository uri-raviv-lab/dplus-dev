from django.db import models
from django.contrib.auth.models import User
import datetime
import os
from django.conf import settings
from dplus.CalculationRunner import LocalRunner, as_job
import json



class DplusSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=None)
    session_guid=models.UUIDField()
    start_time=models.TimeField()
    last_call_time=models.TimeField()
    job=models.CharField(max_length=4096, blank=True, null=True, default=None)

    def __str__(self):
        return str(self.id) +" ("+str(self.user)+")"

    @property
    def directory(self):
        dir= os.path.join(settings.MEDIA_ROOT, 'users', self.user.username, 'sessions', str(self.session_guid))
        return dir

    @property
    def running_job(self):
        return json.loads(self.job, object_hook=as_job)

    @running_job.setter
    def running_job(self, running):
        self.job=json.dumps(running.__dict__)
        self.save()

    @staticmethod
    def get_or_create_session(request):
        session_guid=request.query_params['ID']
        user=request.user
        current_time = datetime.datetime.now().time()
        try:
            curr_session=DplusSession.objects.get(user=user, session_guid=session_guid)
            curr_session.last_call_time=current_time
        except DplusSession.DoesNotExist:
            directory = os.path.join(settings.MEDIA_ROOT, 'users', user.username, 'sessions', session_guid)
            make_session_directories(directory)
            curr_session=DplusSession.objects.create(user=user,session_guid=session_guid, start_time=current_time, last_call_time=current_time, job=json.dumps({"session_directory":directory,"pid":-1}))
        curr_session.save()
        return curr_session


def make_session_directories(directory):
    os.makedirs(os.path.join(directory, 'pdb'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'amp'), exist_ok=True)








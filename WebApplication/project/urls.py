"""project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.8/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Add an import:  from blog import urls as blog_urls
    2. Add a URL to urlpatterns:  url(r'^blog/', include(blog_urls))
"""
import raw_backend.views
from django.conf.urls import include, url
from django.contrib import admin
import file_management.views
from rest_framework import routers
from rest_framework.authtoken import views

urlpatterns = [

    url(r'^admin/', include(admin.site.urls)),
    url(r'^api-token-auth/', views.obtain_auth_token),
    url(r'^api/v1/files$', file_management.views.check_files),
    url(r'^api/v1/files/(?P<id>\d+)$', file_management.views.upload_file),
    url(r'^api/v1/metadata$', raw_backend.views.metadata),
    url(r'^api/v1/generate$', raw_backend.views.generate),
    url(r'^api/v1/fit$', raw_backend.views.fit),
    url(r'^api/v1/job$', raw_backend.views.job),
    url(r'^api/v1/pdb/(?P<modelptr>\d+)$', raw_backend.views.pdb),
    url(r'^api/v1/amplitude/(?P<modelptr>\d+)$', raw_backend.views.amplitude),

]

'''


/api/v1/pdb/<modelptr>		(GetPDB)
	HTTP method: GET
	Returns the PDB of the supplied model pointer

/api/v1/amplitude/<modelptr>		(GetAmplitudte)
	HTTP method: GET
	Returns the Amplitude of the supplied model pointer

/api/v1/files
	HTTP method: POST
	Checks if files need to be uploaded to the server
	Body: JSON with files the client wants to sync to the server
	Returns: JSON containing list of files that needs to be synced and their IDs

/api/v1/files/<id>
	HTTP method: POST
	Uploads a file to the server
	Body: The file's content
	I know if should be PUT, but Django likes files to be uploaded with POST

/api/v1/metadata  (GetAllModelMetadata)
	Returns all the model metadata
	HTTP method: GET
	Returns the model metadata.

/api/v1/generate     (GetGenerateResult)
	HTTP method: GET
	Gets the generate result associated with this session
	Body: The generate results (the 'results' section of the JSON returned by the backend)

/api/v1/generate     (StartGenerate)
	HTTP method: PUT
	Starts a new generate process.
	Args: the generation arguments (the 'args' section of the JSON passed to the backend)


/api/v1/fit			(GetFitResults)
	HTTP method: GET
	Gets the fit results associated with this session
	Body: The fit results (the 'results' section of the JSON returned by the backend)

/api/v1/fit			(StartFit)
	HTTP method: PUT
	Starts a new fitting job
	Body: the fitting arguments (the 'args' section of the JSON passed to the backend)

/api/v1/job				(GetJobStatus)
	HTTP method: GET
	Returns the current job's status
'''
from django.contrib import admin
from django.contrib.admin import register
from file_management.models import InputFile


@register(InputFile)
class InputFileAdmin(admin.ModelAdmin):
    list_display = ('original_filename', 'size', 'sha1_hash')

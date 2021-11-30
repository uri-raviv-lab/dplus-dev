from django.contrib import admin

# Register your models here.
from django.contrib.admin import register
from database_items.models import DplusSession


@register(DplusSession)
class DPlusSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'session_guid', 'start_time', 'last_call_time', 'job')

#@register(Calculation)
#class CalculationAdmin(admin.ModelAdmin):
#    list_display = ('session_id', 'type','is_running','finish_success','start_time','end_time')

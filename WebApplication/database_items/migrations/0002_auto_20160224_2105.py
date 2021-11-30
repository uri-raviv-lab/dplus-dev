# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database_items', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='dplussession',
            name='job_running',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='dplussession',
            name='process_handle',
            field=models.IntegerField(null=True, blank=True, default=-1),
        ),
        migrations.AlterField(
            model_name='calculation',
            name='end_time',
            field=models.TimeField(null=True, blank=True, default=None),
        ),
    ]

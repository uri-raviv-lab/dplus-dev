# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
import file_management.models


class Migration(migrations.Migration):

    dependencies = [
        ('file_management', '0002_inputfile_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='inputfile',
            name='file',
            field=models.FileField(upload_to=file_management.models.get_upload_path),
        ),
    ]

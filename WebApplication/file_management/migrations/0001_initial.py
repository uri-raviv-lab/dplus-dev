# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='InputFile',
            fields=[
                ('id', models.AutoField(serialize=False, verbose_name='ID', primary_key=True, auto_created=True)),
                ('original_filename', models.CharField(max_length=2048)),
                ('size', models.IntegerField()),
                ('sha1_hash', models.CharField(max_length=40)),
                ('file', models.FileField(upload_to='')),
            ],
        ),
    ]

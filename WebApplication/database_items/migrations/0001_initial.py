# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Calculation',
            fields=[
                ('id', models.AutoField(serialize=False, verbose_name='ID', primary_key=True, auto_created=True)),
                ('type', models.CharField(max_length=10)),
                ('is_running', models.BooleanField()),
                ('finish_success', models.BooleanField()),
                ('start_time', models.TimeField()),
                ('end_time', models.TimeField()),
            ],
        ),
        migrations.CreateModel(
            name='DplusSession',
            fields=[
                ('id', models.AutoField(serialize=False, verbose_name='ID', primary_key=True, auto_created=True)),
                ('session_guid', models.UUIDField()),
                ('start_time', models.TimeField()),
                ('last_call_time', models.TimeField()),
                ('directory', models.CharField(max_length=2048)),
                ('most_recent_calc', models.ForeignKey(default=None, blank=True, to='database_items.Calculation', null=True)),
                ('user', models.ForeignKey(default=None, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.AddField(
            model_name='calculation',
            name='session_id',
            field=models.ForeignKey(default=None, to='database_items.DplusSession'),
        ),
    ]

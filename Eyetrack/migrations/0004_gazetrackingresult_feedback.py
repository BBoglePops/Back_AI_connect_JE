# Generated by Django 5.0.4 on 2024-05-24 17:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Eyetrack', '0003_rename_gazetrackingresultnew_gazetrackingresult'),
    ]

    operations = [
        migrations.AddField(
            model_name='gazetrackingresult',
            name='feedback',
            field=models.TextField(default='No feedback provided.'),
        ),
    ]
# Generated by Django 5.0.4 on 2024-05-19 13:22

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='GazeTrackingResult',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]

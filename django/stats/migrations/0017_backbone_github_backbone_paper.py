# Generated by Django 5.1 on 2024-09-23 05:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0016_instanceresult_intermediate_train_dataset_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="backbone",
            name="github",
            field=models.URLField(blank=True),
        ),
        migrations.AddField(
            model_name="backbone",
            name="paper",
            field=models.URLField(blank=True),
        ),
    ]

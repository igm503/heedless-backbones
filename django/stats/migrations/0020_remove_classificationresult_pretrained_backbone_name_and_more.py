# Generated by Django 5.1 on 2024-09-24 05:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0019_remove_pretrainedbackbone_classification_results_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="classificationresult",
            name="pretrained_backbone_name",
        ),
        migrations.RemoveField(
            model_name="instanceresult",
            name="pretrained_backbone_name",
        ),
    ]

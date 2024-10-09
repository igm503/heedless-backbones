# Generated by Django 5.1 on 2024-08-19 00:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0005_alter_pretrainedbackbone_classification_results_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="classificationresult",
            name="pretrained_backbone_name",
            field=models.CharField(default="", max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="instanceresult",
            name="pretrained_backbone_name",
            field=models.CharField(default="", max_length=100),
            preserve_default=False,
        ),
    ]

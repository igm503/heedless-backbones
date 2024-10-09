# Generated by Django 5.1 on 2024-08-19 00:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0003_alter_classificationresult_fine_tune_dataset_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="pretrainedbackbone",
            name="classification_results",
            field=models.ManyToManyField(
                blank=True, null=True, to="stats.classificationresult"
            ),
        ),
        migrations.AlterField(
            model_name="pretrainedbackbone",
            name="instance_results",
            field=models.ManyToManyField(
                blank=True, null=True, to="stats.instanceresult"
            ),
        ),
    ]

# Generated by Django 5.1 on 2024-09-23 03:10

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("view", "0015_classificationresult_intermediate_fine_tune_dataset_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="instanceresult",
            name="intermediate_train_dataset",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="intermediate_instance_train",
                to="view.dataset",
            ),
        ),
        migrations.AddField(
            model_name="instanceresult",
            name="intermediate_train_epochs",
            field=models.IntegerField(
                blank=True, null=True, verbose_name="intermediate training epochs"
            ),
        ),
        migrations.AlterField(
            model_name="classificationresult",
            name="intermediate_fine_tune_dataset",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="intermediate_classification_fine_tune",
                to="view.dataset",
            ),
        ),
        migrations.AlterField(
            model_name="classificationresult",
            name="intermediate_fine_tune_epochs",
            field=models.IntegerField(
                blank=True, null=True, verbose_name="intermediate fine-tuning epochs"
            ),
        ),
        migrations.AlterField(
            model_name="classificationresult",
            name="intermediate_fine_tune_resolution",
            field=models.IntegerField(
                blank=True,
                null=True,
                verbose_name="intermediate fine-tuning resolution",
            ),
        ),
    ]
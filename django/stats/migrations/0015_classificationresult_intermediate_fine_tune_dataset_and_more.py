# Generated by Django 5.1 on 2024-09-22 18:16

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0014_backbonefamily_hierarchical"),
    ]

    operations = [
        migrations.AddField(
            model_name="classificationresult",
            name="intermediate_fine_tune_dataset",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="intermediate_fine_tune",
                to="stats.dataset",
            ),
        ),
        migrations.AddField(
            model_name="classificationresult",
            name="intermediate_fine_tune_epochs",
            field=models.IntegerField(
                blank=True, null=True, verbose_name="fine-tuning epochs"
            ),
        ),
        migrations.AddField(
            model_name="classificationresult",
            name="intermediate_fine_tune_resolution",
            field=models.IntegerField(
                blank=True, null=True, verbose_name="fine-tuning resolution"
            ),
        ),
    ]

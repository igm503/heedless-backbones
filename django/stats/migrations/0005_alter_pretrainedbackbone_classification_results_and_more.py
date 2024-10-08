# Generated by Django 5.1 on 2024-08-19 00:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0004_alter_pretrainedbackbone_classification_results_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="pretrainedbackbone",
            name="classification_results",
            field=models.ManyToManyField(blank=True, to="stats.classificationresult"),
        ),
        migrations.AlterField(
            model_name="pretrainedbackbone",
            name="instance_results",
            field=models.ManyToManyField(blank=True, to="stats.instanceresult"),
        ),
    ]

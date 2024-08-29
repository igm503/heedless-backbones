# Generated by Django 5.1 on 2024-08-18 23:50

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("view", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="pretrainedbackbone",
            name="family",
            field=models.ForeignKey(
                default=0,
                on_delete=django.db.models.deletion.CASCADE,
                to="view.backbonefamily",
            ),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="backbone",
            name="m_parameters",
            field=models.FloatField(verbose_name="parameters (M)"),
        ),
    ]

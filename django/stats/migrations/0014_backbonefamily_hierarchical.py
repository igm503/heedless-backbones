# Generated by Django 5.1 on 2024-09-10 03:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0013_backbonefamily_pretrain_method_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="backbonefamily",
            name="hierarchical",
            field=models.BooleanField(default=True),
            preserve_default=False,
        ),
    ]

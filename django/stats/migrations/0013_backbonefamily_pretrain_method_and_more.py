# Generated by Django 5.1 on 2024-09-10 03:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0012_dataset_eval"),
    ]

    operations = [
        migrations.AddField(
            model_name="backbonefamily",
            name="pretrain_method",
            field=models.CharField(
                choices=[
                    ("Supervised", "Supervised"),
                    ("FCMAE", "FCMAE"),
                    ("MAE", "MAE"),
                    ("CL", "CL"),
                ],
                default="Supervised",
                max_length=100,
                verbose_name="pretraining method",
            ),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="backbonefamily",
            name="model_type",
            field=models.CharField(
                choices=[
                    ("Attention", "Attention"),
                    ("Convolution", "Convolution"),
                    ("LSTM", "LSTM"),
                    ("State Space Model", "State Space Model"),
                    ("Attn + Conv", "Attn + Conv"),
                    ("Conv + SSM", "Conv + SSM"),
                ],
                max_length=100,
                verbose_name="model type",
            ),
        ),
    ]

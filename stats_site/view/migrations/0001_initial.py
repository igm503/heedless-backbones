# Generated by Django 5.1 on 2024-08-18 23:25

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="BackboneFamily",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                (
                    "model_type",
                    models.CharField(
                        choices=[
                            ("CNN", "CNN"),
                            ("Transformer", "Transformer"),
                            ("SSM", "SSM"),
                            ("Hybrid", "Hybrid"),
                        ],
                        max_length=100,
                        verbose_name="model type",
                    ),
                ),
                ("pub_date", models.DateField(verbose_name="publication date")),
                ("paper", models.URLField(blank=True)),
                ("github", models.URLField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name="Dataset",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("website", models.URLField()),
            ],
        ),
        migrations.CreateModel(
            name="DownstreamHead",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("paper", models.URLField(blank=True)),
                ("github", models.URLField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name="Task",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "name",
                    models.CharField(
                        choices=[
                            ("Classification", "Classification"),
                            ("Object Detection", "Object Detection"),
                            ("Instance Segmentation", "Instance Segmentation"),
                            ("Semantic Segmentation", "Semantic Segmentation"),
                        ],
                        max_length=100,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Backbone",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("m_parameters", models.IntegerField(verbose_name="parameters (M)")),
                (
                    "family",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="view.backbonefamily",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="ClassificationResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "fine_tune_epochs",
                    models.IntegerField(
                        blank=True, null=True, verbose_name="fine-tuning epochs"
                    ),
                ),
                (
                    "fine_tune_resolution",
                    models.IntegerField(
                        blank=True, null=True, verbose_name="fine-tuning resolution"
                    ),
                ),
                (
                    "top_1",
                    models.FloatField(blank=True, null=True, verbose_name="top-1"),
                ),
                (
                    "top_5",
                    models.FloatField(blank=True, null=True, verbose_name="top-5"),
                ),
                ("gflops", models.IntegerField(blank=True, null=True)),
                ("paper", models.URLField(blank=True)),
                ("github", models.URLField(blank=True)),
                (
                    "dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="view.dataset"
                    ),
                ),
                (
                    "fine_tune_dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="classification_fine_tune",
                        to="view.dataset",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="InstanceResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "train_epochs",
                    models.IntegerField(
                        blank=True, null=True, verbose_name="training epochs"
                    ),
                ),
                ("mAP", models.FloatField()),
                ("AP50", models.FloatField(blank=True, null=True)),
                ("AP75", models.FloatField(blank=True, null=True)),
                ("mAPs", models.FloatField(blank=True, null=True)),
                ("mAPm", models.FloatField(blank=True, null=True)),
                ("mAPl", models.FloatField(blank=True, null=True)),
                ("gflops", models.IntegerField(blank=True, null=True)),
                ("paper", models.URLField(blank=True)),
                ("github", models.URLField(blank=True)),
                (
                    "dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="view.dataset"
                    ),
                ),
                (
                    "head",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="view.downstreamhead",
                    ),
                ),
                (
                    "train_dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="instance_train",
                        to="view.dataset",
                    ),
                ),
                ("instance_type", models.ManyToManyField(to="view.task")),
            ],
        ),
        migrations.CreateModel(
            name="PretrainedBackbone",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                (
                    "pretrain_method",
                    models.CharField(
                        choices=[
                            ("Supervised", "Supervised"),
                            ("FCMAE", "FCMAE"),
                            ("MAE", "MAE"),
                        ],
                        max_length=100,
                        verbose_name="pretraining method",
                    ),
                ),
                ("pretrain_resolution", models.IntegerField(blank=True, null=True)),
                ("pretrain_epochs", models.IntegerField(blank=True, null=True)),
                ("paper", models.URLField(blank=True)),
                ("github", models.URLField(blank=True)),
                (
                    "backbone",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="view.backbone"
                    ),
                ),
                (
                    "classification_results",
                    models.ManyToManyField(to="view.classificationresult"),
                ),
                ("instance_results", models.ManyToManyField(to="view.instanceresult")),
                (
                    "pretrain_dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="view.dataset"
                    ),
                ),
            ],
        ),
        migrations.AddField(
            model_name="downstreamhead",
            name="tasks",
            field=models.ManyToManyField(to="view.task"),
        ),
        migrations.AddField(
            model_name="dataset",
            name="tasks",
            field=models.ManyToManyField(to="view.task"),
        ),
    ]

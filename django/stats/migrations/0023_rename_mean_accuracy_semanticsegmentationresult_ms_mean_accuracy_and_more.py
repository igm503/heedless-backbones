# Generated by Django 5.1 on 2024-10-15 03:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stats", "0022_alter_backbonefamily_model_type_and_more"),
    ]

    operations = [
        migrations.RenameField(
            model_name="semanticsegmentationresult",
            old_name="mean_accuracy",
            new_name="ms_mean_accuracy",
        ),
        migrations.RenameField(
            model_name="semanticsegmentationresult",
            old_name="pixel_accuracy",
            new_name="ms_pixel_accuracy",
        ),
        migrations.RemoveField(
            model_name="semanticsegmentationresult",
            name="m_iou",
        ),
        migrations.RemoveField(
            model_name="semanticsegmentationresult",
            name="multiscale",
        ),
        migrations.AddField(
            model_name="semanticsegmentationresult",
            name="ms_m_iou",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="semanticsegmentationresult",
            name="ss_m_iou",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="semanticsegmentationresult",
            name="ss_mean_accuracy",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="semanticsegmentationresult",
            name="ss_pixel_accuracy",
            field=models.FloatField(blank=True, null=True),
        ),
    ]

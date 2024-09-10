from django.contrib import admin

from .models import (
    Dataset,
    Task,
    BackboneFamily,
    Backbone,
    PretrainedBackbone,
    DownstreamHead,
    ClassificationResult,
    InstanceResult,
    FPSMeasurement,
)

admin.site.register(Dataset)
admin.site.register(Task)
admin.site.register(BackboneFamily)
admin.site.register(Backbone)
admin.site.register(PretrainedBackbone)
admin.site.register(DownstreamHead)
admin.site.register(ClassificationResult)
admin.site.register(InstanceResult)
admin.site.register(FPSMeasurement)

from django.urls import path

from .views import all, dataset, family, head, datasets, families, heads

urlpatterns = [
    path("", all, name="all"),
    path("families/", families, name="families"),
    path("families/<str:family_name>/", family, name="family"),
    path("datasets", datasets, name="datasets"),
    path("datasets/<str:dataset_name>/", dataset, name="dataset"),
    path("heads/", heads, name="heads"),
    path("heads/<str:head_name>/", head, name="head"),
]

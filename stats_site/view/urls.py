from django.urls import path

from .views import all, dataset, family, head, list_datasets, list_families, list_heads

urlpatterns = [
    path("", all, name="all"),
    path("families/", list_families, name="list_families"),
    path("families/<str:family_name>/", family, name="family"),
    path("datasets", list_datasets, name="list_datasets"),
    path("datasets/<str:dataset_name>/", dataset, name="dataset"),
    path("heads/", list_heads, name="list_heads"),
    path("heads/<str:head_name>/", head, name="head"),
]

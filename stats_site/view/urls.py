from django.urls import path

from .views import show_all, show_family, show_dataset, show_downstream_head

urlpatterns = [
    path('', show_all, name='all'),
    path("<str:family_name>/", show_family, name="backbone_family"),
    path("datasets/<str:dataset_name>/", show_dataset, name="dataset"),
    path("heads/<str:downstream_head_name>/", show_downstream_head, name="downstream_head"),
]

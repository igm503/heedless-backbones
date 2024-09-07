from django.urls import path

from . import views

urlpatterns = [
    path('', views.plot_view, name='results'),
    path("<str:family_name>/", views.show_family, name="backbone_family"),
    path("datasets/<str:dataset_name>/", views.show_dataset, name="dataset"),
    path("heads/<str:downstream_head_name>/", views.show_downstream_head, name="downstream_head"),
]

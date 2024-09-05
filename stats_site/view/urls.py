from django.urls import path

from . import views

urlpatterns = [
    path('', views.plot_view, name='results'),
    path("<str:family_name>/", views.show_family, name="backbone_family"),
    path("<str:dataset_name>/", views.show_dataset, name="dataset"),
    path("<str:downstream_head_name>/", views.show_downstream_head, name="downstream_head"),
]

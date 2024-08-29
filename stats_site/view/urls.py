from django.urls import path

from . import views

urlpatterns = [
    path('', views.plot_view, name='results'),
    # path('get_plot_data/', views.get_plot_data, name='get_plot_data'),
    path("<str:family>/", views.show_family, name="backbone_family"),
    path("<str:dataset>/", views.show_dataset, name="dataset"),
    path("<str:downstream_head>/", views.show_downstream_head, name="downstream_head"),
]

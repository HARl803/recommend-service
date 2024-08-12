from django.urls import path
from . import views

app_name = 'estimate'
urlpatterns = [
    path('', views.index, name='index'),
    path('preview_data/', views.preview_data, name='preview_data'),
]


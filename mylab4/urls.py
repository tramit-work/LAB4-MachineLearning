from django.urls import path
from .views import result_lab  

urlpatterns = [
    path('', result_lab, name='result_lab'),  
]
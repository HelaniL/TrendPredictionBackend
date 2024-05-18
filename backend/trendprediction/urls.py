from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict_colour', views.predict_colour, name='predict_colour'),
    path('predict_pattern', views.predict_pattern, name='predict_pattern'),
    path('predict_style', views.predict_style, name='predict_style'),
    path('currenttrends', views.currenttrends, name='currenttrends'),
    # path('futuretrends', views.futuretrends, name='futuretrends'),
    # path('home', views.api_homepage, name='home'),
    # path('aboutus', views.api_aboutus, name='aboutus'),
    # path('downloadcode', views.api_downloadcode, name='downloadcode'),
    # path('contactus', views.api_contactus, name='contactus'),
]
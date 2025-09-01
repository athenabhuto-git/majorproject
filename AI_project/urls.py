"""
URL configuration for AI_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from AI_app.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index_page, name='front_page'),  
    path('index_page/', index_page, name='index_page'), 
    path('about_page/', about_page, name='about_page'), 
    path('contact_page/', contact_page, name='contact_page'), 
    path('senti_page/', senti_page, name='senti_page'), 
    path('analyze_sentiment/', analyze_sentiment, name='analyze_sentiment'),

]


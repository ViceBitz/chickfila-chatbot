from django.contrib import admin
from django.urls import path, include
from chat.views import interface

urlpatterns = [
    path('', interface, name='interface'),
    path('admin/', admin.site.urls),
    path('api/chat/', include('chat.urls')),
]

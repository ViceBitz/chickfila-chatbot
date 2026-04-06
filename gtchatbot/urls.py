from django.contrib import admin
from django.urls import path, include
from chat.views import interface, dashboard

urlpatterns = [
    path('', interface, name='interface'),
    path('dashboard/', dashboard, name='dashboard'),
    path('admin/', admin.site.urls),
    path('api/chat/', include('chat.urls')),
]

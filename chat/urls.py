from django.urls import path
from . import views

urlpatterns = [
    path('ingest/', views.ingest_documents, name='rag_ingest'),
    path('query/', views.query_chat, name='rag_query'),
]

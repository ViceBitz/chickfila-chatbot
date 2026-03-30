from django.db import models
from pgvector.django import VectorField

class DocumentChunk(models.Model):
    source = models.CharField(max_length=500)
    content = models.TextField()
    embedding = VectorField(dimensions=1536)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        indexes = [
            models.Index(fields=['source']),
        ]

class Conversation(models.Model):
    query = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Conversation {self.id} @ {self.created_at}"
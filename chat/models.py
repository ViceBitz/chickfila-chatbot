from django.db import models


class Conversation(models.Model):
    query = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    response_time_ms = models.IntegerField(null=True, blank=True)
    is_success = models.BooleanField(default=True)
    error_message = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Conversation {self.id} @ {self.created_at}"

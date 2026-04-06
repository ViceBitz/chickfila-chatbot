from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='conversation',
            name='response_time_ms',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='conversation',
            name='is_success',
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name='conversation',
            name='error_message',
            field=models.TextField(blank=True, null=True),
        ),
    ]

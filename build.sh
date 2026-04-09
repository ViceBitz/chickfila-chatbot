#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate

# Create superuser from env vars (only if it doesn't exist yet)
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
import os
username = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin')
email = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@example.com')
password = os.environ.get('DJANGO_SUPERUSER_PASSWORD')
if password and not User.objects.filter(username=username).exists():
    User.objects.create_superuser(username, email, password)
    print(f'Superuser {username} created')
else:
    print('Superuser already exists or no password set')
"

#!/usr/bin/env bash
set -o errexit

# Run migrations
python manage.py migrate

# Enable pgvector extension
python -c "
import os
db_url = os.environ.get('DATABASE_URL')
if db_url:
    import psycopg2
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    conn.cursor().execute('CREATE EXTENSION IF NOT EXISTS vector')
    conn.close()
    print('pgvector extension enabled')
"

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

# Start gunicorn
exec gunicorn gtchatbot.wsgi:application --bind 0.0.0.0:${PORT:-8000}

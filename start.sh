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

# Create superuser (only if it doesn't exist yet)
python manage.py shell -c "
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@example.com', 'admin')
    print('Superuser admin created')
else:
    print('Superuser admin already exists')
"

# Start gunicorn
exec gunicorn gtchatbot.wsgi:application --bind 0.0.0.0:${PORT:-8000}

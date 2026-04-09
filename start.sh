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

# Load knowledge base into pgvector if empty
python manage.py shell -c "
from chat.views import _get_connection_string, get_or_build_store, build_documents, load_location_data, get_embeddings
from langchain_postgres import PGVector
conn = _get_connection_string()
if conn:
    store = PGVector(
        embeddings=get_embeddings(),
        collection_name='chickfila_docs',
        connection=conn,
        use_jsonb=True,
    )
    # Check if collection is empty
    results = store.similarity_search('test', k=1)
    if not results:
        print('Vector store empty — loading documents...')
        docs = build_documents()
        store.add_documents(docs)
        print(f'Loaded {len(docs)} documents')
    else:
        print('Vector store already populated')
    load_location_data()
"

# Start gunicorn
exec gunicorn gtchatbot.wsgi:application --bind 0.0.0.0:${PORT:-8000} --timeout 300

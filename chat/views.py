import os
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI

from .models import Conversation

# Global in-memory vector index in this process.
rag_store = None

# Use a local embedding model if available. Requires sentence-transformers.
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')


def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)


def get_or_build_store(documents):
    global rag_store
    if rag_store is None:
        embeddings = get_embeddings()
        rag_store = FAISS.from_documents(documents, embeddings)
    else:
        rag_store.add_documents(documents)
    return rag_store


_PROMPT = ChatPromptTemplate.from_messages([
    ('system', (
        'You are a Chick-fil-A assistant. '
        'Answer the user\'s question using only the context below. '
        'If the answer is not in the context, say you don\'t have that information.\n\n'
        'Context:\n{context}'
    )),
    ('human', '{question}'),
])


def _format_docs(docs):
    return '\n\n'.join(d.page_content for d in docs)


def create_rag_chain():
    if rag_store is None:
        raise ValueError('RAG store is empty. Please ingest documents first.')
    retriever = rag_store.as_retriever(search_kwargs={'k': 4})
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    return (
        {'context': retriever | _format_docs, 'question': RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )


def interface(request):
    return render(request, 'chat/index.html')


@csrf_exempt
@require_POST
def ingest_documents(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    texts = []
    if 'documents' in data and isinstance(data['documents'], list):
        for item in data['documents']:
            if isinstance(item, dict) and 'text' in item:
                texts.append(str(item['text']))
            elif isinstance(item, str):
                texts.append(item)
    else:
        return JsonResponse({'error': 'Provide "documents" list with {"text":"..."} entries.'}, status=400)

    if not texts:
        return JsonResponse({'error': 'No text to ingest'}, status=400)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for text in texts:
        chunks = splitter.split_text(text)
        docs.extend([Document(page_content=chunk) for chunk in chunks])

    get_or_build_store(docs)
    return JsonResponse({'status': 'ok', 'ingested_chunks': len(docs)})


@csrf_exempt
@require_POST
def query_chat(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    query = data.get('query')
    if not query:
        return JsonResponse({'error': 'Missing query parameter'}, status=400)

    if rag_store is None:
        return JsonResponse({'error': 'No documents ingested yet.'}, status=400)

    try:
        chain = create_rag_chain()
        answer = chain.invoke(query)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    convo = Conversation(query=query, answer=answer)
    convo.save()

    return JsonResponse({'query': query, 'answer': answer})

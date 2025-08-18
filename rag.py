from uuid import uuid4
import sys

# Fix for Streamlit Cloud's old SQLite version
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from prompt import PROMPT, EXAMPLE_PROMPT
import streamlit as st
import os
import tempfile
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

def RequestsURLLoader(urls):
    docs = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract paragraphs only
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            text = "\n".join(paragraphs)

            docs.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            docs.append(Document(page_content=f"Error fetching {url}: {e}", metadata={"source": url}))

    return docs

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lighter model
VECTORSTORE_DIR = Path(tempfile.gettempdir()) / "vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def get_secret(name: str, default: str | None = None):
    """Get secret from Streamlit secrets or environment variables"""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


def initialize_components():
    """Initialize LLM and vector store components"""
    global llm, vector_store
    if llm is None:
        # Get API key
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found in secrets or environment variables")
            st.stop()

        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
        )

        # In-memory vector store for Streamlit Cloud
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=None  # In-memory only
        )


def process_urls(urls):
    """
    Scrape data from URLs and store in vector database
    :param urls: List of input URLs
    :return: Generator yielding status updates
    """
    global vector_store

    try:
        yield "Initializing components..."
        initialize_components()

        # Reset collection for fresh data
        try:
            vector_store.reset_collection()
        except Exception as e:
            # If reset fails, recreate the vector store
            vector_store = None
            initialize_components()

        yield "Loading data..."

        data = RequestsURLLoader(urls)

        if not data:
            yield "Error: No data loaded from URLs"
            return

        yield "Splitting text..."
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=200
        )

        docs = text_splitter.split_documents(data)

        # Ensure proper metadata
        for d in docs:
            if "source" not in d.metadata:
                d.metadata["source"] = d.metadata.get("url", "unknown")

        yield f"Adding {len(docs)} chunks to vector database..."
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(docs, ids=uuids)

        yield "Done! Vector database ready for queries."

    except Exception as e:
        yield f"Error: {str(e)}"
        st.error(f"Failed to process URLs: {str(e)}")


def generate_answer(query):
    """
    Generate answer using RAG with sources
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized. Please process URLs first.")

    try:
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_prompt": EXAMPLE_PROMPT,
            }
        )

        result = chain.invoke({"question": query}, return_only_outputs=True)

        # Extract sources
        sources = []
        if 'sources' in result and result['sources']:
            sources_text = result['sources'].strip()
            if sources_text:
                sources = [s.strip() for s in sources_text.replace(',', '\n').split('\n') if s.strip()]

        # Fallback: get from source_documents
        if not sources and 'source_documents' in result:
            sources = list(set([
                doc.metadata.get('source', 'Unknown source')
                for doc in result['source_documents']
            ]))

        return result['answer'], sources

    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return f"Error: {str(e)}", []
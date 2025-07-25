"""
OpenAI client module for Restaurant Graph Agent.

Handles OpenAI embeddings with connection pooling.
"""

import time
from langchain_openai import OpenAIEmbeddings

# Global embedding model instance
_embedding_model = None


def get_embedding_model():
    """Singleton pattern for embedding model - expensive to create"""
    global _embedding_model
    
    if _embedding_model is None:
        embed_start = time.time()
        print(f"[TIMING] Creating new embedding model...")
        _embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        embed_end = time.time()
        print(f"[TIMING] New embedding model created in {embed_end - embed_start:.3f}s")
    else:
        print(f"[TIMING] Reusing existing embedding model (pooling active)")
    
    return _embedding_model 
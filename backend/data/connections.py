"""
Main database connections module with pooling and caching for Restaurant Graph Agent.

This module manages connections to:
- Neo4j (knowledge graph)
- Supabase (customer data)
- OpenAI (embeddings)
"""

import os
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Global connection instances for pooling
_neo4j_connection = None
_cypher_chain = None
_embedding_model = None
_supabase_client = None

# Caching globals
_schema_cache = {}
_schema_cache_timestamp = 0
_query_cache = {}
_query_cache_timestamps = {}

# Configuration constants - Enhanced for Phase 1 performance
SCHEMA_CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds (extended for better performance)
QUERY_CACHE_DURATION = 60 * 60  # 1 hour for query results (extended for better performance)

load_dotenv()


def get_cached_query_result(query_key: str) -> Optional[Dict[str, Any]]:
    """Get cached query result if available and recent"""
    global _query_cache, _query_cache_timestamps
    
    current_time = time.time()
    
    if query_key in _query_cache:
        cache_age = current_time - _query_cache_timestamps.get(query_key, 0)
        if cache_age < QUERY_CACHE_DURATION:
            print(f"[TIMING] Using cached query result (age: {cache_age:.1f}s)")
            return _query_cache[query_key]
        else:
            # Remove stale cache entry
            del _query_cache[query_key]
            del _query_cache_timestamps[query_key]
    
    return None


def cache_query_result(query_key: str, result: Dict[str, Any]) -> None:
    """Cache query result for future use"""
    global _query_cache, _query_cache_timestamps
    
    _query_cache[query_key] = result
    _query_cache_timestamps[query_key] = time.time()
    print(f"[DEBUG] Cached query result for key: {query_key[:50]}...")


def get_schema_cache():
    """Get the schema cache globals for external access"""
    global _schema_cache, _schema_cache_timestamp
    return _schema_cache, _schema_cache_timestamp, SCHEMA_CACHE_DURATION


def set_schema_cache(schema_cache: dict, timestamp: float):
    """Set the schema cache globals"""
    global _schema_cache, _schema_cache_timestamp
    _schema_cache = schema_cache
    _schema_cache_timestamp = timestamp 
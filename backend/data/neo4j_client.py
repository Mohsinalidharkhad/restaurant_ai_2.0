"""
Neo4j client module for Restaurant Graph Agent.

Handles Neo4j connections, Cypher chain setup, and schema caching.
"""

import os
import time
from typing import Tuple, Any
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
from dotenv import load_dotenv

from .connections import get_schema_cache, set_schema_cache

# Global connection instances
_neo4j_connection = None
_cypher_chain = None

load_dotenv()


def get_neo4j_connection():
    """Singleton pattern for Neo4j connection - reuse existing connection"""
    global _neo4j_connection
    
    if _neo4j_connection is None:
        connection_start = time.time()
        print(f"[TIMING] Creating new Neo4j connection...")
        _neo4j_connection = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True
        )
        connection_end = time.time()
        print(f"[TIMING] New Neo4j connection created in {connection_end - connection_start:.3f}s")
        
        # Test the connection
        try:
            latency_start = time.time()
            _neo4j_connection.query("RETURN 1 as test")
            latency_end = time.time()
            print(f"[TIMING] Connection test latency: {latency_end - latency_start:.3f}s")
        except Exception as e:
            print(f"[ERROR] Connection test failed: {e}")
            _neo4j_connection = None
            raise e
            
        # Get database stats once
        try:
            node_count_result = _neo4j_connection.query("MATCH (n) RETURN count(n) as node_count")
            node_count = node_count_result[0]['node_count'] if node_count_result else 0
            print(f"[DEBUG] Database initialized with {node_count} nodes")
        except Exception as e:
            print(f"[DEBUG] Database stats check failed: {e}")
    else:
        print(f"[TIMING] Reusing existing Neo4j connection (connection pooling active)")
    
    return _neo4j_connection


def get_cypher_chain():
    """Singleton pattern for Cypher chain - reuse existing chain and schema"""
    global _cypher_chain
    
    if _cypher_chain is None:
        chain_setup_start = time.time()
        print(f"[TIMING] Creating new Cypher chain...")
        
        # Get connection (will reuse if available)
        neo4j_graph = get_neo4j_connection()
        
        # LLM setup (lightweight)
        llm_start = time.time()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_end = time.time()
        print(f"[TIMING] LLM setup in {llm_end - llm_start:.3f}s")
        
        # Enhanced schema handling with persistent caching and pre-warming
        current_time = time.time()
        schema_cache, schema_cache_timestamp, cache_duration = get_schema_cache()
        
        # Extended cache duration for better performance (24 hours instead of 1 hour)
        extended_cache_duration = 24 * 60 * 60  # 24 hours in seconds
        
        if schema_cache and (current_time - schema_cache_timestamp) < extended_cache_duration:
            print(f"[DEBUG] Using cached schema (age: {current_time - schema_cache_timestamp:.1f}s)")
            neo4j_graph.schema = schema_cache.get('schema', '')
            schema_time = 0.0
        else:
            print(f"[DEBUG] Refreshing schema...")
            schema_start = time.time()
            neo4j_graph.refresh_schema()
            schema_end = time.time()
            schema_time = schema_end - schema_start
            print(f"[TIMING] Schema refresh took {schema_time:.3f}s")
            
            # Cache the schema persistently
            set_schema_cache({'schema': neo4j_graph.schema}, current_time)
            print(f"[DEBUG] Schema cached for future use")
        
        # Import PROMPTS_CONFIG - will be updated when we extract config
        # For now, import from main.py location
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from main import PROMPTS_CONFIG
        except ImportError:
            # Fallback template if PROMPTS_CONFIG not available
            CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types and properties that are not provided.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Question: {query}"""
            PROMPTS_CONFIG = {
                'cypher_generation': {
                    'template': CYPHER_GENERATION_TEMPLATE
                }
            }
        
        # Create the chain - use config-based template
        CYPHER_GENERATION_TEMPLATE = PROMPTS_CONFIG['cypher_generation']['template']

        cypher_prompt = PromptTemplate(
            input_variables=["schema", "query"], 
            template=CYPHER_GENERATION_TEMPLATE
        )

        chain_create_start = time.time()
        _cypher_chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=neo4j_graph,
            cypher_prompt=cypher_prompt,
            verbose=False,
            return_intermediate_steps=False,
            use_function_response=False,
            allow_dangerous_requests=True  # Acknowledge that we understand the risks and have scoped database permissions appropriately
        )
        chain_create_end = time.time()
        print(f"[TIMING] Cypher chain creation took {chain_create_end - chain_create_start:.3f}s")
        
        chain_setup_end = time.time()
        print(f"[TIMING] Complete Cypher chain setup took {chain_setup_end - chain_setup_start:.3f}s")
    else:
        print(f"[TIMING] Reusing existing Cypher chain (pooling active)")
    
    return _cypher_chain, get_neo4j_connection()


def setup_cypher_chain():
    """Legacy function - now uses connection pooling"""
    setup_start = time.time()
    print(f"[TIMING] setup_cypher_chain started (with pooling)")
    
    cypher_chain, neo4j_graph = get_cypher_chain()
    
    setup_end = time.time()
    print(f"[TIMING] setup_cypher_chain completed in {setup_end - setup_start:.3f}s")
    
    return cypher_chain, neo4j_graph 
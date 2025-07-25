"""
Knowledge Graph service module for Restaurant Graph Agent.

Contains the kg_answer function and related knowledge graph operations.
Separated from main.py to avoid circular imports with tools.
"""

import time
from typing import Dict, Any

# Import dependencies that kg_answer needs
from ..data.neo4j_client import get_neo4j_connection, get_cypher_chain, setup_cypher_chain
from ..data.openai_client import get_embedding_model
from ..data.connections import get_cached_query_result, cache_query_result


def kg_answer(query: str) -> Dict[str, Any]:
    """
    Knowledge Graph Answer function - provides both symbolic and semantic search results.
    
    This function combines:
    1. Symbolic search using Neo4j Cypher queries via LLM
    2. Semantic search using vector embeddings
    
    Args:
        query (str): User's query about the menu
        
    Returns:
        Dict[str, Any]: Contains symbolic_answer, semantic_answer, and formatted results
    """
    kg_start = time.time()
    print(f"[TIMING] kg_answer started for query: {query}")
    
    # Check cache first
    query_cache_key = f"kg_{query.lower().strip()}"
    cached_result = get_cached_query_result(query_cache_key)
    if cached_result:
        kg_end = time.time()
        print(f"[TIMING] kg_answer served from cache in {kg_end - kg_start:.3f}s")
        return cached_result
    
    cypher_chain = None
    neo4j_graph = None
    symbolic_response = "No symbolic answer available"
    vector_result = "No semantic answer available"
    
    try:
        setup_start = time.time()
        cypher_chain, neo4j_graph = get_cypher_chain()  # Now uses pooling
        setup_end = time.time()
        print(f"[TIMING] Cypher chain setup took {setup_end - setup_start:.3f}s")
        
        print(f"[DEBUG] Invoking cypher chain...")
        cypher_start = time.time()
        # Pass both schema and query as the prompt template expects both
        symbolic_response = cypher_chain.invoke({
            "schema": neo4j_graph.schema,
            "query": query
        })
        cypher_end = time.time()
        print(f"[TIMING] Cypher chain invoke took {cypher_end - cypher_start:.3f}s")
        
        # Handle both direct results and dict format
        if isinstance(symbolic_response, dict) and 'result' in symbolic_response:
            print(f"[DEBUG] Symbolic response result: {symbolic_response['result']}")
            actual_result = symbolic_response['result']
        elif isinstance(symbolic_response, list):
            print(f"[DEBUG] Symbolic response (direct list): {symbolic_response}")
            actual_result = symbolic_response
        else:
            print(f"[DEBUG] Symbolic response (other): {symbolic_response}")
            actual_result = symbolic_response
        
        # If we got actual results, use them instead of "I don't know"
        if isinstance(actual_result, list) and len(actual_result) > 0:
            symbolic_response = actual_result
            print(f"[DEBUG] Using direct Cypher results: {len(actual_result)} items found")
        elif isinstance(actual_result, str) and "don't know" in actual_result.lower():
            print(f"[DEBUG] LLM returned 'don't know', but we may have found data")
        
    except Exception as e:
        print(f"[DEBUG] Error in symbolic query: {e}")
        # Let's try a direct Cypher query as fallback
        if neo4j_graph:
            try:
                print(f"[DEBUG] Trying direct Cypher query as fallback...")
                # Extract search term from query - look for ingredient mentions
                search_term = query.lower().replace("items", "").replace("dishes", "").strip()
                direct_cypher = f"""
                MATCH (d:Dish)-[:CONTAINS]->(i:Ingredient) 
                WHERE toLower(i.name) CONTAINS toLower('{search_term}')
                RETURN d.name, d.description, d.price, d.prepTimeMin, d.isSignature, d.vegClass
                LIMIT 10
                """
                fallback_start = time.time()
                symbolic_response = neo4j_graph.query(direct_cypher)
                fallback_end = time.time()
                print(f"[TIMING] Direct fallback query took {fallback_end - fallback_start:.3f}s")
                print(f"[DEBUG] Direct query result: {symbolic_response}")
            except Exception as direct_e:
                print(f"[DEBUG] Direct query also failed: {direct_e}")
                symbolic_response = f"Error in both chain and direct query: {e}"
        else:
            symbolic_response = f"Error generating symbolic answer: {e}"
    
    # Handle semantic search with unified embedding index
    if neo4j_graph:
        try:
            vector_start = time.time()
            print(f"[TIMING] Starting vector search...")
            
            embeddings_init_start = time.time()
            embeddings = get_embedding_model()  # Now uses pooling
            embeddings_init_end = time.time()
            print(f"[TIMING] Embeddings initialization took {embeddings_init_end - embeddings_init_start:.3f}s")
            
            embedding_start = time.time()
            query_embedding = embeddings.embed_query(query)
            embedding_end = time.time()
            print(f"[TIMING] Query embedding generation took {embedding_end - embedding_start:.3f}s")
            
            # Use unified menu_embed index for semantic search
            unified_cypher = """
            WITH $embedding AS queryEmbedding
            CALL db.index.vector.queryNodes('menu_embed', $k, queryEmbedding)
            YIELD node, score
            WHERE score > 0.65
            WITH node, score,
                 CASE 
                     WHEN 'Dish' IN labels(node) THEN node
                     WHEN 'Ingredient' IN labels(node) THEN [(node)<-[:CONTAINS]-(d:Dish) | d][0]
                     WHEN 'Category' IN labels(node) THEN [(node)<-[:IN_CATEGORY]-(d:Dish) | d][0]
                     WHEN 'Cuisine' IN labels(node) THEN [(node)<-[:OF_CUISINE]-(d:Dish) | d][0]
                     ELSE null
                 END as dish,
                 CASE 
                     WHEN 'Dish' IN labels(node) THEN 'dish'
                     WHEN 'Ingredient' IN labels(node) THEN 'ingredient'
                     WHEN 'Category' IN labels(node) THEN 'category'
                     WHEN 'Cuisine' IN labels(node) THEN 'cuisine'
                     ELSE 'unknown'
                 END as search_type
            WHERE dish IS NOT NULL
            WITH DISTINCT dish AS d, max(score) as score, search_type
            OPTIONAL MATCH (d)-[:IN_CATEGORY]->(c:Category)
            RETURN d.name AS name, d.description AS description, d.price AS price,
                   c.name AS category, score, search_type
            ORDER BY score DESC
            """
            
            vector_query_start = time.time()
            vector_result = neo4j_graph.query(unified_cypher, params={"embedding": query_embedding, "k": 15})
            vector_query_end = time.time()
            print(f"[TIMING] Vector query execution took {vector_query_end - vector_query_start:.3f}s")
            
            processing_start = time.time()
            # Remove duplicates and sort by score
            unique_results = {}
            for item in vector_result:
                dish_name = item['name']
                if dish_name not in unique_results or item['score'] > unique_results[dish_name]['score']:
                    unique_results[dish_name] = item
            
            vector_result = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]
            processing_end = time.time()
            print(f"[TIMING] Vector result processing took {processing_end - processing_start:.3f}s")
            
            vector_end = time.time()
            print(f"[TIMING] Total vector search took {vector_end - vector_start:.3f}s")
            print(f"[DEBUG] Vector result using unified embedding: {len(vector_result)} matches")
            
        except Exception as e:
            print(f"[DEBUG] Error in semantic query: {e}")
            vector_result = f"Error in semantic search: {e}"

    result = {
        "symbolic_answer": symbolic_response,
        "semantic_answer": vector_result,
        "formatted": f"SYMBOLIC ANSWER:\n{symbolic_response}\n\nSEMANTIC ANSWER:\n{vector_result}"
    }
    
    # Cache the result
    cache_query_result(query_cache_key, result)
    
    kg_end = time.time()
    print(f"[TIMING] Total kg_answer took {kg_end - kg_start:.3f}s")

    return result 
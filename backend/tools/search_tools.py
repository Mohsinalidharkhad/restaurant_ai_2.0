"""
Search tools module for Restaurant Graph Agent.

Contains tools for semantic search, food term extraction, and debugging embeddings.
"""

import time
import json
from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ..data.neo4j_client import get_neo4j_connection
from ..data.openai_client import get_embedding_model

# Import PROMPTS_CONFIG - will be updated when we extract config
# For now, import from main.py location
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from main import PROMPTS_CONFIG
except ImportError:
    # Fallback if PROMPTS_CONFIG not available
    PROMPTS_CONFIG = {
        'food_extraction': {
            'instruction': '''Please extract food-related terms from this query and return ONLY a JSON object with these categories:
{{
  "dish_names": ["specific dish names mentioned"],
  "ingredients": ["ingredients or food items"],
  "categories": ["appetizer, main course, dessert, etc."],
  "dietary_preferences": ["vegetarian, vegan, etc."],
  "cuisine_types": ["north indian, chinese, etc."],
  "descriptive_terms": ["spicy, creamy, mild, etc."]
}}

Query: {query}

Return only valid JSON, no explanations:'''
        }
    }


@tool
def extract_food_terms(query: str) -> Dict[str, Any]:
    """
    Use LLM to intelligently extract food-related terms optimized for unified embedding search.
    
    Args:
        query: User's question about food/menu
        
    Returns:
        Dict with extracted terms categorized for efficient unified index search
    """
    extract_start = time.time()
    print(f"[TIMING] extract_food_terms started with query: {query}")
    
    try:
        llm_init_start = time.time()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        llm_init_end = time.time()
        print(f"[TIMING] LLM initialization took {llm_init_end - llm_init_start:.3f}s")
        
        # Use config-based extraction prompt
        extraction_prompt = PROMPTS_CONFIG['food_extraction']['instruction'].format(query=query)
        
        llm_call_start = time.time()
        response = llm.invoke(extraction_prompt)
        llm_call_end = time.time()
        print(f"[TIMING] LLM extraction call took {llm_call_end - llm_call_start:.3f}s")
        
        # Parse JSON response with improved error handling
        try:
            parsing_start = time.time()
            content = response.content.strip()
            print(f"[DEBUG] Raw LLM response: {content[:200]}...")
            
            # Remove markdown if present
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            # Try to find JSON in the response
            if not content.startswith('{'):
                # Look for JSON in the response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end > start:
                    content = content[start:end]
            
            extracted_data = json.loads(content)
            parsing_end = time.time()
            print(f"[TIMING] JSON parsing took {parsing_end - parsing_start:.3f}s")
            print(f"[DEBUG] LLM extracted terms: {extracted_data}")
            
            extract_end = time.time()
            print(f"[TIMING] Total extract_food_terms took {extract_end - extract_start:.3f}s")
            
            return {
                "success": True,
                "extracted_terms": extracted_data,
                "primary_search_terms": extracted_data.get("dish_names", []) + extracted_data.get("ingredients", [])[:3],
                "secondary_search_terms": extracted_data.get("categories", []) + extracted_data.get("descriptive_terms", [])[:2]
            }
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing failed: {e}, content: {content[:100]}")
            
            # Enhanced fallback extraction
            fallback_start = time.time()
            words = query.lower().replace(',', ' ').replace('please', '').split()
            food_terms = [word for word in words if len(word) > 2 and word not in ['have', 'what', 'show', 'tell', 'menu', 'dish', 'food', 'items', 'please']]
            
            # Smart categorization based on common patterns
            fallback_data = {
                "dish_names": [],
                "ingredients": [],
                "categories": [],
                "dietary_preferences": [],
                "cuisine_types": [],
                "descriptive_terms": []
            }
            
            # Common ingredient patterns
            common_ingredients = ['paneer', 'chicken', 'mutton', 'lamb', 'rice', 'dal', 'naan', 'roti']
            # Common descriptive terms
            common_descriptors = ['creamy', 'spicy', 'mild', 'tangy', 'sweet', 'hot', 'cold']
            # Common categories
            common_categories = ['appetizer', 'starter', 'main', 'dessert', 'snack', 'beverage']
            
            for term in food_terms:
                if term in common_ingredients:
                    fallback_data["ingredients"].append(term)
                elif term in common_descriptors:
                    fallback_data["descriptive_terms"].append(term)
                elif term in common_categories:
                    fallback_data["categories"].append(term)
                else:
                    # Put unknown terms in ingredients as they're most likely to match
                    fallback_data["ingredients"].append(term)
            
            fallback_end = time.time()
            print(f"[TIMING] Fallback extraction took {fallback_end - fallback_start:.3f}s")
            print(f"[DEBUG] Fallback extraction: {fallback_data}")
            
            extract_end = time.time()
            print(f"[TIMING] Total extract_food_terms (with fallback) took {extract_end - extract_start:.3f}s")
            
            return {
                "success": False,
                "extracted_terms": fallback_data,
                "primary_search_terms": fallback_data["ingredients"] + fallback_data["dish_names"],
                "secondary_search_terms": fallback_data["descriptive_terms"] + fallback_data["categories"],
                "error": "Used enhanced fallback extraction"
            }
            
    except Exception as e:
        extract_end = time.time()
        print(f"[TIMING] extract_food_terms failed after {extract_end - extract_start:.3f}s")
        print(f"[DEBUG] Error in extract_food_terms: {e}")
        return {
            "success": False,
            "extracted_terms": {},
            "primary_search_terms": [],
            "secondary_search_terms": [],
            "error": str(e)
        }


@tool 
def debug_embedding_indexes() -> Dict[str, Any]:
    """
    Check if the unified embedding vector index exists in Neo4j.
    Use this to debug why embedding searches might be failing.
    """
    debug_start = time.time()
    print(f"[TIMING] debug_embedding_indexes started")
    
    try:
        # Use pooled Neo4j connection
        connection_start = time.time()
        neo4j_graph = get_neo4j_connection()  # Now uses pooling
        connection_end = time.time()
        print(f"[TIMING] Neo4j connection for debug took {connection_end - connection_start:.3f}s")
        
        # Check for vector indexes
        index_check_start = time.time()
        index_check_query = "SHOW INDEXES"
        indexes = neo4j_graph.query(index_check_query)
        index_check_end = time.time()
        print(f"[TIMING] Index check query took {index_check_end - index_check_start:.3f}s")
        
        vector_indexes = [idx for idx in indexes if 'vector' in str(idx).lower()]
        
        # Check for the unified menu_embed index
        menu_embed_found = any('menu_embed' in str(idx) for idx in indexes)
        menu_search_found = any('menu_search' in str(idx) for idx in indexes)
        
        # Check if embeddings exist on nodes by label
        embedding_counts = {}
        searchable_labels = ['Dish', 'Ingredient', 'Category', 'Cuisine']
        
        for label in searchable_labels:
            count_query = f"MATCH (n:{label}:Searchable) WHERE n.embedding IS NOT NULL RETURN count(n) as count"
            try:
                count_start = time.time()
                result = neo4j_graph.query(count_query)
                count_end = time.time()
                print(f"[TIMING] Embedding count for {label} took {count_end - count_start:.3f}s")
                embedding_counts[f"{label}_with_embeddings"] = result[0]['count'] if result else 0
            except Exception as e:
                embedding_counts[f"{label}_with_embeddings"] = f"Error: {e}"
        
        # Check total searchable nodes
        try:
            total_start = time.time()
            total_searchable = neo4j_graph.query("MATCH (n:Searchable) RETURN count(n) as count")
            total_end = time.time()
            print(f"[TIMING] Total searchable count took {total_end - total_start:.3f}s")
            total_count = total_searchable[0]['count'] if total_searchable else 0
        except Exception as e:
            total_count = f"Error: {e}"
        
        debug_end = time.time()
        print(f"[TIMING] Total debug_embedding_indexes took {debug_end - debug_start:.3f}s")
        
        return {
            "success": True,
            "total_indexes": len(indexes),
            "vector_indexes_found": len(vector_indexes),
            "menu_embed_index_found": menu_embed_found,
            "menu_search_index_found": menu_search_found,
            "embedding_counts": embedding_counts,
            "total_searchable_nodes": total_count,
            "all_indexes": [str(idx) for idx in indexes],
            "summary": f"Menu embed index: {'✓' if menu_embed_found else '✗'}, Menu search index: {'✓' if menu_search_found else '✗'}"
        }
        
    except Exception as e:
        debug_end = time.time()
        print(f"[TIMING] debug_embedding_indexes failed after {debug_end - debug_start:.3f}s")
        return {
            "success": False,
            "error": str(e),
            "summary": "Failed to check indexes"
        }


@tool
def check_semantic_similarity(extracted_terms: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced semantic similarity search using unified embedding index.
    Uses the single menu_embed index for all node types (:Searchable).
    
    Args:
        extracted_terms: Dictionary containing categorized terms from extract_food_terms
        
    Returns:
        Dict with comprehensive match results and confidence scores
    """
    semantic_start = time.time()
    print(f"[TIMING] check_semantic_similarity started with terms: {extracted_terms}")
    
    try:
        # Use pooled Neo4j connection
        connection_start = time.time()
        neo4j_graph = get_neo4j_connection()  # Now uses pooling
        connection_end = time.time()
        print(f"[TIMING] Neo4j connection for semantic search took {connection_end - connection_start:.3f}s")
        
        embeddings_init_start = time.time()
        embeddings = get_embedding_model()  # Now uses pooling
        embeddings_init_end = time.time()
        print(f"[TIMING] Embeddings initialization took {embeddings_init_end - embeddings_init_start:.3f}s")
        
        all_results = []
        match_confidence = 0.0
        
        # STEP 1: Search by dish names (highest priority)
        dish_names = extracted_terms.get("dish_names", [])
        if dish_names:
            dish_search_start = time.time()
            print(f"[TIMING] Searching for dish names: {dish_names}")
            for dish_name in dish_names[:2]:  # Limit to top 2 dish names
                try:
                    embedding_start = time.time()
                    query_embedding = embeddings.embed_query(dish_name)
                    embedding_end = time.time()
                    print(f"[TIMING] Dish embedding for '{dish_name}' took {embedding_end - embedding_start:.3f}s")
                    
                    dish_name_cypher = """
                    WITH $embedding AS queryEmbedding
                    CALL db.index.vector.queryNodes('menu_embed', $k, queryEmbedding)
                    YIELD node, score
                    WHERE score > 0.75 AND 'Dish' IN labels(node)
                    WITH node AS d, score
                    OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
                    OPTIONAL MATCH (d)-[:IN_CATEGORY]->(c:Category)
                    WITH d, score, collect(DISTINCT i.name) as ingredients, collect(DISTINCT c.name) as categories
                    RETURN 'dish_embedding' as match_type,
                           d.name as name,
                           d.description as description,
                           d.price as price,
                           d.prepTimeMin as prep_time,
                           d.isSignature as is_signature,
                           d.vegClass as veg_class,
                           ingredients,
                           categories,
                           score,
                           $search_term as matched_term
                    ORDER BY score DESC
                    """
                    
                    query_start = time.time()
                    results = neo4j_graph.query(dish_name_cypher, 
                                              params={"embedding": query_embedding, "k": 8, "search_term": dish_name})
                    query_end = time.time()
                    print(f"[TIMING] Dish name query for '{dish_name}' took {query_end - query_start:.3f}s")
                    
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.9)
                        print(f"[DEBUG] Found {len(results)} dish embedding matches for '{dish_name}'")
                
                except Exception as e:
                    print(f"[DEBUG] Error in dish embedding search for '{dish_name}': {e}")
            
            dish_search_end = time.time()
            print(f"[TIMING] Total dish name search took {dish_search_end - dish_search_start:.3f}s")
        
        # STEP 2: Search by ingredients
        ingredients = extracted_terms.get("ingredients", [])
        if ingredients and len(all_results) < 8:  # Only if we need more results
            ingredient_search_start = time.time()
            print(f"[TIMING] Searching for ingredients: {ingredients}")
            for ingredient in ingredients[:3]:  # Limit to top 3 ingredients
                try:
                    embedding_start = time.time()
                    query_embedding = embeddings.embed_query(ingredient)
                    embedding_end = time.time()
                    print(f"[TIMING] Ingredient embedding for '{ingredient}' took {embedding_end - embedding_start:.3f}s")
                    
                    ingredient_cypher = """
                    WITH $embedding AS queryEmbedding
                    CALL db.index.vector.queryNodes('menu_embed', $k, queryEmbedding)
                    YIELD node, score
                    WHERE score > 0.70 AND 'Ingredient' IN labels(node)
                    WITH node AS i, score
                    MATCH (i)<-[:CONTAINS]-(d:Dish)
                    OPTIONAL MATCH (d)-[:CONTAINS]->(all_ingredients:Ingredient)
                    OPTIONAL MATCH (d)-[:IN_CATEGORY]->(c:Category)
                    WITH d, score, collect(DISTINCT all_ingredients.name) as ingredients, collect(DISTINCT c.name) as categories
                    RETURN 'ingredient_embedding' as match_type,
                           d.name as name,
                           d.description as description,
                           d.price as price,
                           d.prepTimeMin as prep_time,
                           d.isSignature as is_signature,
                           d.vegClass as veg_class,
                           ingredients,
                           categories,
                           score * 0.8 as score,
                           $search_term as matched_term
                    ORDER BY score DESC
                    """
                    
                    query_start = time.time()
                    results = neo4j_graph.query(ingredient_cypher, 
                                              params={"embedding": query_embedding, "k": 6, "search_term": ingredient})
                    query_end = time.time()
                    print(f"[TIMING] Ingredient query for '{ingredient}' took {query_end - query_start:.3f}s")
                    
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.75)
                        print(f"[DEBUG] Found {len(results)} ingredient embedding matches for '{ingredient}'")
                
                except Exception as e:
                    print(f"[DEBUG] Error in ingredient embedding search for '{ingredient}': {e}")
            
            ingredient_search_end = time.time()
            print(f"[TIMING] Total ingredient search took {ingredient_search_end - ingredient_search_start:.3f}s")
        
        # STEP 3: Search by categories
        categories = extracted_terms.get("categories", [])
        if categories and len(all_results) < 8:
            category_search_start = time.time()
            print(f"[TIMING] Searching for categories: {categories}")
            for category in categories[:2]:
                try:
                    embedding_start = time.time()
                    query_embedding = embeddings.embed_query(category)
                    embedding_end = time.time()
                    print(f"[TIMING] Category embedding for '{category}' took {embedding_end - embedding_start:.3f}s")
                    
                    category_cypher = """
                    WITH $embedding AS queryEmbedding
                    CALL db.index.vector.queryNodes('menu_embed', $k, queryEmbedding)
                    YIELD node, score
                    WHERE score > 0.65 AND 'Category' IN labels(node)
                    WITH node AS c, score
                    MATCH (c)<-[:IN_CATEGORY]-(d:Dish)
                    OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
                    WITH d, score, collect(DISTINCT i.name) as ingredients, [c.name] as categories
                    RETURN 'category_embedding' as match_type,
                           d.name as name,
                           d.description as description,
                           d.price as price,
                           d.prepTimeMin as prep_time,
                           d.isSignature as is_signature,
                           d.vegClass as veg_class,
                           ingredients,
                           categories,
                           score * 0.7 as score,
                           $search_term as matched_term
                    ORDER BY score DESC
                    """
                    
                    query_start = time.time()
                    results = neo4j_graph.query(category_cypher, 
                                              params={"embedding": query_embedding, "k": 5, "search_term": category})
                    query_end = time.time()
                    print(f"[TIMING] Category query for '{category}' took {query_end - query_start:.3f}s")
                    
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.65)
                        print(f"[DEBUG] Found {len(results)} category embedding matches for '{category}'")
                
                except Exception as e:
                    print(f"[DEBUG] Error in category embedding search for '{category}': {e}")
            
            category_search_end = time.time()
            print(f"[TIMING] Total category search took {category_search_end - category_search_start:.3f}s")
        
        # STEP 4: Mixed search for descriptive terms and general queries
        descriptive_terms = extracted_terms.get("descriptive_terms", []) + extracted_terms.get("dietary_preferences", [])
        if descriptive_terms and len(all_results) < 10:
            mixed_search_start = time.time()
            print(f"[TIMING] Searching for descriptive terms: {descriptive_terms}")
            combined_description = " ".join(descriptive_terms[:3])
            try:
                embedding_start = time.time()
                query_embedding = embeddings.embed_query(combined_description)
                embedding_end = time.time()
                print(f"[TIMING] Mixed embedding took {embedding_end - embedding_start:.3f}s")
                
                mixed_cypher = """
                WITH $embedding AS queryEmbedding
                CALL db.index.vector.queryNodes('menu_embed', $k, queryEmbedding)
                YIELD node, score
                WHERE score > 0.60
                WITH node, score,
                     CASE 
                         WHEN 'Dish' IN labels(node) THEN node
                         WHEN 'Ingredient' IN labels(node) THEN [(node)<-[:CONTAINS]-(d:Dish) | d][0]
                         WHEN 'Category' IN labels(node) THEN [(node)<-[:IN_CATEGORY]-(d:Dish) | d][0]
                         ELSE null
                     END as dish
                WHERE dish IS NOT NULL
                WITH DISTINCT dish AS d, max(score) as score
                OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
                OPTIONAL MATCH (d)-[:IN_CATEGORY]->(c:Category)
                WITH d, score, collect(DISTINCT i.name) as ingredients, collect(DISTINCT c.name) as categories
                RETURN 'mixed_embedding' as match_type,
                       d.name as name,
                       d.description as description,
                       d.price as price,
                       d.prepTimeMin as prep_time,
                       d.isSignature as is_signature,
                       d.vegClass as veg_class,
                       ingredients,
                       categories,
                       score * 0.6 as score,
                       $search_term as matched_term
                ORDER BY score DESC
                """
                
                query_start = time.time()
                results = neo4j_graph.query(mixed_cypher, 
                                          params={"embedding": query_embedding, "k": 12, "search_term": combined_description})
                query_end = time.time()
                print(f"[TIMING] Mixed query took {query_end - query_start:.3f}s")
                
                if results:
                    all_results.extend(results)
                    match_confidence = max(match_confidence, 0.6)
                    print(f"[DEBUG] Found {len(results)} mixed embedding matches")
            
            except Exception as e:
                print(f"[DEBUG] Error in mixed embedding search: {e}")
            
            mixed_search_end = time.time()
            print(f"[TIMING] Total mixed search took {mixed_search_end - mixed_search_start:.3f}s")
        
        # STEP 5: Final fallback to traditional string matching
        if not all_results:
            fallback_start = time.time()
            print(f"[TIMING] No embedding matches found, trying traditional string matching...")
            all_terms = []
            for category, terms in extracted_terms.items():
                if isinstance(terms, list):
                    all_terms.extend(terms)
            
            for term in all_terms[:3]:
                escaped_term = term.replace("'", "\\'")
                fallback_query = f"""
                MATCH (d:Dish)
                WHERE toLower(d.name) CONTAINS toLower('{escaped_term}')
                   OR toLower(d.description) CONTAINS toLower('{escaped_term}')
                OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
                OPTIONAL MATCH (d)-[:IN_CATEGORY]->(c:Category)
                WITH d, collect(DISTINCT i.name) as ingredients, collect(DISTINCT c.name) as categories
                RETURN 'string_match' as match_type,
                       d.name as name,
                       d.description as description,
                       d.price as price,
                       d.prepTimeMin as prep_time,
                       d.isSignature as is_signature,
                       d.vegClass as veg_class,
                       ingredients,
                       categories,
                       0.3 as score,
                       '{term}' as matched_term
                LIMIT 5
                """
                
                try:
                    query_start = time.time()
                    results = neo4j_graph.query(fallback_query)
                    query_end = time.time()
                    print(f"[TIMING] String fallback query for '{term}' took {query_end - query_start:.3f}s")
                    
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.3)
                        print(f"[DEBUG] Found {len(results)} string matches for '{term}'")
                except Exception as e:
                    print(f"[DEBUG] Error in string matching for '{term}': {e}")
            
            fallback_end = time.time()
            print(f"[TIMING] Total fallback search took {fallback_end - fallback_start:.3f}s")
        
        # Remove duplicates and sort by score
        dedup_start = time.time()
        unique_results = {}
        for item in all_results:
            dish_name = item['name']
            if dish_name not in unique_results or item['score'] > unique_results[dish_name]['score']:
                unique_results[dish_name] = item
        
        final_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]
        dedup_end = time.time()
        print(f"[TIMING] Deduplication and sorting took {dedup_end - dedup_start:.3f}s")
        
        # Determine match quality based on embedding types and scores
        has_high_confidence = any(item['score'] > 0.8 for item in final_results)
        has_dish_matches = any(item['match_type'] == 'dish_embedding' for item in final_results)
        has_ingredient_matches = any(item['match_type'] == 'ingredient_embedding' for item in final_results)
        
        result = {
            "matches": {
                "search_terms": extracted_terms,
                "results": final_results
            },
            "found_exact_match": has_dish_matches and has_high_confidence,
            "similar_items": final_results,
            "match_confidence": match_confidence,
            "match_types_found": list(set(item['match_type'] for item in final_results)),
            "total_results": len(final_results),
            "search_summary": f"Found {len(final_results)} matches using unified embedding index with {match_confidence:.2f} confidence"
        }
        
        semantic_end = time.time()
        print(f"[TIMING] Total check_semantic_similarity took {semantic_end - semantic_start:.3f}s")
        print(f"[DEBUG] Unified semantic search result: {len(final_results)} matches, confidence: {match_confidence:.2f}")
        return result
        
    except Exception as e:
        semantic_end = time.time()
        print(f"[TIMING] check_semantic_similarity failed after {semantic_end - semantic_start:.3f}s")
        print(f"[DEBUG] Error in check_semantic_similarity: {e}")
        return {
            "matches": {},
            "found_exact_match": False,
            "similar_items": [],
            "error": str(e),
            "match_confidence": 0.0,
            "total_results": 0
        } 
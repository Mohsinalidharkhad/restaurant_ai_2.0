"""
Menu tools module for Restaurant Graph Agent.

Contains tools for menu recommendations and detailed dish information.
"""

import time
from typing import Dict, Any, List
from langchain_core.tools import tool

from ..data.neo4j_client import get_neo4j_connection

# Import kg_answer from knowledge graph service to avoid circular imports
from ..services.knowledge_graph import kg_answer

print("[DEBUG] Successfully imported kg_answer from knowledge_graph service")

# Simple system initialization check - inline to avoid complex imports
def ensure_system_initialized():
    """Simple system initialization check"""
    print("[DEBUG] System initialization check - using simplified approach")


@tool
def get_recommendations(query: str) -> Dict[str, Any]:
    """
    Handle FAQ questions about menu, dishes, recommendations, ingredients, and dietary information.
    Use this for questions about what's available, dish details, recommendations, or general restaurant info.
    
    Args:
        query: The customer's question about menu/food
    
    Returns:
        Dict with response details including recommendations
    """
    recommendations_start = time.time()
    print(f"[TIMING] get_recommendations started with query: {query}")
    
    # Ensure system is initialized before processing
    ensure_system_initialized()
    
    try:
        kg_call_start = time.time()
        result = kg_answer(query)
        kg_call_end = time.time()
        print(f"[TIMING] kg_answer call took {kg_call_end - kg_call_start:.3f}s")
        
        # Filter out schema from debug output - only show semantic_answer and formatted result
        filtering_start = time.time()
        filtered_result = {
            "semantic_answer": result.get("semantic_answer", "No semantic answer"),
            "formatted": result.get("formatted", "No formatted answer available")
        }
        filtering_end = time.time()
        print(f"[TIMING] Result filtering took {filtering_end - filtering_start:.3f}s")
        
        print(f"[DEBUG] kg_answer returned: {filtered_result}")
        
        recommendations_end = time.time()
        print(f"[TIMING] Total get_recommendations took {recommendations_end - recommendations_start:.3f}s")
        
        return {
            "success": True,
            "response": result,
            # "message": result.get("message", ""),
        }
    except Exception as e:
        recommendations_end = time.time()
        print(f"[TIMING] get_recommendations failed after {recommendations_end - recommendations_start:.3f}s")
        print(f"[DEBUG] Exception in get_recommendations: {e}")
        return {
            "success": False,
            "error": str(e),
            # "message": "I'm having trouble finding menu information. Please try again."
        }


@tool
def get_detailed_dish_info(dish_names: List[str]) -> Dict[str, Any]:
    """
    Get comprehensive information about specific dishes from embedding search results.
    Use this when you have specific dish names from check_semantic_similarity results.
    
    Args:
        dish_names: List of dish names to get detailed information about
        
    Returns:
        Dict with detailed dish information including ingredients, prices, prep times, etc.
    """
    detail_start = time.time()
    print(f"[TIMING] get_detailed_dish_info started for dishes: {dish_names}")
    
    if not dish_names:
        return {
            "success": False,
            "error": "No dish names provided",
            "dishes": []
        }
    
    try:
        # Use pooled Neo4j connection
        connection_start = time.time()
        neo4j_graph = get_neo4j_connection()  # Now uses pooling
        connection_end = time.time()
        print(f"[TIMING] Neo4j connection for detailed info took {connection_end - connection_start:.3f}s")
        
        detailed_info = []
        
        for dish_name in dish_names[:5]:  # Limit to 5 dishes to avoid overwhelming
            escaped_name = dish_name.replace("'", "\\'")
            dish_info_query = f"""
            MATCH (d:Dish {{name: '{escaped_name}'}})
            OPTIONAL MATCH (d)-[:CONTAINS]->(i:Ingredient)
            OPTIONAL MATCH (d)-[:IN_CATEGORY]->(c:Category)
            OPTIONAL MATCH (d)-[:OF_CUISINE]->(cu:Cuisine)
            OPTIONAL MATCH (d)-[:HAS_ALLERGEN]->(a:Allergen)
            OPTIONAL MATCH (d)-[:AVAILABLE_DURING]->(at:AvailabilityTime)
            OPTIONAL MATCH (d)-[:PAIR_WITH]->(paired:Dish)
            WITH d, 
                 collect(DISTINCT i.name) as ingredients,
                 collect(DISTINCT c.name) as categories,
                 collect(DISTINCT cu.name) as cuisines,
                 collect(DISTINCT a.name) as allergens,
                 collect(DISTINCT at.name) as availability,
                 collect(DISTINCT paired.name) as pairings
            RETURN d.name as name,
                   d.description as description,
                   d.price as price,
                   d.currency as currency,
                   d.prepTimeMin as prep_time,
                   d.spiceLevel as spice_level,
                   d.vegClass as veg_class,
                   d.isSignature as is_signature,
                   d.available as available,
                   ingredients,
                   categories,
                   cuisines,
                   allergens,
                   availability,
                   pairings
            """
            
            try:
                query_start = time.time()
                results = neo4j_graph.query(dish_info_query)
                query_end = time.time()
                print(f"[TIMING] Detailed query for '{dish_name}' took {query_end - query_start:.3f}s")
                
                if results:
                    dish_info = results[0]  # Should be only one result per dish
                    detailed_info.append({
                        "name": dish_info["name"],
                        "description": dish_info["description"],
                        "price": f"{dish_info['currency']} {dish_info['price']}" if dish_info['currency'] and dish_info['price'] else "Price not available",
                        "prep_time": f"{dish_info['prep_time']} minutes" if dish_info['prep_time'] else "Time varies",
                        "spice_level": f"Spice level: {dish_info['spice_level']}/5" if dish_info['spice_level'] is not None else "Spice level not specified",
                        "diet_type": dish_info["veg_class"] or "Not specified",
                        "is_signature": dish_info["is_signature"] or False,
                        "available": dish_info["available"] if dish_info["available"] is not None else True,
                        "ingredients": dish_info["ingredients"] or [],
                        "categories": dish_info["categories"] or [],
                        "cuisines": dish_info["cuisines"] or [],
                        "allergens": dish_info["allergens"] or [],
                        "availability_times": dish_info["availability"] or [],
                        "pairs_with": dish_info["pairings"] or []
                    })
                    print(f"[DEBUG] Retrieved detailed info for {dish_name}")
                else:
                    print(f"[DEBUG] No detailed info found for {dish_name}")
                    
            except Exception as e:
                print(f"[DEBUG] Error getting detailed info for '{dish_name}': {e}")
        
        detail_end = time.time()
        print(f"[TIMING] Total get_detailed_dish_info took {detail_end - detail_start:.3f}s")
        
        return {
            "success": True,
            "dishes": detailed_info,
            "total_dishes": len(detailed_info),
            "message": f"Retrieved detailed information for {len(detailed_info)} dishes"
        }
        
    except Exception as e:
        detail_end = time.time()
        print(f"[TIMING] get_detailed_dish_info failed after {detail_end - detail_start:.3f}s")
        print(f"[DEBUG] Error in get_detailed_dish_info: {e}")
        return {
            "success": False,
            "error": str(e),
            "dishes": []
        } 
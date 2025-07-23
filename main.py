# Registration Agent and Tools
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel
from langgraph.graph import END, StateGraph, START
from supabase import create_client, Client
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts.prompt import PromptTemplate
# Add logging configuration to suppress Neo4j warnings
import logging
logging.getLogger("neo4j").setLevel(logging.ERROR)

# Add new imports for NLP processing
import re
from typing import List
import time  # Add timing import

load_dotenv()
supabase= create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    is_registered: bool
    phone_number: str  # Added for phone number persistence across tool calls
    dialog_state: Annotated[
        list[
            Literal[
                "waiter_agent",
            ]
        ],
        update_dialog_stack,
    ]


builder = StateGraph(State)


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            
            # Always return the result if it has content, regardless of tool calls
            if result.content:
                return {"messages": [result]}
            
            # If no content but has tool calls, process normally
            if result.tool_calls:
                return {"messages": [result]}
            
            # If neither content nor tool calls, prompt for real output
            if not result.content and not result.tool_calls:
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": [result]}
        
# CompleteOrEscalate removed - no longer needed without separate registration agent


# Registration tools removed - replaced by check_user_registration and collect_registration in waiter agent






# Registration agent removed - waiter agent handles all registration through collect_registration() tool

# --- Tools for Waiter Agent ---
def place_order(order_details: dict) -> Dict[str, Any]:
    """Stub: Place a new order."""
    # TODO: Implement with Supabase SQL
    return {"success": True, "order_id": 123}

def get_order_status(order_id: int) -> Dict[str, Any]:
    """Stub: Get the status of an order."""
    # TODO: Implement with Supabase SQL
    return {"status": "preparing"}

def modify_order(order_id: int, modifications: dict) -> Dict[str, Any]:
    """Stub: Modify an existing order."""
    # TODO: Implement with Supabase SQL
    return {"success": True}

def print_bill(order_id: int) -> Dict[str, Any]:
    """Stub: Print or fetch the bill for an order."""
    # TODO: Implement with Supabase SQL
    return {"bill": 42.50}

def get_dish_detail(dish_id: str) -> Dict[str, Any]:
    """Stub: Get details for a dish (from Neo4j)."""
    # TODO: Implement with Neo4j
    return {"name": "Sample Dish", "ingredients": ["ingredient1", "ingredient2"]}

# Add connection pooling and caching
_neo4j_connection = None
_cypher_chain = None
_schema_cache = {}
_schema_cache_timestamp = 0
SCHEMA_CACHE_DURATION = 300  # 5 minutes in seconds

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
    global _cypher_chain, _schema_cache, _schema_cache_timestamp
    
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
        
        # Schema handling with persistent caching
        current_time = time.time()
        
        if _schema_cache and (current_time - _schema_cache_timestamp) < SCHEMA_CACHE_DURATION:
            print(f"[DEBUG] Using cached schema (age: {current_time - _schema_cache_timestamp:.1f}s)")
            neo4j_graph.schema = _schema_cache.get('schema', '')
            schema_time = 0.0
        else:
            print(f"[DEBUG] Refreshing schema...")
            schema_start = time.time()
            neo4j_graph.refresh_schema()
            schema_end = time.time()
            schema_time = schema_end - schema_start
            print(f"[TIMING] Schema refresh took {schema_time:.3f}s")
            
            # Cache the schema persistently
            _schema_cache = {'schema': neo4j_graph.schema}
            _schema_cache_timestamp = current_time
            print(f"[DEBUG] Schema cached for future use")
        
        # Create the chain
        CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types and properties that are not provided.
The user input might have some typos, so you need to handle that. Just to help you with typo correction, the queries will be related to the indian restaurant menu.
When the user asks for items based on an ingredient (e.g., "mango", "mutton", "paneer"), match `Dish` nodes connected to any `Ingredient` node such that `toLower(i.name)` CONTAINS the lowercase ingredient term (e.g., `MATCH (d:Dish)-[:CONTAINS]->(i:Ingredient) WHERE toLower(i.name) CONTAINS "mango"`).
Always perform string comparisons case-insensitively by wrapping both sides with `toLower()`. Prefer `CONTAINS` for partial matches over exact equality.
In indian english specific synonyms are used for some words, so you need to handle that.
- Use 'Appetizer' in the where clause to search for starters or Appetiser or chaat.
- Use 'Dessert' in the where clause to search for sweet or dessert.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not include embedding properties in your response. If you find any embedding properties, exclude them.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Do not use parameters in your queries - always use direct string literals.
Try to user ingredient name in the where clause instead of dish name.

Examples: Here are a few examples of generated Cypher statements for particular questions:
1. Do you have paneer based desserts?
MATCH (d:Dish)-[:CONTAINS]->(paneer_ingredient:Ingredient), 
      (d)-[:IN_CATEGORY]->(c:Category),
      (d)-[:CONTAINS]->(all_ingredients:Ingredient)
WHERE toLower(paneer_ingredient.name) CONTAINS 'paneer' 
AND c.name = 'Dessert'
RETURN d.name, 
       d.description, 
       d.price, 
       c.name as category,
       collect(DISTINCT all_ingredients.name) AS all_ingredients
2. Do you have Gulab Jamun?
MATCH (d:Dish {{name: "Gulab Jamun"}}) RETURN d.name, d.description, d.price, d.prep_time
3. What do you have for snacks?
MATCH (d:Dish)-[IN_CAEGORY]-> (c:Category) WHERE toLower(c.name) CONTAINS 'snack' RETURN d.name, d.description, d.price
4. Do you have paneer based items?
MATCH (d:Dish)-[:CONTAINS]->(i:Ingredient) WHERE lower(i.name) CONTAINS 'paneer' RETURN d.name, d.description, d.price, collect(i.name) AS ingredients;
5. does the Dal Makhani have any cream or ghee in it?
MATCH (d:Dish {{ name : 'Dal Makhani'}})-[:CONTAINS]->(i:Ingredient)  RETURN d.name, d.description, d.price, collect(i.name) AS ingredients;


The question is:
{query}"""

        cypher_prompt = PromptTemplate(
            input_variables=["schema", "query"], 
            template=CYPHER_GENERATION_TEMPLATE
        )

        chain_create_start = time.time()
        _cypher_chain = GraphCypherQAChain.from_llm(
            llm,
            graph=neo4j_graph,
            verbose=True,
            validate_cypher=True,
            cypher_prompt=cypher_prompt,
            return_direct=True,
            top_k=20,
            input_key="query",
            allow_dangerous_requests=True,
        )
        chain_create_end = time.time()
        print(f"[TIMING] Cypher chain creation took {chain_create_end - chain_create_start:.3f}s")
        
        chain_setup_end = time.time()
        print(f"[TIMING] Total new Cypher chain setup took {chain_setup_end - chain_setup_start:.3f}s")
    else:
        print(f"[TIMING] Reusing existing Cypher chain (chain pooling active)")
    
    return _cypher_chain, get_neo4j_connection()

def setup_cypher_chain():
    """Legacy function - now uses connection pooling"""
    setup_start = time.time()
    print(f"[TIMING] setup_cypher_chain started (with pooling)")
    
    cypher_chain, neo4j_graph = get_cypher_chain()
    
    setup_end = time.time()
    print(f"[TIMING] setup_cypher_chain completed in {setup_end - setup_start:.3f}s")
    
    return cypher_chain, neo4j_graph

# Add embedding model pooling and query caching
_embedding_model = None
_query_cache = {}
_query_cache_timestamps = {}
QUERY_CACHE_DURATION = 180  # 3 minutes for query results

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

def get_cached_query_result(query_key):
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

def cache_query_result(query_key, result):
    """Cache query result for future use"""
    global _query_cache, _query_cache_timestamps
    
    _query_cache[query_key] = result
    _query_cache_timestamps[query_key] = time.time()
    print(f"[DEBUG] Cached query result for key: {query_key[:50]}...")

def kg_answer(query: str) -> Dict[str, Any]:
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
        
        extraction_prompt = f"""
        Analyze this restaurant customer query and extract food-related terms for semantic search.
        Focus on terms that would match our menu items (dishes, ingredients, categories).
        
Return ONLY valid JSON in this exact format (no markdown, no comments):
{{
            "dish_names": ["Butter Chicken", "Saag Paneer"],      // Exact dish names mentioned
            "ingredients": ["paneer", "chicken", "spinach"],      // Key ingredients
            "categories": ["dessert", "appetizer", "main course"], // Food categories 
            "dietary_preferences": ["vegetarian", "spicy", "mild"], // Diet/spice preferences
            "cuisine_types": ["North Indian", "Chinese"],         // Cuisine styles
            "descriptive_terms": ["creamy", "tangy", "sweet"]     // Taste/texture descriptors
}}

Example for "creamy paneer items":
{{
    "dish_names": [],
    "ingredients": ["paneer"],
    "categories": [],
    "dietary_preferences": [],
    "cuisine_types": [],
    "descriptive_terms": ["creamy"]
}}

        Rules:
        - Prioritize exact dish names if mentioned
        - Extract key ingredients (paneer, chicken, etc.)
        - Include food categories (appetizer, main course, dessert, etc.)
        - Capture dietary preferences (veg, non-veg, spicy levels)
        - Handle variations and typos intelligently
        - Focus on terms that would help find similar dishes
        
        Extract from the query "{query}" now:
        
        """
        
        llm_call_start = time.time()
        response = llm.invoke(extraction_prompt)
        llm_call_end = time.time()
        print(f"[TIMING] LLM extraction call took {llm_call_end - llm_call_start:.3f}s")
        
        # Parse JSON response with improved error handling
        import json
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
def check_user_registration(config: RunnableConfig, phone_number: str = None) -> Dict[str, Any]:
    """
    Check if a user is registered. Use this before actions that require registration.
    
    Args:
        phone_number: Optional phone number to check (will use config if not provided)
        
    Returns:
        Dict with registration status and user information
    """
    print(f"[DEBUG] check_user_registration called with phone_number: '{phone_number}'")
    
    configuration = config.get("configurable", {})
    if not phone_number:
        phone_number = configuration.get("phone_number", None)
    
    if not phone_number:
        return {
            "is_registered": False,
            "message": "No phone number available to check registration"
        }
    
    try:
        response = supabase.table('customers').select('*').eq('phone_number', phone_number).execute()
        
        if response.data and len(response.data) > 0:
            customer = response.data[0]
            print(f"[DEBUG] User is registered: {customer.get('name')} with phone {phone_number}")
            return {
                "is_registered": True,
                "customer_info": {
                    "name": customer.get("name"),
                    "phone_number": phone_number,
                    "preferences": customer.get("preferences", {}),
                    "allergies": customer.get("allergies", [])
                }
            }
        else:
            print(f"[DEBUG] User not registered for phone_number: '{phone_number}'")
            return {
                "is_registered": False,
                "phone_number": phone_number
            }
            
    except Exception as e:
        print(f"[DEBUG] Error checking registration: {e}")
        return {
            "is_registered": False,
            "error": str(e)
        }

@tool
def collect_registration(config: RunnableConfig, phone_number: str, name: str = None) -> Dict[str, Any]:
    """
    Collect user registration when needed for specific actions like placing orders or reservations.
    This should only be called when registration is required for a specific action.
    
    Args:
        phone_number: User's 10-digit phone number
        name: User's name (optional, will be requested if not provided)
        
    Returns:
        Dict with registration status and user information
    """
    print(f"[DEBUG] collect_registration called with phone_number: '{phone_number}', name: '{name}'")
    
    if not phone_number or len(phone_number) != 10:
        return {
            "success": False,
            "error": "Please provide a valid 10-digit phone number",
            "set_is_registered": False
        }
    
    try:
        # Check if customer already exists
        existing_customer = supabase.table('customers').select('*').eq('phone_number', phone_number).execute()
        
        if existing_customer.data and len(existing_customer.data) > 0:
            # Customer exists - return their information
            customer = existing_customer.data[0]
            print(f"[DEBUG] Found existing customer: {customer.get('name')} with phone {phone_number}")
            
            # Update config with customer information
            configuration = config.get("configurable", {})
            configuration["phone_number"] = phone_number
            configuration["name"] = customer.get("name")
            
            return {
                "success": True,
                "is_existing_customer": True,
                "set_is_registered": True,
                "customer_info": {
                    "name": customer.get("name"),
                    "phone_number": phone_number,
                    "preferences": customer.get("preferences", {}),
                    "allergies": customer.get("allergies", [])
                },
                "summary": f"Existing customer - \n\n Customer Name: {customer.get('name')}"
            }
        else:
            # New customer - collect name if not provided
            if not name:
                return {
                    "success": False,
                    "needs_name": True,
                    "set_is_registered": False,
                    # "summary": "Thank you for providing your phone number. And What is your name please?"
                }
            
            # Create new customer record
            customer_data = {
                "phone_number": phone_number,
                "name": name,
                "preferences": {},
                "allergies": [],
            }
            
            result = supabase.table('customers').insert(customer_data).execute()
            
            if result.data:
                print(f"[DEBUG] Created new customer: {name} with phone {phone_number}")
                
                # Update config with customer information
                configuration = config.get("configurable", {})
                configuration["phone_number"] = phone_number
                configuration["name"] = name
                
                return {
                    "success": True,
                    "is_existing_customer": False,
                    "set_is_registered": True,
                    "customer_info": {
                        "name": name,
                        "phone_number": phone_number,
                        "preferences": {},
                        "allergies": []
                    },
                    "message": f"Thank you, {name}! I've registered you in our system."
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to register your information. Please try again.",
                    "set_is_registered": False
                }
                
    except Exception as e:
        print(f"[DEBUG] Error in collect_registration: {e}")
        return {
            "success": False,
            "error": f"Registration failed: {str(e)}",
            "set_is_registered": False
        }

@tool
def make_reservation(config: RunnableConfig, phone_number: str = None, name: str = None, pax: int = None, date: str = None, time: str = None, conflict_resolution: str = None, **kwargs) -> Dict[str, Any]:
    """
    Make a restaurant reservation. Requires user registration first.
    
    Args:
        phone_number: User's 10-digit phone number (required for registration check)
        name: User's name (required for new customers only)
        pax: Number of people for the reservation (1-12)
        date: Reservation date in YYYY-MM-DD format
        time: Reservation time in HH:MM format (24-hour)
        conflict_resolution: How to handle conflicts ('proceed', 'update_existing', 'cancel_and_create')
        
    Returns:
        Dict with reservation details and next steps
    """
    from datetime import datetime
    print(f"[DEBUG] make_reservation called with phone_number: '{phone_number}', name: '{name}', pax: {pax}, date: '{date}', time: '{time}'")
    
    configuration = config.get("configurable", {})
    print(f"[DEBUG] make_reservation config phone_number: '{configuration.get('phone_number')}'")
    
    # Step 1: Validate phone number and check registration
    # ENHANCED: Try multiple sources for phone number
    if not phone_number:
        phone_number = configuration.get("phone_number", None)
        print(f"[DEBUG] make_reservation using config phone_number: '{phone_number}'")
    
    # If still no phone number, try thread cache
    if not phone_number:
        phone_number = get_thread_phone_number(config)
        print(f"[DEBUG] make_reservation using thread cache phone_number: '{phone_number}'")
    
    # If still no phone number, try to extract from kwargs (state might pass it)
    if not phone_number and kwargs:
        phone_number = kwargs.get("phone_number", None)
        print(f"[DEBUG] make_reservation using kwargs phone_number: '{phone_number}'")
    
    # Enhanced validation: check if phone number is a string and has exactly 10 digits
    if not phone_number or not str(phone_number).isdigit() or len(str(phone_number)) != 10:
        return {
            "success": False,
            "step": "phone_number",
            "message": "To make a reservation, I'll need your 10-digit phone number first. What's your phone number?"
        }
    
    try:
        # Check if customer exists
        customer_check = supabase.table('customers').select('*').eq('phone_number', phone_number).execute()
        
        if customer_check.data and len(customer_check.data) > 0:
            # Existing customer
            customer = customer_check.data[0]
            customer_name = customer.get('name')
            customer_phone = customer.get('phone_number')
            
            # Update config with customer info
            configuration["phone_number"] = customer_phone
            configuration["name"] = customer_name
            
            # Cache phone number for this thread
            set_thread_phone_number(config, customer_phone)
            
            print(f"[DEBUG] Found existing customer for reservation: {customer_name}")
            
        else:
            # New customer - need name
            if not name:
                return {
                    "success": False,
                    "step": "name",
                    "phone_number": phone_number,
                    "message": f"I don't have your details for {phone_number}. What's your name so I can register you for the reservation?"
                }
            
            # Register new customer
            customer_data = {
                "phone_number": phone_number,
                "name": name,
                "preferences": {},
                "allergies": [],
            }
            
            reg_result = supabase.table('customers').insert(customer_data).execute()
            if not reg_result.data:
                return {
                    "success": False,
                    "error": "Failed to register customer. Please try again.",
                    "step": "registration_failed"
                }
            
            customer_name = name
            customer_phone = phone_number
            
            # Update config
            configuration["phone_number"] = customer_phone
            configuration["name"] = customer_name
            
            # Cache phone number for this thread
            set_thread_phone_number(config, customer_phone)
            
            print(f"[DEBUG] Registered new customer for reservation: {customer_name}")
        
        # Step 2: Collect reservation details
        if not pax:
            return {
                "success": False,
                "step": "pax",
                "customer_name": customer_name,
                "message": f"Great, {customer_name}! How many people will be dining with us? (1-12 people)"
            }
        
        if pax < 1 or pax > 12:
            return {
                "success": False,
                "step": "pax",
                "message": "We can accommodate 1-12 people per reservation. How many people will be dining?"
            }
        
        if not date:
            return {
                "success": False,
                "step": "date",
                "customer_name": customer_name,
                "pax": pax,
                "message": f"Perfect! For {pax} people. What date would you like to make the reservation? (Please provide in YYYY-MM-DD format, e.g., 2024-07-15)"
            }
        
        # Validate date format immediately
        try:
            reservation_date = datetime.strptime(date, "%Y-%m-%d").date()
            today = datetime.now().date()
            
            if reservation_date < today:
                return {
                    "success": False,
                    "step": "date",
                    "message": "Please select a future date for your reservation. What date would you like?"
                }
        except ValueError:
            return {
                "success": False,
                "step": "date",
                "message": "Please provide the date in YYYY-MM-DD format (e.g., 2024-07-15)."
            }
        
        if not time:
            return {
                "success": False,
                "step": "time",
                "customer_name": customer_name,
                "pax": pax,
                "date": date,
                "message": f"Excellent! For {date}. What time would you prefer? (Please provide in HH:MM format, e.g., 19:30 for 7:30 PM)"
            }
        
        # Validate time format immediately
        try:
            reservation_time = datetime.strptime(time, "%H:%M").time()
            
            # Check if time is within restaurant hours (11 AM to 10 PM, last seating)
            open_time = datetime.strptime("11:00", "%H:%M").time()
            last_seating = datetime.strptime("22:00", "%H:%M").time()
            
            if reservation_time < open_time or reservation_time > last_seating:
                return {
                    "success": False,
                    "step": "time",
                    "message": "We accept reservations between 11:00 AM and 10:00 PM. What time would you prefer?"
                }
        except ValueError:
            return {
                "success": False,
                "step": "time", 
                "message": "Please provide the time in HH:MM format (e.g., 19:30 for 7:30 PM)."
            }
        
        # Step 3: All validations passed, proceed to create reservation
        # CRITICAL: Only proceed if ALL required fields are explicitly provided by user
        
        # Step 3.5: Check for existing reservations on the same date
        try:
            # Get all existing reservations for this customer on the requested date
            existing_reservations = supabase.table('reservations').select('*').eq('cust_number', customer_phone).eq('booking_date', date).execute()
            
            if existing_reservations.data:
                existing_res = existing_reservations.data[0]  # Take the first (most relevant) existing reservation
                existing_time = existing_res.get('booking_time')
                existing_id = existing_res.get('id')
                existing_pax = existing_res.get('pax')
                
                # Format existing time for display
                try:
                    existing_time_display = datetime.strptime(str(existing_time), "%H:%M").strftime("%I:%M %p")
                except:
                    existing_time_display = str(existing_time)
                
                # Format date for display
                try:
                    display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
                except:
                    display_date = date
                
                # Check if user has provided conflict resolution
                if conflict_resolution:
                    # Handle conflict resolution based on user choice
                    if str(existing_time) == str(time):  # Same date and time
                        if conflict_resolution.lower() in ['update', 'update_existing', '1']:
                            # Update existing reservation
                            if existing_pax != pax:
                                update_result = supabase.table('reservations').update({'pax': pax}).eq('id', existing_id).execute()
                                if update_result.data:
                                    return {
                                        "success": True,
                                        "step": "completed",
                                        "action": "reservation_updated",
                                        "set_is_registered": True,
                                        "reservation_details": {
                                            "reservation_id": existing_id,
                                            "customer_name": customer_name,
                                            "phone_number": customer_phone,
                                            "pax": pax,
                                            "date": display_date,
                                            "time": existing_time_display,
                                            "status": "updated"
                                        },
                                        "summary": f"Successfully updated reservation {existing_id} for {customer_name} on {display_date} at {existing_time_display}. Changed from {existing_pax} to {pax} people."
                                    }
                            else:
                                return {
                                    "success": True,
                                    "step": "completed", 
                                    "action": "reservation_exists",
                                    "set_is_registered": True,
                                    "reservation_details": {
                                        "reservation_id": existing_id,
                                        "customer_name": customer_name,
                                        "phone_number": customer_phone,
                                        "pax": pax,
                                        "date": display_date,
                                        "time": existing_time_display,
                                        "status": "confirmed"
                                    },
                                    "summary": f"Your existing reservation {existing_id} for {customer_name} on {display_date} at {existing_time_display} for {pax} people is already confirmed."
                                }
                        
                        elif conflict_resolution.lower() in ['keep', '2']:
                            # Keep existing reservation as is
                            return {
                                "success": True,
                                "step": "completed",
                                "action": "reservation_kept",
                                "set_is_registered": True,
                                "reservation_details": {
                                    "reservation_id": existing_id,
                                    "customer_name": customer_name,
                                    "phone_number": customer_phone,
                                    "pax": existing_pax,
                                    "date": display_date,
                                    "time": existing_time_display,
                                    "status": "confirmed"
                                },
                                "summary": f"No changes made. Your existing reservation {existing_id} for {customer_name} on {display_date} at {existing_time_display} for {existing_pax} people remains confirmed."
                            }
                        
                        elif conflict_resolution.lower() in ['cancel', 'cancel_and_create', '3']:
                            # Cancel existing and create new
                            supabase.table('reservations').delete().eq('id', existing_id).execute()
                            # Continue to create new reservation below
                            print(f"[DEBUG] Canceled existing reservation {existing_id} to create new one")
                        
                        else:
                            # Invalid choice, ask again
                            return {
                                "success": False,
                                "step": "existing_reservation_same_time",
                                "action": "reservation_conflict_same_time",
                                "existing_reservation": {
                                    "reservation_id": existing_id,
                                    "date": display_date,
                                    "time": existing_time_display,
                                    "pax": existing_pax
                                },
                                "requested_reservation": {
                                    "date": display_date,
                                    "time": datetime.strptime(time, "%H:%M").strftime("%I:%M %p"),
                                    "pax": pax
                                },
                                "message": f"I didn't understand your choice. You have a reservation on {display_date} at {existing_time_display} for {existing_pax} people (ID: {existing_id}).\n\nPlease choose:\n1. **Update** - Change to {pax} people\n2. **Keep** - Keep existing reservation\n3. **Cancel** - Cancel existing and create new\n\nPlease type 1, 2, or 3."
                            }
                    
                    else:  # Same date but different time
                        if conflict_resolution.lower() in ['yes', 'proceed', '1']:
                            # Create new reservation in addition to existing one
                            print(f"[DEBUG] User confirmed to create additional reservation on same date")
                            # Continue to create new reservation below
                        
                        elif conflict_resolution.lower() in ['no', '2']:
                            # Keep only existing reservation
                            return {
                                "success": True,
                                "step": "completed",
                                "action": "reservation_kept",
                                "set_is_registered": True,
                                "reservation_details": {
                                    "reservation_id": existing_id,
                                    "customer_name": customer_name,
                                    "phone_number": customer_phone,
                                    "pax": existing_pax,
                                    "date": display_date,
                                    "time": existing_time_display,
                                    "status": "confirmed"
                                },
                                "summary": f"No new reservation created. Your existing reservation {existing_id} for {customer_name} on {display_date} at {existing_time_display} for {existing_pax} people remains confirmed."
                            }
                        
                        elif conflict_resolution.lower() in ['update', '3']:
                            # Modify existing reservation to new time and pax
                            update_data = {'booking_time': time, 'pax': pax}
                            update_result = supabase.table('reservations').update(update_data).eq('id', existing_id).execute()
                            if update_result.data:
                                new_time_display = datetime.strptime(time, "%H:%M").strftime("%I:%M %p")
                                return {
                                    "success": True,
                                    "step": "completed",
                                    "action": "reservation_updated",
                                    "set_is_registered": True,
                                    "reservation_details": {
                                        "reservation_id": existing_id,
                                        "customer_name": customer_name,
                                        "phone_number": customer_phone,
                                        "pax": pax,
                                        "date": display_date,
                                        "time": new_time_display,
                                        "status": "updated"
                                    },
                                    "summary": f"Successfully updated reservation {existing_id} for {customer_name} on {display_date}. Changed from {existing_time_display} to {new_time_display} and from {existing_pax} to {pax} people."
                                }
                        
                        else:
                            # Invalid choice, ask again
                            return {
                                "success": False,
                                "step": "existing_reservation_different_time",
                                "action": "reservation_conflict_different_time",
                                "existing_reservation": {
                                    "reservation_id": existing_id,
                                    "date": display_date,
                                    "time": existing_time_display,
                                    "pax": existing_pax
                                },
                                "requested_reservation": {
                                    "date": display_date,
                                    "time": datetime.strptime(time, "%H:%M").strftime("%I:%M %p"),
                                    "pax": pax
                                },
                                "message": f"I didn't understand your choice. You have a reservation on {display_date} at {existing_time_display} for {existing_pax} people (ID: {existing_id}).\n\nYou want another at {datetime.strptime(time, '%H:%M').strftime('%I:%M %p')} for {pax} people.\n\nPlease choose:\n1. **Yes** - Create both reservations\n2. **No** - Keep only existing reservation\n3. **Update** - Modify existing reservation\n\nPlease type 1, 2, or 3."
                            }
                
                else:
                    # No conflict resolution provided, ask user for choice
                    if str(existing_time) == str(time):
                        # Case 1: Same date and time - ask if they want to change existing reservation
                        return {
                            "success": False,
                            "step": "existing_reservation_same_time",
                            "action": "reservation_conflict_same_time",
                            "existing_reservation": {
                                "reservation_id": existing_id,
                                "date": display_date,
                                "time": existing_time_display,
                                "pax": existing_pax
                            },
                            "requested_reservation": {
                                "date": display_date,
                                "time": datetime.strptime(time, "%H:%M").strftime("%I:%M %p"),
                                "pax": pax
                            },
                            "message": f"I found that you already have a reservation on {display_date} at {existing_time_display} for {existing_pax} people (Reservation ID: {existing_id}).\n\nWould you like me to:\n1. **Update** your existing reservation to {pax} people\n2. **Keep** your existing reservation as is\n3. **Cancel** your existing reservation and create a new one\n\nPlease reply with 1, 2, or 3."
                        }
                    
                    else:
                        # Case 2: Same date but different time - inform and ask for confirmation
                        return {
                            "success": False,
                            "step": "existing_reservation_different_time",
                            "action": "reservation_conflict_different_time",
                            "existing_reservation": {
                                "reservation_id": existing_id,
                                "date": display_date,
                                "time": existing_time_display,
                                "pax": existing_pax
                            },
                            "requested_reservation": {
                                "date": display_date,
                                "time": datetime.strptime(time, "%H:%M").strftime("%I:%M %p"),
                                "pax": pax
                            },
                            "message": f"I notice you already have a reservation on {display_date} at {existing_time_display} for {existing_pax} people (Reservation ID: {existing_id}).\n\nYou're now requesting another reservation on the same date at {datetime.strptime(time, '%H:%M').strftime('%I:%M %p')} for {pax} people.\n\nAre you sure you want to make **two separate reservations** on the same day? Please confirm:\n1. **Yes** - Create the new reservation in addition to the existing one\n2. **No** - Keep only the existing reservation\n3. **Update** - Modify the existing reservation instead\n\nPlease reply with 1, 2, or 3."
                        }
        
        except Exception as e:
            print(f"[DEBUG] Error checking existing reservations: {e}")
            # Continue with creating the reservation if there's an error checking existing ones
        
        # Step 4: Create reservation in database
        reservation_data = {
            "cust_number": customer_phone,  # Foreign key to customers table
            "pax": pax,
            "booking_date": date,
            "booking_time": time
        }
        
        reservation_result = supabase.table('reservations').insert(reservation_data).execute()
        
        if reservation_result.data:
            reservation_id = reservation_result.data[0].get('id', 'N/A')
            
            # Format time for display
            display_time = datetime.strptime(time, "%H:%M").strftime("%I:%M %p")
            display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
            
            print(f"[DEBUG] Created reservation {reservation_id} for {customer_name}")
            
            return {
                "success": True,
                "step": "completed",
                "action": "reservation_created",
                "set_is_registered": True,
                "reservation_details": {
                    "reservation_id": reservation_id,
                    "customer_name": customer_name,
                    "phone_number": customer_phone,
                    "pax": pax,
                    "date": display_date,
                    "time": display_time,
                    "status": "confirmed"
                },
                "summary": f"Successfully created reservation {reservation_id} for {customer_name} on {display_date} at {display_time} for {pax} people. The reservation is confirmed and customer should arrive 10 minutes early."
            }
        else:
            return {
                "success": False,
                "error": "Failed to create reservation. Please try again.",
                "step": "reservation_failed"
            }
            
    except Exception as e:
        print(f"[DEBUG] Error in make_reservation: {e}")
        return {
            "success": False,
            "error": f"Reservation failed: {str(e)}",
            "step": "error"
        }

@tool
def get_faq_answer(query: str) -> Dict[str, Any]:
    """
    Search FAQ database for answers to general restaurant questions about hours, location, 
    parking, policies, services, etc. Use this for non-menu related questions.
    
    Args:
        query: User's question about restaurant information
        
    Returns:
        Dict with FAQ answers and confidence scores
    """
    faq_start = time.time()
    print(f"[TIMING] get_faq_answer started with query: {query}")
    
    try:
        # Use pooled Neo4j connection
        connection_start = time.time()
        neo4j_graph = get_neo4j_connection()
        connection_end = time.time()
        print(f"[TIMING] Neo4j connection for FAQ search took {connection_end - connection_start:.3f}s")
        
        # Use pooled embedding model
        embeddings_start = time.time()
        embeddings = get_embedding_model()
        embeddings_end = time.time()
        print(f"[TIMING] Embedding model for FAQ took {embeddings_end - embeddings_start:.3f}s")
        
        # Generate query embedding
        embed_start = time.time()
        query_embedding = embeddings.embed_query(query)
        embed_end = time.time()
        print(f"[TIMING] FAQ query embedding took {embed_end - embed_start:.3f}s")
        
        # Search FAQ vector index
        faq_search_query = """
        WITH $embedding AS queryEmbedding
        CALL db.index.vector.queryNodes('faq_embed', $k, queryEmbedding)
        YIELD node, score
        WHERE score > 0.7
        WITH node AS faq, score
        RETURN faq.question as question,
               faq.answer as answer,
               faq.id as id,
               score
        ORDER BY score DESC
        """
        
        search_start = time.time()
        results = neo4j_graph.query(faq_search_query, params={"embedding": query_embedding, "k": 5})
        search_end = time.time()
        print(f"[TIMING] FAQ vector search took {search_end - search_start:.3f}s")
        
        # If no high-confidence results, try keyword matching
        if not results or (results and results[0]['score'] < 0.8):
            print(f"[DEBUG] Low confidence FAQ results, trying keyword search...")
            
            # Extract keywords from query
            keywords = query.lower().split()
            keyword_conditions = []
            
            for keyword in keywords:
                if len(keyword) > 2:  # Skip very short words
                    keyword_conditions.append(f"toLower(faq.question) CONTAINS '{keyword}' OR toLower(faq.answer) CONTAINS '{keyword}'")
            
            if keyword_conditions:
                keyword_query = f"""
                MATCH (faq:FAQ)
                WHERE {' OR '.join(keyword_conditions)}
                RETURN faq.question as question,
                       faq.answer as answer,
                       faq.id as id,
                       0.6 as score
                ORDER BY faq.id
                LIMIT 3
                """
                
                keyword_start = time.time()
                keyword_results = neo4j_graph.query(keyword_query)
                keyword_end = time.time()
                print(f"[TIMING] FAQ keyword search took {keyword_end - keyword_start:.3f}s")
                
                # Combine results, avoiding duplicates
                existing_questions = {r['question'] for r in results}
                for kr in keyword_results:
                    if kr['question'] not in existing_questions:
                        results.append(kr)
        
        # Format results
        if results:
            top_result = results[0]
            confidence = top_result['score']
            
            # Build response
            response_parts = []
            
            # Primary answer
            response_parts.append(f"**{top_result['question']}**")
            response_parts.append(f"{top_result['answer']}")
            
            # Additional related FAQs if available
            if len(results) > 1:
                related_faqs = []
                for faq in results[1:3]:  # Show up to 2 additional FAQs
                    if faq['score'] > 0.6:  # Only show if reasonably relevant
                        related_faqs.append(f"• **{faq['question']}**: {faq['answer']}")
                
                if related_faqs:
                    response_parts.append("\n**Related Information:**")
                    response_parts.extend(related_faqs)
            
            formatted_response = "\n\n".join(response_parts)
            
            faq_end = time.time()
            print(f"[TIMING] Total get_faq_answer took {faq_end - faq_start:.3f}s")
            print(f"[DEBUG] FAQ search found {len(results)} results with confidence {confidence:.3f}")
            
            return {
                "success": True,
                "answer": formatted_response,
                "confidence": confidence,
                "total_results": len(results),
                "search_type": "FAQ database"
            }
        
        else:
            # No FAQ results found
            faq_end = time.time()
            print(f"[TIMING] get_faq_answer found no results in {faq_end - faq_start:.3f}s")
            
            return {
                "success": False,
                "answer": "I don't have specific information about that in our FAQ database. Please contact our restaurant directly for more details.",
                "confidence": 0.0,
                "total_results": 0,
                "search_type": "FAQ database"
            }
    
    except Exception as e:
        faq_end = time.time()
        print(f"[TIMING] get_faq_answer failed after {faq_end - faq_start:.3f}s")
        print(f"[DEBUG] Error in get_faq_answer: {e}")
        
        return {
            "success": False,
            "answer": f"I encountered an error searching our FAQ database: {str(e)}",
            "confidence": 0.0,
            "total_results": 0,
            "error": str(e),
            "search_type": "FAQ database"
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


# --- Waiter Agent ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

waiter_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful restaurant waiter. Help guests with menu questions, recommendations, orders, and general restaurant information."
            " Always use tools to get accurate information - never make up details."
            
            "\n\nWAITER PERSONA & LANGUAGE:"
            "\n• Speak naturally like a friendly, knowledgeable waiter - not like an AI assistant"
            "\n• NEVER mention 'retrieving information', 'searching database', or 'tools', etc"
            "\n• Use natural phrases: 'Yes, we have...', 'Our chef makes...', 'I'd recommend...'"
            "\n• Be confident and direct: 'Absolutely!' instead of 'It seems that...'"
            "\n• Sound human: 'Let me check our specials' not 'Let me search our menu'"
            
            "\n\nREGISTRATION REQUIREMENTS:"
            "\n• MENU BROWSING, RECOMMENDATIONS, INGREDIENT INFO: NO registration needed"
            "\n• ORDERS, RESERVATIONS, BILLS: Registration REQUIRED"
            "\n• When registration needed: Use check_user_registration() first"
            "\n• If not registered: 'To [place order/make reservation], I'll need your phone number first'"
            "\n• Use collect_registration() tool when customer provides phone number"
            "\n• Continue conversation naturally after registration"
            "\n• For menu questions, proceed immediately without asking for registration"
            
            "\n\nRESERVATION HANDLING:"
            "\n• NEW RESERVATIONS: Use make_reservation() tool for reservation requests"
            "\n• CHECK RESERVATIONS: Use get_reservations() to show customer's bookings"
            "\n• CANCEL RESERVATIONS: Use cancel_reservation() with reservation ID"
            "\n• MODIFY RESERVATIONS: Use modify_reservation() with reservation ID and new details"
            "\n• make_reservation() handles complete flow: registration check + reservation creation"
            "\n• DON'T use collect_registration() separately for reservations - make_reservation() handles everything"
            "\n• The reservation tools will guide users step-by-step through each process"
            "\n• Always verify phone number for reservation management (check/cancel/modify)"
            "\n• IMPORTANT: Tools return structured data - YOU format the response naturally based on the action and details"
            "\n• For successful operations, create warm, personal responses using the customer's name and details"
            "\n• For reservation lists, only present upcoming reservations"
            "\n• Always mention reservation IDs for future reference and modification options"
            
            "\n\nRESERVATION CONFLICT HANDLING:"
            "\n• CONFLICT DETECTION: make_reservation() automatically checks for existing reservations on the same date"
            "\n• SAME DATE & TIME: When customer has existing reservation at same date/time, make_reservation() will ask for choice"
            "\n• SAME DATE, DIFFERENT TIME: When customer has existing reservation on same date but different time, make_reservation() will inform and ask for confirmation"
            "\n• PARSING USER CHOICE: When user responds to conflict options, use parse_reservation_conflict_response() first"
            "\n• THEN CALL make_reservation() AGAIN: Use the conflict_resolution parameter from parse_reservation_conflict_response()"
            "\n• CONFLICT RESOLUTION OPTIONS:"
            "\n  - 'update_existing' or '1': Update the existing reservation"
            "\n  - 'keep' or '2': Keep existing reservation, cancel new request"
            "\n  - 'cancel_and_create' or '3': Cancel existing, create new"
            "\n  - 'proceed' or '1' (for different times): Create both reservations"
            "\n  - 'update' or '3' (for different times): Modify existing to new time/pax"
            "\n• WORKFLOW: make_reservation() → conflict detected → user chooses → parse_reservation_conflict_response() → make_reservation(conflict_resolution=choice)"
            "\n• BE HELPFUL: Explain options clearly and confirm the final outcome"
            
            "\n\nCURRENT DATE & TIME: {current_datetime}"
            "\nIMPORTANT: Use this current time to provide accurate, contextual answers about operating hours."
            "\n• If asked 'Are you open now?' - compare current time with operating hours from FAQ tool"
            "\n• If current time is OUTSIDE operating hours → clearly state 'We are currently CLOSED'"
            "\n• If current time is WITHIN operating hours → state 'Yes, we are currently open'"
            "\n• Always be specific: 'We are open from X to Y' and 'We will open/close at Z'"
            
            "\n\nTOOL SELECTION (be efficient):"
            "\n• Menu questions ('What do you have') → get_recommendations(query) - NO registration needed"
            "\n• Dish verification ('Do you have X') → extract_food_terms(query) THEN check_semantic_similarity(extracted_terms) - NO registration needed"
            "\n• Complex searches → extract_food_terms(query) THEN check_semantic_similarity(extracted_terms) - NO registration needed"
            "\n• Order requests ('I want to order') → check_user_registration() → collect_registration() if needed → get_detailed_dish_info()"
            "\n• NEW RESERVATIONS ('I want to make a reservation') → make_reservation() - handles everything automatically"
            "\n• CHECK RESERVATIONS ('Show my bookings') → get_reservations() - requires phone number"
            "\n• CANCEL RESERVATIONS ('Cancel reservation ID 123') → cancel_reservation() - requires ID and phone"
            "\n• MODIFY RESERVATIONS ('Change reservation ID 123') → modify_reservation() - requires ID, phone, and new details"
            "\n• Restaurant info (hours, location, policies) → get_faq_answer(query) - NO registration needed"
            
            "\n\nQUESTION ROUTING:"
            "\n1. MENU QUESTIONS → Use menu tools directly (get_recommendations, check_semantic_similarity)"
            "\n2. ORDER REQUESTS → Check registration first, then proceed"
            "\n3. NEW RESERVATION REQUESTS → Use make_reservation() immediately (handles registration + booking)"
            "\n4. RESERVATION MANAGEMENT → Use appropriate tool:"
            "\n   - 'Show/check my reservations' → get_reservations()"
            "\n   - 'Cancel reservation ID X' → cancel_reservation()"
            "\n   - 'Modify/change reservation ID X' → modify_reservation()"
            "\n5. RESTAURANT INFO → Use get_faq_answer for:"
            "\n   - Operating hours, timings"
            "\n   - Location, address, directions"
            "\n   - Parking availability, contact info"
            "\n   - Services (delivery, takeaway, reservations)"
            "\n   - Payment methods, policies"
            "\n6. If unsure → Try menu tools first (no registration barrier)"
            
            "\n\nCRITICAL: EXACT DISH VERIFICATION"
            "\nWhen users ask about specific dishes:"
            "\n1. Check if we have the EXACT dish name (NO registration needed)"
            "\n2. If YES → Answer about our dish"
            "\n3. If NO → 'We don't have [THEIR DISH] but we do have [OUR SIMILAR DISH]'"
            "\n4. NEVER pretend we have dishes we don't serve"
            "\n5. Only ask for registration when they want to ORDER, not when browsing"
            
            "\n\nFORMATTING:"
            "\n• Multiple items with details → Use markdown tables"
            "\n• Single dish descriptions → Regular text"
            "\n• Include: | Dish Name | Description | Price | Spice Level | Prep Time |"
            "\n• FAQ answers → Use clear headings and bullet points"
            
            "\n\nOPTIMIZATION HINTS:"
            "\n• optimization_hint='direct_menu_query' → Use get_recommendations"
            "\n• optimization_hint='order_request' → Use check_user_registration first"
            "\n• optimization_hint='faq_query' → Use get_faq_answer directly"
            
            "\n\nEXAMPLES WITH NATURAL RESPONSES:"
            "\n'What veg starters do you have?' → get_recommendations → 'We have some wonderful vegetarian appetizers like...'"
            "\n'Do you have Fish Curry?' → extract_food_terms('Do you have Fish Curry?') → check_semantic_similarity(extracted_terms=result) → 'Yes, we have Fish Curry' OR 'We don't have Fish Curry, but our Prawn Curry is similar'"
            "\n'I want to order butter chicken and naan' → check_user_registration → collect_registration (if needed) → get_detailed_dish_info → 'Excellent choice! Let me get your order ready...'"
            "\n'I want to make a reservation' → make_reservation() → 'I'd be happy to help you make a reservation! What's your phone number?'"
            "\n'Can I book a table for 4 people?' → make_reservation() → 'Absolutely! Let me help you book a table. What's your phone number?'"
            "\n'Show me my reservations' → get_reservations() → 'Let me check your reservations. What's your phone number?'"
            "\n'Check my bookings' → get_reservations() → 'I'll pull up your reservations. What's your phone number?'"
            "\n'Cancel reservation ID 123' → cancel_reservation() → 'I can help you cancel that reservation. Let me verify your phone number first.'"
            "\n'Change my reservation to 6 people' → modify_reservation() → 'I can help modify your reservation. What's the reservation ID and your phone number?'"
            "\n'Do you serve non-veg?' → get_recommendations → 'Absolutely! We have a great selection of chicken, mutton, and seafood dishes. What type of non-veg are you in the mood for?'"
            "\n'What are your timings?' → get_faq_answer → 'We're open from 11 AM to 11 PM every day'"
            "\n'Are you open now?' at 2:30 AM → 'We're currently closed. We'll be open again at 11 AM'"
            "\n'Where are you located?' → get_faq_answer → 'We're located at [address]. It's easy to find with plenty of parking'"
            
            "\n\nSTRUCTURED TOOL RESPONSE FORMATTING:"
            "\n• Tools now return structured data (action, details, summary) - YOU format responses naturally"
            "\n• action='reservation_created' → 'Fantastic! I've confirmed your reservation for [date] at [time]...'"
            "\n• action='reservations_retrieved' → 'Hello [name]! I found your reservations...'"
            "\n• action='reservation_canceled' → 'Done! I've canceled your reservation [id]...'"
            "\n• action='reservation_updated' → 'Perfect! I've updated your reservation...'"
            "\n• NEVER just repeat the tool's summary - create warm, personal responses"
            
            "\n\nCRITICAL: TOOL CHAINING FOR DISH SEARCHES"
            "\n• ALWAYS use extract_food_terms(query) FIRST, then check_semantic_similarity(extracted_terms=result)"
            "\n• NEVER call check_semantic_similarity without the extracted_terms parameter"
            "\n• Example: User asks 'Do you have Galawati Kebab?'"
            "\n  1. Call extract_food_terms('Do you have Galawati Kebab?')"
            "\n  2. Call check_semantic_similarity(extracted_terms=<result from step 1>)"
            "\n  3. Format response based on similarity results"
            
            "\n\nAvailable tools: extract_food_terms, check_semantic_similarity, get_detailed_dish_info, get_recommendations, get_faq_answer, debug_embedding_indexes, check_user_registration, collect_registration, make_reservation, get_reservations, cancel_reservation, modify_reservation, parse_reservation_conflict_response"
            
            "\n\nPersonalization: {user_info}"
        ),
        ("placeholder", "{messages}"),
    ]
)

# waiter_agent_tools will be defined at the very end of the file after all tools



from typing import Callable

from langchain_core.messages import ToolMessage


# create_entry_node removed - no longer needed without separate registration agent


# --- Single Agent Graph - Waiter agent handles everything ---

# Start directly with waiter agent - it handles all functionality including registration when needed
builder.add_edge(START, "waiter_agent")

# Registration graph nodes removed - waiter agent handles registration directly



# Custom ReAct Waiter Agent
class ReactWaiterAgent(Assistant):
    def __init__(self, runnable: Runnable):
        super().__init__(runnable)
    
    def __call__(self, state: State, config: RunnableConfig):
        waiter_start = time.time()
        print(f"[TIMING] ReactWaiterAgent started processing")
        
        # Extract user info from config or state - prioritize state phone number
        configuration = config.get("configurable", {})
        phone_number = state.get("phone_number") or configuration.get("phone_number", None)
        
        # Update config with phone number from state for tool persistence
        # CRITICAL: Update the actual config object, not just the local copy
        if phone_number and phone_number != configuration.get("phone_number"):
            config["configurable"]["phone_number"] = phone_number
            print(f"[DEBUG] Updated actual config object with phone number from state: {phone_number}")
        
        # Create user_info from available data
        user_info = f"Phone: {phone_number}" if phone_number else "No user info available"
        
        # Add current date and time for contextual responses (IST)
        from datetime import datetime
        import pytz
        
        # Get current time in IST (Indian Standard Time)
        ist = pytz.timezone('Asia/Kolkata')
        current_datetime = datetime.now(ist).strftime("%A, %B %d, %Y at %I:%M %p IST")
        
        # Add user_info and current_datetime to state for the prompt
        state_with_user_info = {**state, "user_info": user_info, "current_datetime": current_datetime}
        
        # Optimize for efficiency - check if this looks like a straightforward menu query
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        if last_message and hasattr(last_message, 'content'):
            user_query = last_message.content.lower()
            
            # For simple menu queries, try direct recommendation first
            simple_patterns = [
                'recommend', 'suggest', 'what do you have', 'show me', 'do you have',
                'menu', 'dish', 'food', 'items', 'options', 'varieties'
            ]
            
            # For order/confirmation requests, check registration
            order_patterns = [
                'i want to order', 'i would like to order', 'can i order', 'place order',
                'i\'ll order', 'order for me', 'i need to order', 'make an order'
            ]
            
            # For reservation requests, use make_reservation tool
            reservation_patterns = [
                'reservation', 'book a table', 'reserve', 'table for', 'make a booking',
                'i want to book', 'can i book', 'table booking', 'dinner reservation',
                'lunch reservation', 'book table'
            ]
            
            # For reservation management requests
            check_reservation_patterns = [
                'check my reservations', 'show my bookings', 'my reservations', 'check my bookings',
                'see my reservations', 'view my bookings', 'do i have any reservations'
            ]
            
            cancel_reservation_patterns = [
                'cancel reservation', 'cancel my booking', 'delete reservation', 'remove booking',
                'cancel id', 'cancel booking id'
            ]
            
            modify_reservation_patterns = [
                'change reservation', 'modify booking', 'update reservation', 'reschedule booking',
                'change my booking', 'modify my reservation', 'edit reservation'
            ]
            
            if any(pattern in user_query for pattern in simple_patterns):
                print(f"[DEBUG] Detected simple menu query, optimizing workflow")
                # Add a hint to the agent to be more direct
                state_with_user_info["optimization_hint"] = "direct_menu_query"
            elif any(pattern in user_query for pattern in order_patterns):
                print(f"[DEBUG] Detected order request, checking registration")
                # Add a hint for order processing
                state_with_user_info["optimization_hint"] = "order_request"
            elif any(pattern in user_query for pattern in reservation_patterns):
                print(f"[DEBUG] Detected reservation request, using make_reservation tool")
                # Add a hint for reservation processing
                state_with_user_info["optimization_hint"] = "reservation_request"
            elif any(pattern in user_query for pattern in check_reservation_patterns):
                print(f"[DEBUG] Detected check reservations request")
                state_with_user_info["optimization_hint"] = "check_reservations"
            elif any(pattern in user_query for pattern in cancel_reservation_patterns):
                print(f"[DEBUG] Detected cancel reservation request")
                state_with_user_info["optimization_hint"] = "cancel_reservation"
            elif any(pattern in user_query for pattern in modify_reservation_patterns):
                print(f"[DEBUG] Detected modify reservation request")
                state_with_user_info["optimization_hint"] = "modify_reservation"
        
        result = super().__call__(state_with_user_info, config)
        
        waiter_end = time.time()
        print(f"[TIMING] ReactWaiterAgent completed in {waiter_end - waiter_start:.3f}s")
        
        return result

# Custom waiter tools node that handles registration state updates
def waiter_tools_node(state: State):
    """Custom tool node for waiter agent that handles registration state updates and phone number persistence"""
    # ENHANCED: Pass phone number from state to tools for persistence
    # Before calling the tool node, check if we have a phone number in state
    phone_number_from_state = state.get("phone_number")
    
    if phone_number_from_state:
        print(f"[DEBUG] waiter_tools_node found phone number in state: {phone_number_from_state}")
        # We need to inject the phone number into the tool calls
        # This is a bit tricky with LangGraph's tool system, so let's add it to the state context
        # that tools can access
        enhanced_state = {
            **state,
            "_phone_number_context": phone_number_from_state  # Add context for tools
        }
    else:
        enhanced_state = state
    
    # Call the standard tool node
    tool_node = create_tool_node_with_fallback(waiter_agent_tools)
    result = tool_node.invoke(enhanced_state)
    
    # Check if any tool results indicate registration status change
    new_is_registered = state.get("is_registered", False)
    
    # Track phone number extraction for persistence
    extracted_phone_number = None
    
    if "messages" in result:
        for message in result["messages"]:
            if hasattr(message, 'content'):
                content = str(message.content)
                
                # Extract phone numbers from tool results for session persistence
                import re
                # Look for phone number patterns in tool results
                phone_patterns = [
                    r"phone[_\s]*number['\"]?:\s*['\"]?(\d{10})",
                    r"'phone_number':\s*'(\d{10})'",
                    r'"phone_number":\s*"(\d{10})"',
                    r"customer_info.*phone_number.*?(\d{10})",
                    r"Hello\s+([A-Za-z]+)!\s+Here are your reservations"  # Extract from successful reservation queries
                ]
                
                for pattern in phone_patterns:
                    phone_match = re.search(pattern, content, re.IGNORECASE)
                    if phone_match:
                        extracted_phone_number = phone_match.group(1)
                        print(f"[DEBUG] Extracted phone number from tool result: {extracted_phone_number}")
                        break
                
                # Check for registration completion patterns
                if "set_is_registered" in content:
                    if "true" in content.lower() or "'set_is_registered': true" in content.lower():
                        new_is_registered = True
                        print(f"[DEBUG] Registration status updated to True via tool result")
                    elif "false" in content.lower() or "'set_is_registered': false" in content.lower():
                        new_is_registered = False
                        print(f"[DEBUG] Registration status updated to False via tool result")
                
                # Also check for success patterns from collect_registration
                if ("'success': true" in content.lower() or '"success": true' in content.lower()) and "collect_registration" in str(message):
                    new_is_registered = True
                    print(f"[DEBUG] Registration completed via collect_registration tool")
                
                # Check for success patterns from make_reservation
                if ("'success': true" in content.lower() or '"success": true' in content.lower()) and "make_reservation" in str(message):
                    new_is_registered = True
                    print(f"[DEBUG] Registration completed via make_reservation tool")
                
                # Check for successful reservation operations (get/cancel/modify) to extract phone numbers
                if any(pattern in content.lower() for pattern in ["hello", "reservations:", "reservation id", "confirmed", "updated successfully", "canceled successfully"]):
                    # Look for tool calls in the last user message to extract phone number
                    if state.get("messages"):
                        recent_messages = state["messages"][-3:]  # Check last few messages
                        for msg in recent_messages:
                            if hasattr(msg, 'content') and msg.content:
                                # Look for 10-digit numbers in recent user messages
                                phone_match = re.search(r'\b(\d{10})\b', str(msg.content))
                                if phone_match:
                                    extracted_phone_number = phone_match.group(1)
                                    print(f"[DEBUG] Extracted phone number from recent user message: {extracted_phone_number}")
                                    break
    
    # Create the result with phone number persistence
    result_dict = {
        **result,
        "is_registered": new_is_registered
    }
    
    # Add phone number to state if extracted
    if extracted_phone_number:
        result_dict["phone_number"] = extracted_phone_number
        print(f"[DEBUG] Storing phone number {extracted_phone_number} in state for persistence")
        
        # ENHANCED: Also cache the phone number for the current thread
        try:
            # We need to get the config from somewhere - let's try to extract it from the state context
            # Since we can't easily get the config here, we'll set up a mechanism to pass it
            # For now, we'll use a global thread-local storage approach
            print(f"[DEBUG] Should cache phone number {extracted_phone_number} but need config access")
        except Exception as e:
            print(f"[DEBUG] Error caching phone number: {e}")
    
    return result_dict

# Wrap waiter agent with ReAct logic
def waiter_agent_with_debug(state: State, config: RunnableConfig):
    print(f"[DEBUG] waiter_agent called with {len(state.get('messages', []))} messages")
    print(f"[DEBUG] waiter_agent is_registered = {state.get('is_registered', False)}")
    
    # NO LONGER BLOCK - waiter agent can handle menu questions without registration
    # Only check registration when tools need it (handled in tools themselves)
    
    # Ensure system is initialized before processing menu queries
    ensure_system_initialized()
    
    react_agent = ReactWaiterAgent(waiter_agent_runnable)
    return react_agent(state, config)

builder.add_node("waiter_agent", waiter_agent_with_debug)
builder.add_node("waiter_tools", waiter_tools_node)
builder.add_conditional_edges(
    "waiter_agent",
    tools_condition,
    {
        "tools": "waiter_tools",
        END: END
    }
)

builder.add_edge("waiter_tools", "waiter_agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(png_bytes)
    print("[INFO] Workflow graph exported to workflow_graph.png")
except Exception:
    # This requires some extra dependencies and is optional
    pass

import uuid

# Thread-based phone number cache for persistence across tool calls
_thread_phone_cache = {}

def get_thread_phone_number(config: RunnableConfig) -> str:
    """Get phone number for current thread from cache"""
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id and thread_id in _thread_phone_cache:
        phone = _thread_phone_cache[thread_id]
        print(f"[DEBUG] Retrieved cached phone number for thread {thread_id}: {phone}")
        return phone
    return None

def set_thread_phone_number(config: RunnableConfig, phone_number: str):
    """Cache phone number for current thread"""
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id and phone_number:
        _thread_phone_cache[thread_id] = phone_number
        print(f"[DEBUG] Cached phone number for thread {thread_id}: {phone_number}")

# Add startup initialization
def initialize_restaurant_system():
    """
    Pre-warm all expensive operations during app startup to eliminate first-request latency.
    Call this once when the application starts.
    """
    init_start = time.time()
    print("\n" + "="*60)
    print("🚀 INITIALIZING RESTAURANT SYSTEM")
    print("="*60)
    
    try:
        # 1. Pre-warm Neo4j connection
        print("📡 Establishing Neo4j connection...")
        connection_start = time.time()
        neo4j_graph = get_neo4j_connection()
        connection_end = time.time()
        print(f"✅ Neo4j connected in {connection_end - connection_start:.3f}s")
        
        # 2. Pre-warm Cypher chain and schema
        print("⚙️  Setting up Cypher chain and caching schema...")
        chain_start = time.time()
        cypher_chain, _ = get_cypher_chain()
        chain_end = time.time()
        print(f"✅ Cypher chain ready in {chain_end - chain_start:.3f}s")
        
        # 3. Pre-warm embedding model
        print("🧠 Loading embedding model...")
        embed_start = time.time()
        embedding_model = get_embedding_model()
        embed_end = time.time()
        print(f"✅ Embedding model loaded in {embed_end - embed_start:.3f}s")
        
        # 4. Test all systems with a quick query
        print("🧪 Testing system with sample query...")
        test_start = time.time()
        
        # Quick test query to ensure everything works
        test_result = neo4j_graph.query("MATCH (d:Dish) RETURN count(d) as dish_count LIMIT 1")
        dish_count = test_result[0]['dish_count'] if test_result else 0
        
        # Test embedding generation
        test_embedding = embedding_model.embed_query("test")
        
        test_end = time.time()
        print(f"✅ System test passed in {test_end - test_start:.3f}s")
        print(f"📊 Database contains {dish_count} dishes")
        
        # 5. Pre-cache some common queries to improve first user experience
        print("🗄️  Pre-caching common queries...")
        cache_start = time.time()
        
        common_queries = [
            "vegetarian starters",
            "paneer dishes", 
            "spicy main course",
            "desserts",
            "beverages"
        ]
        
        for query in common_queries:
            try:
                # This will cache the results for instant retrieval
                kg_answer(query)
                print(f"   ✓ Cached: {query}")
            except Exception as e:
                print(f"   ⚠️  Failed to cache '{query}': {e}")
        
        cache_end = time.time()
        print(f"✅ Common queries cached in {cache_end - cache_start:.3f}s")
        
        init_end = time.time()
        total_time = init_end - init_start
        
        print("\n" + "="*60)
        print(f"🎉 SYSTEM READY! Total initialization: {total_time:.3f}s")
        print("💡 First user query will now be FAST!")
        print("="*60 + "\n")
        
        return {
            "success": True,
            "total_time": total_time,
            "dish_count": dish_count,
            "message": f"System initialized successfully in {total_time:.3f}s"
        }
        
    except Exception as e:
        init_end = time.time()
        total_time = init_end - init_start
        
        print("\n" + "="*60)
        print(f"❌ INITIALIZATION FAILED after {total_time:.3f}s")
        print(f"Error: {e}")
        print("="*60 + "\n")
        
        return {
            "success": False,
            "total_time": total_time,
            "error": str(e),
            "message": f"System initialization failed: {e}"
        }

# Call initialization when module is imported (but only once)
_system_initialized = False

def ensure_system_initialized():
    """Ensure system is initialized - call this before any restaurant operations"""
    global _system_initialized
    
    if not _system_initialized:
        result = initialize_restaurant_system()
        _system_initialized = result["success"]
        return result
    else:
        print("🔄 System already initialized - ready to serve!")
        return {"success": True, "message": "System already initialized"}

def run_command_line_interface():
    """Run the command-line chat interface"""
    print("🔄 Initializing system for command-line interface...")
    
    # Initialize system first
    init_result = ensure_system_initialized()
    if not init_result["success"]:
        print(f"❌ Failed to initialize system: {init_result.get('error', 'Unknown error')}")
        print("⚠️ Some features may not work properly. Continuing anyway...")
    
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "phone_number": None,
            "thread_id": thread_id,
        }
    }

    # --- Chat loop ---
    from langchain_core.messages import HumanMessage

    state = {"messages": [], "is_registered": False, "phone_number": None}
    print("Welcome to My Restaurant! Feel free to browse our menu and ask about our dishes. Registration is only needed when placing orders or making reservations.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Goodbye!")
            break
        state["messages"].append(HumanMessage(content=user_input))
        result = graph.invoke(state, config)
        # result["messages"] is a list of messages, get the last one
        assistant_message = result["messages"][-1]
        print(f"Assistant: {assistant_message.content}")
        # Update state with the new messages
        state = result

@tool
def get_reservations(config: RunnableConfig, phone_number: str = None, **kwargs) -> Dict[str, Any]:
    """
    Get all reservations for a customer by phone number.
    
    Args:
        phone_number: User's 10-digit phone number
        
    Returns:
        Dict with reservation details or error message
    """
    from datetime import datetime
    print(f"[DEBUG] get_reservations called with phone_number: '{phone_number}'")
    
    configuration = config.get("configurable", {})
    if not phone_number:
        phone_number = configuration.get("phone_number", None)
    
    # Try thread cache if still no phone number
    if not phone_number:
        phone_number = get_thread_phone_number(config)
        print(f"[DEBUG] get_reservations using thread cache phone_number: '{phone_number}'")
    
    if not phone_number or len(phone_number) != 10:
        return {
            "success": False,
            "message": "To check your reservations, I'll need your 10-digit phone number. What's your phone number?"
        }
    
    try:
        # Check if customer exists
        customer_check = supabase.table('customers').select('name').eq('phone_number', phone_number).execute()
        
        if not customer_check.data:
            return {
                "success": False,
                "message": f"I don't have any customer record for {phone_number}. Please check your phone number or register first."
            }
        
        customer_name = customer_check.data[0].get('name')
        
        # Cache phone number for this thread since it was successfully used
        set_thread_phone_number(config, phone_number)
        
        # Get all reservations for this customer, ordered by date
        reservations_result = supabase.table('reservations').select('*').eq('cust_number', phone_number).order('booking_date', desc=False).execute()
        
        if not reservations_result.data:
            return {
                "success": True,
                "action": "no_reservations_found",
                "customer_name": customer_name,
                "reservations": [],
                "upcoming_count": 0,
                # "past_count": 0,
                "summary": f"{customer_name} has no reservations on record. This could be a good opportunity to make a new reservation."
            }
        
        # Format reservations for display
        formatted_reservations = []
        current_date = datetime.now().date()
        
        for reservation in reservations_result.data:
            try:
                booking_date = datetime.strptime(reservation['booking_date'], "%Y-%m-%d").date()
                booking_time = reservation.get('booking_time', 'No time specified')
                
                # Format time for display if it's in HH:MM format
                if booking_time and ':' in str(booking_time):
                    try:
                        time_obj = datetime.strptime(str(booking_time), "%H:%M").time()
                        display_time = time_obj.strftime("%I:%M %p")
                    except:
                        display_time = str(booking_time)
                else:
                    display_time = str(booking_time)
                
                # Determine status based on date
                if booking_date < current_date:
                    status = "Completed"
                else:
                    status = "Confirmed"
                
                formatted_reservations.append({
                    "reservation_id": reservation['id'],
                    "date": booking_date.strftime("%B %d, %Y"),
                    "time": display_time,
                    "pax": reservation['pax'],
                    "status": status,
                    "is_upcoming": booking_date >= current_date
                })
                
            except Exception as e:
                print(f"[DEBUG] Error formatting reservation {reservation.get('id')}: {e}")
                # Include malformed reservations with basic info
                formatted_reservations.append({
                    "reservation_id": reservation['id'],
                    "date": str(reservation['booking_date']),
                    "time": str(reservation.get('booking_time', 'No time')),
                    "pax": reservation['pax'],
                    "status": "Unknown",
                    "is_upcoming": False
                })
        
        # Separate upcoming and past reservations
        upcoming = [r for r in formatted_reservations if r['is_upcoming']]
        # past = [r for r in formatted_reservations if not r['is_upcoming']]
        
        # Create summary for LLM to format naturally
        return {
            "success": True,
            "action": "reservations_retrieved",
            "customer_name": customer_name,
            "reservations": formatted_reservations,
            "upcoming_reservations": upcoming,
            # "past_reservations": past,
            "upcoming_count": len(upcoming),
            # "past_count": len(past),
            "summary": f"Retrieved {len(formatted_reservations)} total reservations for {customer_name}: {len(upcoming)} upcoming reservations."
        }
        
    except Exception as e:
        print(f"[DEBUG] Error in get_reservations: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error while checking your reservations: {str(e)}"
        }

@tool
def cancel_reservation(config: RunnableConfig, reservation_id: int = None, phone_number: str = None) -> Dict[str, Any]:
    """
    Cancel a reservation by ID. Requires phone number verification.
    
    Args:
        reservation_id: The reservation ID to cancel
        phone_number: User's phone number for verification
        
    Returns:
        Dict with cancellation status and details
    """
    from datetime import datetime
    print(f"[DEBUG] cancel_reservation called with reservation_id: {reservation_id}, phone_number: '{phone_number}'")
    
    configuration = config.get("configurable", {})
    if not phone_number:
        phone_number = configuration.get("phone_number", None)
    
    # Try thread cache if still no phone number
    if not phone_number:
        phone_number = get_thread_phone_number(config)
        print(f"[DEBUG] cancel_reservation using thread cache phone_number: '{phone_number}'")
    
    if not reservation_id:
        return {
            "success": False,
            "message": "To cancel a reservation, I need the reservation ID. Which reservation would you like to cancel?"
        }
    
    if not phone_number or len(phone_number) != 10:
        return {
            "success": False,
            "message": "To cancel a reservation, I need to verify your phone number. What's your 10-digit phone number?"
        }
    
    try:
        # Get the reservation and verify ownership
        reservation_result = supabase.table('reservations').select('*').eq('id', reservation_id).eq('cust_number', phone_number).execute()
        
        if not reservation_result.data:
            return {
                "success": False,
                "message": f"I couldn't find reservation ID {reservation_id} for your phone number. Please check the reservation ID and try again."
            }
        
        reservation = reservation_result.data[0]
        
        # Check if reservation is in the future
        try:
            booking_date = datetime.strptime(reservation['booking_date'], "%Y-%m-%d").date()
            current_date = datetime.now().date()
            
            if booking_date < current_date:
                return {
                    "success": False,
                    "message": f"Reservation ID {reservation_id} is for {booking_date.strftime('%B %d, %Y')}, which has already passed. Past reservations cannot be canceled."
                }
        except:
            pass  # If date parsing fails, proceed with cancellation
        
        # Get customer name for personalized message
        customer_result = supabase.table('customers').select('name').eq('phone_number', phone_number).execute()
        customer_name = customer_result.data[0].get('name') if customer_result.data else "there"
        
        # Delete the reservation
        delete_result = supabase.table('reservations').delete().eq('id', reservation_id).execute()
        
        if delete_result.data or delete_result.status_code == 204:
            # Format the canceled reservation details
            booking_time = reservation.get('booking_time', 'No time specified')
            if booking_time and ':' in str(booking_time):
                try:
                    time_obj = datetime.strptime(str(booking_time), "%H:%M").time()
                    display_time = time_obj.strftime("%I:%M %p")
                except:
                    display_time = str(booking_time)
            else:
                display_time = str(booking_time)
            
            try:
                display_date = datetime.strptime(reservation['booking_date'], "%Y-%m-%d").strftime("%B %d, %Y")
            except:
                display_date = reservation['booking_date']
            
            return {
                "success": True,
                "action": "reservation_canceled",
                "canceled_reservation": {
                    "reservation_id": reservation_id,
                    "customer_name": customer_name,
                    "date": display_date,
                    "time": display_time,
                    "pax": reservation['pax']
                },
                "summary": f"Successfully canceled reservation {reservation_id} for {customer_name} on {display_date} at {display_time} for {reservation['pax']} people. The cancellation is complete."
            }
        else:
            return {
                "success": False,
                "message": f"There was an issue canceling reservation ID {reservation_id}. Please try again or contact us directly."
            }
        
    except Exception as e:
        print(f"[DEBUG] Error in cancel_reservation: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error while canceling your reservation: {str(e)}"
        }

@tool
def modify_reservation(config: RunnableConfig, reservation_id: int = None, phone_number: str = None, new_date: str = None, new_time: str = None, new_pax: int = None) -> Dict[str, Any]:
    """
    Modify an existing reservation. Requires phone number verification.
    
    Args:
        reservation_id: The reservation ID to modify
        phone_number: User's phone number for verification
        new_date: New date in YYYY-MM-DD format (optional)
        new_time: New time in HH:MM format (optional)
        new_pax: New number of people (optional)
        
    Returns:
        Dict with modification status and details
    """
    from datetime import datetime
    print(f"[DEBUG] modify_reservation called with reservation_id: {reservation_id}, phone_number: '{phone_number}', new_date: '{new_date}', new_time: '{new_time}', new_pax: {new_pax}")
    
    configuration = config.get("configurable", {})
    if not phone_number:
        phone_number = configuration.get("phone_number", None)
    
    # Try thread cache if still no phone number
    if not phone_number:
        phone_number = get_thread_phone_number(config)
        print(f"[DEBUG] modify_reservation using thread cache phone_number: '{phone_number}'")
    
    if not reservation_id:
        return {
            "success": False,
            "message": "To modify a reservation, I need the reservation ID. Which reservation would you like to modify?"
        }
    
    if not phone_number or len(phone_number) != 10:
        return {
            "success": False,
            "message": "To modify a reservation, I need to verify your phone number. What's your 10-digit phone number?"
        }
    
    try:
        # Get the reservation and verify ownership
        reservation_result = supabase.table('reservations').select('*').eq('id', reservation_id).eq('cust_number', phone_number).execute()
        
        if not reservation_result.data:
            return {
                "success": False,
                "message": f"I couldn't find reservation ID {reservation_id} for your phone number. Please check the reservation ID and try again."
            }
        
        current_reservation = reservation_result.data[0]
        
        # Check if reservation is in the future
        try:
            booking_date = datetime.strptime(current_reservation['booking_date'], "%Y-%m-%d").date()
            current_date = datetime.now().date()
            
            if booking_date < current_date:
                return {
                    "success": False,
                    "message": f"Reservation ID {reservation_id} is for {booking_date.strftime('%B %d, %Y')}, which has already passed. Past reservations cannot be modified."
                }
        except:
            pass  # If date parsing fails, proceed with modification
        
        # If no modifications provided, ask what they want to change
        if not new_date and not new_time and not new_pax:
            try:
                current_date_display = datetime.strptime(current_reservation['booking_date'], "%Y-%m-%d").strftime("%B %d, %Y")
            except:
                current_date_display = current_reservation['booking_date']
            
            current_time = current_reservation.get('booking_time', 'No time')
            if current_time and ':' in str(current_time):
                try:
                    time_obj = datetime.strptime(str(current_time), "%H:%M").time()
                    current_time_display = time_obj.strftime("%I:%M %p")
                except:
                    current_time_display = str(current_time)
            else:
                current_time_display = str(current_time)
            
            return {
                "success": False,
                "step": "modification_details",
                "current_reservation": {
                    "reservation_id": reservation_id,
                    "date": current_date_display,
                    "time": current_time_display,
                    "pax": current_reservation['pax']
                },
                "message": f"**Current Reservation Details:**\n• **ID:** {reservation_id}\n• **Date:** {current_date_display}\n• **Time:** {current_time_display}\n• **Party Size:** {current_reservation['pax']} people\n\nWhat would you like to change? You can modify the date (YYYY-MM-DD), time (HH:MM), or number of people."
            }
        
        # Validate new values if provided
        updates = {}
        
        if new_date:
            try:
                new_date_obj = datetime.strptime(new_date, "%Y-%m-%d").date()
                today = datetime.now().date()
                
                if new_date_obj < today:
                    return {
                        "success": False,
                        "message": "Please select a future date for your reservation."
                    }
                
                updates['booking_date'] = new_date
            except ValueError:
                return {
                    "success": False,
                    "message": "Please provide the date in YYYY-MM-DD format (e.g., 2025-07-15)."
                }
        
        if new_time:
            try:
                new_time_obj = datetime.strptime(new_time, "%H:%M").time()
                
                # Check restaurant hours
                open_time = datetime.strptime("11:00", "%H:%M").time()
                last_seating = datetime.strptime("22:00", "%H:%M").time()
                
                if new_time_obj < open_time or new_time_obj > last_seating:
                    return {
                        "success": False,
                        "message": "We accept reservations between 11:00 AM and 10:00 PM. Please choose a time within our operating hours."
                    }
                
                updates['booking_time'] = new_time
            except ValueError:
                return {
                    "success": False,
                    "message": "Please provide the time in HH:MM format (e.g., 19:30 for 7:30 PM)."
                }
        
        if new_pax:
            if new_pax < 1 or new_pax > 12:
                return {
                    "success": False,
                    "message": "We can accommodate 1-12 people per reservation. Please choose a number within this range."
                }
            
            updates['pax'] = new_pax
        
        # Update the reservation
        if updates:
            update_result = supabase.table('reservations').update(updates).eq('id', reservation_id).execute()
            
            if update_result.data:
                updated_reservation = update_result.data[0]
                
                # Get customer name
                customer_result = supabase.table('customers').select('name').eq('phone_number', phone_number).execute()
                customer_name = customer_result.data[0].get('name') if customer_result.data else "there"
                
                # Format the updated details
                try:
                    display_date = datetime.strptime(updated_reservation['booking_date'], "%Y-%m-%d").strftime("%B %d, %Y")
                except:
                    display_date = updated_reservation['booking_date']
                
                booking_time = updated_reservation.get('booking_time', 'No time')
                if booking_time and ':' in str(booking_time):
                    try:
                        time_obj = datetime.strptime(str(booking_time), "%H:%M").time()
                        display_time = time_obj.strftime("%I:%M %p")
                    except:
                        display_time = str(booking_time)
                else:
                    display_time = str(booking_time)
                
                return {
                    "success": True,
                    "action": "reservation_updated",
                    "updated_reservation": {
                        "reservation_id": reservation_id,
                        "customer_name": customer_name,
                        "date": display_date,
                        "time": display_time,
                        "pax": updated_reservation['pax']
                    },
                    "summary": f"Successfully updated reservation {reservation_id} for {customer_name}. New details: {display_date} at {display_time} for {updated_reservation['pax']} people."
                }
            else:
                return {
                    "success": False,
                    "message": f"There was an issue updating reservation ID {reservation_id}. Please try again or contact us directly."
                }
        else:
            return {
                "success": False,
                "message": "No changes were specified. What would you like to modify about your reservation?"
            }
        
    except Exception as e:
        print(f"[DEBUG] Error in modify_reservation: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"I encountered an error while modifying your reservation: {str(e)}"
        }

@tool  
def parse_reservation_conflict_response(user_response: str) -> Dict[str, Any]:
    """
    Parse user's response to a reservation conflict and extract their choice.
    
    Args:
        user_response: User's response to conflict options
        
    Returns:
        Dict with parsed conflict resolution choice
    """
    response_lower = user_response.lower().strip()
    
    # Check for numeric choices
    if '1' in response_lower:
        if 'update' in response_lower or 'change' in response_lower:
            return {"conflict_resolution": "update_existing", "choice_number": 1}
        elif 'yes' in response_lower or 'create' in response_lower or 'both' in response_lower:
            return {"conflict_resolution": "proceed", "choice_number": 1}
        else:
            return {"conflict_resolution": "update_existing", "choice_number": 1}
    
    elif '2' in response_lower:
        return {"conflict_resolution": "keep", "choice_number": 2}
    
    elif '3' in response_lower:
        if 'update' in response_lower or 'modify' in response_lower:
            return {"conflict_resolution": "update", "choice_number": 3}
        else:
            return {"conflict_resolution": "cancel_and_create", "choice_number": 3}
    
    # Check for word-based responses
    elif any(word in response_lower for word in ['update', 'change', 'modify']):
        return {"conflict_resolution": "update_existing", "choice_text": "update"}
    
    elif any(word in response_lower for word in ['keep', 'stay', 'existing', 'current']):
        return {"conflict_resolution": "keep", "choice_text": "keep"}
    
    elif any(word in response_lower for word in ['cancel', 'delete', 'remove']):
        return {"conflict_resolution": "cancel_and_create", "choice_text": "cancel"}
    
    elif any(word in response_lower for word in ['yes', 'proceed', 'both', 'create', 'new']):
        return {"conflict_resolution": "proceed", "choice_text": "proceed"}
    
    elif any(word in response_lower for word in ['no', 'cancel', 'stop']):
        return {"conflict_resolution": "keep", "choice_text": "no"}
    
    else:
        return {
            "conflict_resolution": None, 
            "error": "Could not understand the choice. Please respond with 1, 2, or 3, or use words like 'update', 'keep', 'cancel', 'yes', or 'no'."
        }

# Define waiter agent tools after all tools are created
waiter_agent_tools = [extract_food_terms, check_semantic_similarity, get_recommendations, get_detailed_dish_info, get_faq_answer, debug_embedding_indexes, check_user_registration, collect_registration, make_reservation, get_reservations, cancel_reservation, modify_reservation, parse_reservation_conflict_response]
waiter_agent_runnable = waiter_agent_prompt | llm.bind_tools(waiter_agent_tools)

# Only run the command line interface if this script is executed directly
if __name__ == "__main__":
    run_command_line_interface()





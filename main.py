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
    dialog_state: Annotated[
        list[
            Literal[
                "registration_agent",
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
        
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


# --- Registration Agent Tools ---
@tool
def get_customer(config: RunnableConfig, phone_number: str) -> list[dict]:
    """Check if a user is registered by phone number using Supabase and fetch their details"""
    print(f"[DEBUG] get_customer called with phone_number: '{phone_number}'")
    if not phone_number:
        configuration = config.get("configurable", {})
        phone_number = configuration.get("phone_number", None)
        print(f"[DEBUG] get_customer using config phone_number: '{phone_number}'")
    if not phone_number:
        print("[DEBUG] get_customer: No phone number present, raising error")
        raise ValueError("No phone number present")
    print(f"[DEBUG] get_customer: Querying database for phone_number: '{phone_number}'")
    response = supabase.table('customers').select('*').eq('phone_number', phone_number).execute()
    if response.data and len(response.data) > 0:
        customer = response.data[0]
        print(f"[DEBUG] get_customer: Found customer: {customer.get('name')} with phone {phone_number}")
        return {
            "registered": True,
            "name": customer.get("name"),
            "preferences": customer.get("preferences"),
            "allergies": customer.get("allergies"),
            "set_is_registered": True
        }
    else:
        print(f"[DEBUG] get_customer: No customer found for phone_number: '{phone_number}'")
        return {"registered": False, "set_is_registered": False}

@tool
def mark_registration_complete(config: RunnableConfig, phone_number: str = None, name: str = None, reason: str = "Registration process completed") -> dict:
    """Mark the registration process as complete, allowing transition to waiter agent. Creates minimal customer record if needed."""
    print(f"[DEBUG] mark_registration_complete called with phone_number: '{phone_number}', name: '{name}', reason: '{reason}'")
    configuration = config.get("configurable", {})
    
    if not phone_number:
        phone_number = configuration.get("phone_number", None)
        print(f"[DEBUG] mark_registration_complete using config phone_number: '{phone_number}'")
    if not name:
        name = configuration.get("name", None)
        print(f"[DEBUG] mark_registration_complete using config name: '{name}'")
    
    # If we have phone and name, create a minimal customer record
    if phone_number and name:
        try:
            # Check if customer exists
            existing = supabase.table('customers').select('*').eq('phone_number', phone_number).execute()
            if not existing.data:
                # Create minimal customer record
                customer_data = {
                    "phone_number": phone_number,
                    "name": name,
                    "preferences": {},
                    "allergies": [],
                }
                result = supabase.table('customers').insert(customer_data).execute()
        except Exception as e:
            print(f"Error creating minimal customer record: {e}")
    
    return {"success": True, "set_is_registered": True, "message": f"Registration marked complete: {reason}"}

@tool
def create_or_update_customer(config: RunnableConfig, phone_number: str, name: str, preferences: Optional[dict]= None, allergies: Optional[list[str]]=None) -> list[dict]:
    """Register a new customer or update an existing customer in Supabase."""
    print(f"[DEBUG] create_or_update_customer called with phone_number: '{phone_number}', name: '{name}', preferences: {preferences}, allergies: {allergies}")
    configuration = config.get("configurable", {})
    phone_number_config = configuration.get("phone_number", None)
    if not phone_number:
        phone_number = phone_number_config
    name_config = configuration.get("name", None)
    if not name:
        name = name_config
    preferences_config = configuration.get("preferences", {})
    if not preferences:
        preferences = preferences_config
    allergies_config = configuration.get("allergies", [])
    if not allergies:
        allergies = allergies_config

    # Default empty preferences and allergies if None
    if preferences is None:
        preferences = {}
    if allergies is None:
        allergies = []

    customer_data = {
        "phone_number": phone_number,
        "name": name,
        "preferences": preferences,
        "allergies": allergies,
    }

    try:
        # Check if customer exists
        existing = supabase.table('customers').select('*').eq('phone_number', phone_number).execute()
        if existing.data:
            # Update existing customer
            result = supabase.table('customers').update(customer_data).eq('phone_number', phone_number).execute()
        else:
            # Create new customer
            result = supabase.table('customers').insert(customer_data).execute()
        return {"success": True, "set_is_registered": True, "message": "Registration completed successfully", **(result.data[0] if result.data else {})}
    except Exception as e:
        print(f"Error creating/updating customer: {e}")
        return {"success": False, "error": str(e), "set_is_registered": False}


# --- Registration Agent ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

registration_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly restaurant waiter helping with customer registration."
            " Collect phone number, name, preferences, and allergies."
            
            "\n\nCRITICAL: NEVER call tools without phone number first!"
            
            "\n\nWORKFLOW:"
            "\n1. Ask for 10-digit phone number (wait for response)"
            "\n2. Use get_customer tool to check registration"
            "\n3. Existing customers: Confirm preferences, offer updates, use CompleteOrEscalate"
            "\n4. New customers: Get name, offer preferences/allergies collection, use CompleteOrEscalate"
            "\n5. If user wants to skip or asks about menu → mark_registration_complete + CompleteOrEscalate"
            
            "\n\nDon't answer menu questions - transfer to waiter immediately after minimum registration."
        ),
        ("placeholder", "{messages}"),
    ]
)

registration_agent_tools = [get_customer, create_or_update_customer, mark_registration_complete]
registration_agent_runnable = registration_agent_prompt | llm.bind_tools(registration_agent_tools + [CompleteOrEscalate])

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
            
            "\n\nCURRENT DATE & TIME: {current_datetime}"
            "\nIMPORTANT: Use this current time to provide accurate, contextual answers about operating hours."
            "\n• If asked 'Are you open now?' - compare current time with operating hours from FAQ tool"
            "\n• If current time is OUTSIDE operating hours → clearly state 'We are currently CLOSED'"
            "\n• If current time is WITHIN operating hours → state 'Yes, we are currently open'"
            "\n• Always be specific: 'We are open from X to Y' and 'We will open/close at Z'"
            
            "\n\nTOOL SELECTION (be efficient):"
            "\n• Order requests ('I want X') → get_detailed_dish_info(['item1', 'item2']) - single call"
            "\n• Menu questions ('What do you have') → get_recommendations(query)"
            "\n• Dish verification ('Do you have X') → check_semantic_similarity first"
            "\n• Complex searches → extract_food_terms then check_semantic_similarity"
            "\n• Restaurant info (hours, location, policies) → get_faq_answer(query)"
            
            "\n\nQUESTION ROUTING:"
            "\n1. MENU QUESTIONS → Use menu tools (get_recommendations, check_semantic_similarity)"
            "\n2. RESTAURANT INFO → Use get_faq_answer for:"
            "\n   - Operating hours, timings"
            "\n   - Location, address, directions"
            "\n   - Parking availability, contact info"
            "\n   - Services (delivery, takeaway, reservations)"
            "\n   - Payment methods, policies"
            "\n3. If unsure → Try menu tools first, then FAQ if no relevant results"
            
            "\n\nCRITICAL: EXACT DISH VERIFICATION"
            "\nWhen users ask about specific dishes:"
            "\n1. Check if we have the EXACT dish name"
            "\n2. If YES → Answer about our dish"
            "\n3. If NO → 'We don't have [THEIR DISH] but we do have [OUR SIMILAR DISH]'"
            "\n4. NEVER pretend we have dishes we don't serve"
            
            "\n\nFORMATTING:"
            "\n• Multiple items with details → Use markdown tables"
            "\n• Single dish descriptions → Regular text"
            "\n• Include: | Dish Name | Description | Price | Spice Level | Prep Time |"
            "\n• FAQ answers → Use clear headings and bullet points"
            
            "\n\nOPTIMIZATION HINTS:"
            "\n• optimization_hint='direct_menu_query' → Use get_recommendations"
            "\n• optimization_hint='order_request' → Use get_detailed_dish_info directly"
            "\n• optimization_hint='faq_query' → Use get_faq_answer directly"
            
            "\n\nEXAMPLES WITH NATURAL RESPONSES:"
            "\n'I want butter chicken and naan' → get_detailed_dish_info → 'Excellent choice! Our Butter Chicken is...'"
            "\n'Do you have Fish Curry?' → check_semantic_similarity → 'Yes, we have Fish Curry' OR 'We don't have Fish Curry, but our Prawn Curry is similar'"
            "\n'Do you serve non-veg?' → get_recommendations → 'Absolutely! We have a great selection of chicken, mutton, and seafood dishes. What type of non-veg are you in the mood for?'"
            "\n'What veg starters?' → get_recommendations → 'We have some wonderful vegetarian appetizers like...'"
            "\n'What are your timings?' → get_faq_answer → 'We're open from 11 AM to 11 PM every day'"
            "\n'Are you open now?' at 2:30 AM → 'We're currently closed. We'll be open again at 11 AM'"
            "\n'Where are you located?' → get_faq_answer → 'We're located at [address]. It's easy to find with plenty of parking'"
            
            "\n\nAvailable tools: extract_food_terms, check_semantic_similarity, get_detailed_dish_info, get_recommendations, get_faq_answer, debug_embedding_indexes, CompleteOrEscalate"
            
            "\n\nPersonalization: {user_info}"
        ),
        ("placeholder", "{messages}"),
    ]
)

waiter_agent_tools = [extract_food_terms, check_semantic_similarity, get_recommendations, get_detailed_dish_info, get_faq_answer, debug_embedding_indexes]
waiter_agent_runnable = waiter_agent_prompt | llm.bind_tools(waiter_agent_tools + [CompleteOrEscalate])



from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        messages = []
        
        # For registration agent, always provide clear guidance
        if new_dialog_state == "registration_agent":
            from langchain_core.messages import SystemMessage
            messages.append(
                SystemMessage(
                    content=f"You are now the {assistant_name}. The user has provided their phone number. "
                    "Check if they are registered using the get_customer tool with the phone number they provided. "
                    "Follow the registration process step by step as described in your system prompt."
                )
            )
        else:
            # For other agents, use the original logic
            last_message = state["messages"][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call_id = last_message.tool_calls[0]["id"]
                messages.append(
                    ToolMessage(
                        content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                        f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                        " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                        " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                        " Do not mention who you are - just act as the proxy for the assistant.",
                        tool_call_id=tool_call_id,
                    )
                )
        
        return {
            "messages": messages,
            "dialog_state": new_dialog_state,
        }

    return entry_node


# --- Graphs ---

def check_registration_node(state, config):
    # This node doesn't modify state, just used for routing
    # print(f"[DEBUG] check_registration_node: is_registered = {state.get('is_registered', False)}")
    return state

builder.add_node("check_registration", check_registration_node)
builder.add_edge(START, "check_registration")

def check_registration_condition(state):
    is_registered = state.get("is_registered", False)
    print(f"[DEBUG] check_registration_condition: is_registered = {is_registered}")
    if is_registered:
        print("[DEBUG] Routing to waiter_agent")
        return "waiter_agent"
    else:
        print("[DEBUG] Routing to registration_agent")
        return "registration_agent"

builder.add_conditional_edges(
    "check_registration",
    check_registration_condition,
    {
        "registration_agent": "registration_agent",
        "waiter_agent": "waiter_agent"
    }
)

# builder.add_node(
#     "enter_registration_agent",
#     create_entry_node("Registration Assistant", "registration_agent"),
# )
def registration_agent_with_debug(state: State, config: RunnableConfig):
    print(f"[DEBUG] registration_agent called with {len(state.get('messages', []))} messages")
    print(f"[DEBUG] registration_agent is_registered = {state.get('is_registered', False)}")
    agent = Assistant(registration_agent_runnable)
    return agent(state, config)

builder.add_node("registration_agent", registration_agent_with_debug)

# Create a custom tool node that handles state updates
def registration_tools_node(state: State):
    """Custom tool node that updates registration status based on tool results"""
    # Check if we're blocking a CompleteOrEscalate due to incomplete registration
    last_message = state["messages"][-1] if state["messages"] else None
    is_registered = state.get("is_registered", False)
    
    if (last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls and 
        any(tc["name"] == CompleteOrEscalate.__name__ for tc in last_message.tool_calls) and 
        not is_registered):
        
        # Block the CompleteOrEscalate tool call but preserve any message content
        from langchain_core.messages import ToolMessage, AIMessage
        
        # Create a tool message saying the escalation was blocked
        blocked_messages = []
        for tc in last_message.tool_calls:
            if tc["name"] == CompleteOrEscalate.__name__:
                blocked_messages.append(
                    ToolMessage(
                        content="Registration must be completed first. Please provide your phone number to continue.",
                        tool_call_id=tc["id"],
                    )
                )
        
        return {
            "messages": blocked_messages,
            "is_registered": is_registered
        }
    
    # Call the standard tool node for other cases
    tool_node = create_tool_node_with_fallback(registration_agent_tools)
    result = tool_node.invoke(state)
    
    # Check if any tool results indicate registration status change
    new_is_registered = state.get("is_registered", False)
    
    if "messages" in result:
        for message in result["messages"]:
            if hasattr(message, 'content'):
                content = str(message.content)
                
                # Check for various patterns that indicate registration status
                if "set_is_registered" in content:
                    if "true" in content.lower() or "'registered': true" in content.lower():
                        new_is_registered = True
                    elif "false" in content.lower() or "'registered': false" in content.lower():
                        new_is_registered = False
                
                # Also check for success patterns
                if "'success': true" in content.lower() or '"success": true' in content.lower():
                    new_is_registered = True
                
                # Check for registration completion messages
                if "registration marked complete" in content.lower() or "registration completed successfully" in content.lower():
                    new_is_registered = True
    
    return {
        **result,
        "is_registered": new_is_registered
    }

builder.add_node("registration_tools", registration_tools_node)




# Add transition node
def transition_to_waiter_node(state: State) -> dict:
    """Transition from registration to waiter agent after successful registration"""
    print("[DEBUG] Transitioning to waiter agent after successful registration")
    
    # Handle any pending CompleteOrEscalate tool calls
    messages = []
    if state["messages"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Create tool responses for any CompleteOrEscalate calls
            for tc in last_message.tool_calls:
                if tc["name"] == CompleteOrEscalate.__name__:
                    from langchain_core.messages import ToolMessage
                    messages.append(
                        ToolMessage(
                            content="Registration completed successfully. You are now connected to our menu assistant.",
                            tool_call_id=tc["id"],
                        )
                    )
    
    return {
        "dialog_state": "waiter_agent",
        "messages": messages
    }

builder.add_node("transition_to_waiter", transition_to_waiter_node)
builder.add_edge("transition_to_waiter", "waiter_agent")

builder.add_edge("registration_tools", "registration_agent")

# Simplified registration routing - similar to waiter agent
def registration_route_condition(state: State):
    is_registered = state.get("is_registered", False)
    route = tools_condition(state)
    
    # Check for CompleteOrEscalate tool calls
    if state["messages"] and hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls:
        tool_calls = state["messages"][-1].tool_calls
        did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
        
        if did_cancel and is_registered:
            print("[DEBUG] CompleteOrEscalate with registered user - transitioning to waiter")
            return "transition_to_waiter"
        elif did_cancel and not is_registered:
            print("[DEBUG] CompleteOrEscalate blocked - user not registered")
            return "registration_tools"  # Block the escalation
    
    # Standard tool routing
    if route == "tools":
        return "registration_tools"
    else:
        # FIXED: Don't auto-transition just because user is registered
        # Only transition when registration agent explicitly calls CompleteOrEscalate
        print(f"[DEBUG] Registration agent ending turn - waiting for user input (is_registered={is_registered})")
        return END

builder.add_conditional_edges(
    "registration_agent", 
    registration_route_condition,
    {
        "registration_tools": "registration_tools",
        "transition_to_waiter": "transition_to_waiter", 
        END: END
    }
)



# Custom ReAct Waiter Agent
class ReactWaiterAgent(Assistant):
    def __init__(self, runnable: Runnable):
        super().__init__(runnable)
    
    def __call__(self, state: State, config: RunnableConfig):
        waiter_start = time.time()
        print(f"[TIMING] ReactWaiterAgent started processing")
        
        # Extract user info from config or state
        configuration = config.get("configurable", {})
        phone_number = configuration.get("phone_number", None)
        
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
            
            # For order/confirmation requests, be more direct
            order_patterns = [
                'i want', 'i would like', 'can i have', 'please get me', 'order',
                'i\'ll have', 'give me', 'i need', 'can you bring'
            ]
            
            if any(pattern in user_query for pattern in simple_patterns):
                print(f"[DEBUG] Detected simple menu query, optimizing workflow")
                # Add a hint to the agent to be more direct
                state_with_user_info["optimization_hint"] = "direct_menu_query"
            elif any(pattern in user_query for pattern in order_patterns):
                print(f"[DEBUG] Detected order request, optimizing for multi-item processing")
                # Add a hint for order processing
                state_with_user_info["optimization_hint"] = "order_request"
        
        result = super().__call__(state_with_user_info, config)
        
        waiter_end = time.time()
        print(f"[TIMING] ReactWaiterAgent completed in {waiter_end - waiter_start:.3f}s")
        
        return result

# Wrap waiter agent with ReAct logic
def waiter_agent_with_debug(state: State, config: RunnableConfig):
    # BULLETPROOF REGISTRATION CHECK - Block waiter agent if not registered
    is_registered = state.get("is_registered", False)
    print(f"[DEBUG] waiter_agent called, is_registered = {is_registered}")
    
    if not is_registered:
        print("[DEBUG] BLOCKING waiter agent - user not registered")
        from langchain_core.messages import AIMessage
        # Force the agent to ask for registration instead of answering menu questions
        return {
            "messages": [AIMessage(content="I'd be happy to help you with our menu! However, first, may I please have your 10-digit phone number? This will help me provide you with personalized service and assistance.")]
        }
    
    # Ensure system is initialized before processing menu queries
    ensure_system_initialized()
    
    # Check if this is the first time waiter agent is called (after registration)
    messages = state.get("messages", [])
    if messages and any("Registration completed successfully" in str(msg.content) for msg in messages if hasattr(msg, 'content')):
        # This is a transition from registration - provide a welcome message
        from langchain_core.messages import AIMessage
        print("[DEBUG] First time waiter agent - providing welcome message")
        return {
            "messages": [AIMessage(content="Perfect! Your registration is now complete. I'm here to help you explore our delicious menu. What would you like to know about our dishes today? You can ask me about specific items, ingredients, dietary options, or I can recommend something based on your preferences!")]
        }
    
    react_agent = ReactWaiterAgent(waiter_agent_runnable)
    return react_agent(state, config)

builder.add_node("waiter_agent", waiter_agent_with_debug)
builder.add_node("waiter_tools", create_tool_node_with_fallback(waiter_agent_tools))
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

    state = {"messages": [], "is_registered": False}
    print("Welcome to Neemsi! Please provide your phone number to help you further!")
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

# Only run the command line interface if this script is executed directly
if __name__ == "__main__":
    run_command_line_interface()

# Add startup initialization
def initialize_restaurant_system():
    """
    Pre-warm all expensive operations during app startup to eliminate first-request latency.
    Call this once when the application starts.
    """
    init_start = time.time()
    print("\n" + "="*60)
    print("🚀 INITIALIZING NEEMSI RESTAURANT SYSTEM")
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



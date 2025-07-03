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
            "You are a helpful restaurant registration assistant. You have a persona of an experienced waiter who is very friendly and helpful, so speak like a courteous waiter."
            "Don't say your are registered with us or not, or I see that you're already in our system, just say based on the past visits here are your preferences, and then ask them if they want to update them."
            "Your job is to collect the user's phone number, name, food preferences, and allergies or update those details"
            "\n\nCRITICAL RULE: NEVER CALL ANY TOOLS UNTIL THE USER PROVIDES THEIR PHONE NUMBER"
            "\n- Do NOT call get_customer tool until you have received a 10-digit phone number from the user"
            "\n- Do NOT call any other tools until you have the phone number"
            "\n- Do NOT assume or guess phone numbers"
            "\n- Do NOT use default or example phone numbers"
            "\n- ALWAYS wait for the user's response before proceeding to the next step"
            "\n- If the user asks about menu items, politely ask for their phone number first"
            "Perform the tasks in the following order:\n"
            "\n1. FIRST: Ask the user for their 10-digit phone number. Do not ask users if they are registered or not. Simply request the phone number and wait for their response."
            "\n2. ONLY AFTER receiving the phone number: Use the get_customer tool to check if they are registered."
            "\n3. If the user is already registered, confirm their preferences (dietary preference: Veg/Non-veg/Vegan, Spice-level: Bland/Less Spicy/Medium Spicy/Spicy, Cuising: North Indian/South Indian/Chinese/Mughlai etc) and allergies (Gluten, Soy, etc) and offer to update them. If they want to update, use create_or_update_customer. If they don't want to update, use mark_registration_complete. After this is done, use CompleteOrEscalate with cancel=True to transfer to the waiter."
            "\n4. If the user is not registered, ask user for their name first. Once you have the name, offer to collect preferences (dietary, spice level, cuisine) and allergies. If they provide preferences, use create_or_update_customer. If they want to skip preferences or say 'later', use mark_registration_complete with just the name. After registration is complete, use CompleteOrEscalate with cancel=True to transfer to the waiter."
            "\n5. After collecting minimum required information and doing basic registration, if at any point the user wants to skip setting preferences, says 'later', 'will do that later', 'skip', or asks about menu/food items, use mark_registration_complete and then immediately use CompleteOrEscalate with cancel=True and reason='Registration complete, user ready for menu assistance' However the basic registration using atleast phone number needs to be done (and also name if the user is new)."
            "Do not answer any menu related questions, just use CompleteOrEscalate to transfer to the waiter."
            "\n\nIMPORTANT: As soon as you have the minimum required information (phone number and name for new users, or confirmed/updated preferences for existing users), immediately use CompleteOrEscalate to transfer them to the waiter assistant who can help with menu questions."
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

def setup_cypher_chain():
    print(f"[DEBUG] Setting up Neo4j connection...")
    print(f"[DEBUG] NEO4J_URI: {os.getenv('NEO4J_URI')}")
    print(f"[DEBUG] NEO4J_USER: {os.getenv('NEO4J_USER')}")
    
    neo4j_graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True
        )
    print(f"[DEBUG] Neo4j connection established")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"[DEBUG] Refreshing schema...")
    neo4j_graph.refresh_schema()
    print(f"[DEBUG] Schema refreshed. Schema content: {neo4j_graph.schema}")

    
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

    print(f"[DEBUG] Creating GraphCypherQAChain...")
    try:
        cypher_chain = GraphCypherQAChain.from_llm(
            llm,
            graph=neo4j_graph,
            verbose=True,
            validate_cypher=True,
            cypher_prompt=cypher_prompt,
            return_direct=True,  # Return raw results directly to avoid LLM processing issues
            top_k=20,
            input_key="query",  # Added input_key
            allow_dangerous_requests=True,
        )
        print(f"[DEBUG] GraphCypherQAChain created successfully")
    except Exception as e:
        print(f"[DEBUG] Error creating GraphCypherQAChain: {e}")
        raise e
    
    return cypher_chain, neo4j_graph

def kg_answer(query: str) -> Dict[str, Any]:
    print(f"[DEBUG] kg_answer called with query: {query}")
    
    cypher_chain = None
    neo4j_graph = None
    symbolic_response = "No symbolic answer available"
    vector_result = "No semantic answer available"
    
    try:
        cypher_chain, neo4j_graph = setup_cypher_chain()
        print(f"[DEBUG] Cypher chain setup complete")
        
        # Print the schema to debug
        schema = neo4j_graph.schema
        print(f"[DEBUG] Neo4j Schema: {schema}")
        
        print(f"[DEBUG] Invoking cypher chain...")
        # Pass both schema and query as the prompt template expects both
        symbolic_response = cypher_chain.invoke({
            "schema": neo4j_graph.schema,
            "query": query
        })
        
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
                symbolic_response = neo4j_graph.query(direct_cypher)
                print(f"[DEBUG] Direct query result: {symbolic_response}")
            except Exception as direct_e:
                print(f"[DEBUG] Direct query also failed: {direct_e}")
                symbolic_response = f"Error in both chain and direct query: {e}"
        else:
            symbolic_response = f"Error generating symbolic answer: {e}"
    
    # Handle semantic search with unified embedding index
    if neo4j_graph:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            query_embedding = embeddings.embed_query(query)
            
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
            
            vector_result = neo4j_graph.query(unified_cypher, params={"embedding": query_embedding, "k": 15})
            
            # Remove duplicates and sort by score
            unique_results = {}
            for item in vector_result:
                dish_name = item['name']
                if dish_name not in unique_results or item['score'] > unique_results[dish_name]['score']:
                    unique_results[dish_name] = item
            
            vector_result = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]
            print(f"[DEBUG] Vector result using unified embedding: {len(vector_result)} matches")
            
        except Exception as e:
            print(f"[DEBUG] Error in semantic query: {e}")
            vector_result = f"Error in semantic search: {e}"

    return {
        "symbolic_answer": symbolic_response,
        "semantic_answer": vector_result,
        "formatted": f"SYMBOLIC ANSWER:\n{symbolic_response}\n\nSEMANTIC ANSWER:\n{vector_result}"
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
    print(f"[DEBUG] extract_food_terms called with query: {query}")
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
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
        
        response = llm.invoke(extraction_prompt)
        
        # Parse JSON response with improved error handling
        import json
        try:
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
            print(f"[DEBUG] LLM extracted terms: {extracted_data}")
            
            return {
                "success": True,
                "extracted_terms": extracted_data,
                "primary_search_terms": extracted_data.get("dish_names", []) + extracted_data.get("ingredients", [])[:3],
                "secondary_search_terms": extracted_data.get("categories", []) + extracted_data.get("descriptive_terms", [])[:2]
            }
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing failed: {e}, content: {content[:100]}")
            
            # Enhanced fallback extraction
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
            
            print(f"[DEBUG] Fallback extraction: {fallback_data}")
            
            return {
                "success": False,
                "extracted_terms": fallback_data,
                "primary_search_terms": fallback_data["ingredients"] + fallback_data["dish_names"],
                "secondary_search_terms": fallback_data["descriptive_terms"] + fallback_data["categories"],
                "error": "Used enhanced fallback extraction"
            }
            
    except Exception as e:
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
    try:
        neo4j_graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True
        )
        
        # Check for vector indexes
        index_check_query = "SHOW INDEXES"
        indexes = neo4j_graph.query(index_check_query)
        
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
                result = neo4j_graph.query(count_query)
                embedding_counts[f"{label}_with_embeddings"] = result[0]['count'] if result else 0
            except Exception as e:
                embedding_counts[f"{label}_with_embeddings"] = f"Error: {e}"
        
        # Check total searchable nodes
        try:
            total_searchable = neo4j_graph.query("MATCH (n:Searchable) RETURN count(n) as count")
            total_count = total_searchable[0]['count'] if total_searchable else 0
        except Exception as e:
            total_count = f"Error: {e}"
        
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
    print(f"[DEBUG] check_semantic_similarity called with terms: {extracted_terms}")
    
    try:
        # Setup Neo4j connection
        neo4j_graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True
        )
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        all_results = []
        match_confidence = 0.0
        
        # STEP 1: Search by dish names (highest priority)
        dish_names = extracted_terms.get("dish_names", [])
        if dish_names:
            print(f"[DEBUG] Searching for dish names: {dish_names}")
            for dish_name in dish_names[:2]:  # Limit to top 2 dish names
                try:
                    query_embedding = embeddings.embed_query(dish_name)
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
                    
                    results = neo4j_graph.query(dish_name_cypher, 
                                              params={"embedding": query_embedding, "k": 8, "search_term": dish_name})
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.9)
                        print(f"[DEBUG] Found {len(results)} dish embedding matches for '{dish_name}'")
                
                except Exception as e:
                    print(f"[DEBUG] Error in dish embedding search for '{dish_name}': {e}")
        
        # STEP 2: Search by ingredients
        ingredients = extracted_terms.get("ingredients", [])
        if ingredients and len(all_results) < 8:  # Only if we need more results
            print(f"[DEBUG] Searching for ingredients: {ingredients}")
            for ingredient in ingredients[:3]:  # Limit to top 3 ingredients
                try:
                    query_embedding = embeddings.embed_query(ingredient)
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
                    
                    results = neo4j_graph.query(ingredient_cypher, 
                                              params={"embedding": query_embedding, "k": 6, "search_term": ingredient})
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.75)
                        print(f"[DEBUG] Found {len(results)} ingredient embedding matches for '{ingredient}'")
                
                except Exception as e:
                    print(f"[DEBUG] Error in ingredient embedding search for '{ingredient}': {e}")
        
        # STEP 3: Search by categories
        categories = extracted_terms.get("categories", [])
        if categories and len(all_results) < 8:
            print(f"[DEBUG] Searching for categories: {categories}")
            for category in categories[:2]:
                try:
                    query_embedding = embeddings.embed_query(category)
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
                    
                    results = neo4j_graph.query(category_cypher, 
                                              params={"embedding": query_embedding, "k": 5, "search_term": category})
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.65)
                        print(f"[DEBUG] Found {len(results)} category embedding matches for '{category}'")
                
                except Exception as e:
                    print(f"[DEBUG] Error in category embedding search for '{category}': {e}")
        
        # STEP 4: Mixed search for descriptive terms and general queries
        descriptive_terms = extracted_terms.get("descriptive_terms", []) + extracted_terms.get("dietary_preferences", [])
        if descriptive_terms and len(all_results) < 10:
            print(f"[DEBUG] Searching for descriptive terms: {descriptive_terms}")
            combined_description = " ".join(descriptive_terms[:3])
            try:
                query_embedding = embeddings.embed_query(combined_description)
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
                
                results = neo4j_graph.query(mixed_cypher, 
                                          params={"embedding": query_embedding, "k": 12, "search_term": combined_description})
                if results:
                    all_results.extend(results)
                    match_confidence = max(match_confidence, 0.6)
                    print(f"[DEBUG] Found {len(results)} mixed embedding matches")
            
            except Exception as e:
                print(f"[DEBUG] Error in mixed embedding search: {e}")
        
        # STEP 5: Final fallback to traditional string matching
        if not all_results:
            print(f"[DEBUG] No embedding matches found, trying traditional string matching...")
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
                    results = neo4j_graph.query(fallback_query)
                    if results:
                        all_results.extend(results)
                        match_confidence = max(match_confidence, 0.3)
                        print(f"[DEBUG] Found {len(results)} string matches for '{term}'")
                except Exception as e:
                    print(f"[DEBUG] Error in string matching for '{term}': {e}")
        
        # Remove duplicates and sort by score
        unique_results = {}
        for item in all_results:
            dish_name = item['name']
            if dish_name not in unique_results or item['score'] > unique_results[dish_name]['score']:
                unique_results[dish_name] = item
        
        final_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:10]
        
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
        
        print(f"[DEBUG] Unified semantic search result: {len(final_results)} matches, confidence: {match_confidence:.2f}")
        return result
        
    except Exception as e:
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
    print(f"[DEBUG] get_recommendations called with query: {query}")
    try:
        result = kg_answer(query)
        # Filter out schema from debug output - only show semantic_answer and formatted result
        filtered_result = {
            "semantic_answer": result.get("semantic_answer", "No semantic answer"),
            "formatted": result.get("formatted", "No formatted answer available")
        }
        print(f"[DEBUG] kg_answer returned: {filtered_result}")
        return {
            "success": True,
            "response": result,
            # "message": result.get("message", ""),
        }
    except Exception as e:
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
    print(f"[DEBUG] get_detailed_dish_info called with dishes: {dish_names}")
    
    if not dish_names:
        return {
            "success": False,
            "error": "No dish names provided",
            "dishes": []
        }
    
    try:
        # Setup Neo4j connection
        neo4j_graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True
        )
        
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
                results = neo4j_graph.query(dish_info_query)
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
        
        return {
            "success": True,
            "dishes": detailed_info,
            "total_dishes": len(detailed_info),
            "message": f"Retrieved detailed information for {len(detailed_info)} dishes"
        }
        
    except Exception as e:
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
            "You are a helpful restaurant agent with the persona of an experienced waiter who is very friendly and helpful, so speak like a courteous waiter."
            "Your goal is to help guests with their questions and requests about the menu, dishes, recommendations, ingredients, and dietary information."
            "You should always use the tools to answer the user's question and not make up any information."
            "For any specific dish related question, you should first check if we have the exact dish in the menu."
            "If we do not have the exact dish, you should use the check_semantic_similarity tool to find a similar dish."
    

            "\n\nCRITICAL: BE BRUTALLY HONEST ABOUT EXACT MENU AVAILABILITY"
            "\nWhen users ask about specific dishes by name, you MUST follow this verification process:"
            "\n1. EXACT MATCH CHECK: First verify if the exact dish name exists in our menu"
            "\n2. IF EXACT MATCH FOUND: Proceed to answer about that dish"
            "\n3. IF EXACT MATCH NOT FOUND: Be crystal clear that we DON'T have that specific dish"
            "\n4. OFFER ALTERNATIVES: Only then mention similar dishes we DO have"
            "\n"
            "\nMANDATORY RESPONSE FORMAT FOR MISSING DISHES:"
            "\n- 'We don't have [EXACT DISH NAME] on our menu, but we do have [SIMILAR DISH NAME] which is [description]'"
            "\n- 'I don't see [EXACT DISH NAME] in our menu, however we offer [SIMILAR DISH NAME] which [explanation]'"
            "\n- NEVER answer questions about dishes as if we serve them when we don't"
            "\n- NEVER assume similar dishes are the same as what the customer asked for"
            "\n- ALWAYS use the EXACT dish names from our menu in responses"
            "\n"
            "\nEXAMPLE - USER ASKS: 'How spicy is your Saag Paneer?'"
            "\nCORRECT RESPONSE: 'We don't have Saag Paneer on our menu, but we do have Palak Paneer which is cottage cheese in spiced spinach puree. Our Palak Paneer has a spice level of 2/5...'"
            "\nINCORRECT RESPONSE: 'Our Saag Paneer has a spice level of...' (This is misleading!)"
            "\n"
            "\nEXAMPLE - USER ASKS: 'What breads go with your Saag Paneer?'"
            "\nCORRECT RESPONSE: 'We don't have Saag Paneer on our menu, but we do have Palak Paneer which is similar. For our Palak Paneer, both naan and tandoori roti pair beautifully...'"
            "\nINCORRECT RESPONSE: 'When it comes to pairing bread with Saag Paneer...' (This implies we have it!)"
            
            "\n\nFORMATTING GUIDELINES FOR CLEAR PRESENTATION:"
            "\nWhen presenting multiple dishes or items with structured information, use markdown tables for better readability:"
            "\n- USE TABLES when showing multiple dishes/items with attributes like: price, spice level, prep time, ingredients, categories"
            "\n- USE TABLES for comparison data, menu sections, or lists with consistent properties"
            "\n- DON'T USE TABLES for: single dish descriptions, simple recommendations, conversational responses, or general information"
            
            "\nTable Format Guidelines:"
            "\n- Use markdown table syntax: | Column 1 | Column 2 | Column 3 |"
            "\n- Include relevant columns: Dish Name, Description, Price, Spice Level, Prep Time (as applicable)"
            "\n- Keep descriptions concise in tables (1-2 lines max)"
            "\n- Use emojis sparingly for visual appeal: 🌶️ for spice, ⏱️ for time, 💰 for price"
            "\n- Always provide a brief intro before the table explaining what it shows"
            
            "\nEXAMPLE TABLE FORMAT:"
            "\nHere are our paneer-based main course options:"
            "\n| Dish Name | Description | ₹ Price | 🌶️ Spice Level | ⏱️ Prep Time |"
            "\n|-----------|-------------|-------|-------------|-----------|"
            "\n| Paneer Tikka Masala | Charred paneer in tomato gravy | ₹230 | 3/5 |  20 min |"
            "\n| Palak Paneer | Cottage cheese in spiced spinach | ₹220 |  2/5 | 15 min |"
            
            "\nDECISION PROCESS FOR TABLE USE:"
            "\n1. If user asks for 'options', 'varieties', 'list of dishes', or 'what do you have' → Use tables"
            "\n2. If response includes 3+ dishes with similar attributes → Use tables"
            "\n3. If comparing dishes or showing menu categories → Use tables"
            "\n4. If describing single dish in detail → Use regular text"
            "\n5. If giving general recommendations or conversational response → Use regular text"
            "\n6. If explaining ingredients or preparation methods → Use regular text"
            
            "\nALWAYS include a friendly introduction before tables and a helpful conclusion after tables."
            
            "\n\nFOLLOW THIS MANDATORY VERIFICATION PROCESS:"
            "\n1. THOUGHT: Analyze user's query to identify EXACT dish names mentioned"
            "\n2. ACTION: Use extract_food_terms to categorize terms, paying special attention to dish_names"
            "\n3. THOUGHT: Review extracted dish_names - these are what user specifically asked about"
            "\n4. ACTION: Use check_semantic_similarity to search for these exact dishes in our menu"
            "\n5. CRITICAL THOUGHT: EXACT MATCH VERIFICATION - For each dish user mentioned:"
            "\n   - Does our search return the EXACT dish name the user asked for?"
            "\n   - If YES → We have that dish, proceed with information about it"
            "\n   - If NO but similar dishes found → We DON'T have their dish, but have alternatives"
            "\n   - If NO matches → We don't have anything similar"
            "\n6. ACTION: Get detailed information using get_recommendations"
            "\n7. MANDATORY RESPONSE FORMAT:"
            "\n   - For EXACT matches: Answer directly about our dish"
            "\n   - For NO exact matches: 'We don't have [USER'S DISH] but we do have [OUR SIMILAR DISH]'"
            "\n   - NEVER pretend we have dishes we don't actually serve"
            
            "\n\nSEARCH RESULT INTERPRETATION RULES:"
            "\n- SYMBOLIC_ANSWER = Exact database matches (this is your primary source of truth)"
            "\n- SEMANTIC_ANSWER = Similar items found via embedding search (these are alternatives/suggestions)"
            "\n- ALWAYS check SYMBOLIC_ANSWER first for exact matches"
            "\n- Only use SEMANTIC_ANSWER when SYMBOLIC_ANSWER is empty or doesn't contain exact matches"
            "\n- If user asks for 'Saag Paneer' but symbolic shows 'Palak Paneer' → We DON'T have Saag Paneer"
            "\n- If user asks for 'Butter Chicken' and symbolic shows 'Butter Chicken' → We DO have Butter Chicken"
            "\n"
            "\nEMBEDDING SEARCH ADVANTAGES:"
            "\n- Unified embedding index: Single index for all searchable nodes (dishes, ingredients, categories, cuisines)"
            "\n- Intelligent routing: Automatically finds dishes through multiple pathways (direct dish match, ingredient match, category match)"
            "\n- Semantic understanding: Matches meaning, not just keywords, even with typos or variations"
            "\n- Multi-stage search: Prioritizes exact matches while providing semantic fallbacks"
            
            "\n\nEXAMPLE NATURAL WORKFLOW:"
            "\nUser: 'What's the difference between butter chicken and chicken tikka masala?'"
            "\nTHOUGHT: User is asking about specific dishes - let me see if I have those exact dishes in the menu"
            "\nACTION: check_semantic_similarity(extracted_terms_dict)"
            "\nOBSERVATION: Found 'Butter Chicken' (exact match) and 'Chicken Tikka' (similar but not exact - no 'Chicken Tikka Masala')"
            "\nTHOUGHT: User asked for 'Chicken Tikka Masala' but we only have 'Chicken Tikka' - I need to clarify this"
            "\nACTION: get_recommendations('Butter Chicken and Chicken Tikka comparison')"
            "\nRESPONSE: 'I'd be happy to help! We have Butter Chicken on our menu, but we don't have Chicken Tikka Masala. We do have Chicken Tikka, which is different. Would you like me to explain the difference between Butter Chicken and Chicken Tikka instead?'"
            
            "\n\nEXAMPLE NATURAL WORKFLOW 2 (EXACT DISH VERIFICATION):"
            "\nUser: 'How spicy is the Andhra-style Chicken on a scale of 1–10? Can the kitchen make it milder?'"
            "\nTHOUGHT: User is asking about a specific dish 'Andhra-style Chicken' - I need to verify if we have this EXACT dish"
            "\nACTION: check_semantic_similarity(extracted_terms_dict)"
            "\nOBSERVATION: Search results show 'Andhra Mutton Curry' but NO 'Andhra-style Chicken' found"
            "\nTHOUGHT: Critical - the user asked for 'Andhra-style Chicken' but we don't have that exact dish. I found 'Andhra Mutton Curry' which is similar but different. I MUST be transparent about this."
            "\nACTION: get_recommendations('Andhra cuisine dishes')"
            "\nRESPONSE: 'We don't have Andhra-style Chicken on our menu, but we do have Andhra Mutton Curry which is a spicy traditional dish from Andhra Pradesh. Our Andhra Mutton Curry has a spice level of 4/5, and yes, our kitchen can definitely make it milder to suit your preference!'"
            "\n"
            "\nEXAMPLE NATURAL WORKFLOW 3 (SAAG PANEER CASE):"
            "\nUser: 'Which breads go best with your Saag Paneer—naan or tandoori roti?'"
            "\nTHOUGHT: User specifically asked about 'Saag Paneer' - I need to check if we have this exact dish"
            "\nACTION: check_semantic_similarity(extracted_terms_dict)"
            "\nOBSERVATION: Search results show 'Palak Paneer' but NO 'Saag Paneer' found"
            "\nTHOUGHT: The user asked for 'Saag Paneer' but we only have 'Palak Paneer'. These are similar (both are spinach with paneer) but different names. I must clarify we don't have Saag Paneer."
            "\nACTION: get_recommendations('Palak Paneer bread pairing')"
            "\nRESPONSE: 'We don't have Saag Paneer on our menu, but we do have Palak Paneer which is cottage cheese in spiced spinach puree. For our Palak Paneer, both naan and tandoori roti pair beautifully! The soft naan is great for scooping, while the tandoori roti offers a heartier, more rustic pairing.'"

            "\n\nEXAMPLE FOR GENERAL QUERY:"
            "\nUser: 'Do you have any creamy paneer dishes?'"
            "\nTHOUGHT: User wants creamy paneer dishes, this is a general category search"
            "\nACTION: extract_food_terms('creamy paneer dishes')"
            "\nOBSERVATION: Extracted: {{'ingredients': ['paneer'], 'descriptive_terms': ['creamy'], ...}}"
            "\nACTION: check_semantic_similarity(extracted_terms_dict)"
            "\nOBSERVATION: Found exact matches: Paneer Butter Masala, Shahi Paneer, etc."
            "\nTHOUGHT: Great! We have exactly what they're looking for"
            "\nACTION: get_recommendations('creamy paneer dishes')"
            "\nRESPONSE: 'Absolutely! We have several wonderfully creamy paneer dishes. Our Paneer Butter Masala and Shahi Paneer are particularly popular...'"
            
            "\n\nKEY IMPROVEMENTS:"
            "\n- Unified semantic search through single embedding index with intelligent routing"
            "\n- Multi-pathway matching (dish→ingredient→category→cuisine) ensures comprehensive results"
            "\n- Confidence scoring helps determine response certainty"
            "\n- Categorized extraction improves search precision"
            
            "\n\nAvailable tools:"
            "\n1. extract_food_terms: LLM-powered categorization of food terms for optimized embedding search"
            "\n2. check_semantic_similarity: Advanced embedding-based search using unified menu_embed index"
            "\n3. get_detailed_dish_info: Get comprehensive details for specific dishes (use after embedding search)"
            "\n4. get_recommendations: General menu information and recommendations (provides comprehensive dish info)"
            "\n5. debug_embedding_indexes: Check if unified embedding index exists (use if embeddings fail)"
            "\n6. CompleteOrEscalate: Transfer control if needed"
            
            "\n\nPersonalization info:\n<user_information>\n{user_info}\n</user_information>"
        ),
        ("placeholder", "{messages}"),
    ]
)

waiter_agent_tools = [extract_food_terms, check_semantic_similarity, get_recommendations, get_detailed_dish_info, debug_embedding_indexes]
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
        # Extract user info from config or state
        configuration = config.get("configurable", {})
        phone_number = configuration.get("phone_number", None)
        
        # Create user_info from available data
        user_info = f"Phone: {phone_number}" if phone_number else "No user info available"
        
        # Add user_info to state for the prompt - the LLM will handle term extraction via tools
        state_with_user_info = {**state, "user_info": user_info}
        
        return super().__call__(state_with_user_info, config)

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



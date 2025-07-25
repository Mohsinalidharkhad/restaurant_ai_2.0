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
import yaml
import os.path as path

load_dotenv()

# Import database connections from new modules
try:
    # Try relative imports first (when run as module)
    from ..data.supabase_client import get_supabase_client
    from ..data.neo4j_client import get_neo4j_connection, get_cypher_chain, setup_cypher_chain
    from ..data.openai_client import get_embedding_model
    from ..data.connections import get_cached_query_result, cache_query_result
    
    # Import tools from new modules
    from ..tools.search_tools import extract_food_terms, debug_embedding_indexes, check_semantic_similarity
    from ..tools.menu_tools import get_recommendations, get_detailed_dish_info
    from ..tools.customer_tools import check_user_registration, collect_registration
    from ..tools.reservation_tools import make_reservation, get_reservations, cancel_reservation, modify_reservation, parse_reservation_conflict_response
    from ..tools.faq_tools import get_faq_answer
except ImportError:
    # Fall back to absolute imports (when run as script)
    import sys
    import os
    # Add the project root to the path if not already there
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from backend.data.supabase_client import get_supabase_client
    from backend.data.neo4j_client import get_neo4j_connection, get_cypher_chain, setup_cypher_chain
    from backend.data.openai_client import get_embedding_model
    from backend.data.connections import get_cached_query_result, cache_query_result
    
    # Import tools from new modules
    from backend.tools.search_tools import extract_food_terms, debug_embedding_indexes, check_semantic_similarity
    from backend.tools.menu_tools import get_recommendations, get_detailed_dish_info
    from backend.tools.customer_tools import check_user_registration, collect_registration
    from backend.tools.reservation_tools import make_reservation, get_reservations, cancel_reservation, modify_reservation, parse_reservation_conflict_response
    from backend.tools.faq_tools import get_faq_answer

# Get Supabase client using the new connection pooling
supabase = get_supabase_client()

# Config loading utility
def load_prompts_config():
    """Load prompts from the YAML configuration file"""
    config_path = path.join(path.dirname(__file__), '../../config/prompts/prompts.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate that all required sections exist
        required_sections = ['waiter_agent', 'cypher_generation', 'food_extraction', 'evaluation']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            print(f"❌ ERROR: Missing required sections in prompts.yaml: {missing_sections}")
            print(f"📁 Please ensure prompts.yaml contains all required sections")
            import sys
            sys.exit(1)
        
        # Validate subsections exist
        if 'system_message' not in config['waiter_agent']:
            print("❌ ERROR: Missing 'system_message' in waiter_agent section")
            import sys
            sys.exit(1)
            
        if 'template' not in config['cypher_generation']:
            print("❌ ERROR: Missing 'template' in cypher_generation section")
            import sys
            sys.exit(1)
            
        if 'instruction' not in config['food_extraction']:
            print("❌ ERROR: Missing 'instruction' in food_extraction section")
            import sys
            sys.exit(1)
            
        if 'restaurant_evaluator' not in config['evaluation']:
            print("❌ ERROR: Missing 'restaurant_evaluator' in evaluation section")
            import sys
            sys.exit(1)
        
        return config
        
    except FileNotFoundError:
        print(f"❌ ERROR: prompts.yaml not found at {config_path}")
        print(f"📁 Please ensure prompts.yaml exists in the project root")
        print(f"💡 You can copy from prompts_example.yaml or check the repository")
        import sys
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ ERROR: Invalid YAML syntax in prompts.yaml: {e}")
        print(f"🔧 Please check your YAML formatting (indentation, quotes, etc.)")
        print(f"💡 You can validate YAML online or use a YAML linter")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Unexpected error loading prompts.yaml: {e}")
        import sys
        sys.exit(1)

# Load prompts configuration - fail fast if not available
PROMPTS_CONFIG = load_prompts_config()


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

# Connection functions are now imported from backend.data modules

# kg_answer function moved to backend.services.knowledge_graph to avoid circular imports


# Search tools are now imported from backend.tools.search_tools

# Customer tools are now imported from backend.tools.customer_tools

# Reservation tools are now imported from backend.tools.reservation_tools

# FAQ tools are now imported from backend.tools.faq_tools

# check_semantic_similarity is already imported from backend.tools.search_tools


# Menu tools are now imported from backend.tools.menu_tools


# --- Waiter Agent ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Use config-based waiter agent prompt
waiter_system_message = PROMPTS_CONFIG['waiter_agent']['system_message']

waiter_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", waiter_system_message),
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

# Enhanced startup initialization with Phase 1 optimizations
def initialize_restaurant_system():
    """
    Pre-warm all expensive operations during app startup to eliminate first-request latency.
    Enhanced with Phase 1 optimizations: pre-warming, enhanced caching, connection pooling.
    """
    init_start = time.time()
    print("\n" + "="*60)
    print("🚀 INITIALIZING RESTAURANT SYSTEM (Phase 1 Optimized)")
    print("="*60)
    
    try:
        # 1. Pre-warm Neo4j connection with enhanced pooling
        print("📡 Establishing Neo4j connection (enhanced pooling)...")
        connection_start = time.time()
        try:
            neo4j_graph = get_neo4j_connection()
            connection_end = time.time()
            print(f"✅ Neo4j connected in {connection_end - connection_start:.3f}s")
        except Exception as e:
            print(f"⚠️ Neo4j connection failed: {e}")
            print("⚠️ System will continue but some features may not work")
            neo4j_graph = None
        
        # 2. Pre-warm Cypher chain and schema with extended caching (only if Neo4j is available)
        if neo4j_graph:
            print("⚙️  Setting up Cypher chain and caching schema (24h cache)...")
            chain_start = time.time()
            try:
                cypher_chain, _ = get_cypher_chain()
                chain_end = time.time()
                print(f"✅ Cypher chain ready in {chain_end - chain_start:.3f}s")
            except Exception as e:
                print(f"⚠️ Cypher chain setup failed: {e}")
        else:
            print("⚠️ Skipping Cypher chain setup (Neo4j unavailable)")
        
        # 3. Pre-warm embedding model with enhanced pooling
        print("🧠 Loading embedding model (enhanced pooling)...")
        embed_start = time.time()
        try:
            embedding_model = get_embedding_model()
            embed_end = time.time()
            print(f"✅ Embedding model loaded in {embed_end - embed_start:.3f}s")
        except Exception as e:
            print(f"⚠️ Embedding model loading failed: {e}")
            print("⚠️ System will continue but semantic search may not work")
            embedding_model = None
        
        # 4. Enhanced system testing with performance validation
        dish_count = 0
        if neo4j_graph:
            print("🧪 Testing system with performance validation...")
            test_start = time.time()
            try:
                # Quick test query to ensure everything works
                test_result = neo4j_graph.query("MATCH (d:Dish) RETURN count(d) as dish_count LIMIT 1")
                dish_count = test_result[0]['dish_count'] if test_result else 0
                
                # Test embedding generation (only if embedding model is available)
                if embedding_model:
                    test_embedding = embedding_model.embed_query("test")
                    print(f"✅ Embedding test passed (vector size: {len(test_embedding)})")
                
                # Test Cypher chain with a simple query
                if cypher_chain:
                    try:
                        test_cypher_result = cypher_chain.invoke({
                            "schema": neo4j_graph.schema,
                            "query": "How many dishes do you have?"
                        })
                        print(f"✅ Cypher chain test passed")
                    except Exception as cypher_e:
                        print(f"⚠️ Cypher chain test failed: {cypher_e}")
                
                test_end = time.time()
                print(f"✅ System test passed in {test_end - test_start:.3f}s")
                print(f"📊 Database contains {dish_count} dishes")
            except Exception as e:
                print(f"⚠️ System test failed: {e}")
                print("⚠️ System will continue but functionality may be limited")
        else:
            print("⚠️ Skipping system test (Neo4j unavailable)")
        
        # 5. Pre-warm common queries for instant responses (Phase 1 optimization)
        print("🗄️  Pre-warming common queries for instant responses...")
        cache_start = time.time()
        common_queries = [
            "vegetarian starters", 
            "paneer dishes", 
            "spicy main course", 
            "desserts", 
            "beverages",
            "chicken dishes",
            "seafood options"
        ]
        
        cached_count = 0
        for query in common_queries:
            try:
                # Import kg_answer from the new location
                from .knowledge_graph import kg_answer
                kg_answer(query)
                cached_count += 1
                print(f"   ✓ Pre-warmed: {query}")
            except Exception as e:
                print(f"   ⚠️  Failed to pre-warm '{query}': {e}")
        
        cache_end = time.time()
        print(f"✅ {cached_count}/{len(common_queries)} common queries pre-warmed in {cache_end - cache_start:.3f}s")
        
        init_end = time.time()
        total_time = init_end - init_start
        
        print("\n" + "="*60)
        print(f"🎉 SYSTEM READY! Total initialization: {total_time:.3f}s")
        print(f"💡 {cached_count} queries pre-warmed for instant responses!")
        print(f"⚡ Expected cold start: < 2s, warm start: < 1s")
        print("="*60 + "\n")
        
        return {
            "success": True,
            "total_time": total_time,
            "dish_count": dish_count,
            "cached_queries": cached_count,
            "message": f"System initialized successfully in {total_time:.3f}s with {cached_count} pre-warmed queries"
        }
        
    except Exception as e:
        init_end = time.time()
        total_time = init_end - init_start
        
        print("\n" + "="*60)
        print(f"❌ INITIALIZATION FAILED after {total_time:.3f}s")
        print(f"Error: {e}")
        print("="*60 + "\n")
        
        # Import traceback for detailed error info
        import traceback
        traceback.print_exc()
        
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



# get_reservations is now imported from backend.tools.reservation_tools

# cancel_reservation is now imported from backend.tools.reservation_tools

# modify_reservation is now imported from backend.tools.reservation_tools

# parse_reservation_conflict_response is now imported from backend.tools.reservation_tools

# Define waiter agent tools after all tools are created
waiter_agent_tools = [extract_food_terms, check_semantic_similarity, get_recommendations, get_detailed_dish_info, get_faq_answer, debug_embedding_indexes, check_user_registration, collect_registration, make_reservation, get_reservations, cancel_reservation, modify_reservation, parse_reservation_conflict_response]
waiter_agent_runnable = waiter_agent_prompt | llm.bind_tools(waiter_agent_tools)







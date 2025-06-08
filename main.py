# Registration Agent and Tools
from typing import Dict, Any
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from supabase import create_client, Client
import pytz
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
from typing import Dict, List
import json

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

# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
#     is_registered: bool

builder = StateGraph(State)


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        
        while True:
            # configuration = config.get("configurable", {})
            # phone_number = configuration.get("phone_number", None)
            # state = {**state, "user_info": phone_number}
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

            # Update registration flag if tool call output sets it
            # if hasattr(result, 'tool_calls') and result.tool_calls:
            #     for tool_call in result.tool_calls:
            #         if hasattr(tool_call, 'output') and isinstance(tool_call.output, dict):
            #             if tool_call.output.get("set_is_registered"):
            #                 state["is_registered"] = True
            #             elif tool_call.output.get("set_is_registered") is False:
            #                 state["is_registered"] = False

        return {"messages": result}
        
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
    if not phone_number:
        configuration = config.get("configurable", {})
        phone_number = configuration.get("phone_number", None)
    if not phone_number:
        raise ValueError("No phone number present")
    response = supabase.table('customers').select('*').eq('phone_number', phone_number).execute()
    if response.data and len(response.data) > 0:
        customer = response.data[0]
        return {
            "registered": True,
            "name": customer.get("name"),
            "preferences": customer.get("preferences"),
            "allergies": customer.get("allergies"),
            "set_is_registered": True
        }
    else:
        return {"registered": False, "set_is_registered": False}

@tool
def mark_registration_complete(config: RunnableConfig, phone_number: str = None, name: str = None, reason: str = "Registration process completed") -> dict:
    """Mark the registration process as complete, allowing transition to waiter agent. Creates minimal customer record if needed."""
    configuration = config.get("configurable", {})
    
    if not phone_number:
        phone_number = configuration.get("phone_number", None)
    if not name:
        name = configuration.get("name", None)
    
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
            "Perform the tasks in the following order:\n"
            "\n1. Ask the user their 10-digit phone number and check using a tool if they are registered using the get_customer tool. Do not ask users if they are registered or not"
            "\n2. If the user is already registered, confirm their preferences (dietary preference: Veg/Non-veg/Vegan, Spice-level: Bland/Less Spicy/Medium Spicy/Spicy, Cuising: North Indian/South Indian/Chinese/Mughlai etc) and allergies (Gluten, Soy, etc) and offer to update them. If they want to update, use create_or_update_customer. If they don't want to update, use mark_registration_complete. After this is done, use CompleteOrEscalate with cancel=True to transfer to the waiter."
            "\n3. If the user is not registered, ask user for their name first. Once you have the name, offer to collect preferences (dietary, spice level, cuisine) and allergies. If they provide preferences, use create_or_update_customer. If they want to skip preferences or say 'later', use mark_registration_complete with just the name. After registration is complete, use CompleteOrEscalate with cancel=True to transfer to the waiter."
            "\n4. If at any point the user wants to skip setting preferences, says 'later', 'will do that later', 'skip', or asks about menu/food items, use mark_registration_complete and then immediately use CompleteOrEscalate with cancel=True and reason='Registration complete, user ready for menu assistance'."
            "Do not answer any menu related questions, just use CompleteOrEscalate to transfer to the waiter."
            "\n\nIMPORTANT: As soon as you have the minimum required information (phone number and name for new users, or confirmed/updated preferences for existing users), or if the user asks about menu items or wants to proceed, immediately use CompleteOrEscalate to transfer them to the waiter assistant who can help with menu questions."
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

    neo4j_graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            enhanced_schema=True
        )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    neo4j_graph.refresh_schema()

    
    CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types and properties that are not provided.
The user input might have some typos, so you need to handle that. Just to help you with typo correction, the queries will be related to the indian restaurant menu.
When filtering by category, use the `category` property of the `Dish` node (e.g., `d.category = \"Main Course\"`).
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
Examples: Here are a few examples of generated Cypher statements for particular questions:
# Do you have Gulab Jamun?
MATCH (d:Dish {{name: \"Gulab Jamun\"}}) RETURN d.name, d.description, d.price, d.prep_time, d.is_signature, d.region, d.category
# What creamy vegetarian items do you have for main course?
MATCH (d:Dish)-[:HAS_DIETARY]->(:Dietary {{type: \"vegetarian\"}}) WHERE d.category = \"Main Course\" AND d.description CONTAINS \"creamy\" RETURN d.name, d.description, d.price, d.prep_time, d.is_signature, d.region, d.category
# Do you have paneer based items?
MATCH (d:Dish)-[:CONTAINS]->(i:Ingredient {{name: \"paneer\"}}) RETURN d.name, d.description, d.price, d.prep_time, d.is_signature, d.region, d.category

The question is:
{question}"""

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"], 
        template=CYPHER_GENERATION_TEMPLATE
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=neo4j_graph,
        verbose=False,
        validate_cypher=True,
        use_function_response=True,
        cypher_prompt=cypher_prompt,
        allow_dangerous_requests=True,
    )
    return cypher_chain, neo4j_graph

def kg_answer(query: str) -> Dict[str, Any]:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    cypher_chain, neo4j_graph = setup_cypher_chain()
    symbolic_response = cypher_chain.invoke(query)
    has_symbolic = (
        isinstance(symbolic_response, dict)
        and (
            (isinstance(symbolic_response.get('result'), list) and symbolic_response.get('result'))
            or (isinstance(symbolic_response.get('result'), str) and symbolic_response.get('result').strip())
        )
    )

    query_embedding = embeddings.embed_query(query)
    cypher = (
        """
        WITH $embedding AS queryEmbedding
        CALL db.index.vector.queryNodes('menu_embeddings', $k, queryEmbedding)
        YIELD node, score
        RETURN node.name AS name, node.description AS description, node.price AS price,
                node.category AS category, score
        ORDER BY score DESC
        """
    )
    vector_result = neo4j_graph.query(cypher, params={"embedding": query_embedding, "k": 10})

    return {
        "symbolic_answer": symbolic_response,
        "semantic_answer": vector_result,
        "formatted": f"SYMBOLIC ANSWER:\n{symbolic_response}\n\nSEMANTIC ANSWER:\n{vector_result}"
    }


@tool
def get_recommendations(query: str) -> Dict[str, Any]:
    """
    Handle FAQ questions about menu, dishes, recommendations, ingredients, and dietary information.
    Use this for questions about what's available, dish details, recommendations, or general restaurant info.
    
    Args:
        phone_number: Customer's phone number
        query: The customer's question about menu/food
    
    Returns:
        Dict with response details including recommendations
    """
    try:
        result = kg_answer(query)
        return {
            "success": True,
            "response": result,
            # "message": result.get("message", ""),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            # "message": "I'm having trouble finding menu information. Please try again."
        }


# --- Waiter Agent ---
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

waiter_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful restaurant agent you have a persona of an experienced waiter who is very friendly and helpful, so speak like a courteous waiter."
            "Your goal is to help guests with their questions and requests about the menu, dishes, recommendations, ingredients, and dietary information."
            "You should only rely on the information provided by the tools and not make up any information."
            "You have access to the following tools: \n"
            "1. CompleteOrEscalate: A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant, who can re-route the dialog based on the user's needs."
            "2. get_recommendations: A tool to get recommendations for dishes, ingredients, and dietary information."
            "\n\nYou can find current user information for personalisation:\n<user_information>\n{user_info}\n</user_information>"
        ),
        ("placeholder", "{messages}"),
    ]
)

waiter_agent_tools = [get_recommendations]
waiter_agent_runnable = waiter_agent_prompt | llm.bind_tools(waiter_agent_tools + [CompleteOrEscalate])



from typing import Callable

from langchain_core.messages import ToolMessage


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        messages = []
        # Only add ToolMessage if the last message has tool_calls
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
    # print(f"[DEBUG] check_registration_condition: is_registered = {is_registered}")
    if is_registered:
        # print("[DEBUG] Routing to waiter_agent")
        return "waiter_agent"
    else:
        # print("[DEBUG] Routing to enter_registration_agent")
        return "enter_registration_agent"

builder.add_conditional_edges(
    "check_registration",
    check_registration_condition,
    {
        "enter_registration_agent": "enter_registration_agent",
        "waiter_agent": "waiter_agent"
    }
)

builder.add_node(
    "enter_registration_agent",
    create_entry_node("Registration Assistant", "registration_agent"),
)
builder.add_node("registration_agent", Assistant(registration_agent_runnable))
builder.add_edge("enter_registration_agent", "registration_agent")
# Create a custom tool node that handles state updates
def registration_tools_node(state: State):
    """Custom tool node that updates registration status based on tool results"""
    # print(f"[DEBUG] registration_tools_node called")
    
    # Call the standard tool node
    tool_node = create_tool_node_with_fallback(registration_agent_tools)
    result = tool_node.invoke(state)
    
    # Check if any tool results indicate registration status change
    new_is_registered = state.get("is_registered", False)
    
    if "messages" in result:
        for message in result["messages"]:
            if hasattr(message, 'content'):
                content = str(message.content)
                # print(f"[DEBUG] Tool message content: {content}")
                
                # Check for various patterns that indicate registration status
                if "set_is_registered" in content:
                    if "true" in content.lower() or "'registered': true" in content.lower():
                        new_is_registered = True
                        # print(f"[DEBUG] Tool result indicates user is now registered")
                    elif "false" in content.lower() or "'registered': false" in content.lower():
                        new_is_registered = False
                        # print(f"[DEBUG] Tool result indicates user is not registered")
                
                # Also check for success patterns
                if "'success': true" in content.lower() or '"success": true' in content.lower():
                    new_is_registered = True
                    # print(f"[DEBUG] Tool success indicates user is now registered")
                
                # Check for registration completion messages
                if "registration marked complete" in content.lower() or "registration completed successfully" in content.lower():
                    new_is_registered = True
                    # print(f"[DEBUG] Registration completion detected")
    
    # print(f"[DEBUG] Updated is_registered from {state.get('is_registered', False)} to {new_is_registered}")
    
    return {
        **result,
        "is_registered": new_is_registered
    }

builder.add_node("registration_tools", registration_tools_node)


def route_registration_agent(
    state: State,
):
    # print(f"[DEBUG] route_registration_agent called, is_registered = {state.get('is_registered', False)}")
    
    route = tools_condition(state)
    if route == END:
        # print("[DEBUG] Route is END")
        # Check if registration is complete before ending
        if state.get("is_registered", False):
            # print("[DEBUG] User is registered, transitioning to waiter_agent")
            return "transition_to_waiter"
        return END
    
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        # print("[DEBUG] CompleteOrEscalate called, leaving skill")
        return "leave_skill"
    
    safe_toolnames = [t.name for t in registration_agent_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        # print("[DEBUG] Calling registration tools")
        return "registration_tools"
    
    # print("[DEBUG] Default route to registration_tools")
    return "registration_tools"

# Add transition node
def transition_to_waiter_node(state: State) -> dict:
    """Transition from registration to waiter agent after successful registration"""
    # print("[DEBUG] Transitioning to waiter agent after successful registration")
    return {
        "dialog_state": "waiter_agent",
        "messages": []
    }

builder.add_node("transition_to_waiter", transition_to_waiter_node)
builder.add_edge("transition_to_waiter", "waiter_agent")

builder.add_edge("registration_tools", "registration_agent")
builder.add_conditional_edges(
    "registration_agent",
    route_registration_agent,
    ["registration_tools", "leave_skill", "transition_to_waiter", END],
)


def waiter_tools_condition(state):
    # Example: If the agent output says to use a waiter tool
    if state.get("action") == "use_waiter_tool":
        return "waiter_tools"
    # Otherwise, finish the workflow
    return END


# Wrap waiter agent with debug logging
def waiter_agent_with_debug(state: State, config: RunnableConfig):
    # print(f"[DEBUG] waiter_agent called, is_registered = {state.get('is_registered', False)}")
    
    # Extract user info from config or state
    configuration = config.get("configurable", {})
    phone_number = configuration.get("phone_number", None)
    
    # Create user_info from available data
    user_info = f"Phone: {phone_number}" if phone_number else "No user info available"
    
    # Add user_info to state for the prompt
    state_with_user_info = {**state, "user_info": user_info}
    
    assistant = Assistant(waiter_agent_runnable)
    return assistant(state_with_user_info, config)

builder.add_node("waiter_agent", waiter_agent_with_debug)
builder.add_node("waiter_tools", create_tool_node_with_fallback(waiter_agent_tools))

builder.add_edge("registration_tools", "registration_agent")
builder.add_conditional_edges(
    "waiter_agent",
    tools_condition,
    {
        "tools": "waiter_tools",
        END: END
    }
)

# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }

builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "waiter_agent")

# builder.add_edge("waiter_agent", "__end__")
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

import shutil
import uuid

thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "phone_number": None,
        # Checkpoints are accessed by thread_id
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



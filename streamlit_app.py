import streamlit as st
import streamlit.components.v1 as components
import sys
import os
from typing import Dict, Any
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage

# Import from your main application
from main import graph, ensure_system_initialized
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 🚀 PRE-WARM SYSTEM AT STARTUP - This eliminates first-request latency!
print("🔄 Starting Streamlit app with system pre-warming...")
initialization_result = ensure_system_initialized()
if initialization_result["success"]:
    print("✅ Streamlit app ready with pre-warmed system!")
else:
    print(f"⚠️ Streamlit app started with initialization issues: {initialization_result.get('error', 'Unknown error')}")

# Configure Streamlit page
st.set_page_config(
    page_title="Restaurant Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .welcome-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Removed chat-container and scroll-to-bottom - using Streamlit's native scroll behavior */
    .latency-info {
        font-size: 0.7rem;
        color: #888;
        opacity: 0.6;
        margin-top: 0.2rem;
        font-style: italic;
        text-align: right;
    }
    .latency-info:hover {
        opacity: 0.9;
    }
    
    /* Ensure sidebar remains stable during processing */
    .stSidebar {
        position: fixed !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Prevent sidebar from being affected by spinner or processing states */
    .stSidebar > div {
        visibility: visible !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = {
            "messages": [], 
            "is_registered": False,
            "phone_number": None,  # Added for phone number persistence
            "dialog_state": []
        }
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        
    if "app_config" not in st.session_state:
        st.session_state.app_config = {
            "configurable": {
                "phone_number": None,
                "thread_id": st.session_state.thread_id,
            }
        }
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
        
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0
        
    # Removed pending_user_input and scroll_to_bottom - no longer needed
        
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
        
    if "user_phone" not in st.session_state:
        st.session_state.user_phone = ""
        
    if "name_extracted" not in st.session_state:
        st.session_state.name_extracted = False

def extract_customer_info_from_messages(messages):
    """Extract customer information from tool results in messages - ONLY during registration"""
    import json
    import re
    
    # Only extract name once during registration process
    if st.session_state.name_extracted:
        return
    
    for message in messages:
        if hasattr(message, 'content') and message.content:
            content = str(message.content)
            
            # Look for customer data in tool results (JSON format) - PRIMARY source
            try:
                # Only look for JSON with customer registration data
                json_matches = re.findall(r'\{[^}]*"name"[^}]*"phone_number"[^}]*\}', content)
                for json_str in json_matches:
                    try:
                        # Clean up the JSON string
                        json_str = json_str.replace("'", '"')
                        data = json.loads(json_str)
                        if 'name' in data and data['name'] and 'phone_number' in data:
                            st.session_state.user_name = data['name']
                            st.session_state.name_extracted = True
                            print(f"[STREAMLIT DEBUG] Extracted user name from registration data: {data['name']}")
                            break
                    except:
                        continue
                        
                # Look for phone number patterns in tool results
                phone_patterns = [
                    r"phone[_\s]*number['\"]?:\s*['\"]?(\d{10})",
                    r"'phone_number':\s*'(\d{10})'",
                    r'"phone_number":\s*"(\d{10})"'
                ]
                
                for pattern in phone_patterns:
                    phone_match = re.search(pattern, content, re.IGNORECASE)
                    if phone_match:
                        st.session_state.user_phone = phone_match.group(1)
                        print(f"[STREAMLIT DEBUG] Extracted user phone: {phone_match.group(1)}")
                        break
                
                # Look for customer names ONLY in registration-specific contexts
                # Only extract if we haven't extracted a name yet AND this looks like registration
                if not st.session_state.user_name and not st.session_state.name_extracted:
                    # VERY specific patterns for registration process only
                    registration_name_patterns = [
                        r"Found customer:\s+([A-Za-z]{2,15})\s+with phone\s+\d{10}",  # Debug message during registration
                        r"Thank you,\s+([A-Za-z]{2,15})!\s+Based on your past visits",  # Welcome back message
                    ]
                    
                    for pattern in registration_name_patterns:
                        name_match = re.search(pattern, content, re.IGNORECASE)
                        if name_match:
                            potential_name = name_match.group(1).strip()
                            # Additional validation - must be a reasonable person name
                            if (len(potential_name) >= 2 and len(potential_name) <= 15 and 
                                potential_name.isalpha() and potential_name[0].isupper()):
                                # Check if this doesn't look like a dish name
                                dish_keywords = ['tikka', 'masala', 'curry', 'rice', 'naan', 'roti', 'paneer', 'chicken', 'mutton', 'dosa', 'biryani']
                                if not any(keyword in potential_name.lower() for keyword in dish_keywords):
                                    st.session_state.user_name = potential_name
                                    st.session_state.name_extracted = True
                                    print(f"[STREAMLIT DEBUG] Extracted user name from registration conversation: {potential_name}")
                                    break
                        
            except Exception as e:
                print(f"[STREAMLIT DEBUG] Error extracting customer info: {e}")
                
    # Also try to get phone from app config if not found
    if not st.session_state.user_phone and st.session_state.app_config.get("configurable", {}).get("phone_number"):
        st.session_state.user_phone = st.session_state.app_config["configurable"]["phone_number"]

def add_message_to_history(role: str, content: str, latency: float = None):
    """Add message to Streamlit session state for UI display only"""
    # Create message object with optional latency
    message = {"role": role, "content": content}
    if latency is not None:
        message["latency"] = latency
    
    # Add to Streamlit session state for UI
    st.session_state.messages.append(message)
    
    # Limit UI message history to prevent performance issues (keep last 40 messages)
    if len(st.session_state.messages) > 40:
        st.session_state.messages = st.session_state.messages[-40:]

def process_user_message(user_input: str) -> tuple[str, float]:
    """Process user message through the LangGraph system"""
    start_time = time.time()
    
    try:
        print(f"[STREAMLIT DEBUG] Processing user input: {user_input}")
        print(f"[STREAMLIT DEBUG] Current registration status: {st.session_state.graph_state.get('is_registered', False)}")
        print(f"[STREAMLIT DEBUG] Current graph state messages count: {len(st.session_state.graph_state.get('messages', []))}")
        
        # Create a fresh copy of graph state for this interaction
        current_state = {
            "messages": st.session_state.graph_state["messages"].copy(),
            "is_registered": st.session_state.graph_state.get("is_registered", False),
            "phone_number": st.session_state.graph_state.get("phone_number", None),
            "dialog_state": st.session_state.graph_state.get("dialog_state", [])
        }
        
        # Add user message to state
        current_state["messages"].append(HumanMessage(content=user_input))
        
        print(f"[STREAMLIT DEBUG] Messages before graph invoke: {len(current_state['messages'])}")
        
        # Process through the graph
        result = graph.invoke(current_state, st.session_state.app_config)
        
        print(f"[STREAMLIT DEBUG] Messages after graph invoke: {len(result.get('messages', []))}")
        
        # Calculate latency
        end_time = time.time()
        latency = end_time - start_time
        
        # Update the graph state with results (but preserve important state)
        # Limit message history to prevent memory issues (keep last 20 messages)
        messages = result.get("messages", [])
        if len(messages) > 20:
            messages = messages[-20:]
            
        st.session_state.graph_state.update({
            "messages": messages,
            "is_registered": result.get("is_registered", st.session_state.graph_state.get("is_registered", False)),
            "phone_number": result.get("phone_number", st.session_state.graph_state.get("phone_number", None)),
            "dialog_state": result.get("dialog_state", [])
        })
        
        # Also update app config if phone number was extracted
        if result.get("phone_number") and result.get("phone_number") != st.session_state.app_config["configurable"].get("phone_number"):
            st.session_state.app_config["configurable"]["phone_number"] = result.get("phone_number")
            print(f"[STREAMLIT DEBUG] Updated app config with extracted phone number: {result.get('phone_number')}")
        
        # Extract customer information from tool results when user becomes registered
        if result.get("is_registered", False):
            extract_customer_info_from_messages(messages)
            
        # Also check for phone number in user input
        if user_input and not st.session_state.user_phone:
            # Look for 10-digit phone numbers in user input
            import re
            phone_match = re.search(r'\b(\d{10})\b', user_input)
            if phone_match:
                st.session_state.user_phone = phone_match.group(1)
                # Also store in app config for the agents to use
                st.session_state.app_config["configurable"]["phone_number"] = phone_match.group(1)
                print(f"[STREAMLIT DEBUG] Captured phone number from user input: {phone_match.group(1)}")
        
        # Extract the assistant's response
        if result.get("messages"):
            assistant_message = result["messages"][-1]
            if hasattr(assistant_message, 'content') and assistant_message.content:
                response = assistant_message.content.strip()
                print(f"[STREAMLIT DEBUG] Assistant response: {response[:100]}... (Latency: {latency:.2f}s)")
                return response, latency
            else:
                return "I'm here to help! How can I assist you today?", latency
        else:
            return "I'm here to help! How can I assist you today?", latency
            
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"[STREAMLIT ERROR] Error processing message: {e}")
        import traceback
        traceback.print_exc()
        
        # Reset processing state in case of error
        st.session_state.processing = False
        
        # Try to preserve the conversation state even if there's an error
        error_message = "I apologize, but I encountered an error. Please try again or clear the conversation to start fresh."
        
        return error_message, latency

def display_chat_history():
    """Display chat history with custom styling using Streamlit's native chat components"""
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Add latency info for assistant messages if available
                if "latency" in message:
                    latency = message["latency"]
                    # Format latency appropriately (ms for <1s, s for >=1s)
                    if latency < 1.0:
                        latency_text = f"⚡ {latency*1000:.0f}ms"
                    else:
                        latency_text = f"⚡ {latency:.2f}s"
                    st.markdown(f'<div class="latency-info">{latency_text}</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar - called at the beginning to ensure stability"""
    with st.sidebar:
        st.markdown("### User Information")
        
        # Display current registration status
        is_registered = st.session_state.graph_state.get("is_registered", False)
        has_phone = st.session_state.graph_state.get("phone_number") or st.session_state.user_phone
        
        if is_registered:
            st.success("🟢 Status: Registered")
        elif has_phone:
            st.info("🔵 Status: Phone Verified for Reservations")
        else:
            st.info("🔵 Status: Registration not required for menu browsing")
        
        # Display user information when registered or has phone
        if is_registered or has_phone:
            display_name = st.session_state.user_name if st.session_state.user_name else "User"
            display_phone = has_phone if has_phone else st.session_state.user_phone
            
            if display_name and display_name != "User":
                st.markdown("**User Name:** " + display_name)
            if display_phone:
                st.markdown("**Mobile Number:** " + display_phone)
        
        if not is_registered:
            st.markdown("**Note:** You can browse menu and get recommendations without registration")
            if has_phone:
                st.markdown("**Phone verified for:** Reservations and booking management")
                st.markdown("**Full registration needed for:** Placing orders, billing")
            else:
                st.markdown("**Registration needed for:** Placing orders, making reservations, billing")

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar first to ensure it stays stable
    render_sidebar()
    
    # App header
    st.markdown('<div class="main-header">🍽️ My Restaurant Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-powered dining companion for personalized menu recommendations and reservations management</div>', unsafe_allow_html=True)
    
    # Welcome message for new users
    if not st.session_state.initialized and len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-message">
            <h3>🌟 Welcome to My Restaurant! 🌟</h3>
            <p>I'm your personal dining assistant, ready to help you explore our delicious menu.</p>
            <p><strong>Browse freely!</strong> Ask me about dishes, ingredients, dietary options, or recommendations.</p>
            <p><em>Registration is only needed when you're ready to place an order or make a reservation.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add initial assistant message
        initial_message = "Hello! Welcome to My Restaurant. I'm here to help you explore our menu and answer any questions about our dishes. What would you like to know about our food today?"
        add_message_to_history("assistant", initial_message)
        st.session_state.initialized = True
    
    # Display chat history
    display_chat_history()
    
    # Chat input - process immediately in same render cycle
    if user_input := st.chat_input("Type your message here...", disabled=st.session_state.processing):
        current_time = time.time()
        # Rate limiting: prevent requests within 1 second of each other
        if not st.session_state.processing and (current_time - st.session_state.last_request_time) > 1.0:
            st.session_state.last_request_time = current_time
            st.session_state.processing = True
            
            # Immediately add user message to history
            add_message_to_history("user", user_input)
            
            # Show user message immediately
            st.chat_message("user").write(user_input)
            
            # Process the message and show response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response, latency = process_user_message(user_input)
                
                # Display response
                if response and response.strip():
                    st.write(response)
                    
                    # Add latency info
                    if latency < 1.0:
                        latency_text = f"⚡ {latency*1000:.0f}ms"
                    else:
                        latency_text = f"⚡ {latency:.2f}s"
                    st.markdown(f'<div class="latency-info">{latency_text}</div>', unsafe_allow_html=True)
                    
                    # Add to message history
                    add_message_to_history("assistant", response, latency)
            
            # Reset processing state
            st.session_state.processing = False
            
        elif st.session_state.processing:
            st.info("⏳ Please wait for the current message to be processed...")
        else:
            st.info("⏱️ Please wait a moment before sending another message...")
    


if __name__ == "__main__":
    main() 
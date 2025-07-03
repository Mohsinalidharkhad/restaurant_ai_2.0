import streamlit as st
import streamlit.components.v1 as components
import sys
import os
from typing import Dict, Any
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage

# Import from your main application
from main import graph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Neemsi Restaurant Assistant",
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
    .chat-container {
        height: 600px;
        overflow-y: auto;
    }
    .scroll-to-bottom {
        height: 1px;
    }
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
        
    if "pending_user_input" not in st.session_state:
        st.session_state.pending_user_input = None
        
    if "scroll_to_bottom" not in st.session_state:
        st.session_state.scroll_to_bottom = False
        
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
            "dialog_state": result.get("dialog_state", [])
        })
        
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
    """Display chat history with custom styling"""
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
    
    # Add a unique anchor that changes with each new message to force scroll
    if st.session_state.messages:
        anchor_id = f"msg-{len(st.session_state.messages)}"
        st.markdown(f'<div id="{anchor_id}" style="height: 1px;"></div>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar - called at the beginning to ensure stability"""
    with st.sidebar:
        st.markdown("### User Information")
        
        # Display current registration status
        is_registered = st.session_state.graph_state.get("is_registered", False)
        if is_registered:
            st.success("🟢 Registration Status: Registered")
        else:
            st.error("🔴 Registration Status: Not Registered")
        
        # Display user information when registered
        if is_registered:
            st.markdown("**User Name:** " + (st.session_state.user_name if st.session_state.user_name else ""))
            st.markdown("**Mobile Number:** " + (st.session_state.user_phone if st.session_state.user_phone else ""))
        else:
            st.markdown("**User Name:** ")
            st.markdown("**Mobile Number:** ")

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar first to ensure it stays stable
    render_sidebar()
    
    # App header
    st.markdown('<div class="main-header">🍽️ Neemsi Restaurant Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-powered dining companion for personalized menu recommendations</div>', unsafe_allow_html=True)
    
    # Welcome message for new users
    if not st.session_state.initialized and len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-message">
            <h3>🌟 Welcome to Neemsi Restaurant! 🌟</h3>
            <p>I'm your personal dining assistant, ready to help you explore our delicious menu.</p>
            <p>To get started, I'll need your 10-digit phone number for personalized service.</p>
            <p>Feel free to ask me about our dishes, ingredients, dietary options, or recommendations!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add initial assistant message
        initial_message = "Welcome to Neemsi! Please provide your 10-digit phone number to help you further!"
        add_message_to_history("assistant", initial_message)
        st.session_state.initialized = True
    
    # Create a container for the chat
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        display_chat_history()
        
        # Auto-scroll functionality
        if st.session_state.scroll_to_bottom and len(st.session_state.messages) > 0:
            # Get the latest message anchor ID
            latest_anchor = f"msg-{len(st.session_state.messages)}"
            
            # JavaScript to scroll to the latest message
            st.markdown(f"""
            <script>
                function scrollToLatest() {{
                    setTimeout(function() {{
                        var latestMsg = document.getElementById('{latest_anchor}');
                        if (latestMsg) {{
                            latestMsg.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
                        }}
                    }}, 100);
                }}
                scrollToLatest();
            </script>
            """, unsafe_allow_html=True)
            
            # Reset scroll flag
            st.session_state.scroll_to_bottom = False
        
        # Additional fallback: Use URL fragment to maintain position
        if len(st.session_state.messages) > 0:
            st.markdown(f"<style>#msg-{len(st.session_state.messages)} {{ scroll-margin-top: 20px; }}</style>", unsafe_allow_html=True)
        
        # Check if we need to process a pending user input
        if st.session_state.pending_user_input and not st.session_state.processing:
            st.session_state.processing = True
            
            # Show thinking spinner and process the message
            with st.spinner("🤔 Thinking..."):
                response, latency = process_user_message(st.session_state.pending_user_input)
            
            # Add assistant response to history with latency
            if response and response.strip():
                add_message_to_history("assistant", response, latency)
            
            # Clear pending input and reset processing state
            st.session_state.pending_user_input = None
            st.session_state.processing = False
            
            # Set scroll flag to scroll to bottom after assistant response
            st.session_state.scroll_to_bottom = True
            
            # Rerun to show the assistant response
            st.rerun()
    
    # Chat input
    if user_input := st.chat_input("Type your message here...", disabled=st.session_state.processing):
        current_time = time.time()
        # Rate limiting: prevent requests within 1 second of each other
        if not st.session_state.processing and (current_time - st.session_state.last_request_time) > 1.0:
            st.session_state.last_request_time = current_time
            
            # Immediately add user message to history and show it
            add_message_to_history("user", user_input)
            
            # Set pending input for processing on next render
            st.session_state.pending_user_input = user_input
            
            # Set scroll flag to scroll to bottom after rerun
            st.session_state.scroll_to_bottom = True
            
            # Rerun to immediately show user message
            st.rerun()
        elif st.session_state.processing:
            st.info("⏳ Please wait for the current message to be processed...")
        else:
            st.info("⏱️ Please wait a moment before sending another message...")
    


if __name__ == "__main__":
    main() 
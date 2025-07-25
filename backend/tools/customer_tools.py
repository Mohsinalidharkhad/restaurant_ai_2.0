"""
Customer tools module for Restaurant Graph Agent.

Contains tools for customer registration and management.
"""

from typing import Dict, Any
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from ..data.supabase_client import get_supabase_client


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
        supabase = get_supabase_client()
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
        supabase = get_supabase_client()
        
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
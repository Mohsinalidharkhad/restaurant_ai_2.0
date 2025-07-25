"""
Reservation tools module for Restaurant Graph Agent.

Contains tools for making, viewing, modifying, and canceling restaurant reservations.
"""

from typing import Dict, Any
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from ..data.supabase_client import get_supabase_client

# Helper functions for phone number caching - these would normally be in a utils module
def get_thread_phone_number(config: RunnableConfig):
    """Get cached phone number for this thread"""
    # This would need to be implemented with proper thread-safe caching
    # For now, return None as fallback
    return None

def set_thread_phone_number(config: RunnableConfig, phone_number: str):
    """Cache phone number for this thread"""
    # This would need to be implemented with proper thread-safe caching
    # For now, just pass
    pass


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
        supabase = get_supabase_client()
        
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
                            "message": f"I found that you already have a reservation on {display_date} at {existing_time_display} for {existing_pax} people (Reservation ID: {existing_id}).\n\nYou're requesting another reservation on the same date at {datetime.strptime(time, '%H:%M').strftime('%I:%M %p')} for {pax} people.\n\nWould you like to:\n1. **Yes** - Create both reservations (two separate tables)\n2. **No** - Keep only your existing reservation\n3. **Update** - Modify your existing reservation to the new time and party size\n\nPlease reply with 1, 2, or 3."
                        }
        
        except Exception as e:
            print(f"[DEBUG] Error checking existing reservations: {e}")
            # Continue with creating new reservation
        
        # Step 4: Create new reservation (if we reach here, all validations passed and no conflicts or conflicts resolved)
        reservation_data = {
            "cust_number": customer_phone,
            "booking_date": date,
            "booking_time": time,
            "pax": pax
        }
        
        result = supabase.table('reservations').insert(reservation_data).execute()
        
        if result.data:
            new_reservation = result.data[0]
            reservation_id = new_reservation['id']
            
            # Format details for response
            try:
                display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%B %d, %Y")
            except:
                display_date = date
            
            try:
                display_time = datetime.strptime(time, "%H:%M").strftime("%I:%M %p")
            except:
                display_time = time
            
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
                "summary": f"Excellent! Your reservation is confirmed.\n\n**Reservation Details:**\n• **ID:** {reservation_id}\n• **Name:** {customer_name}\n• **Date:** {display_date}\n• **Time:** {display_time}\n• **Party Size:** {pax} people\n\nWe look forward to serving you!"
            }
        else:
            return {
                "success": False,
                "error": "Failed to create reservation in database",
                "step": "database_error"
            }
            
    except Exception as e:
        print(f"[DEBUG] Error in make_reservation: {e}")
        return {
            "success": False,
            "error": str(e),
            "step": "general_error",
            "message": f"I encountered an error while processing your reservation: {str(e)}"
        }


@tool
def get_reservations(config: RunnableConfig, phone_number: str = None, **kwargs) -> Dict[str, Any]:
    """
    Get all reservations for a customer by phone number.
    
    Args:
        phone_number: User's 10-digit phone number
        
    Returns:
        Dict with reservation details or error message
    """
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
        supabase = get_supabase_client()
        
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
        
        # Create summary for LLM to format naturally
        return {
            "success": True,
            "action": "reservations_retrieved",
            "customer_name": customer_name,
            "reservations": formatted_reservations,
            "upcoming_reservations": upcoming,
            "upcoming_count": len(upcoming),
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
        supabase = get_supabase_client()
        
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
        supabase = get_supabase_client()
        
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
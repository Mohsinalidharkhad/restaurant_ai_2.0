"""
Supabase client module for Restaurant Graph Agent.

Handles Supabase connections for customer data.
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Global Supabase client instance
_supabase_client = None

load_dotenv()


def get_supabase_client() -> Client:
    """Singleton pattern for Supabase client - reuse existing connection"""
    global _supabase_client
    
    if _supabase_client is None:
        print("[TIMING] Creating new Supabase client...")
        _supabase_client = create_client(
            os.getenv("SUPABASE_URL"), 
            os.getenv("SUPABASE_KEY")
        )
        print("[TIMING] Supabase client created")
    else:
        print("[TIMING] Reusing existing Supabase client (pooling active)")
    
    return _supabase_client 
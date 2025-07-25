"""
FAQ tools module for Restaurant Graph Agent.

Contains tools for searching restaurant FAQ database.
"""

import time
from typing import Dict, Any
from langchain_core.tools import tool

from ..data.neo4j_client import get_neo4j_connection
from ..data.openai_client import get_embedding_model


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
#!/usr/bin/env python3
"""
Populate FAQ Embeddings
Generates embeddings for FAQ questions and answers and stores them in Neo4j
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings
import time

load_dotenv()

def populate_faq_embeddings():
    """Generate and store embeddings for all FAQ entries"""
    
    # Initialize connections
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("🧠 Starting FAQ embedding generation...")
    
    try:
        with driver.session() as session:
            # First, check what properties exist on FAQ nodes
            sample_result = session.run("MATCH (f:FAQ) RETURN f LIMIT 1")
            sample_record = sample_result.single()
            
            if not sample_record:
                print("❌ No FAQ nodes found in database!")
                return
            
            sample_faq = dict(sample_record['f'])
            available_props = list(sample_faq.keys())
            print(f"📋 Available FAQ properties: {available_props}")
            
            # Try to identify question and answer fields
            question_field = None
            answer_field = None
            
            # Common variations for question field
            for field in ['question', 'Question', 'q', 'query', 'ask']:
                if field in available_props:
                    question_field = field
                    break
            
            # Common variations for answer field  
            for field in ['answer', 'Answer', 'a', 'response', 'reply']:
                if field in available_props:
                    answer_field = field
                    break
            
            if not question_field or not answer_field:
                print(f"❌ Could not identify question/answer fields!")
                print(f"Available properties: {available_props}")
                print("Please check your CSV column names and re-upload the data.")
                return
            
            print(f"✅ Using question field: '{question_field}', answer field: '{answer_field}'")
            
            # Get all FAQ entries without embeddings
            query = f"""
                MATCH (f:FAQ) 
                WHERE f.embedding IS NULL 
                RETURN f.id as id, f.{question_field} as question, f.{answer_field} as answer
                ORDER BY f.id
            """
            
            result = session.run(query)
            all_faqs = [record.data() for record in result]
            
            # Filter out any FAQs with missing data
            faqs = []
            for faq in all_faqs:
                if faq.get('id') and faq.get('question') and faq.get('answer'):
                    faqs.append(faq)
                else:
                    print(f"⚠️  Skipping FAQ with missing data: {faq}")
            
            if not faqs:
                print("✅ All FAQs already have embeddings!")
                return
            
            print(f"📊 Found {len(faqs)} valid FAQs without embeddings")
            
            # Process in batches
            batch_size = 10
            for i in range(0, len(faqs), batch_size):
                batch = faqs[i:i+batch_size]
                
                print(f"🔄 Processing batch {i//batch_size + 1}/{(len(faqs)-1)//batch_size + 1}...")
                
                # Generate embeddings for combined question + answer
                texts = []
                valid_batch = []
                
                for faq in batch:
                    if faq.get('question') and faq.get('answer'):
                        texts.append(f"{faq['question']} {faq['answer']}")
                        valid_batch.append(faq)
                    else:
                        print(f"  ⚠️ Skipping FAQ {faq.get('id', 'unknown')} - missing question or answer")
                
                if not texts:
                    print("  ⚠️ No valid FAQs in this batch, skipping...")
                    continue
                    
                batch_embeddings = embeddings.embed_documents(texts)
                
                # Update each FAQ with its embedding
                for faq, embedding in zip(valid_batch, batch_embeddings):
                    session.run("""
                        MATCH (f:FAQ {id: $id})
                        SET f.embedding = $embedding,
                            f.embedding_model = 'text-embedding-3-small',
                            f.embedding_status = 'completed',
                            f.updated_at = datetime()
                    """, {
                        "id": faq['id'],
                        "embedding": embedding
                    })
                    
                    print(f"  ✅ Updated embedding for FAQ {faq['id']}")
                
                # Rate limiting
                if i + batch_size < len(faqs):
                    time.sleep(1)
            
            print(f"\n🎉 Successfully generated embeddings for {len(faqs)} FAQs!")
            
            # Verify the results
            verification_result = session.run("""
                MATCH (f:FAQ) 
                WHERE f.embedding IS NOT NULL 
                RETURN count(f) as embedded_count
            """)
            
            embedded_count = verification_result.single()['embedded_count']
            print(f"✅ Verification: {embedded_count} FAQs now have embeddings")
            
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        raise
    
    finally:
        driver.close()

if __name__ == "__main__":
    populate_faq_embeddings() 
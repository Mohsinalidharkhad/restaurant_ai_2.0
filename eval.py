from langsmith import evaluate, Client
from langsmith.wrappers import wrap_openai
from openai import OpenAI
import uuid
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

# Hardcoded customer details for evaluation (as requested)
EVAL_PHONE_NUMBER = "8691990606"
EVAL_CUSTOMER_NAME = "Mohsin"
EVAL_CUSTOMER_PREFERENCES = {
    "dietary_preference": "", 
    "spice_level": "Less Spicy", 
    "cuisine": "North Indian"
}
EVAL_CUSTOMER_ALLERGIES = ["Nuts"]

def chatbot_with_registration(inputs: dict) -> dict:
    """
    Target function that handles both registration and menu queries.
    This simulates the complete chatbot experience for evaluation.
    """
    try:
        # Import the chatbot graph from main.py
        from main import graph
        
        # Create a unique thread ID for this evaluation run
        thread_id = str(uuid.uuid4())
        
        # Set up the configuration with hardcoded phone number
        config = {
            "configurable": {
                "phone_number": EVAL_PHONE_NUMBER,
                "thread_id": thread_id,
            }
        }
        
        # Initialize state
        state = {"messages": [], "is_registered": False}
        
        # Step 1: Handle phone number registration automatically
        phone_message = HumanMessage(content=EVAL_PHONE_NUMBER)
        state["messages"].append(phone_message)
        result = graph.invoke(state, config)
        
        # Step 2: Handle name registration (if new user)
        name_message = HumanMessage(content=EVAL_CUSTOMER_NAME)
        state = result
        state["messages"].append(name_message)
        result = graph.invoke(state, config)
        
        # Step 3: Handle preferences (if needed)
        preferences_message = HumanMessage(content="Yes, I would like to set my preferences. I prefer North Indian food with less spice level. I'm allergic to nuts.")
        state = result
        state["messages"].append(preferences_message)
        result = graph.invoke(state, config)
        
        # Step 4: Now handle the actual query from the dataset
        query_message = HumanMessage(content=inputs["Question"])
        state = result
        state["messages"].append(query_message)
        final_result = graph.invoke(state, config)
        
        # Extract the final assistant response
        if final_result["messages"]:
            assistant_response = final_result["messages"][-1].content
        else:
            assistant_response = "No response generated"
            
        return {
            "answer": assistant_response,
            "thread_id": thread_id
        }
        
    except Exception as e:
        print(f"Error in chatbot_with_registration: {e}")
        return {
            "answer": f"Error occurred: {str(e)}",
            "thread_id": "error"
        }

# Python LLM-as-a-judge evaluators

def combined_restaurant_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> list:
    """
    Single LLM call evaluator that assesses all 4 metrics at once for efficiency.
    Returns all 4 scores in one evaluation to save cost and time.
    """
    try:
        openai_client = wrap_openai(OpenAI())
        
        evaluation_prompt = f"""
        You are evaluating a restaurant chatbot's response on multiple criteria.

        Customer Question: {inputs.get('Question', 'N/A')}
        Chatbot Response: {outputs.get('answer', 'N/A')}
        Reference Answer: {reference_outputs.get('Answer', 'N/A')}

        Please evaluate the chatbot response on the following 4 criteria (score 0.0-1.0 for each):

        1. HELPFULNESS (0.0-1.0): Does the response directly address what the customer asked and provide useful information?
           - 0.0-0.3: Not helpful, doesn't address the question
           - 0.4-0.6: Somewhat helpful, partially addresses the question  
           - 0.7-0.9: Very helpful, addresses the question well
           - 1.0: Extremely helpful, perfectly addresses the question

        2. CORRECTNESS (0.0-1.0): How accurate is the information compared to the reference answer?
           - 0.0-0.3: Incorrect or misleading information
           - 0.4-0.6: Mostly correct with some inaccuracies
           - 0.7-0.9: Very accurate, aligns well with reference
           - 1.0: Completely accurate and correct

        3. COMPLETENESS (0.0-1.0): How comprehensive and detailed is the response?
           - 0.0-0.3: Incomplete, missing key information
           - 0.4-0.6: Partially complete, some details missing
           - 0.7-0.9: Very complete, covers most important aspects
           - 1.0: Completely comprehensive, nothing important missing

        4. OVERALL_QUALITY (0.0-1.0): Holistic assessment considering helpfulness, accuracy, friendliness, completeness, and relevance to restaurant context.

        Respond with ONLY a JSON object with these exact keys and decimal scores:
        {{
            "helpfulness": 0.85,
            "correctness": 0.90,
            "completeness": 0.80,
            "overall_quality": 0.85
        }}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Respond only with valid JSON containing the 4 scores."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0
        )
        
        import json
        response_content = response.choices[0].message.content.strip()
        print(f"[DEBUG] Combined evaluation response: {response_content}")
        
        scores = json.loads(response_content)
        
        # Return list of individual metric results for LangSmith
        results = []
        for metric_name, score in scores.items():
            results.append({
                "key": metric_name,
                "score": score
            })
        
        print(f"[DEBUG] All scores - Helpfulness: {scores.get('helpfulness', 0)}, Correctness: {scores.get('correctness', 0)}, Completeness: {scores.get('completeness', 0)}, Overall: {scores.get('overall_quality', 0)}")
        
        return results
        
    except Exception as e:
        print(f"Error in combined restaurant evaluator: {e}")
        # Return default scores for all metrics
        return [
            {"key": "helpfulness", "score": 0.0},
            {"key": "correctness", "score": 0.0},
            {"key": "completeness", "score": 0.0},
            {"key": "overall_quality", "score": 0.0}
        ]

# Combined evaluator for cost-efficient evaluation of all metrics in one LLM call

def run_evaluation():
    """
    Main function to run the evaluation using Python evaluators.
    """
    print("🍽️ Starting Restaurant Chatbot Evaluation...")
    print("=" * 60)
    
    try:
        dataset_name = "Test base"
        
        print(f"📊 Using dataset: {dataset_name}")
        print(f"👤 Test customer: {EVAL_CUSTOMER_NAME} ({EVAL_PHONE_NUMBER})")
        print(f"🎯 Preferences: {EVAL_CUSTOMER_PREFERENCES}")
        print(f"⚠️  Allergies: {EVAL_CUSTOMER_ALLERGIES}")
        print("-" * 60)
        
        # Test target function first
        print("🔍 Testing chatbot function...")
        test_input = {"Question": "What vegetarian dishes do you have?"}
        test_result = chatbot_with_registration(test_input)
        print(f"✅ Chatbot test successful! Sample response: {test_result.get('answer', 'No answer')[:100]}...")
        print("-" * 60)
        
        # Run evaluation using single combined Python evaluator
        print("🎯 Starting evaluation with combined Python evaluator...")
        experiment_results = evaluate(
            chatbot_with_registration,
            data=dataset_name,
            evaluators=[
                combined_restaurant_evaluator,
            ],
            experiment_prefix="restaurant-chatbot-eval",
            max_concurrency=1,
            metadata={
                "description": "Restaurant chatbot evaluation using single combined Python LLM-as-a-judge evaluator",
                "evaluator_source": "Combined Python LLM evaluator (1 call for 4 metrics)",
                "scoring_scale": "0.0-1.0 for all metrics",
                "efficiency": "75% cost reduction vs separate evaluators",
                "customer_profile": {
                    "phone": EVAL_PHONE_NUMBER,
                    "name": EVAL_CUSTOMER_NAME,
                    "preferences": EVAL_CUSTOMER_PREFERENCES,
                    "allergies": EVAL_CUSTOMER_ALLERGIES
                }
            }
        )
        
        print("\n✅ Evaluation completed successfully!")
        print(f"📈 Experiment URL: {experiment_results}")
        print("\n📋 Summary:")
        print(f"   • Dataset: {dataset_name}")
        print(f"   • Scoring Scale: 0.0-1.0 for all metrics")
        print(f"   • Evaluator: 1 combined Python LLM-as-a-judge (75% cost reduction)")
        print(f"   • Customer Profile: {EVAL_CUSTOMER_NAME}")
        print(f"   • Metrics: Helpfulness, Correctness, Completeness, Overall Quality")
        print(f"   • Efficiency: Single LLM call instead of 4 separate calls")
        print("\n🎯 Check the LangSmith UI for detailed results and insights!")
        
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        # More detailed error information
        import traceback
        print(f"\n🔍 Full error traceback:")
        traceback.print_exc()
        
        print(f"\n💡 Troubleshooting:")
        print("   • Check that 'Test base' dataset exists in LangSmith")
        print("   • Verify LANGSMITH_API_KEY and OPENAI_API_KEY are set")
        print("   • Ensure main.py chatbot is working correctly")

if __name__ == "__main__":
    run_evaluation() 
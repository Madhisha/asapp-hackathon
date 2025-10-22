# response_generator.py
import json
from preprocessor import clean_text
from conversation_model import ChatModel
from context_manager import ContextManager
import os

try:
    from vector_store import PolicyVectorStore
    _HAVE_RAG = True
except Exception:
    PolicyVectorStore = None
    _HAVE_RAG = False


class ResponseGenerator:
    def __init__(self, policy_file='policies.json', policies_jsonl='policies.jsonl', index_dir='vector_index'):
        # Load static policies (fallback)
        with open(policy_file, 'r', encoding='utf-8') as f:
            self.policies = json.load(f)

        self.context_manager = ContextManager()
        self.chat_model = ChatModel()

        # Try to initialize/load vector store if available
        self.vector_store = None
        if _HAVE_RAG:
            try:
                store = PolicyVectorStore()
                # Try to load existing index, if not found it will build later
                if os.path.exists(index_dir):
                    store.load_index(cache_dir=index_dir)
                    self.vector_store = store
                    print("✅ RAG vector store loaded successfully")
                elif os.path.exists(policies_jsonl):
                    # Build index for first time using JSONL file
                    print("Building RAG index for first time...")
                    store.build_index(policies_file=policies_jsonl, cache_dir=index_dir)
                    self.vector_store = store
                    print("✅ RAG index built and loaded")
                else:
                    print(f"Info: {policies_jsonl} not found, RAG not available")
            except Exception as e:
                # If loading fails, keep fallback
                print(f"Info: RAG not available, using static policies. ({e})")

    def _is_airline_related(self, user_input: str) -> bool:
        """Check if the user query is related to airline/travel topics."""
        # Quick check using LLM
        prompt = f"""Is this question related to airlines, flights, travel, or customer service? Answer only 'YES' or 'NO'.

Question: {user_input}

Answer:"""
        response = self.chat_model.generate_response(prompt).strip().upper()
        return response.startswith('YES')
    
    def _build_prompt(self, user_input_clean: str, conversation_context: str) -> str:
        """Build prompt including policy context when available."""
        policy_context = ''
        if self.vector_store:
            try:
                # Retrieve relevant policies using vector search
                results = self.vector_store.retrieve(user_input_clean, top_k=3)
                
                # Display retrieved policies with distances
                if results:
                    print(f"\n📋 Retrieved {len(results)} relevant policies:")
                    for r in results:
                        print(f"  • [{r['section']}] (distance: {r['distance']:.3f})")
                        print(f"    Q: {r['question'][:80]}...")
                
                # Build context from retrieved results
                policy_parts = []
                for r in results:
                    if r['distance'] < 2.0:  # Only include good matches
                        policy_parts.append(f"Q: {r['question']}\nA: {r['answer']}")
                
                policy_context = '\n\n'.join(policy_parts)
                
            except Exception as e:
                print(f"Warning: vector store search failed: {e}")

        if policy_context:
            policy_text = f"Relevant Airline Policies:\n{policy_context}"
        else:
            # Fallback to all policies if retrieval failed
            policy_text = "Airline Policies:\n" + '\n'.join(self.policies.values())

        # Build final prompt with instructions for the LLM
        prompt = f"""You are a helpful airline customer support assistant. Answer questions using ONLY the policy information provided below.

IMPORTANT INSTRUCTIONS:
- Keep answers SHORT and DIRECT (2-4 sentences maximum)
- Only include the most relevant information
- Use bullet points only if listing multiple items
- Don't repeat information
- Be conversational but concise

{policy_text}

{conversation_context}User: {user_input_clean}
Bot:"""
        return prompt

    def generate(self, user_input: str) -> str:
        user_input_clean = clean_text(user_input)
        
        # Check if question is airline-related BEFORE retrieval
        if not self._is_airline_related(user_input_clean):
            return "I'm an airline customer support assistant. I can only help with flight bookings, baggage, cancellations, pets, fares, and other airline-related questions. Please ask something related to air travel."
        
        conversation_context = self.context_manager.get_context()

        prompt = self._build_prompt(user_input_clean, conversation_context)

        # Generate response
        bot_response = self.chat_model.generate_response(prompt)

        # Update conversation context
        self.context_manager.add_turn(user_input_clean, bot_response)
        return bot_response

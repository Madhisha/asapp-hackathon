"# Airline Customer Support Chatbot

An intelligent airline customer support chatbot using RAG (Retrieval Augmented Generation) with LLM integration. The bot answers customer queries using airline policies and maintains context across conversations.

## Features

- **RAG-based Policy Retrieval**: Uses FAISS vector search to find relevant policies
- **LLM Integration**: Powered by Ollama (Mistral Instruct) for natural responses
- **Multi-turn Conversations**: Maintains conversation context and history
- **On-topic Enforcement**: Rejects off-topic questions automatically
- **Distance-based Matching**: Uses L2 distance for accurate policy retrieval
- **Concise Responses**: Generates short, crisp, and accurate answers

## Tech Stack

### Core Technologies

- **Python 3.11.0**: Main programming language
- **Ollama**: Local LLM inference (Mistral Instruct model)
- **FAISS**: Facebook AI Similarity Search for vector indexing
- **Sentence Transformers**: `all-MiniLM-L6-v2` for embedding generation
- **NumPy**: Numerical computations and vector operations

### Libraries

```txt
sentence-transformers
faiss-cpu
numpy
selenium
bs4
ollama (Python client)
```

## Data Sources

- **Airline Policies**: JetBlue airline policies dataset
  - Source: Custom parsed airline policy documents
  - Format: JSONL (JSON Lines) with question-answer pairs
  - File: `policies.jsonl` (263 Q&A pairs)
  - Sections: Baggage, Pet Travel, Cancellations, Fares, Seat Selection, etc.

## Setup Instructions

### Prerequisites

1. **Install Python 3.11+**

   ```cmd
   python --version
   ```

2. **Install Ollama**

   - Download from: https://ollama.ai
   - Pull the Mistral model:

   ```cmd
   ollama pull mistral:instruct
   ```

3. **Create Virtual Environment**

   ```cmd
   python -m venv .hackenv
   .hackenv\Scripts\activate
   ```

4. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

## How to Run

### Step 1: Parse Policies (First Time Setup)

```cmd
python policy_parser.py
```

This parses raw airline policy data and creates `policies.json`.

### Step 2: Convert to JSONL Format

```cmd
python convert_to_jsonl.py
```

This converts `policies.json` to `policies.jsonl` (required for RAG).

### Step 3: Run the Chatbot

```cmd
python main.py
```

**What happens when you run `main.py`:**

- Automatically loads/builds the vector index (calls `vector_store.py` internally)
- Initializes the conversation model (calls `conversation_model.py`)
- Sets up context manager (calls `context_manager.py`)
- Starts the response generator (calls `response_generator.py`)
- Launches the interactive chatbot interface

**Note:** The first time you run `main.py`, it will automatically:

- Generate embeddings for all policy questions
- Build the FAISS index
- Save to `vector_index/` directory

Subsequent runs will load the cached index instantly.

### Step 4: Interact with the Bot

```
You: Can I bring my cat on the flight?
Bot: [Retrieves relevant policies and generates answer]

You: What are the baggage fees?
Bot: [Answers based on retrieved policies]

You: exit
```

## Execution Flow

```
policy_parser.py → policies.json
                      ↓
            convert_to_jsonl.py → policies.jsonl
                                      ↓
                                  main.py
                                      ├─→ vector_store.py (builds/loads index)
                                      └─→ response_generator.py (orchestrates RAG)
                                              ├─→ preprocessor.py (text cleaning)
                                              ├─→ vector_store.py (retrieves policies)
                                              ├─→ conversation_model.py (generates response)
                                              └─→ context_manager.py (conversation history)
```

## File Structure

```
ps3/
├── main.py                      # Main chatbot interface
├── response_generator.py        # Response generation with RAG
├── conversation_model.py        # LLM wrapper (Ollama)
├── context_manager.py           # Conversation context tracking
├── vector_store.py              # RAG implementation (FAISS + embeddings)
├── policy_parser.py             # Policy parsing utilities
├── preprocessor.py              # Text cleaning and preprocessing
├── convert_to_jsonl.py          # JSON to JSONL converter
├── policies.json                # Original policies (nested JSON)
├── policies.jsonl               # Policies in JSONL format
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── vector_index/                # Cached embeddings and FAISS index
    ├── embeddings.npz
    ├── metadata.json
    └── policy_index.index
```

## 🔧 Key Components

### 1. Vector Store (`vector_store.py`)

- Embeds policy questions using `all-MiniLM-L6-v2`
- Uses FAISS for efficient similarity search
- Returns L2 distance (lower = better match)

### 2. Response Generator (`response_generator.py`)

- Checks if query is airline-related
- Retrieves top-k relevant policies (k=3)
- Builds prompt with retrieved context
- Generates concise responses via LLM

### 3. Conversation Model (`conversation_model.py`)

- Wraps Ollama API for LLM calls
- Uses `mistral:instruct` model
- Handles streaming and error cases

### 4. Context Manager (`context_manager.py`)

- Maintains conversation history
- Limits context to last N turns
- Formats context for prompt inclusion

## Example Usage

```
You: can i bring my cat with me

Retrieved 3 relevant policies:
  • [Pet Travel] (distance: 0.950)
  • [Pet Travel] (distance: 1.007)
  • [Fares] (distance: 1.083)

Bot: Yes, you can bring your cat! You'll need to keep it in an approved carrier under the seat. You can book an extra seat for your pet if needed. A pet fee applies, and space is limited to 6 pets per flight.

You: what's the square root of 42

Bot: I'm an airline customer support assistant. I can only help with flight bookings, baggage, cancellations, pets, fares, and other airline-related questions.
```

## Distance Interpretation

- **Distance < 0.5**: Excellent match
- **Distance < 1.0**: Good match
- **Distance < 2.0**: Acceptable match (used as threshold)
- **Distance > 2.0**: Poor match (filtered out)



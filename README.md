# 🍽️ Restaurant Graph Agent

An intelligent restaurant assistant powered by AI, built with LangGraph, Neo4j, and OpenAI. This system provides personalized menu recommendations, ingredient information, and dietary assistance through natural language conversations.

## 🌟 Features

- **🤖 Single Agent Architecture**: Intelligent waiter agent handles both menu assistance and customer registration when needed
- **🔍 Semantic Search**: Vector embeddings and graph-based search for intelligent menu recommendations
- **📊 Knowledge Graph**: Neo4j-powered menu database with relationships between dishes, ingredients, categories, and cuisines
- **👤 Customer Management**: Supabase-backed user profiles with preferences and dietary restrictions
- **🌐 Multiple Interfaces**: Beautiful Streamlit web UI and command-line interface
- **📈 Evaluation System**: LangSmith integration for automated quality assessment
- **🥗 Dietary Intelligence**: Handles vegetarian, vegan, spice preferences, and allergen information

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Command Line   │    │   Evaluation    │
│   (Web App)     │    │   Interface     │    │   (LangSmith)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │       LangGraph Core       │
                    │    (Single Waiter Agent)   │
                    └─────────────┬──────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    ┌─────▼─────┐        ┌───────▼────────┐    ┌───────▼────────┐
    │  Neo4j    │        │    OpenAI      │    │   Supabase     │
    │(Menu KB)  │        │ (LLM + Embed)  │    │ (Customer DB)  │
    └───────────┘        └────────────────┘    └────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database (local or cloud)
- OpenAI API Key
- Supabase Account
- LangSmith Account (for evaluation)

### 1. Installation

```bash
git clone <repository-url>
cd restaurant-graph-agent
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# LangSmith Configuration (for evaluation)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=restaurant-chatbot
```

### 3. Database Setup

#### Step 1: Neo4j Schema Setup

1. Open your Neo4j browser interface
2. Run the schema creation queries from `data_injestion.md` to create constraints and indexes

#### Step 2: Menu Data Upload

1. Create a Google Sheets menu following the format: [Sample Menu](https://drive.google.com/file/d/1M1PRw6jb39JPB-Z0F6ABunzhfaidaeP4/view?usp=sharing)
2. Make the sheet publicly accessible
3. Replace the Google Sheets ID in the Cypher query from `data_injestion.md`
4. Run the data upload query in Neo4j

#### Step 3: Generate Embeddings

```bash
python embed.py
```

This will:
- Generate OpenAI embeddings for all menu items
- Create vector indexes for semantic search
- Tag searchable nodes

### 4. Run the Application

#### Option A: Streamlit Web Interface (Recommended)

```bash
python run_app.py
```

Select option 1 for the web interface. The app will launch at `http://localhost:8501`

#### Option B: Command Line Interface

```bash
python run_app.py
```

Select option 2 for the terminal-based chat interface.

## 📋 Usage Workflow

### Customer Journey

1. **Registration Phase**
   - Provide 10-digit phone number
   - Share name and preferences (dietary, spice level, cuisine)
   - Set allergen information

2. **Menu Assistance Phase**
   - Ask about dishes, ingredients, categories
   - Get personalized recommendations
   - Inquire about allergens and dietary restrictions
   - Request preparation details and pricing

### Sample Conversations

```
User: 8691990606
Bot: Thank you! What's your name?

User: John
Bot: Nice to meet you, John! Would you like to set your food preferences?

User: I prefer vegetarian North Indian food with medium spice
Bot: Perfect! I've saved your preferences. How can I help you with our menu today?

User: Do you have any paneer dishes?
Bot: Yes! We have several delicious paneer options: Paneer Tikka, Paneer Makhani, Palak Paneer...
```

## 🔧 Technical Details

### Core Components

- **`main.py`**: LangGraph application with single agent architecture
- **`embed.py`**: Vector embedding generation for semantic search
- **`streamlit_app.py`**: Modern web interface with real-time chat
- **`run_app.py`**: Application launcher with port management
- **`eval.py`**: Automated evaluation using LangSmith

### Agent Architecture

**Waiter Agent**
   - Processes menu queries using Neo4j knowledge graph
   - Provides semantic search capabilities
   - Offers personalized recommendations
   - Handles customer registration when needed for orders/reservations
   - Manages Supabase customer records

### Search Capabilities

- **Symbolic Search**: Cypher queries for exact matches and relationships
- **Semantic Search**: Vector similarity for fuzzy matching and recommendations
- **Hybrid Approach**: Combines both methods for comprehensive results

## 📊 Evaluation

### Running Evaluations

```bash
python eval.py
```

The evaluation system:
- Uses predefined question datasets in LangSmith
- Tests helpfulness, correctness, completeness, and overall quality
- Provides detailed scoring and insights
- Simulates complete user registration and interaction flows

### Evaluation Metrics

- **Helpfulness** (0.0-1.0): How well the response addresses the customer's question
- **Correctness** (0.0-1.0): Accuracy of information compared to reference answers
- **Completeness** (0.0-1.0): Comprehensiveness of the response
- **Overall Quality** (0.0-1.0): Holistic assessment of the interaction

## 🗂️ File Structure

```
restaurant-graph-agent/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
├── data_injestion.md        # Neo4j setup instructions
├── main.py                  # Core LangGraph application
├── embed.py                 # Embedding generation script
├── run_app.py               # Application launcher
├── streamlit_app.py         # Web interface
├── eval.py                  # Evaluation system
└── workflow_graph.png       # Architecture diagram
```

## 🛠️ Configuration

### Neo4j Schema

The knowledge graph includes:
- **Dishes**: Menu items with descriptions, prices, preparation time
- **Ingredients**: Components of each dish
- **Categories**: Appetizers, main courses, desserts, etc.
- **Cuisines**: North Indian, South Indian, Chinese, etc.
- **Allergens**: Nuts, gluten, dairy, etc.
- **Diet Types**: Vegetarian, vegan, non-vegetarian

### Relationships

- `CONTAINS`: Dish → Ingredient
- `IN_CATEGORY`: Dish → Category  
- `OF_CUISINE`: Dish → Cuisine
- `HAS_ALLERGEN`: Dish → Allergen
- `HAS_DIET_TYPE`: Dish → DietType

## 🚦 Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Verify Neo4j is running
   - Check connection credentials in `.env`
   - Ensure firewall allows port 7687

2. **Embeddings Not Generated**
   - Verify OpenAI API key
   - Check Neo4j data upload completion
   - Run `embed.py` manually

3. **Streamlit Port Issues**
   - Run `python run_app.py` and select option 1
   - The launcher automatically handles port conflicts

4. **Evaluation Failures**
   - Ensure LangSmith API key is set
   - Verify dataset "Test base" exists in LangSmith
   - Check that main chatbot is working correctly

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_VERBOSE=true
```

## 🔮 Future Enhancements

- Order management and checkout functionality
- Multi-language support
- Voice interface integration
- Real-time inventory tracking
- Advanced recommendation algorithms
- Integration with POS systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support, please create an issue in the repository or contact the development team.

---

**Built with ❤️ using LangGraph, Neo4j, and OpenAI** 
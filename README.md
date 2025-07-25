# Restaurant Graph Agent

An AI-powered restaurant assistant that combines advanced menu knowledge with intelligent reservation management. This application uses LangGraph for agent orchestration, Neo4j for menu knowledge graphs, and Supabase for customer data, with a modern Next.js frontend.

## 🏗️ Project Structure

```
restaurant-graph-agent/
├── 📱 frontend/                  # Frontend Application (Next.js + React)
│   ├── app/                      # Next.js 13+ App Router
│   │   ├── api/                  # API route handlers
│   │   ├── page.tsx              # Main application page
│   │   ├── layout.tsx            # Root layout component
│   │   └── globals.css           # Global styles
│   ├── src/                      # Organized source code
│   │   ├── components/           # React components (chat, layout, ui)
│   │   ├── lib/                  # Utilities and helper functions
│   │   ├── types/                # TypeScript definitions
│   │   ├── constants/            # Application constants
│   │   ├── hooks/                # Custom React hooks
│   │   └── styles/               # Additional stylesheets
│   ├── package.json              # Frontend dependencies
│   ├── tsconfig.json             # TypeScript configuration
│   ├── tailwind.config.js        # Tailwind CSS configuration
│   └── next.config.js            # Next.js configuration
│
├── 🧠 backend/                   # Backend Logic (Python + LangGraph)
│   ├── services/                 # Core backend services
│   │   ├── main.py               # Core LangGraph application
│   │   └── backend_api.py        # FastAPI bridge server
│   ├── tools/                    # Agent tool functions
│   │   ├── search_tools.py       # AI-powered search and debugging
│   │   ├── menu_tools.py         # Menu recommendations and details
│   │   ├── customer_tools.py     # Customer registration and management
│   │   ├── reservation_tools.py  # Reservation management
│   │   └── faq_tools.py          # FAQ search and responses
│   ├── data/                     # Database connections and clients
│   │   ├── connections.py        # Global connection pooling
│   │   ├── neo4j_client.py       # Neo4j knowledge graph client
│   │   ├── openai_client.py      # OpenAI embedding client
│   │   └── supabase_client.py    # Supabase customer data client
│   ├── core/                     # Core application logic
│   ├── agents/                   # LangGraph agent definitions
│   └── utils/                    # Backend utilities
│
├── 📊 config/                    # Configuration files
├── 📜 scripts/                   # Utility scripts
├── 🧪 evaluation/                # Testing and evaluation
├── 📖 docs/                      # Documentation
│
├── 🧠 backend/                   # Backend Logic (Python + LangGraph)
│   ├── services/                 # Core backend services
│   │   ├── main.py               # Core LangGraph application
│   │   └── backend_api.py        # FastAPI bridge server  
├── 📋 requirements.txt           # Python dependencies
└── 📖 README.md                  # This file
```

## ✨ Key Features

### 🤖 **Intelligent Agent System**
- **Single Waiter Agent** powered by LangGraph for natural conversation flow
- **13 Specialized Tools** for menu search, reservations, customer management, and FAQ
- **Context-aware responses** with memory across conversation threads
- **Tool-based architecture** for extensible functionality

### 🗃️ **Advanced Knowledge Management**
- **Neo4j Knowledge Graph** storing 1000+ menu items with detailed relationships
- **Vector embeddings** for semantic search and recommendations  
- **Dynamic schema introspection** for intelligent query generation
- **FAQ database** with 15+ common restaurant questions

### 📅 **Smart Reservation System**
- **Real-time availability checking** with conflict resolution
- **Customer phone verification** with registration management
- **Reservation modification and cancellation** with business rule validation
- **Supabase integration** for reliable customer data storage

### 🎨 **Modern Frontend**
- **Next.js 13+ with App Router** for optimal performance
- **Real-time chat interface** with typing indicators and smooth animations
- **Responsive design** optimized for desktop and mobile
- **Professional UI components** built with Tailwind CSS and Radix UI

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Neo4j database** (local or cloud)
- **Supabase account** (free tier available)
- **OpenAI API key** for embeddings and chat

### Environment Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd restaurant-graph-agent
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Fill in your API keys and database credentials
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend && npm install && cd ..
   ```

5. **Start the application**
   ```bash
   python scripts/development/run_next_app.py
   ```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

## 🛠️ Technical Architecture

### Frontend Stack
- **Next.js 13+** - React framework with App Router
- **TypeScript** - Type safety and developer experience
- **Tailwind CSS** - Utility-first styling framework
- **Radix UI** - Accessible component primitives
- **Lucide Icons** - Modern icon library

### Backend Stack
- **LangGraph** - Agent orchestration and workflow management
- **FastAPI** - High-performance API bridge
- **LangChain** - LLM integration and tool management
- **OpenAI GPT-4** - Language model and embeddings

### Data Layer
- **Neo4j** - Graph database for menu knowledge
- **Supabase** - Customer data and reservations
- **Vector Search** - Semantic similarity matching
- **Connection Pooling** - Optimized database performance

## 📊 Performance & Reliability

### System Capabilities
- **Sub-second response times** for menu queries
- **Concurrent user support** with connection pooling
- **Fault-tolerant design** with graceful error handling
- **Automatic system initialization** with health monitoring

### Data Features
- **1000+ menu items** with rich metadata and relationships
- **Vector embeddings** for intelligent recommendations
- **Real-time reservation availability** with conflict resolution
- **Customer history tracking** with privacy protection

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# LangSmith (Optional - for debugging)
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
```

### System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB available space
- **Network**: Internet connection for API services
- **Ports**: 3000 (frontend), 8000 (backend), 7687 (Neo4j)

## 🧪 Development & Testing

### Development Workflow
```bash
# Backend development
python backend/services/main.py  # Direct LangGraph testing

# Frontend development  
cd frontend && npm run dev

# Full stack development
python scripts/development/run_next_app.py  # Both services with live logs
```

### Code Organization
- **Frontend**: All UI code organized in `frontend/` with clear component structure
- **Backend**: All Python code organized in `backend/` with domain-specific modules
- **Tools**: Each tool type (search, menu, customer, reservation, FAQ) in separate modules
- **Configuration**: All config files properly organized by purpose

## 📚 Documentation

- **Frontend README**: `frontend/README.md` - Detailed frontend development guide
- **Component Documentation**: `frontend/src/components/` - Individual component docs
- **Backend API**: `backend_api.py` - FastAPI interactive docs at `/docs`
- **Tool Documentation**: `backend/tools/` - Individual tool module documentation

## 🤝 Contributing

This codebase is designed for maintainability and extensibility:

1. **Frontend**: Follow the established component patterns in `frontend/src/components/`
2. **Backend**: Add new tools in `backend/tools/` following the `@tool` decorator pattern
3. **Types**: Add TypeScript types in `frontend/src/types/` for frontend, Python types in backend modules
4. **Testing**: Use the evaluation framework in `evaluation/` for testing changes

## 🔍 Troubleshooting

### Common Issues
- **Port conflicts**: The launcher automatically handles port conflicts
- **Database connections**: Check environment variables and network access  
- **Missing dependencies**: Run `pip install -r requirements.txt` and `cd frontend && npm install`
- **Performance**: Monitor the evaluation metrics and optimize based on bottlenecks

### Debug Mode
```bash
# Enable detailed logging
export LANGCHAIN_TRACING_V2=true
python scripts/development/run_next_app.py
```

## 📞 Support

For technical issues or questions about this restaurant agent system, please check:
1. The troubleshooting section above
2. Component-specific READMEs in their directories
3. The evaluation framework for testing tools and changes 
# Backend Services

This directory contains the core backend services for the Restaurant Graph Agent application.

## 📁 Directory Contents

```
backend/services/
├── main.py          # Core LangGraph application and agent orchestration
├── backend_api.py   # FastAPI bridge server for frontend communication
└── README.md       # This file
```

## 🚀 Core Services

### `main.py` - LangGraph Application Core

**Purpose**: Main LangGraph application with the restaurant waiter agent
**Technology**: LangGraph + LangChain + OpenAI GPT-4

**Key Features**:
- **Single Waiter Agent**: Handles all restaurant interactions
- **13 Specialized Tools**: Menu search, reservations, customer management, FAQ
- **Neo4j Integration**: Knowledge graph queries for menu intelligence  
- **Supabase Integration**: Customer data and reservation management
- **Vector Search**: Semantic menu search with OpenAI embeddings
- **Configuration-driven**: Uses `config/prompts/prompts.yaml` for behavior

**Usage**:
```bash
# Run directly for testing/debugging
python backend/services/main.py

# Or use the full launcher
python scripts/development/run_next_app.py
```

**Architecture**:
- **State Management**: LangGraph state for conversation context
- **Tool Orchestration**: Automatic tool selection based on user intent
- **Memory**: Conversation memory with phone number persistence
- **Error Handling**: Graceful fallbacks and error recovery

### `backend_api.py` - FastAPI Bridge Server

**Purpose**: REST API bridge between Next.js frontend and LangGraph backend
**Technology**: FastAPI + Uvicorn

**Key Features**:
- **HTTP Endpoints**: RESTful API for frontend communication
- **CORS Support**: Cross-origin requests from Next.js frontend
- **Background Initialization**: System warm-up during startup
- **Timeout Protection**: Prevents hanging requests
- **Error Handling**: Comprehensive error responses with details

**API Endpoints**:
- `POST /api/chat` - Main chat endpoint
- `POST /api/initialize` - System initialization
- `GET /api/initialization-status` - Initialization progress
- `GET /api/health` - Health check
- `GET /` - Basic status

**Usage**:
```bash
# Run directly
python backend/services/backend_api.py

# Or use the full launcher  
python scripts/development/run_next_app.py
```

**Configuration**:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **CORS Origins**: localhost:3000, 127.0.0.1:3000
- **Timeouts**: 3 minutes per request

## 🔧 Integration Architecture

### Service Communication Flow

```
Next.js Frontend (Port 3000)
    ↓ HTTP/JSON
FastAPI Bridge (backend_api.py)
    ↓ Python function calls
LangGraph Core (main.py)
    ↓ Tool calls
Backend Tools (../tools/)
    ↓ Database queries
Data Layer (../data/)
```

### Dependencies Between Services

**main.py Dependencies**:
- `backend.tools.*` - All agent tools
- `backend.data.*` - Database connections
- `config/prompts/prompts.yaml` - Agent behavior configuration
- Environment variables for API keys

**backend_api.py Dependencies**:
- `main.py` - Core LangGraph functionality (lazy loaded)
- FastAPI framework for HTTP handling
- Concurrent futures for timeout protection

## 🚀 Development Workflow

### Local Development

```bash
# Backend-only development (direct LangGraph testing)
python backend/services/main.py

# API server development  
python backend/services/backend_api.py

# Full-stack development (recommended)
python scripts/development/run_next_app.py
```

### Debugging

```bash
# Enable detailed logging
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_VERBOSE=true

# Run with debugging
python backend/services/main.py
```

### Testing

```bash
# Test LangGraph core
python -c "
import sys
sys.path.append('.')
from backend.services.main import graph, ensure_system_initialized
ensure_system_initialized()
print('✅ LangGraph core working')
"

# Test FastAPI bridge
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "config": {"configurable": {"thread_id": "test"}}}'
```

## 📋 Configuration

### Environment Variables

Both services require these environment variables:

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

# LangSmith (Optional)
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
```

### Configuration Files

- **Prompts**: `config/prompts/prompts.yaml` - Agent behavior and instructions
- **Dependencies**: `requirements.txt` - Python package dependencies

## 🔍 Monitoring & Observability

### Logging

Both services provide detailed logging:
- **Timing Information**: Performance metrics for optimization
- **Debug Output**: Tool calls, database queries, decisions
- **Error Tracking**: Comprehensive error reporting with stack traces

### Health Checks

```bash
# Check if backend API is running
curl http://localhost:8000/api/health

# Check initialization status
curl http://localhost:8000/api/initialization-status
```

### Performance Monitoring

- **Request Latency**: Tracked for all chat requests
- **Database Performance**: Connection pooling and query caching
- **Memory Usage**: LangGraph state management
- **Tool Execution Time**: Individual tool performance metrics

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Run from project root
cd /path/to/restaurant-graph-agent
python backend/services/main.py
```

**Database Connection Issues**:
```bash
# Test Neo4j connection
python -c "from backend.data.neo4j_client import get_neo4j_connection; print(get_neo4j_connection())"

# Test Supabase connection  
python -c "from backend.data.supabase_client import get_supabase_client; print(get_supabase_client())"
```

**Configuration Issues**:
```bash
# Test config loading
python -c "from backend.services.main import load_prompts_config; print('✅ Config loaded')"
```

**Port Conflicts**:
```bash
# Use the launcher script (handles port conflicts automatically)
python scripts/development/run_next_app.py
```

### Performance Issues

**Slow Initialization**:
- Check database connectivity
- Verify API keys are valid
- Monitor system resource usage

**Slow Chat Responses**:
- Check Neo4j query performance
- Monitor OpenAI API latency
- Review tool execution times in logs

## 🚀 Deployment Considerations

### Production Deployment

**main.py** (LangGraph Core):
- Can be deployed as a separate service
- Requires all database connections
- Memory usage scales with concurrent conversations

**backend_api.py** (FastAPI Bridge):
- Deploy with ASGI server (Uvicorn, Gunicorn)
- Configure proper CORS origins for production
- Set appropriate timeout values
- Enable request logging and monitoring

### Scaling

- **Horizontal Scaling**: Multiple backend_api.py instances behind load balancer
- **Database Scaling**: Connection pooling handles concurrent requests
- **Caching**: Query results cached to reduce database load
- **Memory Management**: LangGraph state cleanup for long conversations

## 📚 Additional Documentation

- **Tools Documentation**: `../tools/README.md`
- **Data Layer**: `../data/README.md`
- **Scripts**: `../../scripts/README.md`
- **Configuration**: `../../config/README.md`
- **Architecture**: `../../docs/architecture/workflow_graph.png` 
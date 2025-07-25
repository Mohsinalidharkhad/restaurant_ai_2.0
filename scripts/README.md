# Scripts Directory

This directory contains all utility scripts for the Restaurant Graph Agent project, organized by purpose and usage phase.

## 📁 Directory Structure

```
scripts/
├── setup/              # Initial setup and data preparation scripts
├── development/        # Development and deployment scripts  
├── evaluation/         # Testing and evaluation scripts
└── README.md          # This file
```

## 🛠️ Setup Scripts (`setup/`)

Scripts for initial system setup and data preparation:

### `embed.py`
**Purpose**: Generate vector embeddings for menu items
**Usage**: `python scripts/setup/embed.py`
**Prerequisites**: 
- Neo4j database with menu data loaded
- OpenAI API key configured
- All Python dependencies installed

**What it does**:
- Connects to Neo4j and retrieves all menu items
- Generates OpenAI embeddings for dish names and descriptions
- Stores embeddings in Neo4j for semantic search
- Creates vector indexes for efficient similarity search

### `populate_faq_embeddings.py`
**Purpose**: Generate embeddings for FAQ database
**Usage**: `python scripts/setup/populate_faq_embeddings.py`
**Prerequisites**:
- Neo4j database connection
- FAQ data loaded in Neo4j
- OpenAI API key configured

**What it does**:
- Reads FAQ questions and answers from Neo4j
- Generates embeddings for semantic FAQ search
- Updates Neo4j with vector embeddings
- Enables intelligent FAQ matching

## 🚀 Development Scripts (`development/`)

Scripts for running and deploying the application:

### `run_next_app.py`
**Purpose**: Complete application launcher with monitoring
**Usage**: `python scripts/development/run_next_app.py`
**Features**:
- Starts both backend (FastAPI) and frontend (Next.js)
- Handles port conflicts automatically
- Live log streaming with color coding
- Graceful shutdown handling
- Dependency checking and installation

**Ports**:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

**What it does**:
1. Validates Node.js and Python dependencies
2. Checks and clears port conflicts
3. Starts FastAPI backend with live logging
4. Starts Next.js frontend with live logging
5. Provides unified monitoring and shutdown

## 🧪 Evaluation Scripts (`evaluation/`)

Scripts for testing and quality assessment:

### `eval.py`
**Purpose**: Automated evaluation using LangSmith
**Usage**: `python scripts/evaluation/eval.py`
**Prerequisites**:
- LangSmith API key configured
- Main application running and functional
- Test datasets available in LangSmith

**What it does**:
- Runs automated conversation tests
- Evaluates response quality on multiple metrics
- Generates detailed performance reports
- Tests complete user interaction flows

**Metrics Evaluated**:
- Helpfulness (0.0-1.0)
- Correctness (0.0-1.0)  
- Completeness (0.0-1.0)
- Overall Quality (0.0-1.0)

## 🚀 Quick Start Guide

### First-Time Setup
```bash
# 1. Setup database and generate embeddings
python scripts/setup/embed.py
python scripts/setup/populate_faq_embeddings.py

# 2. Start the application
python scripts/development/run_next_app.py
```

### Development Workflow
```bash
# For full-stack development with live logs
python scripts/development/run_next_app.py

# For backend-only development  
python main.py

# For frontend-only development
cd frontend && npm run dev
```

### Testing and Evaluation
```bash
# Run comprehensive evaluation
python scripts/evaluation/eval.py
```

## 📋 Script Dependencies

### Common Requirements
- Python 3.8+
- Virtual environment activated
- `.env` file with API keys configured
- `requirements.txt` dependencies installed

### Database Requirements
- Neo4j database running and accessible
- Supabase project configured
- Menu data loaded (see `docs/setup/data_injestion.md`)

### API Requirements
- OpenAI API key for embeddings and chat
- LangSmith API key for evaluation (optional)
- Supabase API keys for customer data

## 🔧 Environment Variables

All scripts require these environment variables (configured in `.env`):

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

## 🐛 Troubleshooting

### Common Issues

**Script not found**: Scripts moved to organized directories
```bash
# Old: python run_next_app.py
# New: python scripts/development/run_next_app.py
```

**Import errors**: Run scripts from project root
```bash
# Correct usage from project root
cd /path/to/restaurant-graph-agent
python scripts/setup/embed.py
```

**Database connection issues**: Check environment variables and database status
```bash
# Test Neo4j connection
python -c "from neo4j import GraphDatabase; print('Neo4j connected')"
```

**Port conflicts**: Let run_next_app.py handle automatically
```bash
# The launcher script automatically handles port conflicts
python scripts/development/run_next_app.py
```

## 📖 Additional Documentation

- **Setup Guide**: `docs/setup/data_injestion.md`
- **FAQ Management**: `docs/guides/faq.md`
- **Architecture**: `docs/architecture/workflow_graph.png`
- **Frontend Development**: `frontend/README.md`
- **Backend Development**: `backend/README.md` (if exists)

## 🤝 Contributing

When adding new scripts:

1. **Choose the right directory**:
   - `setup/` - One-time setup or data preparation
   - `development/` - Running, building, or deploying
   - `evaluation/` - Testing, benchmarking, or validation

2. **Follow naming conventions**:
   - Use descriptive names (e.g., `generate_menu_embeddings.py`)
   - Use underscores for multi-word names
   - Include `.py` extension for Python scripts

3. **Add documentation**:
   - Include docstring with purpose and usage
   - Add to this README if it's a major script
   - Document any special requirements

4. **Test thoroughly**:
   - Ensure script works from project root
   - Test with clean environment
   - Verify all dependencies are documented 
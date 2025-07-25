# Documentation Directory

This directory contains all documentation for the Restaurant Graph Agent project, organized by purpose and target audience.

## 📁 Directory Structure

```
docs/
├── setup/              # Installation and configuration guides
├── guides/             # User guides and how-to documentation
├── architecture/       # System architecture and design documents
└── README.md          # This file
```

## 🚀 Setup Documentation (`setup/`)

Essential guides for getting the system up and running:

### `data_injestion.md`
**Target Audience**: Developers, System Administrators
**Purpose**: Complete guide for setting up the Neo4j knowledge graph database

**Contents**:
- Neo4j schema creation and constraints
- Menu data upload from Google Sheets
- Database indexing and optimization
- Vector embedding setup instructions
- Troubleshooting common database issues

**Key Sections**:
1. **Schema Setup** - Creating nodes, relationships, and constraints
2. **Data Import** - Loading menu data from external sources
3. **Indexing** - Creating performance indexes
4. **Verification** - Testing database setup
5. **Troubleshooting** - Common issues and solutions

## 📖 User Guides (`guides/`)

Practical guides for using and maintaining the system:

### `faq.md`
**Target Audience**: End Users, Customer Support, Content Managers
**Purpose**: Comprehensive FAQ database for the restaurant assistant

**Contents**:
- Restaurant hours and location information
- Menu categories and dish descriptions
- Dietary restrictions and allergen information
- Reservation policies and procedures
- Special services and accommodations

**Key Sections**:
1. **General Information** - Basic restaurant details
2. **Menu Questions** - Common food and drink inquiries
3. **Reservations** - Booking policies and procedures
4. **Dietary Needs** - Vegetarian, vegan, allergen info
5. **Special Occasions** - Events and group bookings

## 🏗️ Architecture Documentation (`architecture/`)

Technical documentation about system design and architecture:

### `workflow_graph.png`
**Target Audience**: Developers, Technical Stakeholders
**Purpose**: Visual representation of the LangGraph agent workflow

**Diagram Contents**:
- Agent decision flow and tool selection
- Database interaction patterns
- Frontend-backend communication flow
- Error handling and fallback mechanisms
- User registration and authentication flow

**Technical Details**:
- **Format**: PNG image diagram
- **Tools Used**: LangGraph visualization
- **Updated**: Automatically when workflow changes
- **Size**: Optimized for documentation viewing

## 📋 Documentation Standards

### Writing Guidelines

**Style**:
- Use clear, concise language
- Write for the target audience skill level
- Include code examples where relevant
- Use consistent formatting and structure

**Structure**:
- Start with purpose and target audience
- Include prerequisites and assumptions
- Provide step-by-step instructions
- Add troubleshooting sections
- Include examples and screenshots where helpful

**Markdown Conventions**:
- Use `#` for main sections, `##` for subsections
- Use `**bold**` for emphasis, `*italic*` for mild emphasis
- Use `` `code` `` for inline code, `` ```language `` for code blocks
- Use tables for structured data
- Use bullet points for lists and steps

### File Naming

**Conventions**:
- Use lowercase with underscores: `data_setup.md`
- Be descriptive: `neo4j_schema_setup.md` not `setup.md`
- Use `.md` extension for Markdown files
- Use descriptive image names: `agent_workflow_diagram.png`

### Organization

**By Directory**:
- **setup/**: Getting started, installation, configuration
- **guides/**: Using features, managing content, troubleshooting
- **architecture/**: System design, technical specifications, diagrams

**By Audience**:
- **End Users**: FAQ, feature guides, troubleshooting
- **Developers**: Architecture, API docs, setup guides
- **Administrators**: Deployment, configuration, maintenance

## 🔗 Cross-References

### Internal Links
- **Main README**: `../README.md` - Project overview and quick start
- **Frontend Docs**: `../frontend/README.md` - Frontend development guide
- **Backend Docs**: `../backend/README.md` - Backend development guide
- **Scripts**: `../scripts/README.md` - Available utility scripts

### External Resources
- **LangGraph Documentation**: https://python.langchain.com/docs/langgraph
- **Neo4j Documentation**: https://neo4j.com/docs/
- **Next.js Documentation**: https://nextjs.org/docs
- **FastAPI Documentation**: https://fastapi.tiangolo.com/

## 🚀 Quick Navigation

### For New Developers
1. **Start Here**: `../README.md` - Project overview
2. **Setup Database**: `setup/data_injestion.md` - Database configuration
3. **Run Application**: `../scripts/README.md` - Launch scripts
4. **Architecture**: `architecture/workflow_graph.png` - System design

### For Content Managers
1. **FAQ Management**: `guides/faq.md` - Managing restaurant information
2. **Menu Updates**: `setup/data_injestion.md` - Updating menu data
3. **Testing**: `../scripts/README.md` - Evaluation scripts

### For End Users
1. **Using the Assistant**: `guides/faq.md` - Common questions and answers
2. **Getting Help**: `../README.md` - Support and troubleshooting

## 🔄 Maintenance

### Keeping Documentation Updated

**When to Update**:
- After adding new features or tools
- When changing system architecture
- After modifying setup procedures
- When fixing bugs that affect documentation

**Update Checklist**:
- [ ] Update relevant section content
- [ ] Check all internal links still work
- [ ] Update screenshots or diagrams if needed
- [ ] Review for accuracy and clarity
- [ ] Test any included code examples

### Version Control

**Documentation Versioning**:
- Keep documentation in sync with code changes
- Use clear commit messages for doc updates
- Tag major documentation updates
- Maintain backward compatibility notes when needed

## 🤝 Contributing to Documentation

### Adding New Documentation

1. **Choose the right directory**:
   - `setup/` - Installation, configuration, first-time setup
   - `guides/` - Feature usage, how-to guides, troubleshooting
   - `architecture/` - System design, technical specifications

2. **Follow the standards**:
   - Use the writing guidelines above
   - Include target audience in the header
   - Follow the established file naming conventions
   - Add cross-references to related documentation

3. **Review process**:
   - Test any included procedures or code
   - Check for spelling and grammar
   - Ensure links work correctly
   - Verify screenshots are current

### Updating Existing Documentation

1. **Identify what needs updating**
2. **Make changes following style guidelines**
3. **Test any modified procedures**
4. **Update cross-references if needed**
5. **Add notes about what changed**

## 📞 Documentation Support

For questions about documentation:
1. Check the relevant guide first
2. Look for related information in cross-referenced docs  
3. Review the troubleshooting sections
4. Create an issue for missing or unclear documentation 
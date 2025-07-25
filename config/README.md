# Configuration Directory

This directory contains all configuration files for the Restaurant Graph Agent project, organized by component and purpose.

## 📁 Directory Structure

```
config/
├── prompts/            # LLM prompts and instructions
├── environment/        # Environment-specific configurations
└── README.md          # This file
```

## 🤖 Prompts Configuration (`prompts/`)

LLM prompts and instructions for the AI agent system:

### `prompts.yaml`
**Purpose**: Centralized configuration for all AI agent prompts and behaviors
**Format**: YAML configuration file
**Updated**: When fine-tuning agent behavior or adding new features

**Key Sections**:

#### 1. **Waiter Agent Configuration**
```yaml
waiter_agent:
  personality: "Friendly, knowledgeable restaurant waiter"
  tone: "Professional yet warm and approachable"
  conversation_style: "Natural, helpful, and efficient"
```

#### 2. **Tool Instructions**
- **Menu Search**: Instructions for semantic search and recommendations
- **Customer Management**: User registration and preference handling
- **Reservation System**: Booking, modification, and cancellation workflows
- **FAQ Handling**: Restaurant information and common questions

#### 3. **Database Query Generation**
- **Neo4j Cypher**: Templates and examples for graph queries
- **Error Handling**: Fallback strategies for failed queries
- **Optimization**: Performance-focused query patterns

#### 4. **Response Formatting**
- **Message Structure**: How to format responses for the UI
- **Error Messages**: User-friendly error communication
- **Success Confirmations**: Positive feedback patterns

#### 5. **Evaluation Criteria**
- **Quality Metrics**: Rubrics for response evaluation
- **Scoring Guidelines**: How to assess helpfulness and accuracy
- **Test Scenarios**: Standard evaluation cases

### Configuration Benefits

**Externalized Prompts Enable**:
- ✅ **No-Code Behavior Changes** - Modify AI personality without coding
- ✅ **Version Control** - Track prompt changes and effectiveness
- ✅ **A/B Testing** - Compare different prompt strategies
- ✅ **Team Collaboration** - Non-technical team members can improve prompts
- ✅ **Rapid Iteration** - Quick adjustments without redeployment

**Use Cases**:
- Adjusting agent personality and tone
- Fine-tuning restaurant-specific information
- Customizing conversation flows
- Optimizing database query generation
- Improving error handling messages

## 🌍 Environment Configuration (`environment/`)

Environment-specific settings and configurations:

### Purpose
Directory for environment-specific configuration files that may be added in the future:

**Potential Configurations**:
- **Development**: Local development settings
- **Staging**: Pre-production testing configurations  
- **Production**: Live deployment configurations
- **Testing**: Automated testing environment settings

**Example Files** (future):
- `development.yaml` - Local development overrides
- `production.yaml` - Production-specific settings
- `testing.yaml` - Test environment configurations
- `feature_flags.yaml` - Feature enablement flags

## 🔧 Configuration Management

### Loading Configuration

**In Python Code**:
```python
import yaml
from pathlib import Path

def load_prompts():
    """Load prompts from YAML configuration"""
    config_path = Path("config/prompts/prompts.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Usage in main.py
prompts_config = load_prompts()
waiter_prompt = prompts_config['waiter_agent']['system_prompt']
```

**Configuration Validation**:
- YAML syntax validation on load
- Required field checking
- Type validation for structured data
- Default value handling for optional fields

### Best Practices

**Security**:
- ❌ **Never store secrets** in configuration files
- ✅ **Use environment variables** for sensitive data (API keys, passwords)
- ✅ **Keep configs in version control** (except secrets)
- ✅ **Use .env files** for local development secrets

**Organization**:
- ✅ **Group related settings** in logical sections
- ✅ **Use descriptive names** for configuration keys
- ✅ **Include comments** explaining complex configurations
- ✅ **Maintain consistency** across different environments

**Versioning**:
- ✅ **Document changes** in commit messages
- ✅ **Test configuration changes** before deployment
- ✅ **Backup working configurations** before major changes
- ✅ **Use semantic versioning** for major config updates

## 📝 Configuration Schema

### prompts.yaml Structure

```yaml
# Agent Personality and Behavior
waiter_agent:
  system_prompt: "Detailed agent instructions..."
  personality: "Professional and friendly"
  conversation_guidelines: ["Be helpful", "Stay on topic", ...]

# Tool-Specific Instructions  
tools:
  menu_search:
    instructions: "How to search and recommend menu items..."
    examples: ["Example queries and responses..."]
  
  reservations:
    booking_flow: "Step-by-step reservation process..."
    policies: "Restaurant policies and rules..."

# Database Query Templates
database:
  neo4j:
    cypher_examples: ["MATCH patterns...", ...]
    error_handling: "Fallback strategies..."

# Response Formatting
formatting:
  success_messages: "How to format positive responses..."
  error_messages: "How to handle and format errors..."
  
# Evaluation and Quality
evaluation:
  metrics: ["helpfulness", "correctness", ...]
  scoring_rubric: "How to assess response quality..."
```

## 🔄 Configuration Updates

### When to Update Configurations

**Agent Behavior Changes**:
- Adjusting conversation tone or personality
- Adding new response patterns
- Improving error handling
- Customizing for different restaurant types

**Feature Additions**:
- New tool integrations
- Additional database queries
- Enhanced evaluation metrics
- New conversation flows

**Performance Optimization**:
- Better prompt engineering
- More efficient query patterns
- Improved error recovery
- Enhanced user experience

### Update Process

1. **Backup Current Config**: Save working configuration
2. **Make Changes**: Edit YAML files carefully
3. **Validate Syntax**: Check YAML structure and format
4. **Test Changes**: Run application with new configuration
5. **Monitor Performance**: Check if changes improve metrics
6. **Document Changes**: Update this README if needed

### Testing Configuration Changes

**Local Testing**:
```bash
# Test configuration loading
python -c "
import yaml
with open('config/prompts/prompts.yaml') as f:
    config = yaml.safe_load(f)
    print('✅ Configuration loaded successfully')
    print(f'Sections: {list(config.keys())}')
"

# Test with application
python scripts/development/run_next_app.py
```

**Validation Checklist**:
- [ ] YAML syntax is valid
- [ ] All required sections are present
- [ ] No sensitive data in config files
- [ ] Agent responses match expected behavior
- [ ] Database queries work correctly
- [ ] Error handling functions properly

## 🚀 Quick Configuration Guide

### Common Adjustments

**Change Agent Personality**:
```yaml
waiter_agent:
  personality: "Casual and friendly" # or "Formal and professional"
  tone: "Conversational" # or "Business-like"
```

**Adjust Error Messages**:
```yaml
formatting:
  error_messages:
    database_error: "I'm having trouble accessing our menu right now..."
    invalid_input: "I didn't quite understand that. Could you rephrase?"
```

**Customize Restaurant Info**:
```yaml
restaurant:
  name: "Your Restaurant Name"
  hours: "11 AM - 10 PM daily"
  phone: "(555) 123-4567"
  specialty: "Modern Italian cuisine"
```

### Environment Variables

**Required for Configuration**:
```bash
# No environment variables needed for basic config loading
# Secrets should be in .env file, not in config files
```

**Optional for Advanced Features**:
```bash
# Feature flags (if implemented)
ENABLE_ADVANCED_PROMPTS=true
PROMPT_VERSION=v2
DEBUG_PROMPTS=false
```

## 🤝 Contributing to Configuration

### Adding New Configurations

1. **Identify the Purpose**: Setup, prompts, environment-specific, etc.
2. **Choose the Right Directory**: Follow the organization pattern
3. **Use Clear Structure**: Follow YAML best practices
4. **Document the Configuration**: Add comments and examples
5. **Test Thoroughly**: Ensure configuration loads and works correctly

### Configuration Guidelines

**YAML Style**:
- Use 2-space indentation
- Use lowercase with underscores for keys
- Include descriptive comments
- Group related settings together
- Maintain consistent formatting

**Documentation**:
- Comment complex configurations
- Provide examples for non-obvious settings
- Update this README when adding new config types
- Include validation instructions

## 📞 Configuration Support

For configuration-related issues:
1. **Check YAML syntax** using online validators
2. **Review this documentation** for guidance
3. **Test changes locally** before deploying
4. **Backup working configurations** before experimenting
5. **Check application logs** for configuration errors 
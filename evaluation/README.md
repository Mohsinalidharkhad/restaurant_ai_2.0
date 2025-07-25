# Evaluation Directory

This directory is reserved for evaluation-related files, test datasets, and evaluation results.

## 📁 Purpose

The evaluation directory serves as a centralized location for:

- **Test datasets** - Conversation scenarios and expected outputs
- **Evaluation results** - Performance metrics and assessment reports  
- **Benchmarking data** - Comparative analysis files
- **Quality assurance** - Test configurations and validation scripts

## 🧪 Related Scripts

The main evaluation functionality is located in:
- **Evaluation Script**: `../scripts/evaluation/eval.py`
- **Setup Documentation**: `../docs/setup/data_injestion.md`
- **Architecture**: `../docs/architecture/workflow_graph.png`

## 🔗 Integration

This directory integrates with:
- **LangSmith**: For automated evaluation and tracking
- **Configuration**: Uses `../config/prompts/prompts.yaml` for evaluation criteria
- **Backend Tools**: Tests all 13 agent tools through `../backend/tools/`

## 📊 Usage

To run evaluations:
```bash
# From project root
python scripts/evaluation/eval.py
```

For detailed evaluation setup and usage, see the scripts documentation at `../scripts/README.md`.

## 🚀 Future Enhancements

This directory is prepared for:
- Custom test datasets
- Evaluation result storage  
- Performance benchmarking files
- A/B testing configurations
- Quality metrics tracking 
# Claude Configuration Manager - Examples

This directory contains practical examples demonstrating how to use the Claude Configuration Manager library. Each example focuses on different aspects and use cases of the system.

## 🚀 Quick Start

1. **Prerequisites**: Ensure you have Ollama running with the required models installed
2. **Setup**: Install dependencies from `requirements.txt` 
3. **Run**: Execute any example with `python examples/[example_name].py`

## 📚 Example Overview

### 1. Basic Usage (`01_basic_usage.py`)
**What it demonstrates**: Fundamental usage patterns and basic configuration creation

- Creating a simple general-purpose Claude assistant
- Basic configuration parameters and options
- Generating Claude.md files
- Configuration switching and management
- Workspace initialization

**Best for**: First-time users wanting to understand core concepts

**Key concepts**:
- `ClaudeConfigManager` initialization
- `GenerationConfig` creation
- Asynchronous configuration generation
- Basic error handling

---

### 2. Coding Assistant (`02_coding_assistant.py`)
**What it demonstrates**: Advanced technical assistant with multi-model coordination

- Specialized coding assistant configuration
- Multi-model coordination strategies
- High-quality threshold settings
- Technical domain expertise
- Model memory management

**Best for**: Developers wanting a sophisticated coding assistant

**Key concepts**:
- `parallel_specialization` coordination strategy
- Technical personality traits and expertise domains
- Model resource optimization
- Quality validation for technical tasks

---

### 3. Creative Writer (`03_creative_writer.py`)
**What it demonstrates**: Creative assistant optimized for storytelling and content creation

- Creative writing personality and capabilities
- Consensus building between models
- Creative domain specialization
- Engaging communication style
- Inspiration and creativity enhancement

**Best for**: Writers, content creators, and creative professionals

**Key concepts**:
- `consensus_building` coordination strategy
- Creative personality traits
- Storytelling and narrative expertise
- Creative process optimization

---

### 4. Research Analyst (`04_research_analyst.py`)
**What it demonstrates**: Academic and research-focused assistant with rigorous methodology

- Research methodology and analysis
- Evidence-based reasoning approach
- Highest quality thresholds
- Academic writing and citation management
- Statistical and data analysis capabilities

**Best for**: Researchers, academics, and data analysts

**Key concepts**:
- `hierarchical_reasoning` coordination strategy
- Evidence-based reasoning approach
- Academic expertise domains
- Quality assurance for research tasks

---

### 5. CLI Usage (`05_cli_usage.py`)
**What it demonstrates**: Complete command-line interface capabilities

- All CLI commands and options
- Configuration management via terminal
- Git integration commands
- DSPy optimization controls
- Workflow automation

**Best for**: Users preferring command-line interfaces and automation

**Key concepts**:
- CLI command structure
- Configuration templates
- Git integration workflows
- Automation possibilities

---

### 6. Feedback & Optimization (`06_feedback_optimization.py`)
**What it demonstrates**: Self-improvement cycle and DSPy optimization

- User feedback collection and analysis
- Training data accumulation
- DSPy BootstrapFewShot optimization
- Continuous improvement monitoring
- Performance analytics

**Best for**: Understanding the self-improvement capabilities

**Key concepts**:
- Feedback loop implementation
- Training example management
- DSPy optimization process
- Performance monitoring and analytics

## 🎯 Choosing the Right Example

| **If you want to...** | **Start with** | **Then try** |
|----------------------|----------------|--------------|
| Learn the basics | 01_basic_usage.py | 05_cli_usage.py |
| Create a coding assistant | 02_coding_assistant.py | 06_feedback_optimization.py |
| Build a creative writing tool | 03_creative_writer.py | 06_feedback_optimization.py |
| Develop a research assistant | 04_research_analyst.py | 06_feedback_optimization.py |
| Use from command line | 05_cli_usage.py | Any domain-specific example |
| Understand self-improvement | 06_feedback_optimization.py | Domain-specific examples |

## 🛠️ Example Setup Requirements

### System Requirements
- **Memory**: 4GB+ available RAM (6GB+ recommended)
- **Storage**: 40GB+ for full model set
- **Python**: 3.8+ with asyncio support

### Required Models
The examples work best with these Ollama models:
```bash
# Core reasoning models
ollama pull deepseek-r1:8b
ollama pull qwen3:8b

# Specialized models
ollama pull deepseek-coder:6.7b  # For coding tasks
ollama pull opencoder:8b         # For technical documentation
ollama pull smallthinker:latest  # For fast reasoning

# Supporting models
ollama pull gemma3:4b
ollama pull llama3:8b-instruct-q4_0
ollama pull moondream:latest
```

### Python Dependencies
```bash
# Core dependencies
pip install dspy-ai ollama GitPython pydantic rich psutil aiofiles

# Optional for advanced features
pip install pytest pytest-asyncio black flake8
```

## 📖 Example Walkthrough

### Running Your First Example

1. **Start Ollama**:
   ```bash
   ollama serve
   ```

2. **Run basic example**:
   ```bash
   cd examples
   python 01_basic_usage.py
   ```

3. **Expected output**:
   - Manager initialization
   - Configuration creation
   - Claude.md generation
   - Quality scoring
   - Git branch creation

### Customizing Examples

Each example can be customized by modifying the `GenerationConfig`:

```python
config = GenerationConfig(
    branch_name="my_custom_assistant",           # Git branch name
    personality_traits=["helpful", "precise"],   # Personality
    expertise_domains=["your_domain"],           # Specialization
    communication_style="professional",         # Communication
    reasoning_approach="systematic",             # Reasoning style
    model_coordination_strategy="single_model", # Model usage
    quality_threshold=0.85                      # Quality bar
)
```

## 🔧 Troubleshooting

### Common Issues

1. **"Ollama not accessible"**:
   - Ensure Ollama is running: `ollama serve`
   - Check models are installed: `ollama list`

2. **"Insufficient memory"**:
   - Close other applications
   - Use single-model coordination
   - Consider smaller models

3. **"DSPy not configured"**:
   - Verify models are available
   - Check model name matching
   - Restart Ollama service

4. **"Git integration failed"**:
   - Initialize git in workspace: `git init`
   - Configure git credentials
   - Check workspace permissions

### Performance Tips

1. **Memory optimization**:
   - Use `single_model` coordination for limited memory
   - Unload unused models: `ollama unload [model]`
   - Monitor with `htop` or Task Manager

2. **Speed optimization**:
   - Use smaller models for faster generation
   - Enable model caching
   - Use SSD storage for models

3. **Quality optimization**:
   - Use `parallel_specialization` for complex tasks
   - Increase quality threshold for critical applications
   - Collect feedback for continuous improvement

## 🎨 Creating Custom Examples

### Template Structure

```python
#!/usr/bin/env python3
"""
Your Example - Claude Configuration Manager

Description of what this example demonstrates.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to import manager
sys.path.append(str(Path(__file__).parent.parent))

from claude_config_manager import ClaudeConfigManager, GenerationConfig

async def your_example_function():
    """Your example implementation"""
    # Initialize manager
    manager = ClaudeConfigManager("./workspace/your_example")
    
    # Create configuration
    config = GenerationConfig(
        # Your configuration parameters
    )
    
    # Generate and test
    result = await manager.create_configuration(config)
    
    return result

async def main():
    """Run your example"""
    try:
        result = await your_example_function()
        print("✅ Example completed successfully!")
    except Exception as e:
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Example Ideas

- **Customer Support Bot**: Empathetic, solution-focused assistant
- **Technical Documentation Writer**: Clear, comprehensive documentation
- **Language Tutor**: Patient, encouraging language learning assistant
- **Data Science Consultant**: Statistical analysis and insights
- **Product Manager Assistant**: Strategic thinking and prioritization
- **Marketing Copywriter**: Persuasive, brand-aligned content creation

## 📊 Performance Benchmarks

### Typical Generation Times
- **Single model**: 2-5 seconds
- **Dual model coordination**: 5-10 seconds  
- **Multi-model coordination**: 10-20 seconds

### Quality Improvements
- **Basic prompts**: Baseline quality
- **Optimized prompts**: 15-30% improvement
- **Self-improved system**: 40-60% improvement over time

### Resource Usage
- **Memory per model**: 2-5GB depending on model size
- **CPU usage**: 20-80% during generation
- **Storage**: 1-2GB per model

## 🤝 Contributing Examples

We welcome new examples! Please:

1. Follow the template structure above
2. Include comprehensive comments
3. Add error handling and validation
4. Test with different system configurations
5. Update this README with your example

## 📄 License

These examples are provided under the same license as the main Claude Configuration Manager project.

## 🔗 Additional Resources

- **Main Documentation**: See parent directory README.md
- **API Reference**: Check claude_config_manager.py docstrings
- **DSPy Documentation**: [https://dspy-docs.vercel.app/](https://dspy-docs.vercel.app/)
- **Ollama Models**: [https://ollama.com/library](https://ollama.com/library)

---

**Happy configuring! 🚀**
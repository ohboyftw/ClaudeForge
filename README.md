# Claude Configuration Manager

> AI-powered Claude configuration generation with DSPy + Ollama orchestration

## Overview

The Claude Configuration Manager is a sophisticated system that orchestrates multiple Ollama models through DSPy to generate intelligent Claude.md files and associated prompts. It provides git-based version management, performance evaluation through log analysis, and dynamic behavior switching capabilities.

## 🚀 Features

### Core Capabilities
- **🤖 Multi-Model Orchestration**: Coordinate multiple Ollama models for intelligent prompt generation
- **📝 Claude.md Generation**: Automatically generate comprehensive Claude configuration files
- **🔄 Git Version Control**: Track configuration changes with automatic branching and merging
- **📊 Performance Analysis**: Analyze Claude interaction logs to measure effectiveness
- **⚡ Resource Management**: Intelligent model loading/unloading based on system resources
- **🎯 Quality Optimization**: Continuous improvement through feedback loops

### Advanced Features
- **🎨 Multiple Behavioral Profiles**: Create specialized configurations for different use cases
- **📈 Real-time Monitoring**: Track resource usage and model performance
- **🔍 Configuration Validation**: Ensure generated configurations are valid and effective
- **💾 Backup & Restore**: Automatic backups with rollback capabilities
- **📤 Import/Export**: Share configurations across teams and systems
- **🎪 Rich CLI Interface**: Beautiful terminal interface with progress indicators

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Git (for version management)
- 16GB+ RAM recommended for full model set

### Quick Install
```bash
# Clone the repository
git clone https://github.com/example/claude-config-manager.git
cd claude-config-manager

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Ollama Models Setup
Install the recommended models for optimal performance:

```bash
# Core reasoning models
ollama pull deepseek-r1:8b
ollama pull qwen3:8b

# Specialized models
ollama pull deepseek-coder:6.7b
ollama pull opencoder:8b
ollama pull smallthinker:latest
ollama pull gemma3:4b

# Optional models
ollama pull llama3:8b-instruct-q4_0
ollama pull gurubot/phi3-mini-abliterated:latest
ollama pull mistral-openorca:7b-q4_K_M
ollama pull moondream:latest
```

## 🎯 Quick Start

### 1. Initialize Workspace
```bash
python claude_config_manager.py status --show-models
```

### 2. Create Your First Configuration
```bash
python claude_config_manager.py create \
  --name coding-assistant \
  --style technical \
  --domains software_development debugging \
  --traits analytical precise helpful \
  --reasoning systematic
```

### 3. Switch Between Configurations
```bash
# List available configurations
python claude_config_manager.py list --detailed

# Switch to a configuration
python claude_config_manager.py switch --name coding-assistant
```

### 4. Analyze Performance
```bash
python claude_config_manager.py analyze --export metrics.json
```

### 5. Optimize Based on Usage
```bash
python claude_config_manager.py optimize --name coding-assistant
```

## 📖 Usage Examples

### Creating Specialized Configurations

#### Coding Assistant
```bash
python claude_config_manager.py create \
  --name coding-assistant \
  --style technical \
  --domains software_development debugging code_review \
  --traits analytical precise systematic \
  --reasoning evidence_based \
  --quality-threshold 0.9
```

#### Creative Writer
```bash
python claude_config_manager.py create \
  --name creative-writer \
  --style engaging \
  --domains creative_writing storytelling content_creation \
  --traits creative imaginative expressive \
  --reasoning intuitive \
  --strategy parallel
```

#### Research Analyst
```bash
python claude_config_manager.py create \
  --name research-analyst \
  --style formal \
  --domains research data_analysis academic_writing \
  --traits thorough objective detail_oriented \
  --reasoning evidence_based \
  --quality-threshold 0.95
```

### Advanced Usage

#### Batch Operations
```bash
# Create multiple configurations from a config file
python claude_config_manager.py import --file configs/team-configs.yaml

# Backup all configurations
python claude_config_manager.py backup --compress

# Export configuration for sharing
python claude_config_manager.py export \
  --name coding-assistant \
  --output shared/coding-assistant.json \
  --format json
```

#### Performance Monitoring
```bash
# System status with resource monitoring
python claude_config_manager.py status --show-resources --show-models

# Detailed performance analysis
python claude_config_manager.py analyze \
  --config coding-assistant \
  --days 30 \
  --export reports/monthly-performance.yaml

# Validate configuration integrity
python claude_config_manager.py validate --name coding-assistant --fix
```

## 🏗️ Architecture

### Model Coordination Strategy

The system uses a three-tier model architecture:

#### Tier 1: Primary Orchestration
- **qwen3:8b**: Master orchestrator for high-level strategy
- **deepseek-r1:8b**: Advanced reasoning and logic generation

#### Tier 2: Specialized Processing
- **deepseek-coder:6.7b**: Code-focused prompt generation
- **opencoder:8b**: Technical documentation patterns

#### Tier 3: Supporting Models
- **smallthinker**: Fast optimization and refinement
- **gemma3:4b**: General purpose generation
- **moondream**: Vision and multimodal tasks

### Coordination Strategies

1. **Hierarchical**: Sequential processing with validation
2. **Parallel**: Concurrent specialized processing
3. **Consensus**: Multiple perspectives with agreement

## 📊 Performance Metrics

The system tracks comprehensive metrics:

- **Response Quality Score**: AI-generated content effectiveness
- **Task Completion Rate**: Successful configuration deployments
- **Error Rate**: Configuration and generation failures
- **Resource Efficiency**: CPU/Memory/GPU utilization
- **User Satisfaction**: Feedback-based quality assessment
- **Context Retention**: Configuration consistency over time

## 🔧 Configuration Options

### Generation Settings
```python
GenerationConfig(
    personality_traits=["analytical", "precise", "helpful"],
    expertise_domains=["software_development", "debugging"],
    communication_style="technical",  # professional, casual, technical, creative, formal, engaging
    reasoning_approach="systematic",   # analytical, systematic, intuitive, evidence_based, creative
    model_coordination_strategy="hierarchical",  # hierarchical, parallel, consensus
    quality_threshold=0.8,
    use_fast_mode=False,
    parallel_processing=True,
    auto_backup=True,
    max_generation_time=300
)
```

### Resource Management
- **CPU Limit**: Maximum CPU usage percentage
- **Memory Limit**: Maximum memory usage percentage
- **Model Caching**: Intelligent model loading/unloading
- **Timeout Protection**: Prevent hung generations

## 🚀 Advanced Features

### Git Integration
- Automatic branch creation for each configuration
- Commit history tracking with performance metrics
- Merge conflict resolution for collaborative configurations
- Tag-based release management

### Performance Optimization
- **Smart Caching**: Reuse generated content patterns
- **Load Balancing**: Distribute work across available models
- **Resource Monitoring**: Prevent system overload
- **Quality Thresholds**: Ensure output meets standards

### Extensibility
- **Plugin Architecture**: Add custom model coordinators
- **Template System**: Create reusable prompt patterns
- **API Integration**: Connect with external systems
- **Custom Metrics**: Define domain-specific quality measures

## 🔍 Troubleshooting

### Common Issues

#### Model Not Available
```bash
# Check model availability
python claude_config_manager.py status --show-models

# Install missing models
ollama pull <model-name>
```

#### Resource Limitations
```bash
# Check system resources
python claude_config_manager.py status --show-resources

# Use fast mode for limited resources
python claude_config_manager.py create --name test --fast-mode
```

#### Generation Timeouts
```bash
# Increase timeout for complex configurations
python claude_config_manager.py create --name complex --timeout 600

# Use simpler coordination strategy
python claude_config_manager.py create --name simple --strategy hierarchical
```

### Performance Tuning

#### For High-Performance Systems (32GB+ RAM)
```bash
# Enable all models with parallel processing
python claude_config_manager.py create \
  --name high-performance \
  --strategy consensus \
  --quality-threshold 0.95 \
  --parallel-processing
```

#### For Resource-Constrained Systems (16GB RAM)
```bash
# Use fast mode with minimal models
python claude_config_manager.py create \
  --name lightweight \
  --fast-mode \
  --strategy hierarchical \
  --timeout 120
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black claude_config_manager.py
flake8 claude_config_manager.py

# Type checking
mypy claude_config_manager.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) for declarative AI programming
- [Ollama](https://ollama.ai/) for local model serving
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- The Claude AI team for inspiration and guidance

## 📞 Support

- **Documentation**: [Read the Docs](https://claude-config-manager.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/example/claude-config-manager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/example/claude-config-manager/discussions)
- **Email**: support@example.com

---

*Built with ❤️ for the Claude AI community*
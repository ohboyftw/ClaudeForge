#!/usr/bin/env python3
"""
Demonstration script for Claude Configuration Manager

This script shows how to use the DSPy + Ollama system to generate
and manage Claude.md configurations with different behavioral profiles.
"""

import asyncio
import json
from claude_config_manager import (
    ClaudeConfigManager, 
    GenerationConfig, 
    PerformanceMetrics
)

async def demo_basic_usage():
    """Demonstrate basic usage of the system"""
    print("🚀 Claude Configuration Manager Demo")
    print("=" * 50)
    
    # Initialize the manager
    manager = ClaudeConfigManager("./demo_workspace")
    
    # Create different configuration profiles
    configurations = [
        GenerationConfig(
            branch_name="coding_assistant",
            personality_traits=["analytical", "precise", "helpful"],
            expertise_domains=["software_development", "debugging", "code_review"],
            communication_style="technical",
            reasoning_approach="systematic",
            commit_message="Create coding assistant configuration"
        ),
        GenerationConfig(
            branch_name="creative_writer",
            personality_traits=["creative", "imaginative", "expressive"],
            expertise_domains=["creative_writing", "storytelling", "content_creation"],
            communication_style="engaging",
            reasoning_approach="intuitive",
            commit_message="Create creative writer configuration"
        ),
        GenerationConfig(
            branch_name="research_analyst",
            personality_traits=["thorough", "objective", "detail_oriented"],
            expertise_domains=["research", "data_analysis", "academic_writing"],
            communication_style="formal",
            reasoning_approach="evidence_based",
            commit_message="Create research analyst configuration"
        )
    ]
    
    print("\n📝 Creating configurations...")
    
    for config in configurations:
        try:
            print(f"\nCreating '{config.branch_name}' configuration...")
            content = await manager.create_configuration(config)
            print(f"✅ Successfully created '{config.branch_name}'")
            
            # Show a preview of the generated content
            lines = content.split('\n')
            preview = '\n'.join(lines[:10]) + '\n...'
            print(f"Preview:\n{preview}")
            
        except Exception as e:
            print(f"❌ Error creating '{config.branch_name}': {e}")
    
    print("\n📋 Available configurations:")
    configs = manager.list_configurations()
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}")
    
    return manager

async def demo_configuration_switching(manager):
    """Demonstrate configuration switching"""
    print("\n🔄 Configuration Switching Demo")
    print("=" * 50)
    
    configs = manager.list_configurations()
    
    for config in configs[:2]:  # Switch between first two configs
        print(f"\n🔀 Switching to '{config}' configuration...")
        try:
            await manager.switch_configuration(config)
            print(f"✅ Successfully switched to '{config}'")
            
            # Read the current Claude.md file
            claude_md_path = manager.workspace_path / "Claude.md"
            if claude_md_path.exists():
                with open(claude_md_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    preview = '\n'.join(lines[:5])
                    print(f"Current configuration preview:\n{preview}...")
            
        except Exception as e:
            print(f"❌ Error switching to '{config}': {e}")

async def demo_performance_analysis(manager):
    """Demonstrate performance analysis"""
    print("\n📊 Performance Analysis Demo")
    print("=" * 50)
    
    try:
        print("Analyzing performance metrics...")
        metrics = await manager.analyze_performance()
        
        print("\n📈 Performance Metrics:")
        print(f"  • Response Quality Score: {metrics.response_quality_score:.2f}")
        print(f"  • Task Completion Rate: {metrics.task_completion_rate:.2f}")
        print(f"  • Error Rate: {metrics.error_rate:.2f}")
        print(f"  • Average Response Time: {metrics.response_time:.2f}s")
        print(f"  • User Satisfaction: {metrics.user_satisfaction:.2f}")
        print(f"  • Context Retention: {metrics.context_retention:.2f}")
        
        if metrics.improvement_suggestions:
            print("\n💡 Improvement Suggestions:")
            for suggestion in metrics.improvement_suggestions:
                print(f"  • {suggestion}")
        
    except Exception as e:
        print(f"❌ Error analyzing performance: {e}")

async def demo_optimization(manager):
    """Demonstrate configuration optimization"""
    print("\n🎯 Configuration Optimization Demo")
    print("=" * 50)
    
    configs = manager.list_configurations()
    if not configs:
        print("No configurations available for optimization")
        return
    
    config_name = configs[0]  # Optimize first configuration
    
    try:
        print(f"Optimizing '{config_name}' configuration...")
        suggestions = await manager.optimize_configuration(config_name)
        
        print(f"\n🔧 Optimization Suggestions for '{config_name}':")
        print("=" * 40)
        print(suggestions)
        
    except Exception as e:
        print(f"❌ Error optimizing configuration: {e}")

async def demo_model_coordination():
    """Demonstrate model coordination capabilities"""
    print("\n🤖 Model Coordination Demo")
    print("=" * 50)
    
    from claude_config_manager import OllamaModelManager
    
    model_manager = OllamaModelManager()
    
    print("Available model categories:")
    for category, models in model_manager.models.items():
        print(f"  • {category}: {', '.join(models)}")
    
    print("\nChecking model availability...")
    for category, models in model_manager.models.items():
        print(f"\n{category.upper()} models:")
        for model in models:
            try:
                available = await model_manager.check_model_availability(model)
                status = "✅ Available" if available else "❌ Not available"
                print(f"  • {model}: {status}")
            except Exception as e:
                print(f"  • {model}: ❌ Error checking - {e}")

async def demo_git_integration(manager):
    """Demonstrate git integration features"""
    print("\n📚 Git Integration Demo")
    print("=" * 50)
    
    print("Git repository information:")
    print(f"  • Repository path: {manager.git_manager.repo_path}")
    print(f"  • Current branch: {manager.git_manager.repo.active_branch.name}")
    
    print("\nBranch history:")
    try:
        for branch in manager.git_manager.repo.heads:
            print(f"  • {branch.name}")
            
        print("\nCommit history (last 5):")
        commits = list(manager.git_manager.repo.iter_commits(max_count=5))
        for commit in commits:
            print(f"  • {commit.hexsha[:8]}: {commit.message.strip()}")
            
    except Exception as e:
        print(f"❌ Error accessing git information: {e}")

async def main():
    """Run the complete demonstration"""
    print("🎭 Claude Configuration Manager - Complete Demo")
    print("=" * 60)
    
    try:
        # Basic usage demo
        manager = await demo_basic_usage()
        
        # Model coordination demo
        await demo_model_coordination()
        
        # Configuration switching demo
        await demo_configuration_switching(manager)
        
        # Performance analysis demo
        await demo_performance_analysis(manager)
        
        # Optimization demo
        await demo_optimization(manager)
        
        # Git integration demo
        await demo_git_integration(manager)
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install and configure Ollama models")
        print("2. Customize configurations for your use cases")
        print("3. Integrate with your Claude Desktop/Code setup")
        print("4. Monitor performance and iterate")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())


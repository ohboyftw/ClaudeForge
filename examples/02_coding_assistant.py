#!/usr/bin/env python3
"""
Coding Assistant Example - Claude Configuration Manager

This example creates a specialized coding assistant configuration optimized for
software development tasks, with advanced model coordination and quality validation.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import the manager
sys.path.append(str(Path(__file__).parent.parent))

from claude_config_manager import ClaudeConfigManager, GenerationConfig

async def create_coding_assistant():
    """Create a specialized coding assistant configuration"""
    print("💻 Coding Assistant Example - Claude Configuration Manager")
    print("=" * 65)
    
    workspace_path = "./workspace/coding_assistant"
    manager = ClaudeConfigManager(workspace_path)
    
    print(f"✅ Initialized Claude Config Manager for coding assistant")
    
    # Create advanced coding assistant configuration
    config = GenerationConfig(
        branch_name="coding_assistant_v1",
        personality_traits=[
            "analytical", 
            "precise", 
            "systematic", 
            "detail_oriented",
            "patient"
        ],
        expertise_domains=[
            "software_development",
            "code_review", 
            "debugging",
            "system_architecture",
            "performance_optimization",
            "testing_strategies",
            "documentation"
        ],
        communication_style="technical",
        reasoning_approach="systematic",
        model_coordination_strategy="parallel_specialization",  # Use multiple models
        quality_threshold=0.9,  # High quality threshold for coding tasks
        commit_message="Add specialized coding assistant configuration v1"
    )
    
    print(f"\n📋 Advanced Coding Assistant Configuration:")
    print(f"   🎯 Specialization: Software Development & Code Review")
    print(f"   🧠 Multi-model coordination: {config.model_coordination_strategy}")
    print(f"   📊 Quality threshold: {config.quality_threshold}")
    print(f"   🔧 Expertise areas: {len(config.expertise_domains)} domains")
    
    try:
        print(f"\n🔄 Generating coding assistant configuration...")
        print(f"   (This may take longer due to multi-model coordination)")
        
        result = await manager.create_configuration(config)
        
        print(f"✅ Advanced coding assistant configuration generated!")
        print(f"📄 Location: {result.get('claude_md_path')}")
        print(f"🔗 Git branch: {result.get('branch_name')}")
        print(f"📊 Quality score: {result.get('quality_score', 'N/A')}")
        
        # Show key sections of the generated configuration
        if 'claude_md_content' in result:
            content = result['claude_md_content']
            
            # Extract and show key sections
            sections = extract_key_sections(content)
            print(f"\n📖 Key Configuration Sections Generated:")
            print("=" * 50)
            
            for section_name, section_content in sections.items():
                print(f"\n{section_name}:")
                print("─" * 30)
                preview = section_content[:300] + "..." if len(section_content) > 300 else section_content
                print(preview)
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating coding assistant: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_key_sections(content):
    """Extract key sections from generated Claude.md content"""
    sections = {}
    
    # Common section patterns
    section_patterns = [
        "## System Instructions",
        "## Coding Guidelines", 
        "## Code Review Process",
        "## Debugging Approach",
        "## Best Practices",
        "## Examples"
    ]
    
    lines = content.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if this line starts a new section
        is_section_header = any(line.startswith(pattern) for pattern in section_patterns)
        
        if is_section_header:
            # Save previous section if it exists
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.strip()
            current_content = []
        elif current_section:
            current_content.append(line)
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

async def demonstrate_model_coordination():
    """Demonstrate the multi-model coordination capabilities"""
    print(f"\n🤖 Demonstrating Multi-Model Coordination")
    print("=" * 45)
    
    workspace_path = "./workspace/coding_assistant"
    manager = ClaudeConfigManager(workspace_path)
    
    try:
        # Get available models for coordination
        model_info = manager.model_manager.get_model_info()
        
        print(f"🔧 Available Models for Coding Tasks:")
        for category, models in model_info.items():
            if category in ['coding', 'reasoning', 'orchestration']:
                print(f"   📂 {category.title()}: {', '.join(models)}")
        
        # Show model memory usage
        print(f"\n💾 Model Memory Requirements:")
        memory_usage = manager.model_manager.model_memory_usage
        coding_models = model_info.get('coding', []) + model_info.get('reasoning', [])
        
        total_memory = 0
        for model in coding_models[:3]:  # Show top 3 models
            if model in memory_usage:
                mem = memory_usage[model]
                print(f"   🧠 {model}: {mem}GB")
                total_memory += mem
        
        print(f"   📊 Total for coordination: ~{total_memory:.1f}GB")
        
        # Check if system can handle multi-model coordination
        resources = manager.model_manager.resource_monitor.get_system_resources()
        available_memory = resources.get('memory_available_gb', 0)
        
        if available_memory >= total_memory:
            print(f"   ✅ System can handle multi-model coordination ({available_memory:.1f}GB available)")
        else:
            print(f"   ⚠️  May need memory optimization ({available_memory:.1f}GB available)")
            
    except Exception as e:
        print(f"❌ Error demonstrating coordination: {e}")

async def test_coding_assistant():
    """Test the generated coding assistant configuration"""
    print(f"\n🧪 Testing Coding Assistant Configuration")
    print("=" * 40)
    
    workspace_path = "./workspace/coding_assistant"
    manager = ClaudeConfigManager(workspace_path)
    
    try:
        # Switch to the coding assistant configuration
        await manager.switch_configuration("coding_assistant_v1")
        print(f"✅ Switched to coding assistant configuration")
        
        # Simulate getting configuration info (would be used by Claude)
        config_info = manager.get_current_configuration()
        if config_info:
            print(f"📋 Active configuration: {config_info.get('name', 'Unknown')}")
            print(f"🔗 Branch: {config_info.get('branch', 'Unknown')}")
        
        print(f"\n💡 Usage Instructions:")
        print(f"   1. Copy the generated Claude.md to your Claude settings")
        print(f"   2. Test with coding tasks like:")
        print(f"      • Code review requests")
        print(f"      • Debugging assistance") 
        print(f"      • Architecture discussions")
        print(f"      • Performance optimization")
        print(f"   3. Provide feedback to improve the assistant")
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")

async def main():
    """Run the coding assistant example"""
    try:
        # Create the coding assistant
        result = await create_coding_assistant()
        
        if result:
            # Demonstrate multi-model coordination
            await demonstrate_model_coordination()
            
            # Test the configuration
            await test_coding_assistant()
            
            print(f"\n🎉 Coding Assistant Example Complete!")
            print(f"🚀 Your specialized coding assistant is ready to use!")
        else:
            print(f"\n❌ Coding assistant example failed")
            
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Basic Usage Example - Claude Configuration Manager

This example demonstrates the fundamental usage of the Claude Configuration Manager
to create a simple general-purpose Claude assistant configuration.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import the manager
sys.path.append(str(Path(__file__).parent.parent))

from claude_config_manager import ClaudeConfigManager, GenerationConfig

async def basic_usage_example():
    """Demonstrate basic usage of the Claude Configuration Manager"""
    print("🚀 Basic Usage Example - Claude Configuration Manager")
    print("=" * 60)
    
    # Initialize the manager with a workspace directory
    workspace_path = "./workspace/basic_example"
    manager = ClaudeConfigManager(workspace_path)
    
    print(f"✅ Initialized Claude Config Manager")
    print(f"📁 Workspace: {workspace_path}")
    
    # Create a basic configuration
    config = GenerationConfig(
        branch_name="general_assistant",
        personality_traits=["helpful", "friendly", "knowledgeable"],
        expertise_domains=["general_knowledge", "problem_solving", "communication"],
        communication_style="conversational",
        reasoning_approach="balanced",
        model_coordination_strategy="single_model",
        quality_threshold=0.8
    )
    
    print(f"\n📋 Configuration Created:")
    print(f"   🌟 Name: {config.branch_name}")
    print(f"   🎭 Personality: {', '.join(config.personality_traits)}")
    print(f"   🧠 Expertise: {', '.join(config.expertise_domains)}")
    print(f"   💬 Style: {config.communication_style}")
    
    try:
        # Generate the Claude.md configuration
        print(f"\n🔄 Generating Claude.md configuration...")
        
        result = await manager.create_configuration(config)
        
        print(f"✅ Configuration generated successfully!")
        print(f"📄 Claude.md location: {result.get('claude_md_path')}")
        print(f"🔗 Git branch: {result.get('branch_name')}")
        print(f"📊 Quality score: {result.get('quality_score', 'N/A')}")
        
        # Show a preview of the generated content
        if 'claude_md_content' in result:
            content_preview = result['claude_md_content'][:500] + "..." if len(result['claude_md_content']) > 500 else result['claude_md_content']
            print(f"\n📖 Generated Content Preview:")
            print("─" * 40)
            print(content_preview)
            print("─" * 40)
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating configuration: {e}")
        return None

async def demonstrate_switching():
    """Demonstrate configuration switching"""
    print(f"\n🔄 Demonstrating Configuration Switching")
    print("=" * 40)
    
    workspace_path = "./workspace/basic_example"
    manager = ClaudeConfigManager(workspace_path)
    
    try:
        # Switch to the configuration we just created
        await manager.switch_configuration("general_assistant")
        print(f"✅ Switched to 'general_assistant' configuration")
        
        # Get current configuration info
        current_config = manager.get_current_configuration()
        if current_config:
            print(f"📋 Current configuration: {current_config}")
        
    except Exception as e:
        print(f"❌ Error switching configuration: {e}")

async def main():
    """Run the basic usage example"""
    try:
        # Run basic usage
        result = await basic_usage_example()
        
        if result:
            # Demonstrate switching
            await demonstrate_switching()
            
            print(f"\n🎉 Basic Usage Example Complete!")
            print(f"💡 Next steps:")
            print(f"   • Copy the generated Claude.md to your Claude settings")
            print(f"   • Test the configuration with Claude")
            print(f"   • Provide feedback to improve future generations")
        else:
            print(f"\n❌ Basic example failed - check your Ollama setup")
            
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
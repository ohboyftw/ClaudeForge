#!/usr/bin/env python3
"""
Creative Writer Example - Claude Configuration Manager

This example creates a creative writing assistant configuration that emphasizes
creativity, storytelling, and engaging content generation.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import the manager
sys.path.append(str(Path(__file__).parent.parent))

from claude_config_manager import ClaudeConfigManager, GenerationConfig

async def create_creative_writer():
    """Create a specialized creative writing assistant"""
    print("✨ Creative Writer Example - Claude Configuration Manager")
    print("=" * 65)
    
    workspace_path = "./workspace/creative_writer"
    manager = ClaudeConfigManager(workspace_path)
    
    print(f"✅ Initialized Claude Config Manager for creative writing")
    
    # Create creative writer configuration
    config = GenerationConfig(
        branch_name="creative_writer_v1",
        personality_traits=[
            "creative",
            "imaginative", 
            "expressive",
            "empathetic",
            "curious",
            "inspiring"
        ],
        expertise_domains=[
            "creative_writing",
            "storytelling",
            "character_development",
            "world_building",
            "dialogue_writing",
            "poetry",
            "content_creation",
            "narrative_structure"
        ],
        communication_style="engaging",
        reasoning_approach="intuitive",
        model_coordination_strategy="consensus_building",  # Use multiple models for creativity
        quality_threshold=0.85,
        commit_message="Add creative writer configuration with enhanced storytelling capabilities"
    )
    
    print(f"\n📋 Creative Writer Configuration:")
    print(f"   🎨 Focus: Creative Writing & Storytelling")
    print(f"   🌟 Personality: {', '.join(config.personality_traits[:3])}...")
    print(f"   📚 Specializations: {len(config.expertise_domains)} areas")
    print(f"   🤝 Coordination: {config.model_coordination_strategy}")
    
    try:
        print(f"\n🔄 Generating creative writer configuration...")
        print(f"   (Using consensus building for enhanced creativity)")
        
        result = await manager.create_configuration(config)
        
        print(f"✅ Creative writer configuration generated!")
        print(f"📄 Location: {result.get('claude_md_path')}")
        print(f"🎭 Git branch: {result.get('branch_name')}")
        print(f"📊 Quality score: {result.get('quality_score', 'N/A')}")
        
        # Show creative writing specific sections
        if 'claude_md_content' in result:
            content = result['claude_md_content']
            creative_sections = extract_creative_sections(content)
            
            print(f"\n📖 Creative Writing Features Generated:")
            print("=" * 50)
            
            for section_name, section_content in creative_sections.items():
                print(f"\n{section_name}:")
                print("─" * 30)
                preview = section_content[:250] + "..." if len(section_content) > 250 else section_content
                print(preview)
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating creative writer: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_creative_sections(content):
    """Extract creative writing specific sections"""
    sections = {}
    
    # Creative writing section patterns
    creative_patterns = [
        "## Writing Style",
        "## Storytelling Approach", 
        "## Character Development",
        "## Creative Process",
        "## Inspiration Sources",
        "## Writing Techniques",
        "## Examples"
    ]
    
    lines = content.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if this line starts a creative section
        is_creative_section = any(line.startswith(pattern) for pattern in creative_patterns)
        
        if is_creative_section:
            # Save previous section
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.strip()
            current_content = []
        elif current_section:
            current_content.append(line)
        elif "creative" in line.lower() or "story" in line.lower() or "writing" in line.lower():
            # Capture any creative writing related content
            if "🎨 Creative Features" not in sections:
                sections["🎨 Creative Features"] = ""
            sections["🎨 Creative Features"] += line + "\n"
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

async def demonstrate_creative_capabilities():
    """Demonstrate the creative writing capabilities"""
    print(f"\n🎨 Creative Writing Capabilities")
    print("=" * 35)
    
    workspace_path = "./workspace/creative_writer"
    manager = ClaudeConfigManager(workspace_path)
    
    try:
        # Show which models are good for creative tasks
        model_info = manager.model_manager.get_model_info()
        
        print(f"🤖 Models Optimized for Creative Tasks:")
        
        # Models particularly good for creative writing
        creative_models = []
        if 'general' in model_info:
            creative_models.extend(model_info['general'])
        if 'reasoning' in model_info:
            creative_models.extend(model_info['reasoning'][:2])  # Top 2 reasoning models
            
        for model in creative_models[:4]:  # Show top 4 models for creativity
            print(f"   ✨ {model}")
        
        print(f"\n🎭 Creative Writing Use Cases:")
        use_cases = [
            "📖 Short story generation",
            "🎬 Screenplay and dialogue writing", 
            "🏰 Fantasy world building",
            "👥 Character development and backstories",
            "📝 Poetry and creative prose",
            "🎪 Interactive storytelling",
            "📚 Novel chapter outlines",
            "✍️  Creative writing prompts and exercises"
        ]
        
        for use_case in use_cases:
            print(f"   {use_case}")
            
        print(f"\n💡 Creative Enhancement Features:")
        features = [
            "🌈 Multi-perspective storytelling",
            "🎯 Genre-specific writing styles", 
            "🔄 Iterative creative refinement",
            "🎨 Mood and tone adaptation",
            "🌟 Originality and uniqueness focus"
        ]
        
        for feature in features:
            print(f"   {feature}")
            
    except Exception as e:
        print(f"❌ Error demonstrating capabilities: {e}")

async def creative_writing_examples():
    """Show example creative writing prompts"""
    print(f"\n📚 Example Creative Writing Prompts")
    print("=" * 40)
    
    examples = [
        {
            "type": "🏗️ World Building",
            "prompt": "Create a unique fantasy world with its own magic system, geography, and cultures",
            "expected": "Detailed world with consistent rules and rich descriptions"
        },
        {
            "type": "👤 Character Development", 
            "prompt": "Develop a complex protagonist with flaws, motivations, and a compelling backstory",
            "expected": "Multi-dimensional character with clear personality and growth arc"
        },
        {
            "type": "📖 Story Structure",
            "prompt": "Outline a mystery novel with red herrings, clues, and a satisfying resolution",
            "expected": "Well-paced plot with proper setup, investigation, and revelation"
        },
        {
            "type": "✍️ Style Adaptation",
            "prompt": "Write the same scene in the styles of different famous authors",
            "expected": "Distinct voice and techniques matching each author's style"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['type']}")
        print(f"   Prompt: {example['prompt']}")
        print(f"   Expected: {example['expected']}")

async def test_creative_writer():
    """Test the creative writer configuration"""
    print(f"\n🧪 Testing Creative Writer Configuration")
    print("=" * 40)
    
    workspace_path = "./workspace/creative_writer"
    manager = ClaudeConfigManager(workspace_path)
    
    try:
        # Switch to creative writer configuration
        await manager.switch_configuration("creative_writer_v1")
        print(f"✅ Switched to creative writer configuration")
        
        # Show configuration details
        config_info = manager.get_current_configuration()
        if config_info:
            print(f"🎨 Active configuration: {config_info.get('name', 'Unknown')}")
            
        print(f"\n🎭 Ready for Creative Writing Tasks!")
        print(f"   • Character development and backstories")
        print(f"   • Plot outlines and story structures") 
        print(f"   • Dialogue and scene writing")
        print(f"   • World building and setting creation")
        print(f"   • Poetry and creative prose")
        print(f"   • Genre-specific writing assistance")
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")

async def main():
    """Run the creative writer example"""
    try:
        # Create the creative writer
        result = await create_creative_writer()
        
        if result:
            # Demonstrate creative capabilities
            await demonstrate_creative_capabilities()
            
            # Show example prompts
            await creative_writing_examples()
            
            # Test the configuration
            await test_creative_writer()
            
            print(f"\n🎉 Creative Writer Example Complete!")
            print(f"✨ Your creative writing assistant is ready to inspire!")
        else:
            print(f"\n❌ Creative writer example failed")
            
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
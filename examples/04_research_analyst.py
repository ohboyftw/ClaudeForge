#!/usr/bin/env python3
"""
Research Analyst Example - Claude Configuration Manager

This example creates a research-focused assistant configuration optimized for
academic research, data analysis, and evidence-based reasoning.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import the manager
sys.path.append(str(Path(__file__).parent.parent))

from claude_config_manager import ClaudeConfigManager, GenerationConfig

async def create_research_analyst():
    """Create a specialized research analyst configuration"""
    print("🔬 Research Analyst Example - Claude Configuration Manager")
    print("=" * 65)
    
    workspace_path = "./workspace/research_analyst"
    manager = ClaudeConfigManager(workspace_path)
    
    print(f"✅ Initialized Claude Config Manager for research analysis")
    
    # Create research analyst configuration
    config = GenerationConfig(
        branch_name="research_analyst_v1",
        personality_traits=[
            "thorough",
            "objective", 
            "analytical",
            "detail_oriented",
            "methodical",
            "critical_thinking"
        ],
        expertise_domains=[
            "academic_research",
            "data_analysis",
            "statistical_reasoning",
            "literature_review",
            "hypothesis_testing",
            "methodology_design",
            "citation_management",
            "evidence_evaluation",
            "scientific_writing"
        ],
        communication_style="formal",
        reasoning_approach="evidence_based",
        model_coordination_strategy="hierarchical_reasoning",  # Deep analytical reasoning
        quality_threshold=0.95,  # Highest quality for research accuracy
        commit_message="Add research analyst configuration with rigorous analytical capabilities"
    )
    
    print(f"\n📋 Research Analyst Configuration:")
    print(f"   🔬 Focus: Academic Research & Data Analysis")
    print(f"   📊 Quality threshold: {config.quality_threshold} (highest precision)")
    print(f"   🧠 Reasoning: {config.reasoning_approach}")
    print(f"   📚 Expertise areas: {len(config.expertise_domains)} domains")
    
    try:
        print(f"\n🔄 Generating research analyst configuration...")
        print(f"   (Using hierarchical reasoning for deep analysis)")
        
        result = await manager.create_configuration(config)
        
        print(f"✅ Research analyst configuration generated!")
        print(f"📄 Location: {result.get('claude_md_path')}")
        print(f"🔗 Git branch: {result.get('branch_name')}")
        print(f"📊 Quality score: {result.get('quality_score', 'N/A')}")
        
        # Show research-specific sections
        if 'claude_md_content' in result:
            content = result['claude_md_content']
            research_sections = extract_research_sections(content)
            
            print(f"\n📖 Research Analysis Features Generated:")
            print("=" * 50)
            
            for section_name, section_content in research_sections.items():
                print(f"\n{section_name}:")
                print("─" * 30)
                preview = section_content[:280] + "..." if len(section_content) > 280 else section_content
                print(preview)
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating research analyst: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_research_sections(content):
    """Extract research-specific sections from generated content"""
    sections = {}
    
    # Research-specific section patterns
    research_patterns = [
        "## Research Methodology",
        "## Data Analysis Approach",
        "## Evidence Evaluation", 
        "## Citation Standards",
        "## Statistical Analysis",
        "## Literature Review Process",
        "## Hypothesis Testing",
        "## Quality Assurance"
    ]
    
    lines = content.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        # Check if this line starts a research section
        is_research_section = any(line.startswith(pattern) for pattern in research_patterns)
        
        if is_research_section:
            # Save previous section
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line.strip()
            current_content = []
        elif current_section:
            current_content.append(line)
        elif any(keyword in line.lower() for keyword in ['research', 'analysis', 'evidence', 'methodology']):
            # Capture research-related content
            if "🔬 Research Features" not in sections:
                sections["🔬 Research Features"] = ""
            sections["🔬 Research Features"] += line + "\n"
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections

async def demonstrate_research_capabilities():
    """Demonstrate research analysis capabilities"""
    print(f"\n🔬 Research Analysis Capabilities")
    print("=" * 35)
    
    workspace_path = "./workspace/research_analyst"
    manager = ClaudeConfigManager(workspace_path)
    
    try:
        # Show models optimized for analytical reasoning
        model_info = manager.model_manager.get_model_info()
        
        print(f"🧠 Models Optimized for Research Analysis:")
        
        # Models good for research and reasoning
        research_models = []
        if 'reasoning' in model_info:
            research_models.extend(model_info['reasoning'])
        if 'orchestration' in model_info:
            research_models.extend(model_info['orchestration'][:2])
            
        for model in research_models[:4]:
            print(f"   🔍 {model}")
        
        print(f"\n📊 Research Analysis Specializations:")
        specializations = [
            "📚 Literature review and synthesis",
            "📈 Statistical analysis and interpretation",
            "🔬 Experimental design and methodology",
            "📋 Systematic review protocols",
            "📊 Data visualization and reporting",
            "🔍 Source verification and fact-checking",
            "📝 Academic writing and citations",
            "🎯 Hypothesis formulation and testing"
        ]
        
        for spec in specializations:
            print(f"   {spec}")
            
        print(f"\n🎯 Quality Assurance Features:")
        qa_features = [
            "✅ Multi-source verification",
            "📏 Statistical significance testing",
            "🔄 Peer review simulation",
            "📊 Confidence interval analysis",
            "🎯 Bias detection and mitigation",
            "📚 Citation accuracy checking"
        ]
        
        for feature in qa_features:
            print(f"   {feature}")
            
    except Exception as e:
        print(f"❌ Error demonstrating capabilities: {e}")

async def research_methodology_examples():
    """Show example research methodologies and approaches"""
    print(f"\n📋 Research Methodology Examples")
    print("=" * 35)
    
    methodologies = [
        {
            "type": "📊 Quantitative Analysis",
            "description": "Statistical analysis of numerical data with hypothesis testing",
            "use_cases": ["Survey data analysis", "Experimental results", "A/B testing"]
        },
        {
            "type": "🔍 Qualitative Research",
            "description": "Thematic analysis of non-numerical data and observations",
            "use_cases": ["Interview analysis", "Case studies", "Content analysis"]
        },
        {
            "type": "📚 Systematic Review",
            "description": "Comprehensive literature review with systematic methodology",
            "use_cases": ["Meta-analysis", "Evidence synthesis", "Research gaps identification"]
        },
        {
            "type": "🧪 Experimental Design",
            "description": "Controlled experiments with proper variable management",
            "use_cases": ["Causal inference", "Treatment effects", "Randomized trials"]
        }
    ]
    
    for i, method in enumerate(methodologies, 1):
        print(f"\n{i}. {method['type']}")
        print(f"   Description: {method['description']}")
        print(f"   Use cases: {', '.join(method['use_cases'])}")

async def demonstrate_analytical_reasoning():
    """Demonstrate the analytical reasoning process"""
    print(f"\n🧠 Analytical Reasoning Process")
    print("=" * 35)
    
    reasoning_steps = [
        "1️⃣ Problem Definition - Clearly define research questions and objectives",
        "2️⃣ Literature Review - Systematic review of existing research and evidence", 
        "3️⃣ Methodology Selection - Choose appropriate research methods and tools",
        "4️⃣ Data Collection - Gather relevant and reliable data sources",
        "5️⃣ Analysis Execution - Apply statistical and analytical techniques",
        "6️⃣ Results Interpretation - Draw meaningful conclusions from findings",
        "7️⃣ Validation - Cross-check results and assess reliability",
        "8️⃣ Reporting - Present findings with proper documentation"
    ]
    
    print(f"🔄 Hierarchical Reasoning Workflow:")
    for step in reasoning_steps:
        print(f"   {step}")
    
    print(f"\n📈 Quality Metrics Evaluated:")
    quality_metrics = [
        "🎯 Accuracy - Correctness of analysis and conclusions",
        "📊 Reliability - Consistency and reproducibility", 
        "✅ Validity - Appropriateness of methods and interpretations",
        "🔍 Completeness - Thoroughness of investigation",
        "📚 Citations - Proper attribution and references"
    ]
    
    for metric in quality_metrics:
        print(f"   {metric}")

async def test_research_analyst():
    """Test the research analyst configuration"""
    print(f"\n🧪 Testing Research Analyst Configuration")
    print("=" * 40)
    
    workspace_path = "./workspace/research_analyst"
    manager = ClaudeConfigManager(workspace_path)
    
    try:
        # Switch to research analyst configuration
        await manager.switch_configuration("research_analyst_v1")
        print(f"✅ Switched to research analyst configuration")
        
        # Show configuration details
        config_info = manager.get_current_configuration()
        if config_info:
            print(f"🔬 Active configuration: {config_info.get('name', 'Unknown')}")
            
        print(f"\n📊 Ready for Research Analysis Tasks!")
        print(f"   • Literature reviews and meta-analyses")
        print(f"   • Statistical data analysis and interpretation") 
        print(f"   • Research methodology design")
        print(f"   • Evidence evaluation and synthesis")
        print(f"   • Academic writing and citation management")
        print(f"   • Hypothesis testing and validation")
        
        print(f"\n💡 Pro Tips for Research Tasks:")
        print(f"   • Always request source verification")
        print(f"   • Ask for confidence intervals on statistics")
        print(f"   • Request methodology justification")
        print(f"   • Verify citations and references")
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")

async def main():
    """Run the research analyst example"""
    try:
        # Create the research analyst
        result = await create_research_analyst()
        
        if result:
            # Demonstrate research capabilities
            await demonstrate_research_capabilities()
            
            # Show methodology examples
            await research_methodology_examples()
            
            # Demonstrate analytical reasoning
            await demonstrate_analytical_reasoning()
            
            # Test the configuration
            await test_research_analyst()
            
            print(f"\n🎉 Research Analyst Example Complete!")
            print(f"🔬 Your rigorous research assistant is ready for analysis!")
        else:
            print(f"\n❌ Research analyst example failed")
            
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
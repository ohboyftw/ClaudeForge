#!/usr/bin/env python3
"""
Feedback and Optimization Example - Claude Configuration Manager

This example demonstrates the self-improvement cycle: collecting user feedback,
training the DSPy system, and automatically optimizing configurations.
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import the manager
sys.path.append(str(Path(__file__).parent.parent))

from claude_config_manager import ClaudeConfigManager, GenerationConfig

async def setup_feedback_optimization_demo():
    """Set up the feedback and optimization demonstration"""
    print("🔄 Feedback & Optimization Example - Claude Configuration Manager")
    print("=" * 70)
    
    workspace_path = "./workspace/feedback_optimization"
    manager = ClaudeConfigManager(workspace_path)
    
    print(f"✅ Initialized Claude Config Manager for feedback demo")
    print(f"📂 Workspace: {workspace_path}")
    
    return manager

async def create_initial_configuration(manager):
    """Create an initial configuration for optimization"""
    print(f"\n1️⃣ Creating Initial Configuration")
    print("=" * 40)
    
    config = GenerationConfig(
        branch_name="feedback_demo_v1",
        personality_traits=["helpful", "responsive", "adaptable"],
        expertise_domains=["general_assistance", "problem_solving"],
        communication_style="conversational",
        reasoning_approach="balanced",
        model_coordination_strategy="single_model",
        quality_threshold=0.8,
        commit_message="Initial configuration for feedback optimization demo"
    )
    
    print(f"📋 Creating baseline configuration...")
    print(f"   🎯 Name: {config.branch_name}")
    print(f"   💫 Initial quality threshold: {config.quality_threshold}")
    
    try:
        result = await manager.create_configuration(config)
        print(f"✅ Initial configuration created successfully!")
        print(f"📊 Baseline quality score: {result.get('quality_score', 'N/A')}")
        return result
    except Exception as e:
        print(f"❌ Error creating initial configuration: {e}")
        return None

async def simulate_user_feedback_collection(manager):
    """Simulate collecting user feedback over time"""
    print(f"\n2️⃣ Collecting User Feedback")
    print("=" * 35)
    
    # Simulate feedback from different users over time
    feedback_scenarios = [
        {
            "day": 1,
            "config_name": "feedback_demo_v1",
            "satisfaction": 0.7,
            "comments": "Good responses but could be more detailed",
            "user_type": "casual_user"
        },
        {
            "day": 2, 
            "config_name": "feedback_demo_v1",
            "satisfaction": 0.8,
            "comments": "Helpful but sometimes too verbose",
            "user_type": "power_user"
        },
        {
            "day": 3,
            "config_name": "feedback_demo_v1", 
            "satisfaction": 0.6,
            "comments": "Needs better technical explanations",
            "user_type": "technical_user"
        },
        {
            "day": 4,
            "config_name": "feedback_demo_v1",
            "satisfaction": 0.9,
            "comments": "Excellent for creative tasks!",
            "user_type": "creative_user"
        },
        {
            "day": 5,
            "config_name": "feedback_demo_v1",
            "satisfaction": 0.75,
            "comments": "Generally good, could be more consistent",
            "user_type": "business_user"
        }
    ]
    
    print(f"📝 Simulating {len(feedback_scenarios)} days of user feedback...")
    
    for scenario in feedback_scenarios:
        print(f"\n📅 Day {scenario['day']} - {scenario['user_type']}:")
        print(f"   ⭐ Satisfaction: {scenario['satisfaction']:.1f}/1.0")
        print(f"   💬 Comment: \"{scenario['comments']}\"")
        
        # Add feedback to the system
        try:
            manager.optimization_framework.add_user_feedback(
                scenario['config_name'],
                scenario['satisfaction'],
                scenario['comments']
            )
            print(f"   ✅ Feedback recorded in system")
        except Exception as e:
            print(f"   ❌ Error recording feedback: {e}")
        
        # Small delay to simulate time passing
        await asyncio.sleep(0.1)
    
    # Show feedback summary
    print(f"\n📊 Feedback Summary:")
    try:
        feedback_history = manager.optimization_framework.effectiveness_metric.feedback_history
        if feedback_history:
            avg_satisfaction = sum(fb['satisfaction'] for fb in feedback_history) / len(feedback_history)
            print(f"   📈 Average satisfaction: {avg_satisfaction:.2f}/1.0")
            print(f"   📝 Total feedback entries: {len(feedback_history)}")
            
            # Show satisfaction trend
            satisfactions = [fb['satisfaction'] for fb in feedback_history[-5:]]
            trend = "📈 Improving" if satisfactions[-1] > satisfactions[0] else "📉 Declining" if satisfactions[-1] < satisfactions[0] else "➡️ Stable"
            print(f"   {trend}")
        else:
            print(f"   ⚠️ No feedback recorded")
    except Exception as e:
        print(f"   ❌ Error analyzing feedback: {e}")

async def demonstrate_training_data_accumulation(manager):
    """Show how training data accumulates from usage"""
    print(f"\n3️⃣ Training Data Accumulation")
    print("=" * 35)
    
    # Simulate various successful configuration requests
    training_scenarios = [
        {
            "requirements": "Technical writing assistant for documentation",
            "performance": "High user satisfaction with technical accuracy"
        },
        {
            "requirements": "Creative writing helper for storytelling",
            "performance": "Excellent creativity scores and user engagement"
        },
        {
            "requirements": "Code review assistant for Python projects", 
            "performance": "Accurate bug detection and helpful suggestions"
        },
        {
            "requirements": "Research assistant for academic papers",
            "performance": "Thorough analysis and proper citation formatting"
        },
        {
            "requirements": "Customer support chatbot personality",
            "performance": "Empathetic responses and problem resolution"
        },
        {
            "requirements": "Data analysis helper for business reports",
            "performance": "Clear explanations and actionable insights"
        }
    ]
    
    print(f"📚 Adding training examples from successful configurations...")
    
    initial_count = len(manager.optimization_framework.training_examples)
    
    for i, scenario in enumerate(training_scenarios, 1):
        print(f"\n📖 Training Example {i}:")
        print(f"   🎯 Requirements: {scenario['requirements']}")
        print(f"   📊 Performance: {scenario['performance']}")
        
        try:
            manager.optimization_framework.add_training_example(
                scenario['requirements'],
                scenario['performance']
            )
            print(f"   ✅ Added to training set")
        except Exception as e:
            print(f"   ❌ Error adding training example: {e}")
    
    final_count = len(manager.optimization_framework.training_examples)
    added_count = final_count - initial_count
    
    print(f"\n📈 Training Data Status:")
    print(f"   📚 Total training examples: {final_count}")
    print(f"   ➕ Added in this demo: {added_count}")
    print(f"   🎯 Ready for optimization: {'Yes' if final_count >= 5 else f'No (need {5 - final_count} more)'}")

async def trigger_dspy_optimization(manager):
    """Trigger the DSPy optimization process"""
    print(f"\n4️⃣ DSPy Optimization Process")
    print("=" * 35)
    
    try:
        # Check if optimization should run
        should_optimize = await manager.optimization_framework.should_optimize()
        print(f"🔍 Optimization check: {'Ready to optimize' if should_optimize else 'Not ready yet'}")
        
        if should_optimize:
            print(f"\n🚀 Running DSPy BootstrapFewShot optimization...")
            print(f"   (This would normally take several minutes)")
            
            # Note: In a real scenario, this would run actual DSPy optimization
            # For demo purposes, we'll simulate the process
            
            print(f"   🔄 Analyzing training examples...")
            await asyncio.sleep(0.5)
            
            print(f"   🧠 Optimizing prompts with BootstrapFewShot...")
            await asyncio.sleep(0.5)
            
            print(f"   📊 Evaluating performance improvements...")
            await asyncio.sleep(0.5)
            
            # Simulate optimization results
            optimization_results = {
                "improvement_score": 0.15,
                "new_quality_threshold": 0.85,
                "optimized_prompts": 3,
                "validation_score": 0.92
            }
            
            print(f"   ✅ Optimization completed!")
            print(f"\n📈 Optimization Results:")
            print(f"   🎯 Performance improvement: +{optimization_results['improvement_score']:.1%}")
            print(f"   📊 New quality threshold: {optimization_results['new_quality_threshold']}")
            print(f"   🔧 Prompts optimized: {optimization_results['optimized_prompts']}")
            print(f"   ✅ Validation score: {optimization_results['validation_score']:.2f}")
            
        else:
            print(f"\n⏳ Optimization not triggered yet")
            stats = manager.optimization_framework.get_optimization_stats()
            print(f"   📚 Training examples: {stats.get('training_examples', 0)}")
            print(f"   🎯 Need: 5+ examples and performance below threshold")
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")

async def demonstrate_continuous_improvement():
    """Demonstrate the continuous improvement cycle"""
    print(f"\n5️⃣ Continuous Improvement Cycle")
    print("=" * 40)
    
    improvement_cycle = [
        {
            "step": "📊 Performance Monitoring",
            "description": "System continuously monitors configuration effectiveness",
            "metrics": ["Response quality", "User satisfaction", "Task completion rate"]
        },
        {
            "step": "📝 Feedback Collection", 
            "description": "User feedback is automatically collected and analyzed",
            "metrics": ["Satisfaction scores", "Improvement suggestions", "Usage patterns"]
        },
        {
            "step": "📚 Training Data Curation",
            "description": "Successful configurations become training examples",
            "metrics": ["Example quality", "Diversity", "Relevance"]
        },
        {
            "step": "🧠 DSPy Optimization",
            "description": "BootstrapFewShot optimizes prompts using training data",
            "metrics": ["Prompt effectiveness", "Quality improvement", "Consistency"]
        },
        {
            "step": "🚀 Deployment & Testing",
            "description": "Optimized configurations are tested and deployed",
            "metrics": ["A/B test results", "Performance validation", "User acceptance"]
        },
        {
            "step": "🔄 Cycle Repeat",
            "description": "Process repeats continuously for ongoing improvement",
            "metrics": ["Long-term trends", "Cumulative gains", "System evolution"]
        }
    ]
    
    print(f"🔄 Self-Improvement Process:")
    
    for i, step in enumerate(improvement_cycle, 1):
        print(f"\n{i}. {step['step']}")
        print(f"   {step['description']}")
        print(f"   📊 Key metrics: {', '.join(step['metrics'])}")
    
    print(f"\n📈 Expected Long-term Outcomes:")
    outcomes = [
        "🎯 Increasing accuracy and relevance of responses",
        "😊 Higher user satisfaction scores over time", 
        "🚀 Faster generation of high-quality configurations",
        "🧠 Better understanding of user preferences and patterns",
        "⚡ Reduced manual configuration and tuning effort"
    ]
    
    for outcome in outcomes:
        print(f"   {outcome}")

async def show_optimization_analytics(manager):
    """Show optimization analytics and metrics"""
    print(f"\n6️⃣ Optimization Analytics")
    print("=" * 30)
    
    try:
        # Get current optimization statistics
        stats = manager.optimization_framework.get_optimization_stats()
        
        print(f"📊 Current System Statistics:")
        print(f"   🔧 Total optimizations: {stats.get('total_optimizations', 0)}")
        print(f"   📚 Training examples: {stats.get('training_examples', 0)}")
        print(f"   📈 Average improvement: {stats.get('average_improvement', 0):.1%}")
        print(f"   ✅ System optimized: {'Yes' if stats.get('is_optimized', False) else 'No'}")
        
        last_optimization = stats.get('last_optimization')
        if last_optimization:
            print(f"   🕐 Last optimization: {last_optimization}")
        else:
            print(f"   🕐 Last optimization: Never")
        
        # Show feedback analytics
        feedback_history = manager.optimization_framework.effectiveness_metric.feedback_history
        if feedback_history:
            print(f"\n📝 Feedback Analytics:")
            
            # Calculate satisfaction trends
            satisfactions = [fb['satisfaction'] for fb in feedback_history]
            avg_satisfaction = sum(satisfactions) / len(satisfactions)
            
            print(f"   ⭐ Average satisfaction: {avg_satisfaction:.2f}/1.0")
            print(f"   📊 Feedback entries: {len(feedback_history)}")
            
            # Show recent trend
            if len(satisfactions) >= 3:
                recent_avg = sum(satisfactions[-3:]) / 3
                early_avg = sum(satisfactions[:3]) / 3
                trend = recent_avg - early_avg
                trend_emoji = "📈" if trend > 0.05 else "📉" if trend < -0.05 else "➡️"
                print(f"   {trend_emoji} Recent trend: {trend:+.2f}")
        
        print(f"\n🎯 Optimization Recommendations:")
        recommendations = []
        
        if stats.get('training_examples', 0) < 10:
            recommendations.append("📚 Collect more diverse training examples")
        
        if avg_satisfaction < 0.8:
            recommendations.append("💡 Focus on improving user satisfaction")
            
        if stats.get('total_optimizations', 0) == 0:
            recommendations.append("🚀 Run first optimization when ready")
        
        if not recommendations:
            recommendations.append("✅ System performing well, continue monitoring")
        
        for rec in recommendations:
            print(f"   {rec}")
            
    except Exception as e:
        print(f"❌ Error getting analytics: {e}")

async def main():
    """Run the feedback and optimization example"""
    try:
        # Setup
        manager = await setup_feedback_optimization_demo()
        
        # 1. Create initial configuration
        result = await create_initial_configuration(manager)
        if not result:
            print("❌ Failed to create initial configuration")
            return
        
        # 2. Simulate user feedback collection
        await simulate_user_feedback_collection(manager)
        
        # 3. Demonstrate training data accumulation
        await demonstrate_training_data_accumulation(manager)
        
        # 4. Trigger DSPy optimization
        await trigger_dspy_optimization(manager)
        
        # 5. Show continuous improvement cycle
        await demonstrate_continuous_improvement()
        
        # 6. Show analytics
        await show_optimization_analytics(manager)
        
        print(f"\n🎉 Feedback & Optimization Example Complete!")
        print(f"🔄 The self-improvement cycle is now active!")
        print(f"💡 Key takeaways:")
        print(f"   • User feedback drives system improvement")
        print(f"   • Training data accumulates from successful usage")
        print(f"   • DSPy automatically optimizes prompts")
        print(f"   • Performance improves continuously over time")
        
    except Exception as e:
        print(f"❌ Feedback optimization example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
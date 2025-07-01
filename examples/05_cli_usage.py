#!/usr/bin/env python3
"""
CLI Usage Examples - Claude Configuration Manager

This example demonstrates all the command-line interface features of the
Claude Configuration Manager, showing how to use it from the terminal.
"""

import asyncio
import sys
import subprocess
from pathlib import Path

# Add parent directory to path to import the manager
sys.path.append(str(Path(__file__).parent.parent))

from claude_config_manager import ClaudeConfigManager

def run_cli_command(command, workspace_path="./workspace/cli_examples"):
    """Helper function to run CLI commands and capture output"""
    try:
        # Prepare the full command
        full_command = f"cd {workspace_path} && python ../claude_config_manager.py {command}"
        
        print(f"💻 Running: {command}")
        print(f"📂 Workspace: {workspace_path}")
        
        # Run the command (note: this is a simulation since the CLI might need the actual script)
        # In practice, you would run the actual CLI commands
        print(f"   Command: {full_command}")
        print(f"   ✅ Command prepared (run manually for actual execution)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

async def demonstrate_basic_cli_commands():
    """Demonstrate basic CLI commands"""
    print("💻 Basic CLI Commands - Claude Configuration Manager")
    print("=" * 60)
    
    # Ensure workspace exists
    workspace_path = "./workspace/cli_examples"
    Path(workspace_path).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 Basic Configuration Commands:")
    
    # 1. Create a new configuration
    print(f"\n1️⃣ Creating a new configuration:")
    create_command = "create --name general_helper --style conversational --domains general_knowledge problem_solving --traits helpful friendly"
    run_cli_command(create_command, workspace_path)
    
    # 2. List available configurations
    print(f"\n2️⃣ Listing available configurations:")
    list_command = "list"
    run_cli_command(list_command, workspace_path)
    
    # 3. Switch to a configuration
    print(f"\n3️⃣ Switching to a configuration:")
    switch_command = "switch --name general_helper"
    run_cli_command(switch_command, workspace_path)
    
    # 4. Show current configuration
    print(f"\n4️⃣ Showing current configuration:")
    current_command = "current"
    run_cli_command(current_command, workspace_path)

async def demonstrate_advanced_cli_commands():
    """Demonstrate advanced CLI commands"""
    print(f"\n🚀 Advanced CLI Commands:")
    
    workspace_path = "./workspace/cli_examples"
    
    # 5. Create with advanced options
    print(f"\n5️⃣ Creating advanced configuration:")
    advanced_create = "create --name coding_expert --style technical --domains software_development code_review debugging --traits analytical precise systematic --coordination parallel_specialization --quality 0.9"
    run_cli_command(advanced_create, workspace_path)
    
    # 6. Analyze performance
    print(f"\n6️⃣ Analyzing configuration performance:")
    analyze_command = "analyze --name coding_expert"
    run_cli_command(analyze_command, workspace_path)
    
    # 7. Add user feedback
    print(f"\n7️⃣ Adding user feedback:")
    feedback_command = "feedback --name coding_expert --satisfaction 0.9 --comments 'Excellent for debugging tasks'"
    run_cli_command(feedback_command, workspace_path)
    
    # 8. Check DSPy optimization status
    print(f"\n8️⃣ Checking DSPy optimization status:")
    dspy_status_command = "dspy-status"
    run_cli_command(dspy_status_command, workspace_path)

async def demonstrate_dspy_cli_commands():
    """Demonstrate DSPy-specific CLI commands"""
    print(f"\n🧠 DSPy Self-Improvement Commands:")
    
    workspace_path = "./workspace/cli_examples"
    
    # 9. Manual DSPy optimization
    print(f"\n9️⃣ Running manual DSPy optimization:")
    optimize_command = "optimize-dspy --name coding_expert"
    run_cli_command(optimize_command, workspace_path)
    
    # 10. View optimization history
    print(f"\n🔟 Viewing optimization history:")
    history_command = "optimization-history"
    run_cli_command(history_command, workspace_path)
    
    # 11. Export training data
    print(f"\n1️⃣1️⃣ Exporting training data:")
    export_command = "export-training-data --format json"
    run_cli_command(export_command, workspace_path)

async def demonstrate_git_integration_commands():
    """Demonstrate git integration commands"""
    print(f"\n📚 Git Integration Commands:")
    
    workspace_path = "./workspace/cli_examples"
    
    # 12. Show git status
    print(f"\n1️⃣2️⃣ Showing git status:")
    git_status_command = "git-status"
    run_cli_command(git_status_command, workspace_path)
    
    # 13. Create new branch for configuration
    print(f"\n1️⃣3️⃣ Creating new git branch:")
    branch_command = "create-branch --name experimental_writer --from general_helper"
    run_cli_command(branch_command, workspace_path)
    
    # 14. Merge configurations
    print(f"\n1️⃣4️⃣ Merging configurations:")
    merge_command = "merge-config --source experimental_writer --target general_helper"
    run_cli_command(merge_command, workspace_path)

async def show_complete_cli_workflow():
    """Show a complete CLI workflow example"""
    print(f"\n🔄 Complete CLI Workflow Example:")
    print("=" * 40)
    
    workflow_steps = [
        ("1. Initialize", "init --workspace ./my_claude_configs"),
        ("2. Create config", "create --name my_assistant --style helpful --domains general"),
        ("3. Test config", "test --name my_assistant --sample-prompt 'Help me write an email'"),
        ("4. Get feedback", "feedback --name my_assistant --satisfaction 0.8 --comments 'Good but could be more formal'"),
        ("5. Optimize", "optimize-dspy --name my_assistant"),
        ("6. Deploy", "export --name my_assistant --format claude-desktop"),
        ("7. Monitor", "analyze --name my_assistant --time-range 7d")
    ]
    
    for step, command in workflow_steps:
        print(f"\n{step}:")
        print(f"   💻 {command}")

def show_cli_help_information():
    """Show comprehensive CLI help information"""
    print(f"\n📖 CLI Help Information:")
    print("=" * 30)
    
    command_categories = {
        "🏗️ Configuration Management": [
            "create          - Create new Claude configuration",
            "list            - List all configurations", 
            "switch          - Switch to a configuration",
            "current         - Show current configuration",
            "delete          - Delete a configuration",
            "copy            - Copy existing configuration"
        ],
        "📊 Analysis & Monitoring": [
            "analyze         - Analyze configuration performance",
            "metrics         - Show detailed metrics",
            "compare         - Compare multiple configurations",
            "benchmark       - Run performance benchmarks"
        ],
        "🧠 DSPy Self-Improvement": [
            "dspy-status     - Show DSPy optimization status",
            "optimize-dspy   - Run manual DSPy optimization", 
            "feedback        - Add user feedback for training",
            "training-stats  - Show training data statistics"
        ],
        "📚 Git Integration": [
            "git-status      - Show git repository status",
            "create-branch   - Create new configuration branch",
            "merge-config    - Merge configuration changes",
            "commit          - Commit configuration changes"
        ],
        "🔧 Utilities": [
            "export          - Export configuration to various formats",
            "import          - Import configuration from file",
            "validate        - Validate configuration syntax", 
            "backup          - Backup all configurations"
        ]
    }
    
    for category, commands in command_categories.items():
        print(f"\n{category}:")
        for command in commands:
            print(f"   {command}")

async def demonstrate_configuration_templates():
    """Demonstrate using configuration templates via CLI"""
    print(f"\n📋 Configuration Templates via CLI:")
    print("=" * 40)
    
    templates = [
        {
            "name": "coding-assistant",
            "command": "create --template coding-assistant --name my_coder --quality 0.9"
        },
        {
            "name": "creative-writer", 
            "command": "create --template creative-writer --name my_writer --style engaging"
        },
        {
            "name": "research-analyst",
            "command": "create --template research-analyst --name my_researcher --quality 0.95"
        },
        {
            "name": "customer-support",
            "command": "create --template customer-support --name support_bot --traits empathetic helpful"
        }
    ]
    
    print(f"🎨 Available Templates:")
    for template in templates:
        print(f"\n📌 {template['name']}:")
        print(f"   💻 {template['command']}")

async def main():
    """Run the CLI usage examples"""
    try:
        print("🎯 CLI Usage Examples - Claude Configuration Manager")
        print("=" * 65)
        
        # Basic CLI commands
        await demonstrate_basic_cli_commands()
        
        # Advanced CLI commands
        await demonstrate_advanced_cli_commands()
        
        # DSPy CLI commands
        await demonstrate_dspy_cli_commands()
        
        # Git integration commands
        await demonstrate_git_integration_commands()
        
        # Complete workflow
        await show_complete_cli_workflow()
        
        # Templates
        await demonstrate_configuration_templates()
        
        # Help information
        show_cli_help_information()
        
        print(f"\n🎉 CLI Usage Examples Complete!")
        print(f"💡 To run actual commands:")
        print(f"   1. Navigate to your workspace directory")
        print(f"   2. Run: python claude_config_manager.py [command]")
        print(f"   3. Use --help with any command for detailed options")
        
    except Exception as e:
        print(f"❌ CLI examples failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
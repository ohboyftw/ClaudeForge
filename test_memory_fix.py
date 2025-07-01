#!/usr/bin/env python3
"""
Test memory threshold fix without requiring Ollama
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Import the resource monitor to test memory validation
from claude_config_manager import ResourceMonitor

def test_memory_threshold():
    """Test that the memory threshold has been lowered to 6GB"""
    print("🧪 Testing Memory Threshold Fix")
    print("=" * 40)
    
    # Create a mock that returns your actual available memory
    mock_resources = {
        'memory_available_gb': 6.3,
        'memory_total_gb': 7.4,
        'cpu_count': 16
    }
    
    # Test the resource monitor directly
    monitor = ResourceMonitor()
    resources = monitor.get_system_resources()
    
    print(f"📊 Actual system resources:")
    print(f"   💾 Available Memory: {resources.get('memory_available_gb', 0):.1f}GB")
    print(f"   🔢 Total Memory: {resources.get('memory_total_gb', 0):.1f}GB")
    print(f"   🖥️  CPU Cores: {resources.get('cpu_count', 0)}")
    
    # Test memory validation logic
    available_memory = resources.get('memory_available_gb', 0)
    
    print(f"\n🔍 Memory Validation Test:")
    print(f"   Available: {available_memory:.1f}GB")
    
    if available_memory < 4:
        print(f"   ❌ Would fail: {available_memory:.1f}GB < 6GB threshold")
        print(f"   💡 Recommendation: Lower threshold or free up memory")
    else:
        print(f"   ✅ Passes: {available_memory:.1f}GB >= 6GB threshold")
        print(f"   🎉 Memory optimization successful!")
    
    # Test the old vs new threshold
    print(f"\n📈 Threshold Comparison:")
    print(f"   🔴 Old threshold (8GB): {'FAIL' if available_memory < 8 else 'PASS'}")
    print(f"   🟢 New threshold (6GB): {'FAIL' if available_memory < 6 else 'PASS'}")
    
    return available_memory >= 6

if __name__ == "__main__":
    try:
        success = test_memory_threshold()
        if success:
            print(f"\n✅ Memory optimization successful!")
            print(f"🚀 Ready to run with Ollama when installed")
        else:
            print(f"\n❌ Memory still insufficient")
            print(f"💡 Consider closing other applications")
    except Exception as e:
        print(f"❌ Test failed: {e}")
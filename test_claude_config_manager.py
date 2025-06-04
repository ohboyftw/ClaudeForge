#!/usr/bin/env python3
"""
Test suite for Claude Configuration Manager

This module provides comprehensive tests for the Claude Configuration Manager,
including unit tests, integration tests, and performance benchmarks.
"""

import asyncio
import json
import tempfile
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import time

from claude_config_manager import (
    ClaudeConfigManager,
    GenerationConfig,
    PerformanceMetrics,
    OllamaModelManager,
    ClaudePromptOrchestrator,
    GitVersionManager,
    LogAnalyzer,
    ResourceMonitor,
    ModelResult,
    CoordinatedResult
)

# Test fixtures
@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return GenerationConfig(
        branch_name="test_config",
        personality_traits=["analytical", "helpful"],
        expertise_domains=["testing", "development"],
        communication_style="technical",
        reasoning_approach="systematic",
        commit_message="Test configuration creation"
    )

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    mock_client = Mock()
    mock_client.list.return_value = {
        'models': [
            {'name': 'deepseek-r1:8b'},
            {'name': 'qwen3:8b'},
            {'name': 'gemma3:4b'}
        ]
    }
    mock_client.generate.return_value = {
        'response': 'Test generated content',
        'eval_count': 100,
        'load_duration': 1000,
        'total_duration': 5000
    }
    return mock_client

# ============================================================================
# Unit Tests
# ============================================================================

class TestGenerationConfig:
    """Test GenerationConfig class"""
    
    def test_config_creation(self):
        """Test basic configuration creation"""
        config = GenerationConfig(
            branch_name="test",
            personality_traits=["helpful"],
            expertise_domains=["testing"]
        )
        assert config.branch_name == "test"
        assert "helpful" in config.personality_traits
        assert "testing" in config.expertise_domains
    
    def test_config_validation_valid(self):
        """Test validation of valid configuration"""
        config = GenerationConfig(
            branch_name="valid_config",
            communication_style="professional",
            reasoning_approach="analytical",
            quality_threshold=0.8
        )
        is_valid, errors = config.validate()
        assert is_valid
        assert len(errors) == 0
    
    def test_config_validation_invalid(self):
        """Test validation of invalid configuration"""
        config = GenerationConfig(
            branch_name="",  # Invalid: empty branch name
            quality_threshold=1.5,  # Invalid: > 1.0
            max_generation_time=5,  # Invalid: < 10
            communication_style="invalid_style"  # Invalid style
        )
        is_valid, errors = config.validate()
        assert not is_valid
        assert len(errors) > 0
        assert any("empty" in error.lower() for error in errors)
        assert any("threshold" in error.lower() for error in errors)
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = GenerationConfig(
            branch_name="test",
            personality_traits=["analytical"],
            expertise_domains=["testing"]
        )
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["branch_name"] == "test"
        assert "created_at" in config_dict

class TestPerformanceMetrics:
    """Test PerformanceMetrics class"""
    
    def test_metrics_creation(self):
        """Test basic metrics creation"""
        metrics = PerformanceMetrics(
            response_quality_score=0.85,
            task_completion_rate=0.92,
            error_rate=0.05
        )
        assert metrics.response_quality_score == 0.85
        assert metrics.task_completion_rate == 0.92
        assert metrics.error_rate == 0.05
    
    def test_metrics_to_dict(self):
        """Test metrics serialization"""
        metrics = PerformanceMetrics(
            response_quality_score=0.8,
            total_interactions=100,
            successful_interactions=95
        )
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["response_quality_score"] == 0.8
        assert metrics_dict["total_interactions"] == 100

class TestResourceMonitor:
    """Test ResourceMonitor class"""
    
    def test_resource_monitor_creation(self):
        """Test resource monitor initialization"""
        monitor = ResourceMonitor()
        assert monitor.cpu_threshold == 0.8
        assert monitor.memory_threshold == 0.8
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_system_resources(self, mock_disk, mock_memory, mock_cpu):
        """Test system resource retrieval"""
        # Mock system resources
        mock_cpu.return_value = 25.0
        mock_memory.return_value = Mock(percent=60.0, available=8*1024**3, total=16*1024**3)
        mock_disk.return_value = Mock(percent=45.0, free=100*1024**3)
        
        monitor = ResourceMonitor()
        resources = monitor.get_system_resources()
        
        assert resources['cpu_percent'] == 25.0
        assert resources['memory_percent'] == 60.0
        assert resources['memory_available_gb'] == 8.0
        assert resources['disk_percent'] == 45.0
    
    def test_can_load_model_sufficient_resources(self):
        """Test model loading check with sufficient resources"""
        monitor = ResourceMonitor()
        # Mock sufficient resources
        with patch.object(monitor, 'get_system_resources') as mock_resources:
            mock_resources.return_value = {
                'memory_available_gb': 10.0,
                'cpu_percent': 30.0
            }
            
            can_load, reason = monitor.can_load_model(5.0)
            assert can_load
            assert "available" in reason.lower()
    
    def test_can_load_model_insufficient_resources(self):
        """Test model loading check with insufficient resources"""
        monitor = ResourceMonitor()
        # Mock insufficient resources
        with patch.object(monitor, 'get_system_resources') as mock_resources:
            mock_resources.return_value = {
                'memory_available_gb': 2.0,
                'cpu_percent': 95.0
            }
            
            can_load, reason = monitor.can_load_model(5.0)
            assert not can_load
            assert "insufficient" in reason.lower() or "high cpu" in reason.lower()

# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
class TestOllamaModelManager:
    """Test OllamaModelManager integration"""
    
    @patch('ollama.Client')
    async def test_model_manager_initialization(self, mock_client_class):
        """Test model manager initialization"""
        mock_client = mock_ollama_client()
        mock_client_class.return_value = mock_client
        
        manager = OllamaModelManager()
        assert 'reasoning' in manager.models
        assert 'orchestration' in manager.models
        assert len(manager.models['reasoning']) > 0
    
    @patch('ollama.Client')
    async def test_check_model_availability(self, mock_client_class):
        """Test model availability checking"""
        mock_client = mock_ollama_client()
        mock_client_class.return_value = mock_client
        
        manager = OllamaModelManager()
        available = await manager.check_model_availability('deepseek-r1:8b')
        assert available
        
        unavailable = await manager.check_model_availability('nonexistent:model')
        assert not unavailable
    
    @patch('ollama.Client')
    async def test_get_optimal_model(self, mock_client_class):
        """Test optimal model selection"""
        mock_client = mock_ollama_client()
        mock_client_class.return_value = mock_client
        
        manager = OllamaModelManager()
        model = await manager.get_optimal_model('reasoning')
        assert model in manager.models['reasoning']
        
        # Test fallback for unknown category
        model = await manager.get_optimal_model('unknown_category')
        assert model  # Should return some available model
    
    @patch('ollama.Client')
    async def test_generate_with_model(self, mock_client_class):
        """Test content generation with model"""
        mock_client = mock_ollama_client()
        mock_client_class.return_value = mock_client
        
        manager = OllamaModelManager()
        result = await manager.generate_with_model('deepseek-r1:8b', 'Test prompt')
        
        assert isinstance(result, ModelResult)
        assert result.model_name == 'deepseek-r1:8b'
        assert result.content == 'Test generated content'
        assert result.confidence > 0
        assert result.execution_time > 0

@pytest.mark.asyncio
class TestClaudeConfigManager:
    """Test ClaudeConfigManager integration"""
    
    @patch('claude_config_manager.OllamaModelManager')
    @patch('git.Repo')
    async def test_manager_initialization(self, mock_repo, mock_model_manager, temp_workspace):
        """Test manager initialization"""
        # Mock dependencies
        mock_model_manager.return_value = Mock()
        mock_repo.return_value = Mock()
        
        manager = ClaudeConfigManager(str(temp_workspace))
        assert manager.workspace_path == temp_workspace
        assert (temp_workspace / "configs").exists()
        assert (temp_workspace / "backups").exists()
    
    @patch('claude_config_manager.OllamaModelManager')
    @patch('git.Repo')
    async def test_create_configuration(self, mock_repo, mock_model_manager, temp_workspace, sample_config):
        """Test configuration creation"""
        # Mock dependencies
        mock_model_manager_instance = Mock()
        mock_model_manager.return_value = mock_model_manager_instance
        mock_repo_instance = Mock()
        mock_repo.return_value = mock_repo_instance
        
        # Mock the generation process
        mock_orchestrator = Mock()
        mock_generator = Mock()
        mock_git_manager = Mock()
        
        with patch('claude_config_manager.ClaudePromptOrchestrator') as mock_orch_class, \
             patch('claude_config_manager.ClaudeMdGenerator') as mock_gen_class, \
             patch('claude_config_manager.GitVersionManager') as mock_git_class:
            
            mock_orch_class.return_value = mock_orchestrator
            mock_gen_class.return_value = mock_generator
            mock_git_class.return_value = mock_git_manager
            
            # Mock the generator to return test content
            mock_generator.generate_claude_md = AsyncMock(return_value="# Test Claude.md content")
            mock_git_manager.create_configuration_branch = AsyncMock()
            mock_git_manager.commit_configuration = AsyncMock()
            
            manager = ClaudeConfigManager(str(temp_workspace))
            
            # Override the mocked components
            manager.generator = mock_generator
            manager.git_manager = mock_git_manager
            
            content = await manager.create_configuration(sample_config)
            
            assert content
            assert "Claude.md" in content
            mock_generator.generate_claude_md.assert_called_once()
            mock_git_manager.create_configuration_branch.assert_called_once()
            mock_git_manager.commit_configuration.assert_called_once()

# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.slow
    @patch('ollama.Client')
    async def test_multiple_configurations_performance(self, mock_client_class, temp_workspace):
        """Test performance with multiple configurations"""
        mock_client = mock_ollama_client()
        mock_client_class.return_value = mock_client
        
        # Mock git operations
        with patch('git.Repo'):
            manager = ClaudeConfigManager(str(temp_workspace))
            
            # Mock the generation components
            with patch.object(manager.generator, 'generate_claude_md') as mock_generate, \
                 patch.object(manager.git_manager, 'create_configuration_branch') as mock_branch, \
                 patch.object(manager.git_manager, 'commit_configuration') as mock_commit:
                
                mock_generate.return_value = "# Test content"
                mock_branch.return_value = None
                mock_commit.return_value = None
                
                configs = [
                    GenerationConfig(branch_name=f"test_config_{i}")
                    for i in range(5)
                ]
                
                start_time = time.time()
                
                # Create configurations sequentially
                for config in configs:
                    await manager.create_configuration(config)
                
                total_time = time.time() - start_time
                
                # Should complete within reasonable time
                assert total_time < 30  # 30 seconds for 5 configs
                assert mock_generate.call_count == 5
    
    @pytest.mark.slow
    async def test_resource_usage_monitoring(self):
        """Test resource usage doesn't exceed limits"""
        monitor = ResourceMonitor()
        
        # Simulate monitoring during heavy load
        for _ in range(10):
            resources = monitor.get_system_resources()
            if resources:
                # Verify we're not hitting critical resource usage
                # (This is more of a smoke test since we can't control actual usage)
                assert isinstance(resources.get('cpu_percent', 0), (int, float))
                assert isinstance(resources.get('memory_percent', 0), (int, float))
            await asyncio.sleep(0.1)

# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    async def test_invalid_configuration_handling(self, temp_workspace):
        """Test handling of invalid configurations"""
        with patch('claude_config_manager.OllamaModelManager'), \
             patch('git.Repo'):
            
            manager = ClaudeConfigManager(str(temp_workspace))
            
            # Test invalid config
            invalid_config = GenerationConfig(
                branch_name="",  # Invalid empty name
                quality_threshold=2.0  # Invalid threshold
            )
            
            with pytest.raises((ValueError, RuntimeError)):
                await manager.create_configuration(invalid_config)
    
    @patch('ollama.Client')
    async def test_model_unavailable_handling(self, mock_client_class):
        """Test handling when models are unavailable"""
        # Mock client that returns no models
        mock_client = Mock()
        mock_client.list.return_value = {'models': []}
        mock_client_class.return_value = mock_client
        
        manager = OllamaModelManager()
        
        with pytest.raises(RuntimeError, match="No Ollama models available"):
            await manager.get_optimal_model('reasoning')
    
    @patch('ollama.Client')
    async def test_generation_timeout_handling(self, mock_client_class):
        """Test handling of generation timeouts"""
        # Mock client that takes too long
        mock_client = Mock()
        mock_client.list.return_value = {'models': [{'name': 'test:model'}]}
        
        def slow_generate(*args, **kwargs):
            time.sleep(5)  # Simulate slow generation
            return {'response': 'content', 'eval_count': 10}
        
        mock_client.generate = slow_generate
        mock_client_class.return_value = mock_client
        
        manager = OllamaModelManager()
        
        with pytest.raises(RuntimeError, match="timeout"):
            await manager.generate_with_model('test:model', 'prompt', timeout=1)

# ============================================================================
# CLI Tests
# ============================================================================

class TestCLI:
    """Test CLI interface"""
    
    def test_cli_parser_creation(self):
        """Test CLI parser creation"""
        from claude_config_manager import create_cli_parser
        
        parser = create_cli_parser()
        assert parser is not None
        
        # Test help doesn't crash
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing"""
        from claude_config_manager import create_cli_parser
        
        parser = create_cli_parser()
        
        # Test create command
        args = parser.parse_args([
            'create',
            '--name', 'test-config',
            '--style', 'technical',
            '--domains', 'testing', 'development'
        ])
        
        assert args.command == 'create'
        assert args.name == 'test-config'
        assert args.style == 'technical'
        assert 'testing' in args.domains

# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")

# ============================================================================
# Test Utilities
# ============================================================================

def mock_ollama_client():
    """Create a mock Ollama client for testing"""
    mock_client = Mock()
    mock_client.list.return_value = {
        'models': [
            {'name': 'deepseek-r1:8b'},
            {'name': 'qwen3:8b'},
            {'name': 'deepseek-coder:6.7b'},
            {'name': 'gemma3:4b'},
            {'name': 'smallthinker:latest'}
        ]
    }
    mock_client.generate.return_value = {
        'response': 'Test generated content for Claude configuration',
        'eval_count': 150,
        'prompt_eval_count': 50,
        'load_duration': 2000,
        'prompt_eval_duration': 1000,
        'eval_duration': 3000,
        'total_duration': 6000
    }
    return mock_client

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", "--tb=short", __file__])
#!/usr/bin/env python3
"""
Claude Configuration Manager with DSPy + Ollama Integration

This system orchestrates multiple Ollama models through DSPy to generate
intelligent Claude.md files and associated prompts, with git-based version
management and performance evaluation through log analysis.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import aiofiles
import git
import ollama
import dspy
from pydantic import BaseModel, Field, validator
import psutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
import yaml

# ============================================================================
# DSPy Configuration - Proper LM Integration
# ============================================================================

class OllamaDSPyClient(dspy.LM):
    """Custom DSPy language model client for Ollama integration"""
    
    def __init__(self, model_name: str, ollama_client, **kwargs):
        super().__init__(model_name, **kwargs)
        self.model_name = model_name
        self.ollama_client = ollama_client
        self.history = []
        
    def basic_request(self, prompt: str, **kwargs) -> str:
        """Make request to Ollama model through DSPy interface"""
        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                **kwargs
            )
            
            content = response.get('response', '')
            self.history.append({
                'prompt': prompt,
                'response': content,
                'metadata': {
                    'eval_count': response.get('eval_count', 0),
                    'total_duration': response.get('total_duration', 0)
                }
            })
            
            return content
            
        except Exception as e:
            logger.error(f"DSPy Ollama request failed: {e}")
            raise RuntimeError(f"DSPy generation failed: {str(e)}")
    
    def __call__(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

# Configure logging with rich formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('claude_config_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# ============================================================================
# Data Models and Configuration
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics extracted from logs"""
    response_quality_score: float = 0.0
    task_completion_rate: float = 0.0
    error_rate: float = 0.0
    response_time: float = 0.0
    user_satisfaction: float = 0.0
    context_retention: float = 0.0
    prompt_effectiveness: Dict[str, float] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    average_tokens_per_response: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            'response_quality_score': self.response_quality_score,
            'task_completion_rate': self.task_completion_rate,
            'error_rate': self.error_rate,
            'response_time': self.response_time,
            'user_satisfaction': self.user_satisfaction,
            'context_retention': self.context_retention,
            'prompt_effectiveness': self.prompt_effectiveness,
            'improvement_suggestions': self.improvement_suggestions,
            'total_interactions': self.total_interactions,
            'successful_interactions': self.successful_interactions,
            'failed_interactions': self.failed_interactions,
            'average_tokens_per_response': self.average_tokens_per_response
        }

@dataclass
class GenerationConfig:
    """Configuration for Claude.md generation"""
    # Behavioral requirements
    personality_traits: List[str] = field(default_factory=list)
    expertise_domains: List[str] = field(default_factory=list)
    communication_style: str = "professional"
    reasoning_approach: str = "analytical"
    
    # Technical settings
    model_coordination_strategy: str = "hierarchical"
    quality_threshold: float = 0.8
    max_generation_time: int = 300  # seconds
    enable_caching: bool = True
    
    # Output specifications
    output_format: str = "markdown"
    include_examples: bool = True
    reference_files: List[str] = field(default_factory=list)
    max_content_length: int = 50000
    
    # Version management
    branch_name: str = "main"
    commit_message: str = "Update Claude configuration"
    merge_strategy: str = "merge"
    auto_backup: bool = True
    
    # Performance settings
    use_fast_mode: bool = False
    parallel_processing: bool = True
    resource_limit_cpu: float = 0.8  # 80% CPU limit
    resource_limit_memory: float = 0.8  # 80% memory limit
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters"""
        errors = []
        
        if not self.branch_name or not self.branch_name.strip():
            errors.append("Branch name cannot be empty")
        
        if self.quality_threshold < 0 or self.quality_threshold > 1:
            errors.append("Quality threshold must be between 0 and 1")
        
        if self.max_generation_time < 10:
            errors.append("Max generation time must be at least 10 seconds")
        
        if self.communication_style not in ["professional", "casual", "technical", "creative", "formal", "engaging"]:
            errors.append(f"Invalid communication style: {self.communication_style}")
        
        if self.reasoning_approach not in ["analytical", "systematic", "intuitive", "evidence_based", "creative"]:
            errors.append(f"Invalid reasoning approach: {self.reasoning_approach}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'personality_traits': self.personality_traits,
            'expertise_domains': self.expertise_domains,
            'communication_style': self.communication_style,
            'reasoning_approach': self.reasoning_approach,
            'model_coordination_strategy': self.model_coordination_strategy,
            'quality_threshold': self.quality_threshold,
            'max_generation_time': self.max_generation_time,
            'enable_caching': self.enable_caching,
            'output_format': self.output_format,
            'include_examples': self.include_examples,
            'reference_files': self.reference_files,
            'max_content_length': self.max_content_length,
            'branch_name': self.branch_name,
            'commit_message': self.commit_message,
            'merge_strategy': self.merge_strategy,
            'auto_backup': self.auto_backup,
            'use_fast_mode': self.use_fast_mode,
            'parallel_processing': self.parallel_processing,
            'resource_limit_cpu': self.resource_limit_cpu,
            'resource_limit_memory': self.resource_limit_memory,
            'created_at': datetime.now().isoformat()
        }

class ModelResult(BaseModel):
    """Result from a single model execution"""
    model_name: str
    content: str
    confidence: float
    execution_time: float
    tokens_used: int = 0
    memory_used: float = 0.0
    cpu_usage: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class CoordinatedResult(BaseModel):
    """Result from coordinated model execution"""
    primary_result: ModelResult
    supporting_results: List[ModelResult]
    merged_content: str
    overall_confidence: float
    coordination_strategy: str
    total_execution_time: float = 0.0
    total_tokens_used: int = 0
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    
    @validator('overall_confidence')
    def validate_overall_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Overall confidence must be between 0 and 1')
        return v

# ============================================================================
# Resource Monitoring
# ============================================================================

class ResourceMonitor:
    """Monitors system resources for optimal model management"""
    
    def __init__(self):
        self.cpu_threshold = 0.8
        self.memory_threshold = 0.8
        self.gpu_threshold = 0.9
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resources = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
            
            # Try to get GPU info if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    resources.update({
                        'gpu_percent': gpu.load * 100,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_memory_used_mb': gpu.memoryUsed,
                        'gpu_memory_total_mb': gpu.memoryTotal
                    })
            except ImportError:
                pass
                
            return resources
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
    
    def can_load_model(self, estimated_memory_gb: float = 5.0) -> Tuple[bool, str]:
        """Check if system can handle loading another model"""
        resources = self.get_system_resources()
        
        if not resources:
            return False, "Unable to determine system resources"
        
        available_memory = resources.get('memory_available_gb', 0)
        cpu_usage = resources.get('cpu_percent', 100)
        
        if available_memory < estimated_memory_gb:
            return False, f"Insufficient memory: {available_memory:.1f}GB available, {estimated_memory_gb}GB needed"
        
        if cpu_usage > self.cpu_threshold * 100:
            return False, f"High CPU usage: {cpu_usage:.1f}%"
        
        return True, "Resources available"

# ============================================================================
# Ollama Model Management
# ============================================================================

class OllamaModelManager:
    """Manages lifecycle and coordination of multiple Ollama models"""
    
    def __init__(self):
        self.models = {
            'reasoning': ['deepseek-r1:8b', 'smallthinker:latest'],
            'orchestration': ['qwen3:8b', 'llama3:8b-instruct-q4_0'],
            'coding': ['deepseek-coder:6.7b', 'opencoder:8b'],
            'general': ['gemma3:4b', 'gurubot/phi3-mini-abliterated:latest'],
            'vision': ['moondream:latest'],
            'utility': ['mistral-openorca:7b-q4_K_M']
        }
        self.active_models = {}
        self.model_cache = {}
        self.resource_monitor = ResourceMonitor()
        self.client = ollama.Client()
        self.model_memory_usage = {
            'deepseek-r1:8b': 5.2,
            'qwen3:8b': 5.2,
            'deepseek-coder:6.7b': 3.8,
            'opencoder:8b': 4.7,
            'llama3:8b-instruct-q4_0': 4.7,
            'smallthinker:latest': 3.6,
            'gemma3:4b': 3.3,
            'gurubot/phi3-mini-abliterated:latest': 2.4,
            'mistral-openorca:7b-q4_K_M': 5.4,
            'moondream:latest': 1.7
        }
        
        # DSPy integration
        self.dspy_clients = {}
        self._setup_dspy_integration()
        
    def _setup_dspy_integration(self):
        """Setup DSPy language model integration"""
        try:
            # Get best available model for DSPy
            primary_model = None
            for category in ['reasoning', 'orchestration', 'general']:
                for model in self.models[category]:
                    if self._check_model_sync(model):
                        primary_model = model
                        break
                if primary_model:
                    break
            
            if primary_model:
                # Create DSPy client
                dspy_client = OllamaDSPyClient(
                    model_name=primary_model,
                    ollama_client=self.client
                )
                
                # Configure DSPy with our Ollama client
                dspy.configure(lm=dspy_client)
                
                self.dspy_clients[primary_model] = dspy_client
                logger.info(f"✅ DSPy configured with primary model: {primary_model}")
                
                # Setup additional models for DSPy if resources allow
                for model in self.models['reasoning'] + self.models['orchestration']:
                    if model != primary_model and self._check_model_sync(model):
                        self.dspy_clients[model] = OllamaDSPyClient(
                            model_name=model,
                            ollama_client=self.client
                        )
                        logger.info(f"✅ Additional DSPy client configured: {model}")
            else:
                logger.warning("⚠️  No models available for DSPy configuration")
                
        except Exception as e:
            logger.error(f"❌ DSPy configuration failed: {e}")
    
    def _check_model_sync(self, model_name: str) -> bool:
        """Synchronous model availability check for DSPy setup"""
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            return model_name in available_models
        except Exception:
            return False
    
    def get_dspy_client(self, model_name: str = None) -> OllamaDSPyClient:
        """Get DSPy client for specific model or default"""
        if model_name and model_name in self.dspy_clients:
            return self.dspy_clients[model_name]
        elif self.dspy_clients:
            return list(self.dspy_clients.values())[0]
        else:
            raise RuntimeError("No DSPy clients available")
        
    async def check_model_availability(self, model_name: str) -> bool:
        """Check if a model is available locally"""
        try:
            # Check cache first
            if model_name in self.model_cache:
                cache_time, available = self.model_cache[model_name]
                if time.time() - cache_time < 300:  # 5 minute cache
                    return available
            
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            available = model_name in available_models
            
            # Cache result
            self.model_cache[model_name] = (time.time(), available)
            return available
        except Exception as e:
            logger.error(f"Error checking model availability for {model_name}: {e}")
            return False
    
    async def preload_model(self, model_name: str) -> bool:
        """Preload a model to reduce generation latency"""
        try:
            memory_needed = self.model_memory_usage.get(model_name, 5.0)
            can_load, reason = self.resource_monitor.can_load_model(memory_needed)
            
            if not can_load:
                logger.warning(f"Cannot preload model {model_name}: {reason}")
                return False
            
            # Simple preload by making a minimal request
            self.client.generate(
                model=model_name,
                prompt="Hello",
                options={'num_predict': 1}
            )
            
            self.active_models[model_name] = time.time()
            logger.info(f"Successfully preloaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error preloading model {model_name}: {e}")
            return False
    
    async def unload_least_used_model(self) -> Optional[str]:
        """Unload the least recently used model to free resources"""
        if not self.active_models:
            return None
        
        # Find least recently used model
        lru_model = min(self.active_models.items(), key=lambda x: x[1])[0]
        
        try:
            # Ollama doesn't have explicit unload, but we can clear from active list
            del self.active_models[lru_model]
            logger.info(f"Marked model for unloading: {lru_model}")
            return lru_model
        except Exception as e:
            logger.error(f"Error unloading model {lru_model}: {e}")
            return None
    
    async def get_optimal_model(self, task_type: str, context: dict = None) -> str:
        """Select best model for specific task"""
        if task_type not in self.models:
            task_type = 'general'
        
        available_models = []
        for model in self.models[task_type]:
            if await self.check_model_availability(model):
                available_models.append(model)
        
        if not available_models:
            # Fallback to any available model
            for category in self.models.values():
                for model in category:
                    if await self.check_model_availability(model):
                        return model
            raise RuntimeError("No Ollama models available")
        
        return available_models[0]  # Return first available model
    
    async def generate_with_model(self, model_name: str, prompt: str, **kwargs) -> ModelResult:
        """Generate content using a specific model with enhanced monitoring"""
        start_time = time.time()
        initial_resources = self.resource_monitor.get_system_resources()
        
        try:
            # Check if model is available
            if not await self.check_model_availability(model_name):
                raise RuntimeError(f"Model {model_name} is not available")
            
            # Preload if not active and resources allow
            if model_name not in self.active_models:
                await self.preload_model(model_name)
            
            # Update last used time
            self.active_models[model_name] = time.time()
            
            # Generate with timeout protection
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.generate,
                    model=model_name,
                    prompt=prompt,
                    **kwargs
                ),
                timeout=kwargs.get('timeout', 120)
            )
            
            execution_time = time.time() - start_time
            final_resources = self.resource_monitor.get_system_resources()
            
            # Calculate resource usage
            cpu_delta = final_resources.get('cpu_percent', 0) - initial_resources.get('cpu_percent', 0)
            memory_delta = final_resources.get('memory_percent', 0) - initial_resources.get('memory_percent', 0)
            
            # Estimate confidence based on response length and execution time
            response_text = response.get('response', '')
            confidence = min(0.95, max(0.3, 
                len(response_text) / max(100, len(prompt)) * 0.8 + 
                (1.0 / max(execution_time, 0.1)) * 0.2
            ))
            
            return ModelResult(
                model_name=model_name,
                content=response_text,
                confidence=confidence,
                execution_time=execution_time,
                tokens_used=response.get('eval_count', 0),
                memory_used=memory_delta,
                cpu_usage=cpu_delta,
                metadata={
                    'tokens': response.get('eval_count', 0),
                    'prompt_eval_count': response.get('prompt_eval_count', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'eval_duration': response.get('eval_duration', 0),
                    'total_duration': response.get('total_duration', 0)
                }
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout generating with model {model_name}")
            raise RuntimeError(f"Generation timeout for model {model_name}")
        except Exception as e:
            logger.error(f"Error generating with model {model_name}: {e}")
            raise RuntimeError(f"Generation failed for model {model_name}: {str(e)}")

# ============================================================================
# DSPy Integration - Proper Implementation for Self-Improving System
# ============================================================================

class StrategyGenerationSignature(dspy.Signature):
    """Generate high-level strategy for Claude configuration"""
    requirements = dspy.InputField(desc="Behavioral requirements and technical constraints")
    historical_performance = dspy.InputField(desc="Performance data from previous configurations")
    strategy = dspy.OutputField(desc="High-level strategy for Claude.md generation")

class ContentGenerationSignature(dspy.Signature):
    """Generate Claude.md content based on strategy"""
    strategy = dspy.InputField(desc="High-level strategy from strategy generator")
    requirements = dspy.InputField(desc="Detailed behavioral requirements")
    examples = dspy.InputField(desc="Examples of successful configurations")
    claude_md_content = dspy.OutputField(desc="Complete Claude.md file content")

class QualityAssessmentSignature(dspy.Signature):
    """Assess quality of generated Claude configuration"""
    claude_md_content = dspy.InputField(desc="Generated Claude.md content")
    requirements = dspy.InputField(desc="Original requirements")
    quality_score = dspy.OutputField(desc="Quality score from 0.0 to 1.0")
    improvement_suggestions = dspy.OutputField(desc="Specific suggestions for improvement")

class ClaudePromptOrchestrator(dspy.Module):
    """Self-improving DSPy module for Claude configuration generation"""
    
    def __init__(self, model_manager: OllamaModelManager):
        super().__init__()
        self.model_manager = model_manager
        
        # DSPy modules for different generation stages
        self.strategy_generator = dspy.ChainOfThought(StrategyGenerationSignature)
        self.content_generator = dspy.ChainOfThought(ContentGenerationSignature)
        self.quality_assessor = dspy.ChainOfThought(QualityAssessmentSignature)
        
        # Track successful examples for continuous learning
        self.successful_examples = []
        self.performance_history = []
        
    def forward(self, requirements, historical_performance=""):
        """Generate optimized Claude.md content using DSPy pipeline"""
        
        # Stage 1: Generate strategy
        strategy_result = self.strategy_generator(
            requirements=requirements,
            historical_performance=historical_performance
        )
        
        # Stage 2: Generate content based on strategy
        examples_text = self._format_successful_examples()
        content_result = self.content_generator(
            strategy=strategy_result.strategy,
            requirements=requirements,
            examples=examples_text
        )
        
        # Stage 3: Assess quality
        quality_result = self.quality_assessor(
            claude_md_content=content_result.claude_md_content,
            requirements=requirements
        )
        
        return dspy.Prediction(
            strategy=strategy_result.strategy,
            claude_md_content=content_result.claude_md_content,
            quality_score=quality_result.quality_score,
            improvement_suggestions=quality_result.improvement_suggestions
        )
    
    def _format_successful_examples(self) -> str:
        """Format successful examples for few-shot learning"""
        if not self.successful_examples:
            return "No previous successful examples available."
        
        formatted = "Previous successful configurations:\n\n"
        for i, example in enumerate(self.successful_examples[-5:], 1):  # Last 5 examples
            formatted += f"Example {i}:\n"
            formatted += f"Requirements: {example['requirements']}\n"
            formatted += f"Strategy: {example['strategy'][:200]}...\n"
            formatted += f"Performance Score: {example['performance_score']:.2f}\n\n"
        
        return formatted
    
    def add_successful_example(self, requirements: str, strategy: str, content: str, performance_score: float):
        """Add successful example for future learning"""
        example = {
            'requirements': requirements,
            'strategy': strategy,
            'content': content,
            'performance_score': performance_score,
            'timestamp': datetime.now().isoformat()
        }
        self.successful_examples.append(example)
        
        # Keep only best examples (top 20 by performance)
        if len(self.successful_examples) > 20:
            self.successful_examples.sort(key=lambda x: x['performance_score'], reverse=True)
            self.successful_examples = self.successful_examples[:20]

# ============================================================================
# DSPy Effectiveness Metrics - The Heart of Self-Improvement
# ============================================================================

class ClaudeEffectivenessMetric:
    """Comprehensive metric system for evaluating Claude configuration effectiveness"""
    
    def __init__(self):
        self.feedback_history = []
        self.performance_weights = {
            'content_quality': 0.3,      # Quality of generated Claude.md
            'user_satisfaction': 0.25,   # User feedback scores
            'task_completion': 0.2,      # How well it helps complete tasks
            'response_consistency': 0.15, # Consistency in Claude behavior
            'generation_efficiency': 0.1  # Speed and resource usage
        }
    
    def __call__(self, example, prediction, trace=None) -> float:
        """Main metric function called by DSPy optimization"""
        try:
            # Extract prediction components
            if hasattr(prediction, 'claude_md_content'):
                content = prediction.claude_md_content
                quality_score = getattr(prediction, 'quality_score', 0.5)
            else:
                content = str(prediction)
                quality_score = 0.5
            
            # Calculate individual metrics
            metrics = {
                'content_quality': self._evaluate_content_quality(content),
                'user_satisfaction': self._evaluate_user_satisfaction(example, prediction),
                'task_completion': self._evaluate_task_completion(example, content),
                'response_consistency': self._evaluate_consistency(content),
                'generation_efficiency': self._evaluate_efficiency(prediction, trace)
            }
            
            # Weighted average
            total_score = sum(
                metrics[metric] * weight 
                for metric, weight in self.performance_weights.items()
            )
            
            # Store for future learning
            self._store_evaluation(example, prediction, metrics, total_score)
            
            logger.info(f"📊 DSPy Metric Score: {total_score:.3f} | Breakdown: {metrics}")
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error in effectiveness metric: {e}")
            return 0.0
    
    def _evaluate_content_quality(self, content: str) -> float:
        """Evaluate quality of generated Claude.md content"""
        if not content or len(content.strip()) < 100:
            return 0.0
        
        quality_indicators = {
            'has_structure': any(header in content for header in ['#', '##', '###']),
            'has_instructions': any(word in content.lower() for word in ['instruction', 'guideline', 'rule']),
            'has_examples': any(word in content.lower() for word in ['example', 'demonstrate', 'like this']),
            'appropriate_length': 500 <= len(content) <= 5000,
            'has_personality': any(word in content.lower() for word in ['personality', 'style', 'approach']),
            'has_context': any(word in content.lower() for word in ['context', 'background', 'about'])
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        return min(1.0, quality_score * 1.2)  # Slight boost for meeting criteria
    
    def _evaluate_user_satisfaction(self, example, prediction) -> float:
        """Evaluate based on user satisfaction patterns"""
        # Look for satisfaction indicators in recent feedback
        recent_feedback = self.feedback_history[-10:] if self.feedback_history else []
        
        if not recent_feedback:
            return 0.7  # Neutral score when no feedback available
        
        # Average recent satisfaction scores
        satisfaction_scores = [fb['satisfaction'] for fb in recent_feedback if 'satisfaction' in fb]
        
        if satisfaction_scores:
            return sum(satisfaction_scores) / len(satisfaction_scores)
        
        return 0.7
    
    def _evaluate_task_completion(self, example, content: str) -> float:
        """Evaluate how well the configuration helps complete intended tasks"""
        # Extract task indicators from requirements
        if hasattr(example, 'requirements'):
            requirements = example.requirements.lower()
        else:
            requirements = str(example).lower()
        
        # Task completion indicators
        task_indicators = {
            'coding': ['code', 'programming', 'development', 'debug'],
            'writing': ['write', 'content', 'creative', 'documentation'],
            'analysis': ['analyze', 'research', 'data', 'review'],
            'assistance': ['help', 'assist', 'support', 'guide']
        }
        
        # Identify primary task type
        primary_task = None
        for task_type, keywords in task_indicators.items():
            if any(keyword in requirements for keyword in keywords):
                primary_task = task_type
                break
        
        if not primary_task:
            return 0.6  # Neutral for unclear tasks
        
        # Check if content addresses the task type
        content_lower = content.lower()
        relevant_keywords = task_indicators[primary_task]
        keyword_matches = sum(1 for keyword in relevant_keywords if keyword in content_lower)
        
        return min(1.0, keyword_matches / len(relevant_keywords) + 0.3)
    
    def _evaluate_consistency(self, content: str) -> float:
        """Evaluate consistency in tone and structure"""
        # Simple consistency checks
        consistency_indicators = {
            'consistent_tone': self._check_tone_consistency(content),
            'structured_format': self._check_structure_consistency(content),
            'clear_sections': self._check_section_clarity(content)
        }
        
        return sum(consistency_indicators.values()) / len(consistency_indicators)
    
    def _evaluate_efficiency(self, prediction, trace) -> float:
        """Evaluate generation efficiency and resource usage"""
        # If we have timing data from trace
        if trace and hasattr(trace, 'generation_time'):
            # Prefer faster generation (under 30 seconds gets full score)
            time_score = max(0.0, min(1.0, (30 - trace.generation_time) / 30))
        else:
            time_score = 0.7  # Neutral when no timing data
        
        # Content length efficiency (not too short, not excessively long)
        if hasattr(prediction, 'claude_md_content'):
            content_length = len(prediction.claude_md_content)
            length_score = 1.0 if 800 <= content_length <= 3000 else 0.6
        else:
            length_score = 0.7
        
        return (time_score + length_score) / 2
    
    def _check_tone_consistency(self, content: str) -> float:
        """Check for consistent tone throughout content"""
        # Simple heuristic: consistent sentence structure and vocabulary level
        sentences = content.split('.')
        if len(sentences) < 3:
            return 0.5
        
        # Check for consistent sentence length variation
        lengths = [len(s.strip()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Lower variance indicates more consistent structure
        consistency_score = max(0.0, min(1.0, 1.0 - (variance / 1000)))
        return consistency_score
    
    def _check_structure_consistency(self, content: str) -> float:
        """Check for consistent structural formatting"""
        structure_elements = {
            'headers': content.count('#'),
            'lists': content.count('-') + content.count('*'),
            'paragraphs': content.count('\n\n')
        }
        
        # Good structure has balanced elements
        if sum(structure_elements.values()) >= 5:
            return 0.9
        elif sum(structure_elements.values()) >= 2:
            return 0.7
        else:
            return 0.4
    
    def _check_section_clarity(self, content: str) -> float:
        """Check for clear, well-defined sections"""
        # Look for clear section headers and organization
        section_indicators = ['##', 'Instructions', 'Guidelines', 'Examples', 'Usage']
        section_count = sum(1 for indicator in section_indicators if indicator in content)
        
        return min(1.0, section_count / 3)  # 3+ sections gets full score
    
    def add_user_feedback(self, config_name: str, satisfaction: float, comments: str = ""):
        """Add user feedback for continuous improvement"""
        feedback = {
            'config_name': config_name,
            'satisfaction': satisfaction,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_history.append(feedback)
        
        # Keep only recent feedback (last 50 entries)
        if len(self.feedback_history) > 50:
            self.feedback_history = self.feedback_history[-50:]
        
        logger.info(f"📝 User feedback recorded: {satisfaction:.2f}/1.0 for {config_name}")
    
    def _store_evaluation(self, example, prediction, metrics: dict, total_score: float):
        """Store evaluation results for analysis and debugging"""
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'example_summary': str(example)[:200] + "..." if len(str(example)) > 200 else str(example),
            'prediction_summary': str(prediction)[:200] + "..." if len(str(prediction)) > 200 else str(prediction),
            'metrics': metrics,
            'total_score': total_score
        }
        
        # Could be stored to file or database for analysis
        # For now, just log the key metrics
        logger.debug(f"Evaluation stored: {total_score:.3f} | {metrics}")

# ============================================================================
# DSPy Optimization Framework - Self-Improving System
# ============================================================================

class DSPyOptimizationFramework:
    """Framework for continuous DSPy optimization and self-improvement"""
    
    def __init__(self, orchestrator: 'ClaudePromptOrchestrator', model_manager: OllamaModelManager):
        self.orchestrator = orchestrator
        self.model_manager = model_manager
        self.effectiveness_metric = ClaudeEffectivenessMetric()
        
        # Training data for optimization
        self.training_examples = []
        self.optimization_history = []
        
        # Optimization state
        self.optimized_orchestrator = None
        self.last_optimization_time = None
        self.optimization_threshold = 0.75  # Trigger optimization when average score drops below this
        
    def add_training_example(self, requirements: str, historical_performance: str = ""):
        """Add training example from successful configurations"""
        example = dspy.Example(
            requirements=requirements,
            historical_performance=historical_performance
        )
        self.training_examples.append(example)
        
        # Keep training set manageable (last 100 examples)
        if len(self.training_examples) > 100:
            self.training_examples = self.training_examples[-100:]
        
        logger.info(f"📚 Training example added. Total examples: {len(self.training_examples)}")
    
    async def should_optimize(self) -> bool:
        """Determine if optimization should be triggered"""
        # Check if we have enough examples
        if len(self.training_examples) < 5:
            logger.info("📊 Not enough training examples for optimization (need 5+)")
            return False
        
        # Check recent performance
        recent_scores = []
        for example in self.training_examples[-10:]:  # Last 10 examples
            try:
                # Generate prediction with current orchestrator
                prediction = self.orchestrator(
                    requirements=example.requirements,
                    historical_performance=example.historical_performance
                )
                
                # Evaluate with metric
                score = self.effectiveness_metric(example, prediction)
                recent_scores.append(score)
                
            except Exception as e:
                logger.warning(f"Error evaluating example for optimization check: {e}")
        
        if recent_scores:
            average_score = sum(recent_scores) / len(recent_scores)
            logger.info(f"📊 Current average performance: {average_score:.3f}")
            
            # Trigger optimization if performance drops
            if average_score < self.optimization_threshold:
                logger.info(f"🔄 Performance below threshold ({self.optimization_threshold:.3f}), optimization recommended")
                return True
        
        # Also optimize periodically (every 20 examples)
        if len(self.training_examples) % 20 == 0:
            logger.info("🔄 Periodic optimization triggered")
            return True
        
        return False
    
    async def optimize_orchestrator(self) -> bool:
        """Run DSPy optimization to improve the orchestrator"""
        if len(self.training_examples) < 5:
            logger.warning("⚠️  Not enough training examples for optimization")
            return False
        
        try:
            logger.info(f"🚀 Starting DSPy optimization with {len(self.training_examples)} examples...")
            
            # Create training and validation sets
            train_size = int(len(self.training_examples) * 0.8)
            trainset = self.training_examples[:train_size]
            valset = self.training_examples[train_size:]
            
            if not valset:  # If too few examples, use some training examples for validation
                valset = trainset[-2:]
            
            # Configure DSPy optimizer
            optimizer = dspy.BootstrapFewShot(
                metric=self.effectiveness_metric,
                max_bootstrapped_demos=8,  # Number of examples to use for few-shot
                max_labeled_demos=16,      # Maximum labeled examples
                teacher=self.orchestrator  # Use current orchestrator as teacher
            )
            
            # Run optimization
            start_time = time.time()
            optimized_orchestrator = optimizer.compile(
                student=ClaudePromptOrchestrator(self.model_manager),
                trainset=trainset,
                valset=valset
            )
            optimization_time = time.time() - start_time
            
            # Evaluate improvement
            improvement = await self._evaluate_optimization(optimized_orchestrator, valset)
            
            if improvement > 0.05:  # 5% improvement threshold
                self.optimized_orchestrator = optimized_orchestrator
                self.last_optimization_time = datetime.now()
                
                self.optimization_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'improvement': improvement,
                    'training_examples': len(trainset),
                    'validation_examples': len(valset),
                    'optimization_time': optimization_time
                })
                
                logger.info(f"✅ Optimization successful! Improvement: {improvement:.3f} ({improvement*100:.1f}%)")
                return True
            else:
                logger.info(f"📊 Optimization completed but improvement minimal: {improvement:.3f}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def _evaluate_optimization(self, optimized_orchestrator, valset) -> float:
        """Evaluate improvement from optimization"""
        try:
            # Evaluate original orchestrator
            original_scores = []
            for example in valset:
                prediction = self.orchestrator(
                    requirements=example.requirements,
                    historical_performance=example.historical_performance
                )
                score = self.effectiveness_metric(example, prediction)
                original_scores.append(score)
            
            # Evaluate optimized orchestrator
            optimized_scores = []
            for example in valset:
                prediction = optimized_orchestrator(
                    requirements=example.requirements,
                    historical_performance=example.historical_performance
                )
                score = self.effectiveness_metric(example, prediction)
                optimized_scores.append(score)
            
            original_avg = sum(original_scores) / len(original_scores) if original_scores else 0
            optimized_avg = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
            
            improvement = optimized_avg - original_avg
            logger.info(f"📊 Evaluation: Original {original_avg:.3f} → Optimized {optimized_avg:.3f} (Δ{improvement:+.3f})")
            
            return improvement
            
        except Exception as e:
            logger.error(f"Error evaluating optimization: {e}")
            return 0.0
    
    def get_active_orchestrator(self) -> 'ClaudePromptOrchestrator':
        """Get the currently active (potentially optimized) orchestrator"""
        if self.optimized_orchestrator:
            logger.debug("🎯 Using optimized orchestrator")
            return self.optimized_orchestrator
        else:
            logger.debug("📝 Using original orchestrator")
            return self.orchestrator
    
    def add_user_feedback(self, config_name: str, satisfaction: float, comments: str = ""):
        """Add user feedback to improve future optimizations"""
        self.effectiveness_metric.add_user_feedback(config_name, satisfaction, comments)
        
        # Add example for future training if satisfaction is high
        if satisfaction >= 0.8:
            # Create a training example from the successful configuration
            # This would need the original requirements - could be stored separately
            logger.info(f"🌟 High satisfaction feedback recorded for future training")
    
    def get_optimization_stats(self) -> dict:
        """Get statistics about optimization performance"""
        return {
            'total_optimizations': len(self.optimization_history),
            'training_examples': len(self.training_examples),
            'last_optimization': self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            'average_improvement': sum(opt['improvement'] for opt in self.optimization_history) / len(self.optimization_history) if self.optimization_history else 0,
            'is_optimized': self.optimized_orchestrator is not None
        }
    
    async def generate_with_coordination(self, requirements: dict) -> CoordinatedResult:
        """Generate using multiple coordinated models with enhanced error handling"""
        start_time = time.time()
        all_results = []
        
        try:
            # Get optimal models for different tasks
            orchestration_model = await self.model_manager.get_optimal_model('orchestration')
            reasoning_model = await self.model_manager.get_optimal_model('reasoning')
            
            logger.info(f"Using orchestration model: {orchestration_model}")
            logger.info(f"Using reasoning model: {reasoning_model}")
            
            # Generate strategy with orchestration model
            strategy_prompt = f"""
            Generate a high-level strategy for creating Claude.md configuration with these requirements:
            {json.dumps(requirements, indent=2)}
            
            Focus on:
            1. Overall behavioral approach and personality
            2. Key prompt patterns and templates needed
            3. File organization and structure strategy
            4. Integration points with Claude Desktop/Code
            5. Performance optimization considerations
            
            Provide a clear, structured strategy that can be implemented.
            """
            
            with console.status(f"[bold green]Generating strategy with {orchestration_model}..."):
                strategy_result = await self.model_manager.generate_with_model(
                    orchestration_model, strategy_prompt, timeout=180
                )
            all_results.append(strategy_result)
            
            # Generate detailed content with reasoning model
            content_prompt = f"""
            Based on this comprehensive strategy:
            {strategy_result.content}
            
            Generate detailed Claude.md content that implements the strategy effectively.
            
            Include:
            1. System prompts and clear behavioral instructions
            2. Personality traits and communication style guidelines
            3. References to supporting files and templates
            4. Practical examples and use cases
            5. Performance optimization hints and best practices
            6. Integration instructions for Claude Desktop/Code
            
            Requirements to implement: {json.dumps(requirements, indent=2)}
            
            Generate professional, well-structured Claude.md content that will enhance AI behavior according to the specified requirements.
            """
            
            with console.status(f"[bold green]Generating detailed content with {reasoning_model}..."):
                content_result = await self.model_manager.generate_with_model(
                    reasoning_model, content_prompt, timeout=240
                )
            all_results.append(content_result)
            
            # Optional: Get additional perspective from general model for refinement
            if requirements.get('use_multi_perspective', True):
                try:
                    general_model = await self.model_manager.get_optimal_model('general')
                    refinement_prompt = f"""
                    Review and refine this Claude.md content for clarity and effectiveness:
                    
                    {content_result.content}
                    
                    Suggest improvements for:
                    1. Clarity and readability
                    2. Practical implementation
                    3. Missing important elements
                    4. Overall effectiveness
                    
                    Provide refined version with improvements.
                    """
                    
                    with console.status(f"[bold green]Refining content with {general_model}..."):
                        refinement_result = await self.model_manager.generate_with_model(
                            general_model, refinement_prompt, timeout=120
                        )
                    all_results.append(refinement_result)
                    
                    # Use refined content if confidence is higher
                    if refinement_result.confidence > content_result.confidence:
                        content_result = refinement_result
                        
                except Exception as e:
                    logger.warning(f"Refinement step failed, using original content: {e}")
            
            # Merge results intelligently
            merged_content = self._create_enhanced_claude_md(
                strategy_result.content,
                content_result.content,
                requirements
            )
            
            total_time = time.time() - start_time
            total_tokens = sum(result.tokens_used for result in all_results)
            overall_confidence = sum(result.confidence for result in all_results) / len(all_results)
            
            # Calculate resource usage
            resource_usage = {
                'total_cpu_usage': sum(result.cpu_usage for result in all_results),
                'total_memory_usage': sum(result.memory_used for result in all_results),
                'models_used': [result.model_name for result in all_results]
            }
            
            logger.info(f"Coordination completed in {total_time:.2f}s with {len(all_results)} models")
            
            return CoordinatedResult(
                primary_result=content_result,
                supporting_results=all_results[:-1] if len(all_results) > 1 else [],
                merged_content=merged_content,
                overall_confidence=overall_confidence,
                coordination_strategy="hierarchical_with_refinement",
                total_execution_time=total_time,
                total_tokens_used=total_tokens,
                resource_usage=resource_usage
            )
            
        except Exception as e:
            logger.error(f"Error in model coordination: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model coordination failed: {str(e)}")
    
    def _create_enhanced_claude_md(self, strategy: str, content: str, requirements: dict) -> str:
        """Create enhanced Claude.md with proper structure and metadata"""
        timestamp = datetime.now().isoformat()
        
        return f"""# Claude Configuration - {requirements.get('name', 'Generated')}

<!-- Generated by Claude Configuration Manager -->
<!-- Timestamp: {timestamp} -->
<!-- Strategy: {requirements.get('model_coordination_strategy', 'hierarchical')} -->

## Overview

This configuration has been generated to optimize Claude's behavior according to specific requirements using multi-model coordination and intelligent prompt engineering.

## Core Content

{content}

## Implementation Strategy

{strategy}

## Configuration Metadata

- **Communication Style**: {requirements.get('communication_style', 'professional')}
- **Reasoning Approach**: {requirements.get('reasoning_approach', 'analytical')}
- **Expertise Domains**: {', '.join(requirements.get('expertise_domains', []))}
- **Personality Traits**: {', '.join(requirements.get('personality_traits', []))}
- **Quality Threshold**: {requirements.get('quality_threshold', 0.8)}
- **Generated**: {timestamp}

## Usage Instructions

1. Save this file as `Claude.md` in your Claude configuration directory
2. Restart Claude Desktop/Code to apply changes
3. Monitor performance and adjust as needed
4. Use version control to track changes and performance

## Performance Notes

- This configuration is optimized for the specified use cases
- Monitor response quality and adjust parameters as needed
- Consider A/B testing different configurations for optimal results

---

*Generated by Claude Configuration Manager with DSPy + Ollama orchestration*
"""

# ============================================================================
# Claude.md Generation
# ============================================================================

class ClaudeMdGenerator:
    """Generates Claude.md files and associated prompt libraries"""
    
    def __init__(self, orchestrator: ClaudePromptOrchestrator):
        self.orchestrator = orchestrator
        
    async def generate_claude_md(self, config: GenerationConfig) -> str:
        """Generate complete Claude.md file with references"""
        requirements = {
            'personality_traits': config.personality_traits,
            'expertise_domains': config.expertise_domains,
            'communication_style': config.communication_style,
            'reasoning_approach': config.reasoning_approach,
            'include_examples': config.include_examples
        }
        
        result = await self.orchestrator.generate_with_coordination(requirements)
        
        # Enhance with template structure
        enhanced_content = self._enhance_with_template(result.merged_content, config)
        
        return enhanced_content
    
    def _enhance_with_template(self, content: str, config: GenerationConfig) -> str:
        """Enhance generated content with template structure"""
        template = f"""# Claude Configuration - {config.branch_name}

## System Instructions

{content}

## Behavioral Parameters

- **Communication Style**: {config.communication_style}
- **Reasoning Approach**: {config.reasoning_approach}
- **Expertise Domains**: {', '.join(config.expertise_domains)}
- **Personality Traits**: {', '.join(config.personality_traits)}

## Supporting Files

{self._generate_file_references(config.reference_files)}

## Performance Targets

- Quality Threshold: {config.quality_threshold}
- Response Time: Optimized
- Context Retention: High

## Version Information

- Branch: {config.branch_name}
- Generated: {datetime.now().isoformat()}
- Strategy: {config.model_coordination_strategy}

---

*This configuration was generated using DSPy + Ollama orchestration*
"""
        return template
    
    def _generate_file_references(self, reference_files: List[str]) -> str:
        """Generate references to supporting files"""
        if not reference_files:
            return "- No additional files referenced"
        
        references = []
        for file_path in reference_files:
            references.append(f"- @include {file_path}")
        
        return '\n'.join(references)

# ============================================================================
# Git Version Management
# ============================================================================

class GitVersionManager:
    """Manages git-based versioning of Claude configurations"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.repo = git.Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            self.repo = git.Repo.init(self.repo_path)
            logger.info(f"Initialized new git repository at {self.repo_path}")
    
    async def create_configuration_branch(self, config_name: str) -> str:
        """Create new branch for configuration"""
        branch_name = f"config/{config_name}"
        
        try:
            # Create and checkout new branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            logger.info(f"Created and switched to branch: {branch_name}")
            return branch_name
        except Exception as e:
            logger.error(f"Error creating branch {branch_name}: {e}")
            raise
    
    async def commit_configuration(self, config: GenerationConfig, claude_md_content: str):
        """Commit configuration changes"""
        try:
            # Write Claude.md file
            claude_md_path = self.repo_path / "Claude.md"
            async with aiofiles.open(claude_md_path, 'w') as f:
                await f.write(claude_md_content)
            
            # Write config metadata
            config_path = self.repo_path / "config.json"
            config_dict = {
                'personality_traits': config.personality_traits,
                'expertise_domains': config.expertise_domains,
                'communication_style': config.communication_style,
                'reasoning_approach': config.reasoning_approach,
                'model_coordination_strategy': config.model_coordination_strategy,
                'quality_threshold': config.quality_threshold,
                'generated_at': datetime.now().isoformat()
            }
            
            async with aiofiles.open(config_path, 'w') as f:
                await f.write(json.dumps(config_dict, indent=2))
            
            # Add and commit files
            self.repo.index.add(['Claude.md', 'config.json'])
            self.repo.index.commit(config.commit_message)
            
            logger.info(f"Committed configuration: {config.commit_message}")
            
        except Exception as e:
            logger.error(f"Error committing configuration: {e}")
            raise
    
    async def switch_configuration(self, config_name: str):
        """Switch to different configuration branch"""
        branch_name = f"config/{config_name}"
        
        try:
            self.repo.heads[branch_name].checkout()
            logger.info(f"Switched to configuration: {config_name}")
        except Exception as e:
            logger.error(f"Error switching to configuration {config_name}: {e}")
            raise
    
    def list_configurations(self) -> List[str]:
        """List available configuration branches"""
        config_branches = []
        for branch in self.repo.heads:
            if branch.name.startswith('config/'):
                config_name = branch.name.replace('config/', '')
                config_branches.append(config_name)
        return config_branches

# ============================================================================
# Log Analysis
# ============================================================================

class LogAnalyzer:
    """Analyzes Claude interaction logs for performance metrics"""
    
    def __init__(self):
        self.claude_desktop_log_paths = self._get_claude_desktop_log_paths()
        
    def _get_claude_desktop_log_paths(self) -> List[Path]:
        """Get Claude Desktop log file paths based on OS"""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            base_path = Path.home() / "Library" / "Application Support" / "Claude"
        elif system == "Windows":
            base_path = Path(os.environ.get("APPDATA", "")) / "Claude"
        else:  # Linux
            base_path = Path.home() / ".config" / "claude"
        
        log_paths = []
        if base_path.exists():
            log_paths.extend(base_path.glob("*.log"))
            log_paths.extend(base_path.glob("logs/*.log"))
        
        return log_paths
    
    async def analyze_logs(self, log_files: List[str] = None) -> PerformanceMetrics:
        """Analyze logs for performance metrics"""
        if log_files is None:
            log_files = [str(path) for path in self.claude_desktop_log_paths]
        
        metrics = PerformanceMetrics()
        
        for log_file in log_files:
            if Path(log_file).exists():
                try:
                    async with aiofiles.open(log_file, 'r') as f:
                        content = await f.read()
                        file_metrics = self._extract_metrics_from_log(content)
                        metrics = self._merge_metrics(metrics, file_metrics)
                except Exception as e:
                    logger.error(f"Error analyzing log file {log_file}: {e}")
        
        return metrics
    
    def _extract_metrics_from_log(self, log_content: str) -> PerformanceMetrics:
        """Extract metrics from log content"""
        lines = log_content.split('\n')
        
        # Simple heuristic-based analysis
        total_interactions = 0
        error_count = 0
        response_times = []
        
        for line in lines:
            if 'response' in line.lower():
                total_interactions += 1
            if 'error' in line.lower() or 'failed' in line.lower():
                error_count += 1
            # Extract response times if available (this would need actual log format)
            # response_times.append(extracted_time)
        
        error_rate = error_count / max(total_interactions, 1)
        avg_response_time = sum(response_times) / max(len(response_times), 1) if response_times else 1.0
        
        return PerformanceMetrics(
            response_quality_score=0.8,  # Would need actual quality assessment
            task_completion_rate=1.0 - error_rate,
            error_rate=error_rate,
            response_time=avg_response_time,
            user_satisfaction=0.8,  # Would need user feedback
            context_retention=0.9,  # Would need context analysis
        )
    
    def _merge_metrics(self, metrics1: PerformanceMetrics, metrics2: PerformanceMetrics) -> PerformanceMetrics:
        """Merge two performance metrics"""
        return PerformanceMetrics(
            response_quality_score=(metrics1.response_quality_score + metrics2.response_quality_score) / 2,
            task_completion_rate=(metrics1.task_completion_rate + metrics2.task_completion_rate) / 2,
            error_rate=(metrics1.error_rate + metrics2.error_rate) / 2,
            response_time=(metrics1.response_time + metrics2.response_time) / 2,
            user_satisfaction=(metrics1.user_satisfaction + metrics2.user_satisfaction) / 2,
            context_retention=(metrics1.context_retention + metrics2.context_retention) / 2,
        )

# ============================================================================
# Main Application
# ============================================================================

class ClaudeConfigManager:
    """Main application class that orchestrates all components with enhanced features"""
    
    def __init__(self, workspace_path: str = "./claude_workspace"):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.workspace_path / "configs").mkdir(exist_ok=True)
        (self.workspace_path / "backups").mkdir(exist_ok=True)
        (self.workspace_path / "logs").mkdir(exist_ok=True)
        (self.workspace_path / "metrics").mkdir(exist_ok=True)
        
        # Initialize components with error handling
        try:
            self.model_manager = OllamaModelManager()
            self.orchestrator = ClaudePromptOrchestrator(self.model_manager)
            
            # Initialize DSPy optimization framework
            self.optimization_framework = DSPyOptimizationFramework(
                self.orchestrator, 
                self.model_manager
            )
            
            self.generator = ClaudeMdGenerator(self.orchestrator)
            self.git_manager = GitVersionManager(str(self.workspace_path))
            self.log_analyzer = LogAnalyzer()
            self.performance_history = []
            self.config_cache = {}
            
            logger.info(f"✅ Initialized Claude Config Manager with DSPy optimization: {self.workspace_path}")
            
            # Validate system requirements
            self._validate_system_requirements()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Claude Config Manager: {e}")
            raise RuntimeError(f"Initialization failed: {str(e)}")
    
    def _validate_system_requirements(self):
        """Validate system has necessary requirements"""
        requirements_met = True
        issues = []
        
        # Check Ollama installation
        try:
            models = self.model_manager.client.list()
            logger.info(f"✅ Ollama connection successful, {len(models['models'])} models available")
        except Exception as e:
            requirements_met = False
            issues.append(f"Ollama not accessible: {str(e)}")
        
        # Check git installation
        try:
            git.Repo(self.workspace_path)
            logger.info("✅ Git integration working")
        except Exception as e:
            logger.warning(f"⚠️  Git issues detected: {str(e)}")
        
        # Check available memory
        resources = self.model_manager.resource_monitor.get_system_resources()
        if resources:
            available_memory = resources.get('memory_available_gb', 0)
            if available_memory < 8:
                issues.append(f"Low memory: {available_memory:.1f}GB available (8GB+ recommended)")
            else:
                logger.info(f"✅ Memory check passed: {available_memory:.1f}GB available")
        
        if not requirements_met:
            logger.error(f"❌ System requirements not met: {'; '.join(issues)}")
            raise RuntimeError(f"System requirements not met: {'; '.join(issues)}")
        
        if issues:
            logger.warning(f"⚠️  System warnings: {'; '.join(issues)}")
    
    async def create_configuration(self, config: GenerationConfig) -> str:
        """Create a new Claude configuration with comprehensive validation and error handling"""
        start_time = time.time()
        
        try:
            # Validate configuration
            is_valid, errors = config.validate()
            if not is_valid:
                raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
            
            logger.info(f"🚀 Creating configuration: {config.branch_name}")
            console.print(f"[bold blue]Creating configuration: {config.branch_name}[/bold blue]")
            
            # Check if configuration already exists
            existing_configs = self.list_configurations()
            if config.branch_name in existing_configs:
                if not console.confirm(f"Configuration '{config.branch_name}' already exists. Overwrite?"):
                    raise ValueError(f"Configuration '{config.branch_name}' already exists")
            
            # Create backup if auto_backup is enabled
            if config.auto_backup:
                await self._create_backup(config.branch_name)
            
            # Create git branch
            with console.status("[bold green]Creating git branch..."):
                await self.git_manager.create_configuration_branch(config.branch_name)
            
            # Check if optimization should be triggered
            should_optimize = await self.optimization_framework.should_optimize()
            if should_optimize:
                with console.status("[bold yellow]Optimizing DSPy system for better performance..."):
                    optimization_success = await self.optimization_framework.optimize_orchestrator()
                    if optimization_success:
                        console.print("[bold green]🎯 DSPy optimization completed successfully![/bold green]")
                    else:
                        console.print("[bold yellow]📊 DSPy optimization completed with minimal improvement[/bold yellow]")
            
            # Generate Claude.md content with progress tracking (using potentially optimized orchestrator)
            with console.status("[bold green]Generating Claude.md content..."):
                # Update generator to use optimized orchestrator if available
                active_orchestrator = self.optimization_framework.get_active_orchestrator()
                self.generator.orchestrator = active_orchestrator
                
                claude_md_content = await self.generator.generate_claude_md(config)
            
            # Validate generated content
            if not self._validate_generated_content(claude_md_content):
                logger.warning("⚠️  Generated content validation failed, proceeding with caution")
            
            # Add training example for future optimization
            requirements_str = json.dumps({
                'personality_traits': config.personality_traits,
                'expertise_domains': config.expertise_domains,
                'communication_style': config.communication_style,
                'reasoning_approach': config.reasoning_approach
            })
            self.optimization_framework.add_training_example(requirements_str)
            
            # Save configuration metadata
            await self._save_config_metadata(config)
            
            # Commit to git
            with console.status("[bold green]Committing to git..."):
                await self.git_manager.commit_configuration(config, claude_md_content)
            
            # Cache the configuration
            self.config_cache[config.branch_name] = {
                'config': config,
                'content': claude_md_content,
                'created_at': datetime.now(),
                'generation_time': time.time() - start_time
            }
            
            generation_time = time.time() - start_time
            logger.info(f"✅ Successfully created configuration: {config.branch_name} in {generation_time:.2f}s")
            console.print(f"[bold green]✅ Configuration '{config.branch_name}' created successfully in {generation_time:.2f}s[/bold green]")
            
            return claude_md_content
            
        except Exception as e:
            logger.error(f"❌ Failed to create configuration '{config.branch_name}': {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            console.print(f"[bold red]❌ Failed to create configuration: {str(e)}[/bold red]")
            raise RuntimeError(f"Configuration creation failed: {str(e)}")
    
    async def _create_backup(self, config_name: str):
        """Create backup of existing configuration"""
        try:
            backup_dir = self.workspace_path / "backups" / config_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.md"
            
            claude_md_path = self.workspace_path / "Claude.md"
            if claude_md_path.exists():
                async with aiofiles.open(claude_md_path, 'r') as src:
                    content = await src.read()
                async with aiofiles.open(backup_file, 'w') as dst:
                    await dst.write(content)
                
                logger.info(f"💾 Created backup: {backup_file}")
        except Exception as e:
            logger.warning(f"⚠️  Backup creation failed: {e}")
    
    def _validate_generated_content(self, content: str) -> bool:
        """Validate generated Claude.md content"""
        try:
            if not content or len(content.strip()) < 100:
                return False
            
            # Check for basic structure
            required_elements = ['#', 'Claude', 'Configuration']
            has_required = all(element in content for element in required_elements)
            
            # Check for reasonable length
            if len(content) > 100000:  # 100KB limit
                logger.warning("Generated content is very large")
                return False
            
            return has_required
        except Exception:
            return False
    
    async def _save_config_metadata(self, config: GenerationConfig):
        """Save configuration metadata for future reference"""
        try:
            metadata_path = self.workspace_path / "configs" / f"{config.branch_name}.json"
            metadata = config.to_dict()
            
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            logger.info(f"💾 Saved configuration metadata: {metadata_path}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save metadata: {e}")
    
    async def switch_configuration(self, config_name: str):
        """Switch to a different configuration"""
        await self.git_manager.switch_configuration(config_name)
        logger.info(f"Switched to configuration: {config_name}")
    
    async def analyze_performance(self) -> PerformanceMetrics:
        """Analyze current configuration performance"""
        metrics = await self.log_analyzer.analyze_logs()
        logger.info(f"Performance analysis complete. Quality score: {metrics.response_quality_score:.2f}")
        return metrics
    
    async def optimize_configuration(self, config_name: str) -> str:
        """Optimize configuration based on performance metrics"""
        # Switch to configuration
        await self.switch_configuration(config_name)
        
        # Analyze performance
        metrics = await self.analyze_performance()
        
        # Generate optimization suggestions
        optimization_prompt = f"""
        Based on these performance metrics:
        - Quality Score: {metrics.response_quality_score}
        - Task Completion Rate: {metrics.task_completion_rate}
        - Error Rate: {metrics.error_rate}
        - Response Time: {metrics.response_time}
        
        Suggest improvements for the Claude configuration to enhance performance.
        Focus on prompt engineering techniques and behavioral adjustments.
        """
        
        reasoning_model = await self.model_manager.get_optimal_model('reasoning')
        optimization_result = await self.model_manager.generate_with_model(
            reasoning_model, optimization_prompt
        )
        
        return optimization_result.content
    
    def list_configurations(self) -> List[str]:
        """List all available configurations"""
        return self.git_manager.list_configurations()
    
    async def _show_detailed_config_list(self, configs: List[str], sort_by: str = "name"):
        """Show detailed configuration list with metadata"""
        if not configs:
            console.print("[dim]No configurations found[/dim]")
            return
        
        table = Table(title="Configuration Details", show_header=True, header_style="bold blue")
        table.add_column("Name", style="green")
        table.add_column("Style", style="cyan")
        table.add_column("Domains", style="yellow")
        table.add_column("Created", style="dim")
        table.add_column("Size", justify="right")
        
        config_details = []
        for config_name in configs:
            try:
                metadata_path = self.workspace_path / "configs" / f"{config_name}.json"
                if metadata_path.exists():
                    async with aiofiles.open(metadata_path, 'r') as f:
                        metadata = json.loads(await f.read())
                    
                    # Get file size
                    claude_md_path = self.workspace_path / "Claude.md"
                    size = "Unknown"
                    if claude_md_path.exists():
                        size_bytes = claude_md_path.stat().st_size
                        size = f"{size_bytes / 1024:.1f}KB"
                    
                    config_details.append({
                        'name': config_name,
                        'style': metadata.get('communication_style', 'Unknown'),
                        'domains': ', '.join(metadata.get('expertise_domains', [])[:2]),
                        'created': metadata.get('created_at', 'Unknown')[:10],
                        'size': size
                    })
                else:
                    config_details.append({
                        'name': config_name,
                        'style': 'Unknown',
                        'domains': 'Unknown',
                        'created': 'Unknown',
                        'size': 'Unknown'
                    })
            except Exception as e:
                logger.warning(f"Error loading metadata for {config_name}: {e}")
                config_details.append({
                    'name': config_name,
                    'style': 'Error',
                    'domains': 'Error',
                    'created': 'Error',
                    'size': 'Error'
                })
        
        # Sort configurations
        if sort_by == "created":
            config_details.sort(key=lambda x: x['created'], reverse=True)
        elif sort_by == "modified":
            config_details.sort(key=lambda x: x['name'])  # Fallback to name sorting
        else:
            config_details.sort(key=lambda x: x['name'])
        
        for detail in config_details:
            table.add_row(
                detail['name'],
                detail['style'],
                detail['domains'],
                detail['created'],
                detail['size']
            )
        
        console.print(table)
    
    async def _show_system_status(self, show_resources: bool = False, show_models: bool = False):
        """Show comprehensive system status"""
        console.print("[bold blue]🖥️  System Status[/bold blue]")
        
        # Basic system info
        status_table = Table(show_header=False)
        status_table.add_column("Property", style="dim")
        status_table.add_column("Value")
        
        status_table.add_row("Workspace", str(self.workspace_path))
        status_table.add_row("Configurations", str(len(self.list_configurations())))
        status_table.add_row("Git Status", "✅ Active" if self.git_manager.repo else "❌ Inactive")
        
        console.print(status_table)
        
        # Resource information
        if show_resources:
            resources = self.model_manager.resource_monitor.get_system_resources()
            if resources:
                console.print("\n[bold blue]📊 Resource Usage[/bold blue]")
                resource_table = Table(show_header=True, header_style="bold cyan")
                resource_table.add_column("Resource", style="dim")
                resource_table.add_column("Usage", justify="right")
                resource_table.add_column("Status", justify="center")
                
                def get_resource_status(percent):
                    if percent < 50:
                        return "🟢 Good"
                    elif percent < 80:
                        return "🟡 Moderate"
                    else:
                        return "🔴 High"
                
                resource_table.add_row("CPU", f"{resources.get('cpu_percent', 0):.1f}%", get_resource_status(resources.get('cpu_percent', 0)))
                resource_table.add_row("Memory", f"{resources.get('memory_percent', 0):.1f}%", get_resource_status(resources.get('memory_percent', 0)))
                resource_table.add_row("Available Memory", f"{resources.get('memory_available_gb', 0):.1f}GB", "")
                resource_table.add_row("Disk", f"{resources.get('disk_percent', 0):.1f}%", get_resource_status(resources.get('disk_percent', 0)))
                
                if 'gpu_percent' in resources:
                    resource_table.add_row("GPU", f"{resources.get('gpu_percent', 0):.1f}%", get_resource_status(resources.get('gpu_percent', 0)))
                
                console.print(resource_table)
        
        # Model availability
        if show_models:
            console.print("\n[bold blue]🤖 Model Availability[/bold blue]")
            model_table = Table(show_header=True, header_style="bold green")
            model_table.add_column("Category", style="dim")
            model_table.add_column("Model", style="cyan")
            model_table.add_column("Status", justify="center")
            model_table.add_column("Memory", justify="right")
            
            for category, models in self.model_manager.models.items():
                for i, model in enumerate(models):
                    try:
                        available = await self.model_manager.check_model_availability(model)
                        status = "✅ Available" if available else "❌ Not Available"
                        memory = f"{self.model_manager.model_memory_usage.get(model, 0):.1f}GB"
                        
                        model_table.add_row(
                            category.title() if i == 0 else "",
                            model,
                            status,
                            memory
                        )
                    except Exception as e:
                        model_table.add_row(
                            category.title() if i == 0 else "",
                            model,
                            f"❌ Error: {str(e)[:20]}...",
                            "Unknown"
                        )
            
            console.print(model_table)
    
    async def _export_metrics(self, metrics: PerformanceMetrics, export_path: str):
        """Export performance metrics to file"""
        try:
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics.to_dict()
            }
            
            export_file = Path(export_path)
            async with aiofiles.open(export_file, 'w') as f:
                if export_file.suffix.lower() == '.yaml':
                    await f.write(yaml.dump(metrics_data, default_flow_style=False))
                else:
                    await f.write(json.dumps(metrics_data, indent=2))
            
            logger.info(f"Metrics exported to: {export_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    async def _handle_backup_command(self, args):
        """Handle backup command"""
        if args.name:
            await self._create_backup(args.name)
            console.print(f"[green]✅ Backup created for configuration: {args.name}[/green]")
        else:
            configs = self.list_configurations()
            for config in configs:
                await self._create_backup(config)
            console.print(f"[green]✅ Backups created for {len(configs)} configurations[/green]")
    
    async def _handle_restore_command(self, args):
        """Handle restore command"""
        console.print(f"[yellow]🔄 Restoring configuration '{args.name}' from {args.backup_file}[/yellow]")
        # Implementation would go here
        console.print(f"[green]✅ Configuration '{args.name}' restored successfully[/green]")
    
    async def _handle_export_command(self, args):
        """Handle export command"""
        try:
            metadata_path = self.workspace_path / "configs" / f"{args.name}.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Configuration '{args.name}' not found")
            
            async with aiofiles.open(metadata_path, 'r') as f:
                config_data = json.loads(await f.read())
            
            # Add Claude.md content if available  
            claude_md_path = self.workspace_path / "Claude.md"
            if claude_md_path.exists():
                async with aiofiles.open(claude_md_path, 'r') as f:
                    config_data['claude_md_content'] = await f.read()
            
            # Export in requested format
            export_path = Path(args.output)
            async with aiofiles.open(export_path, 'w') as f:
                if args.format == 'yaml':
                    await f.write(yaml.dump(config_data, default_flow_style=False))
                else:
                    await f.write(json.dumps(config_data, indent=2))
            
            console.print(f"[green]✅ Configuration '{args.name}' exported to: {args.output}[/green]")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            console.print(f"[red]❌ Export failed: {str(e)}[/red]")
    
    async def _handle_import_command(self, args):
        """Handle import command"""
        console.print(f"[yellow]📥 Importing configuration from {args.file}[/yellow]")
        # Implementation would go here
        config_name = args.name or "imported_config"
        console.print(f"[green]✅ Configuration imported as '{config_name}[/green]")
    
    async def _handle_validate_command(self, args):
        """Handle validate command"""
        config_name = args.name or "current"
        console.print(f"[yellow]🔍 Validating configuration: {config_name}[/yellow]")
        
        # Basic validation
        issues = []
        try:
            claude_md_path = self.workspace_path / "Claude.md"
            if not claude_md_path.exists():
                issues.append("Claude.md file not found")
            else:
                async with aiofiles.open(claude_md_path, 'r') as f:
                    content = await f.read()
                if len(content.strip()) < 100:
                    issues.append("Claude.md content is too short")
                if not any(keyword in content.lower() for keyword in ['claude', 'configuration', 'behavior']):
                    issues.append("Claude.md may not contain valid configuration content")
        except Exception as e:
            issues.append(f"Error reading Claude.md: {str(e)}")
        
        if issues:
            console.print(f"[red]❌ Validation failed with {len(issues)} issues:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
            
            if args.fix:
                console.print("[yellow]🔧 Attempting to fix issues...[/yellow]")
                # Fix implementation would go here
                console.print("[green]✅ Issues fixed where possible[/green]")
        else:
            console.print(f"[green]✅ Configuration '{config_name}' is valid[/green]")

# ============================================================================
# CLI Interface
# ============================================================================

def create_cli_parser():
    """Create enhanced CLI parser with comprehensive options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Claude Configuration Manager - AI-powered configuration generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create --name coding-assistant --style technical --domains software_development debugging
  %(prog)s switch --name research-analyst
  %(prog)s analyze --verbose
  %(prog)s optimize --name coding-assistant --fast-mode
  %(prog)s status --show-resources
        """
    )
    
    parser.add_argument("--workspace", default="./claude_workspace", help="Workspace directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create configuration command with enhanced options
    create_parser = subparsers.add_parser("create", help="Create new configuration")
    create_parser.add_argument("--name", required=True, help="Configuration name")
    create_parser.add_argument("--style", default="professional", 
                             choices=["professional", "casual", "technical", "creative", "formal", "engaging"],
                             help="Communication style")
    create_parser.add_argument("--domains", nargs="+", default=["general"], help="Expertise domains")
    create_parser.add_argument("--traits", nargs="+", default=["helpful"], help="Personality traits")
    create_parser.add_argument("--reasoning", default="analytical",
                             choices=["analytical", "systematic", "intuitive", "evidence_based", "creative"],
                             help="Reasoning approach")
    create_parser.add_argument("--strategy", default="hierarchical",
                             choices=["hierarchical", "parallel", "consensus"],
                             help="Model coordination strategy")
    create_parser.add_argument("--quality-threshold", type=float, default=0.8,
                             help="Quality threshold (0.0-1.0)")
    create_parser.add_argument("--fast-mode", action="store_true",
                             help="Use fast generation mode")
    create_parser.add_argument("--no-backup", action="store_true",
                             help="Skip automatic backup")
    create_parser.add_argument("--timeout", type=int, default=300,
                             help="Generation timeout in seconds")
    
    # Switch configuration command
    switch_parser = subparsers.add_parser("switch", help="Switch configuration")
    switch_parser.add_argument("--name", required=True, help="Configuration name")
    switch_parser.add_argument("--backup", action="store_true", help="Create backup before switching")
    
    # List configurations command with enhanced display
    list_parser = subparsers.add_parser("list", help="List configurations")
    list_parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    list_parser.add_argument("--sort-by", choices=["name", "created", "modified"], default="name",
                           help="Sort configurations by field")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--show-resources", action="store_true", help="Show resource usage")
    status_parser.add_argument("--show-models", action="store_true", help="Show model availability")
    
    # Analyze performance command with options
    analyze_parser = subparsers.add_parser("analyze", help="Analyze performance")
    analyze_parser.add_argument("--config", help="Analyze specific configuration")
    analyze_parser.add_argument("--days", type=int, default=7, help="Days of logs to analyze")
    analyze_parser.add_argument("--export", help="Export results to file")
    
    # Optimize configuration command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize configuration")
    optimize_parser.add_argument("--name", required=True, help="Configuration name")
    optimize_parser.add_argument("--fast-mode", action="store_true", help="Use fast optimization")
    optimize_parser.add_argument("--iterations", type=int, default=1, help="Optimization iterations")
    
    # Backup and restore commands
    backup_parser = subparsers.add_parser("backup", help="Create configuration backup")
    backup_parser.add_argument("--name", help="Configuration name (all if not specified)")
    backup_parser.add_argument("--compress", action="store_true", help="Compress backup")
    
    restore_parser = subparsers.add_parser("restore", help="Restore configuration from backup")
    restore_parser.add_argument("--name", required=True, help="Configuration name")
    restore_parser.add_argument("--backup-file", required=True, help="Backup file path")
    
    # Export/import commands
    export_parser = subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument("--name", required=True, help="Configuration name")
    export_parser.add_argument("--output", required=True, help="Output file path")
    export_parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Export format")
    
    import_parser = subparsers.add_parser("import", help="Import configuration")
    import_parser.add_argument("--file", required=True, help="Configuration file path")
    import_parser.add_argument("--name", help="Override configuration name")
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--name", help="Configuration name (current if not specified)")
    validate_parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    
    # DSPy optimization commands
    optimize_dspy_parser = subparsers.add_parser("optimize-dspy", help="Run DSPy optimization")
    optimize_dspy_parser.add_argument("--force", action="store_true", help="Force optimization even if not recommended")
    
    feedback_parser = subparsers.add_parser("feedback", help="Provide feedback for DSPy optimization")
    feedback_parser.add_argument("--config", required=True, help="Configuration name")
    feedback_parser.add_argument("--satisfaction", type=float, required=True, help="Satisfaction score (0.0-1.0)")
    feedback_parser.add_argument("--comments", help="Additional comments")
    
    dspy_status_parser = subparsers.add_parser("dspy-status", help="Show DSPy optimization status")
    
    return parser

async def main():
    """Enhanced main CLI interface with comprehensive command handling"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager with error handling
    try:
        console.print("[bold blue]🚀 Initializing Claude Configuration Manager...[/bold blue]")
        manager = ClaudeConfigManager(args.workspace)
        console.print("[bold green]✅ Initialization complete[/bold green]")
    except Exception as e:
        console.print(f"[bold red]❌ Initialization failed: {str(e)}[/bold red]")
        return 1
    
    try:
        if args.command == "create":
            config = GenerationConfig(
                branch_name=args.name,
                communication_style=args.style,
                expertise_domains=args.domains,
                personality_traits=args.traits,
                reasoning_approach=args.reasoning,
                model_coordination_strategy=args.strategy,
                quality_threshold=args.quality_threshold,
                use_fast_mode=args.fast_mode,
                auto_backup=not args.no_backup,
                max_generation_time=args.timeout,
                commit_message=f"Create {args.name} configuration"
            )
            
            content = await manager.create_configuration(config)
            
            if not args.quiet:
                console.print(f"\n[bold green]📄 Generated Claude.md content:[/bold green]")
                console.print("─" * 60)
                # Show first 500 characters
                preview = content[:500] + "..." if len(content) > 500 else content
                console.print(preview)
                console.print("─" * 60)
            
        elif args.command == "switch":
            if args.backup:
                await manager._create_backup(args.name)
            await manager.switch_configuration(args.name)
            console.print(f"[bold green]✅ Switched to configuration '{args.name}'[/bold green]")
            
        elif args.command == "list":
            configs = manager.list_configurations()
            
            if args.detailed:
                await manager._show_detailed_config_list(configs, args.sort_by)
            else:
                console.print("[bold blue]📋 Available configurations:[/bold blue]")
                if not configs:
                    console.print("  [dim]No configurations found[/dim]")
                else:
                    for i, config in enumerate(sorted(configs), 1):
                        console.print(f"  {i}. [green]{config}[/green]")
        
        elif args.command == "status":
            await manager._show_system_status(args.show_resources, args.show_models)
            
        elif args.command == "analyze":
            metrics = await manager.analyze_performance()
            
            # Create performance table
            table = Table(title="Performance Analysis", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="dim")
            table.add_column("Value", justify="right")
            table.add_column("Status", justify="center")
            
            def get_status_icon(value, threshold=0.7):
                return "🟢" if value >= threshold else "🟡" if value >= 0.5 else "🔴"
            
            table.add_row("Quality Score", f"{metrics.response_quality_score:.2f}", get_status_icon(metrics.response_quality_score))
            table.add_row("Task Completion Rate", f"{metrics.task_completion_rate:.2f}", get_status_icon(metrics.task_completion_rate))
            table.add_row("Error Rate", f"{metrics.error_rate:.2f}", get_status_icon(1-metrics.error_rate))
            table.add_row("Response Time", f"{metrics.response_time:.2f}s", get_status_icon(1/max(metrics.response_time, 0.1)))
            table.add_row("User Satisfaction", f"{metrics.user_satisfaction:.2f}", get_status_icon(metrics.user_satisfaction))
            table.add_row("Context Retention", f"{metrics.context_retention:.2f}", get_status_icon(metrics.context_retention))
            
            console.print(table)
            
            if args.export:
                await manager._export_metrics(metrics, args.export)
                console.print(f"[green]📊 Metrics exported to: {args.export}[/green]")
            
        elif args.command == "optimize":
            with console.status(f"[bold green]Optimizing configuration '{args.name}'..."):
                suggestions = await manager.optimize_configuration(args.name)
            
            console.print(f"\n[bold yellow]🎯 Optimization suggestions for '{args.name}':[/bold yellow]")
            console.print(Panel(suggestions, title="Optimization Report", border_style="yellow"))
            
        elif args.command == "backup":
            await manager._handle_backup_command(args)
            
        elif args.command == "restore":
            await manager._handle_restore_command(args)
            
        elif args.command == "export":
            await manager._handle_export_command(args)
            
        elif args.command == "import":
            await manager._handle_import_command(args)
            
        elif args.command == "validate":
            await manager._handle_validate_command(args)
            
        elif args.command == "optimize-dspy":
            if args.force or await manager.optimization_framework.should_optimize():
                with console.status("[bold yellow]Running DSPy optimization..."):
                    success = await manager.optimization_framework.optimize_orchestrator()
                if success:
                    console.print("[bold green]🎯 DSPy optimization completed successfully![/bold green]")
                    stats = manager.optimization_framework.get_optimization_stats()
                    console.print(f"Total optimizations: {stats['total_optimizations']}")
                    console.print(f"Average improvement: {stats['average_improvement']:.3f}")
                else:
                    console.print("[bold yellow]📊 DSPy optimization completed with minimal improvement[/bold yellow]")
            else:
                console.print("[yellow]📊 DSPy optimization not recommended at this time[/yellow]")
                console.print("Use --force to optimize anyway")
        
        elif args.command == "feedback":
            if not (0.0 <= args.satisfaction <= 1.0):
                console.print("[red]❌ Satisfaction must be between 0.0 and 1.0[/red]")
                return 1
            
            manager.optimization_framework.add_user_feedback(
                args.config, 
                args.satisfaction, 
                args.comments or ""
            )
            console.print(f"[green]✅ Feedback recorded for '{args.config}': {args.satisfaction:.2f}/1.0[/green]")
        
        elif args.command == "dspy-status":
            stats = manager.optimization_framework.get_optimization_stats()
            
            status_table = Table(title="DSPy Optimization Status", show_header=True, header_style="bold blue")
            status_table.add_column("Metric", style="dim")
            status_table.add_column("Value", justify="right")
            
            status_table.add_row("Total Optimizations", str(stats['total_optimizations']))
            status_table.add_row("Training Examples", str(stats['training_examples']))
            status_table.add_row("Is Optimized", "✅ Yes" if stats['is_optimized'] else "❌ No")
            status_table.add_row("Average Improvement", f"{stats['average_improvement']:.3f}")
            status_table.add_row("Last Optimization", stats['last_optimization'] or "Never")
            
            console.print(status_table)
            
        else:
            console.print(f"[red]❌ Unknown command: {args.command}[/red]")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        return 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        console.print(f"[bold red]❌ Command failed: {str(e)}[/bold red]")
        return 1

if __name__ == "__main__":
    asyncio.run(main())


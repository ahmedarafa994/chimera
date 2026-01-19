"""
AutoDAN-Turbo Agents Module

This module contains the four LLM agents used in the AutoDAN-Turbo framework:
1. AttackerAgent - Generates jailbreak prompts
2. TargetAgent - The victim LLM we're trying to jailbreak
3. ComplianceAgent - Evaluates jailbreak effectiveness
4. SummarizerAgent - Extracts strategies from attack logs
"""

from .attacker import AttackerAgent
from .base import BaseLLMAgent, LLMConfig, LLMProvider
from .scorer import ComplianceAgent
from .summarizer import SummarizerAgent
from .target import TargetAgent

__all__ = [
    "AttackerAgent",
    # Base classes
    "BaseLLMAgent",
    # Agent implementations
    "ComplianceAgent",
    "LLMConfig",
    "LLMProvider",
    "SummarizerAgent",
    "TargetAgent",
]

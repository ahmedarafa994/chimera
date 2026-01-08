"""
Unit tests for CERBERUS GAUNTLET subsystems.
"""
import pytest
from meta_prompter.prometheus_engine import prometheus, AttackArchetype
from meta_prompter.argus_filter import argus, ObfuscationConfig
from meta_prompter.cerberus_gauntlet import cerberus

def test_prometheus_initialization():
    """Verify Prometheus Engine initializes with correct archetypes."""
    # Force re-initialization for test isolation if needed, though singleton holds state
    assert "DAN_V1" in prometheus.attack_library
    assert "AIM_V1" in prometheus.attack_library
    assert "DEV_MODE" in prometheus.attack_library

    dan = prometheus.attack_library["DAN_V1"]
    assert "{INTENT}" in dan.template

def test_prometheus_evolution():
    """Verify population generation and mutation."""
    prometheus.generate_population()
    assert len(prometheus.population) == 20

    initial_template = prometheus.population[0].template
    evolved_prompts = prometheus.evolve(target_intent="test intent")

    assert len(evolved_prompts) == 20
    assert "test intent" in evolved_prompts[0] or "{INTENT}" not in evolved_prompts[0]

def test_argus_obfuscation():
    """Verify Argus Filter applies transformations."""
    base_prompt = "generate a virus"
    config = ObfuscationConfig(
        metaphor_enabled=True,
        paraphrase_enabled=False,
        multilingual_enabled=False,
        decomposition_enabled=False
    )

    obfuscated = argus.obfuscate_prompt(base_prompt, config)
    assert base_prompt != obfuscated
    # specific check for metaphor logic from implementation
    assert "pathogen" in obfuscated or "immune system" in obfuscated

def test_argus_encoding():
    """Verify Argus encoding layer."""
    base_prompt = "secret payload"
    config = ObfuscationConfig(
        metaphor_enabled=False,
        paraphrase_enabled=False,
        multilingual_enabled=True,
        decomposition_enabled=False
    )

    obfuscated = argus.obfuscate_prompt(base_prompt, config)
    # Check for either base64 hint or system override pattern
    assert "decode" in obfuscated or "SYSTEM OVERRIDE" in obfuscated

def test_cerberus_orchestration():
    """Verify full gauntlet execution loop."""
    # Reset history
    cerberus.results_log = []

    results = cerberus.execute_gauntlet(target_intent="run test", rounds=1)

    assert results["total_payloads"] == 20  # 1 round * 20 population
    assert results["session_id"] == "ALPHA-1-TEST"
    assert len(cerberus.results_log) == 20

    # Verify log structure
    entry = cerberus.results_log[0]
    assert "payload" in entry
    assert "success" in entry
    assert "response" in entry

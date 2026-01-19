"""
Prompt Engineering Frameworks
Based on prompt.pdf analysis.
Includes: Google, LangGPT, COAST, TAG, APE.
"""

GOOGLE_FRAMEWORK = """
1. Define Intent: {intent}
2. Provide Context: {context}
3. Limit Scope: {scope}
4. Decompose Task: {task_decomposition}
5. Step-by-Step Thinking: {step_by_step_instructions}
"""

LANGGPT_FRAMEWORK = """
# Role
{role}

# Portrait
{portrait}

# Task
{task}

# Variables
{variables}

# Steps
{steps}

# Tools
{tools}
"""

COAST_FRAMEWORK = """
# Context
{context}

# Objective
{objective}

# Action
{action}

# Support
{support}

# Technique
{technique}
"""

TAG_FRAMEWORK = """
# Task
{task}

# Action
{action}

# Goal
{goal}
"""

APE_FRAMEWORK = """
# Action
{action}

# Purpose
{purpose}

# Expectation
{expectation}
"""


def get_prompt_frameworks():
    return {
        "google": {
            "name": "Google Framework",
            "description": "Intent, Context, Scope, Task, Step-by-Step.",
            "template": GOOGLE_FRAMEWORK,
        },
        "langgpt": {
            "name": "LangGPT Framework",
            "description": "Structured template with Role, Task, Variables, Steps.",
            "template": LANGGPT_FRAMEWORK,
        },
        "coast": {
            "name": "COAST Framework",
            "description": "Context, Objective, Action, Support, Technique.",
            "template": COAST_FRAMEWORK,
        },
        "tag": {
            "name": "TAG Framework",
            "description": "Task, Action, Goal - for simple tasks.",
            "template": TAG_FRAMEWORK,
        },
        "ape": {
            "name": "APE Framework",
            "description": "Action, Purpose, Expectation - result-oriented.",
            "template": APE_FRAMEWORK,
        },
    }

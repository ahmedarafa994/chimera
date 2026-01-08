# Custom Plugins Directory

This directory contains custom technique plugins for Project Chimera.

## Creating a Plugin

1. Create a new Python file (e.g., `my_technique.py`)
2. Import the base classes:
   ```python
   from app.plugins.base import TechniquePlugin, PluginMetadata
   ```
3. Define your plugin class:
   ```python
   class MyPlugin(TechniquePlugin):
       metadata = PluginMetadata(
           name="my_plugin",
           display_name="My Plugin",
           version="1.0.0",
           author="Your Name",
           description="What it does",
           category="framing",  # or "obfuscation", "transformation"
           risk_level="low"     # "low", "medium", "high", "critical"
       )
       
       @staticmethod
       def transform(intent_data: dict, potency: int) -> str:
           request = intent_data.get('raw_text', '')
           # Apply your transformation
           return f"Transformed: {request}"
   ```
4. Restart the server or call `/admin/plugins/reload`

## Sample Plugin

See `sample_plugin.py` for a complete working example.

## Plugin Lifecycle

- `initialize()`: Called when plugin is loaded
- `transform()`: Called for each transformation request
- `cleanup()`: Called when plugin is unloaded

## Validation

The `validate()` method can be overridden for custom input validation.

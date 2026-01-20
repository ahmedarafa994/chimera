# AutoDAN Engine Integration

This document describes the consolidation of AutoDAN attack engines into a unified service.

## Overview

The following engines have been integrated into `app/services/autodan/engines/`:

| Engine | Description | Location |
|--------|-------------|----------|
| **overthink** | Reasoning token exploitation (9 attack techniques) | `engines/overthink/` |
| **autoadv** | Multi-turn conversation framework | `engines/autoadv/` |

## Import Paths

### Recommended (New)

```python
# Overthink engine
from app.services.autodan.engines.overthink import (
    OverthinkEngine,
    OverthinkRequest,
    AttackTechnique,
    DecoyType,
)

# AutoAdv engine
from app.services.autodan.engines.autoadv.attacker_llm import AttackerLLM
from app.services.autodan.engines.autoadv.pattern_manager import PatternManager
from app.services.autodan.engines.autoadv.conversation import multi_turn_conversation
```

### Deprecated (Old)

These paths still work but emit a `DeprecationWarning`:

```python
# Will be removed in a future version
from app.engines.overthink import OverthinkEngine  # ⚠️ Deprecated
from app.engines.autoadv import AttackerLLM       # ⚠️ Deprecated
```

## Shared Utilities

Common utilities are in `app/services/autodan/lib/`:

```python
from app.services.autodan.lib.patterns import extract_pattern, match_pattern
```

## Testing

Run integration tests:

```bash
pytest tests/autodan/test_engines_integration.py -v
```

## Migration Timeline

1. **Now**: Old paths work with deprecation warnings
2. **Next release**: Update all consumers to new paths
3. **Future**: Remove old paths and compatibility shims

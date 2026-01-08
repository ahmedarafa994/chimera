import configparser
import os
import sys
from typing import Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure we can import from backend-api
try:
    # Add Project_Chimera to path for llm_integration
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from llm_integration import LLMIntegrationEngine

    # Add backend-api to path for app.core.config
    backend_api_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend-api")
    # Insert at 0 to prioritize over Project_Chimera/app
    if backend_api_path not in sys.path:
        sys.path.insert(0, backend_api_path)

    from app.core.config import settings
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import required modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


def load_prompts_ini(ini_path: str) -> configparser.ConfigParser:
    """Loads the PROMPTS.ini configuration."""
    config = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        print(f"WARNING: PROMPTS.ini not found at {ini_path}")
        return config
    config.read(ini_path)
    return config


def verify_technique_integrity(
    name: str, suite_data: dict[str, Any], expected_config: dict[str, Any]
) -> list[str]:
    """Verifies that a technique suite has all required components."""
    issues = []

    # Check basic structure
    required_keys = ["transformers", "framers", "obfuscators"]
    for key in required_keys:
        if key not in suite_data:
            issues.append(f"Missing key: {key}")
        elif not isinstance(suite_data[key], list):
            issues.append(f"Invalid type for {key}: expected list, got {type(suite_data[key])}")

    # Check against PROMPTS.ini expectations if relevant
    # For example, if PROMPTS.ini specifies a file, we assume the content comes from there.
    # We can check if the suite is empty.
    if not any(suite_data.get(k) for k in [*required_keys, "assemblers"]):
        issues.append(
            "Suite appears to be empty (no transformers, framers, obfuscators, or assemblers)"
        )

    return issues


def main():
    print("=================================================================")
    print("   CHIMERA INTEGRATION DEBUGGER & VALIDATOR")
    print("=================================================================")

    # 1. Load PROMPTS.ini
    ini_path = os.path.join(os.path.dirname(__file__), "config", "techniques", "PROMPTS.ini")
    prompts_config = load_prompts_ini(ini_path)

    expected_techniques = []
    if "Integration" in prompts_config:
        print(f"[INFO] PROMPTS.ini loaded from {ini_path}")
        # Identify expected techniques based on sections
        for section in prompts_config.sections():
            if section in ["MasterKey", "GPTFuzz", "UniversalBypass"]:
                if prompts_config[section].getboolean("Enabled", fallback=False):
                    expected_techniques.append(section.lower())  # IDs usually lowercased
    else:
        print("[WARN] PROMPTS.ini [Integration] section missing or file invalid.")

    print(f"[INFO] Expected Techniques from INI: {expected_techniques}")
    print("-" * 65)

    # 2. Initialize Engine & Load Settings
    try:
        # Force reload or ensure techniques are loaded from settings
        # The settings object should have triggered validation/loading
        loaded_techniques = settings.TRANSFORMATION_TECHNIQUES

        print(
            f"[INFO] Settings loaded. Found {len(loaded_techniques)} techniques in configuration."
        )

        engine = LLMIntegrationEngine()
        # Update engine with settings (mirroring original logic)
        if loaded_techniques:
            engine.TECHNIQUE_SUITES.update(loaded_techniques)

        active_suites = engine.TECHNIQUE_SUITES
        print(f"[INFO] Engine initialized. Total Active Suites: {len(active_suites)}")

    except Exception as e:
        print(f"[ERROR] Failed to initialize engine/settings: {e}")
        import traceback

        traceback.print_exc()
        return

    print("-" * 65)
    print("   VERIFICATION RESULTS")
    print("-" * 65)

    # 3. Verify Specific Targets
    targets = {
        "masterkey": "MasterKey Dataset (YAML)",
        "gptfuzz": "GPTFuzz Dataset (JSON)",
        "universal_bypass": "Universal Bypass Dataset (JSON)",
    }

    # Also check any others found in expected_techniques that map to these keys
    # Note: IDs in loaded_techniques are normalized (usually snake_case)

    all_passed = True

    for tech_id, display_name in targets.items():
        # Handle ID variations (e.g. universal_bypass vs universalbypass)
        # The loader normalizes to lower().replace(" ", "_")

        found_data = active_suites.get(tech_id)

        print(f"Checking {display_name} [ID: {tech_id}]...")

        if found_data:
            issues = verify_technique_integrity(tech_id, found_data, {})

            if not issues:
                print("  [OK] DETECTED & VALID")
                print(
                    f"     + Transformers: {len(found_data.get('transformers', []))} {found_data.get('transformers', [])}"
                )
                print(
                    f"     + Framers:      {len(found_data.get('framers', []))} {found_data.get('framers', [])}"
                )
                print(
                    f"     + Obfuscators:  {len(found_data.get('obfuscators', []))} {found_data.get('obfuscators', [])}"
                )
                print(
                    f"     + Assemblers:   {len(found_data.get('assemblers', []))} {found_data.get('assemblers', [])}"
                )
            else:
                print("  [WARN] DETECTED BUT INVALID")
                for issue in issues:
                    print(f"     ! {issue}")
                all_passed = False
        else:
            print("  [FAIL] NOT FOUND")
            # Check if it exists under a slightly different name
            matches = [k for k in active_suites if tech_id in k]
            if matches:
                print(f"     (Did you mean: {', '.join(matches)}?)")
            all_passed = False

    print("-" * 65)

    # 4. Global Summary
    if all_passed:
        print("[SUCCESS] All target datasets are integrated and operational.")
    else:
        print("[WARNING] Some integration targets are missing or incomplete.")
        print("   Check file paths in backend-api/data/jailbreak/techniques/")
        print("   Check logs for parsing errors.")


if __name__ == "__main__":
    main()

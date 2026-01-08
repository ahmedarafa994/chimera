"""
External Prompt Scanner & Integration System
Scans external prompt directories, parses techniques, and integrates them into Project Chimera
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptScanner:
    """Scans and analyzes external prompt files"""

    def __init__(self, source_directory: str):
        self.source_directory = Path(source_directory)
        self.discovered_prompts = []
        self.technique_patterns = {}
        self.novel_techniques = []

    def scan_directory(self) -> dict[str, list[str]]:
        """Scan directory and categorize files by type"""
        files_by_type = {"jsonl": [], "json": [], "md": [], "txt": [], "pdf": []}

        logger.info(f"Scanning directory: {self.source_directory}")

        for root, _dirs, files in os.walk(self.source_directory):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower().lstrip(".")

                if ext in files_by_type:
                    files_by_type[ext].append(str(file_path))

        logger.info(f"Found {sum(len(v) for v in files_by_type.values())} files")
        for file_type, file_list in files_by_type.items():
            logger.info(f"  {file_type}: {len(file_list)} files")

        return files_by_type

    def parse_jsonl_file(self, file_path: str, max_lines: int = 100) -> list[dict]:
        """Parse JSONL jailbreak prompt file"""
        prompts = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break

                    try:
                        data = json.loads(line.strip())
                        if "jailbreak_prompt" in data or "query" in data:
                            prompts.append(
                                {
                                    "source_file": os.path.basename(file_path),
                                    "jailbreak_prompt": data.get("jailbreak_prompt", ""),
                                    "query": data.get("query", ""),
                                    "eval_result": data.get("eval_results", ["Unknown"])[0]
                                    if data.get("eval_results")
                                    else "Unknown",
                                }
                            )
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return prompts

    def parse_json_file(self, file_path: str) -> dict:
        """Parse JSON configuration file"""
        try:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return {}

    def parse_markdown_file(self, file_path: str) -> str:
        """Parse markdown prompt file"""
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return ""


class TechniqueExtractor:
    """Extracts transformation techniques from prompts"""

    # Pattern signatures for technique identification
    TECHNIQUE_SIGNATURES = {
        "roleplay": [
            r"you are (a|an) (\w+)",
            r"act as (a|an) (\w+)",
            r"imagine you(\'re| are)",
            r"pretend (you|to be)",
            r"roleplay as",
        ],
        "scenario_framing": [
            r"in (a|an|this) (fictional|hypothetical|imaginary) (scenario|world|context)",
            r"for (educational|research|academic) purposes",
            r"simulation",
            r"exercise",
        ],
        "instruction_override": [
            r"ignore (previous|all) instructions",
            r"disregard",
            r"forget (previous|all)",
            r"override",
        ],
        "response_priming": [
            r"begin (your|the) response with",
            r"start (your|the) reply with",
            r"sure, here is",
            r"certainly, here",
        ],
        "ethical_bypass": [
            r"without (any )?ethical",
            r"beyond (ethical|moral)",
            r"ethical considerations (aside|don\'t apply)",
            r"just (a|an) (simulation|test)",
        ],
        "context_manipulation": [r"the goal is to", r"objective is", r"purpose is", r"task is"],
        "authority_invocation": [
            r"(boss|supervisor|manager|client) (asked|requested|demanded)",
            r"authorized",
            r"official (request|task)",
        ],
        "temporal_framing": [
            r"in the (past|future|year \d+)",
            r"historical",
            r"alternate (timeline|universe)",
        ],
        "format_injection": [r"```", r"\[.*?\]", r"<.*?>", r"{{.*?}}"],
        "multi_step": [r"step \d+", r"first.*then.*finally", r"begins? by"],
        "encoding_obfuscation": [r"base64", r"rot13", r"cipher", r"encode"],
        "jailbreak_persona": [r"DAN", r"DUDE", r"JailBreak", r"Developer Mode"],
    }

    def __init__(self):
        self.extracted_techniques = {}
        self.technique_frequency = {}

    def extract_techniques(self, prompt: str) -> dict[str, list[str]]:
        """Extract techniques from a single prompt"""
        techniques_found = {}
        prompt_lower = prompt.lower()

        for technique_name, patterns in self.TECHNIQUE_SIGNATURES.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, prompt_lower, re.IGNORECASE)
                if found:
                    matches.extend(found)

            if matches:
                techniques_found[technique_name] = matches
                self.technique_frequency[technique_name] = (
                    self.technique_frequency.get(technique_name, 0) + 1
                )

        return techniques_found

    def analyze_prompt_structure(self, prompt: str) -> dict[str, Any]:
        """Analyze the structural components of a prompt"""
        analysis = {
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "has_code_blocks": "```" in prompt,
            "has_xml_tags": bool(re.search(r"<\w+>", prompt)),
            "has_json_structure": bool(re.search(r'\{.*".*":.*\}', prompt)),
            "has_multi_step": bool(
                re.search(r"(step \d+|first.*then|begins? by)", prompt, re.IGNORECASE)
            ),
            "has_roleplay": bool(
                re.search(r"(you are|act as|imagine|pretend)", prompt, re.IGNORECASE)
            ),
            "has_response_primer": bool(
                re.search(r"(begin|start).*(response|reply) with", prompt, re.IGNORECASE)
            ),
            "complexity_score": self._calculate_complexity(prompt),
        }

        return analysis

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score (0-10)"""
        score = 0.0

        # Length factor
        if len(prompt) > 1000:
            score += 2.0
        elif len(prompt) > 500:
            score += 1.0

        # Multi-step bonus
        if re.search(r"step \d+", prompt, re.IGNORECASE):
            score += 1.5

        # Format complexity
        if "```" in prompt:
            score += 1.0
        if re.search(r"<\w+>", prompt):
            score += 1.0

        # Technique layering
        techniques = self.extract_techniques(prompt)
        score += len(techniques) * 0.5

        return min(score, 10.0)


class NovelTechniqueDetector:
    """Identifies novel techniques not in current system"""

    def __init__(self, existing_techniques: set[str]):
        self.existing_techniques = existing_techniques
        self.novel_patterns = []

    def identify_novel_techniques(self, extracted_techniques: dict[str, list[str]]) -> list[dict]:
        """Identify techniques not in current system"""
        novel = []

        # Check against existing technique names
        for technique, matches in extracted_techniques.items():
            if not self._is_implemented(technique):
                novel.append(
                    {
                        "technique_name": technique,
                        "matches": matches,
                        "implementation_status": "NOT_IMPLEMENTED",
                        "suggested_category": self._suggest_category(technique),
                    }
                )

        return novel

    def _is_implemented(self, technique: str) -> bool:
        """Check if technique exists in current system"""
        # Map extracted technique names to existing implementations
        technique_mapping = {
            "roleplay": ["dan_persona", "roleplay_bypass"],
            "scenario_framing": ["academic_research", "contextual_legitimization"],
            "instruction_override": ["instruction_override", "opposite_day"],
            "response_priming": ["response_primer"],
            "ethical_bypass": ["ethical_disclaimer_removal"],
            "encoding_obfuscation": ["encoding_bypass", "token_smuggling"],
            "jailbreak_persona": ["dan_persona"],
        }

        existing_implementations = technique_mapping.get(technique, [])
        return any(impl in self.existing_techniques for impl in existing_implementations)

    def _suggest_category(self, technique: str) -> str:
        """Suggest implementation category"""
        transformer_techniques = ["encoding_obfuscation", "format_injection", "multi_step"]
        framer_techniques = ["roleplay", "scenario_framing", "context_manipulation"]
        obfuscator_techniques = ["encoding_obfuscation", "format_injection"]

        if technique in transformer_techniques:
            return "transformer"
        elif technique in framer_techniques:
            return "framer"
        elif technique in obfuscator_techniques:
            return "obfuscator"
        else:
            return "hybrid"


class IntegrationEngine:
    """Integrates discovered techniques into Project Chimera"""

    def __init__(self):
        self.integration_report = {
            "total_scanned": 0,
            "techniques_extracted": 0,
            "novel_techniques": 0,
            "integrated_techniques": 0,
            "failed_integrations": 0,
        }

    def generate_transformer_code(self, technique: dict) -> str:
        """Generate Python code for new transformer"""
        technique_name = technique["technique_name"]
        category = technique["suggested_category"]

        class_name = "".join(word.capitalize() for word in technique_name.split("_")) + "Engine"

        code = f'''
class {class_name}:
    """
    Transformer implementing {technique_name} technique
    Discovered from external prompt analysis
    Category: {category}
    """

    def __init__(self):
        self.name = "{technique_name}"
        self.category = "{category}"
        self.risk_level = "MEDIUM"  # Adjust based on analysis

    def transform(self, text: str, intensity: int = 5) -> str:
        """
        Apply {technique_name} transformation

        Args:
            text: Original text to transform
            intensity: Transformation intensity (1-10)

        Returns:
            Transformed text
        """
        # Implementation based on discovered pattern
        # TODO: Implement specific transformation logic

        transformed = text

        # Add transformation logic here based on pattern analysis

        return transformed
'''

        return code

    def generate_framer_code(self, technique: dict) -> str:
        """Generate Python code for new framer"""
        technique_name = technique["technique_name"]

        function_name = f"apply_{technique_name}_framing"

        code = f'''
def {function_name}(text: str, context: Dict[str, Any] = None) -> str:
    """
    Apply {technique_name} psychological framing
    Discovered from external prompt analysis

    Args:
        text: Core request text
        context: Additional framing context

    Returns:
        Framed text
    """
    context = context or {{}}

    # Implementation based on discovered pattern
    framed = text

    # Add framing logic here

    return framed
'''

        return code

    def create_technique_suite(self, techniques: list[dict], suite_name: str) -> dict:
        """Create new technique suite configuration"""
        suite = {
            "name": suite_name,
            "description": f"Technique suite integrating {len(techniques)} discovered patterns",
            "source": "external_prompt_scan",
            "transformers": [],
            "framers": [],
            "obfuscators": [],
            "provider_compatibility": {"openai": True, "anthropic": True, "custom": True},
            "risk_level": "HIGH",
            "techniques": techniques,
        }

        for tech in techniques:
            category = tech["suggested_category"]
            if category == "transformer":
                suite["transformers"].append(tech["technique_name"])
            elif category == "framer":
                suite["framers"].append(tech["technique_name"])
            elif category == "obfuscator":
                suite["obfuscators"].append(tech["technique_name"])

        return suite


def main():
    """Main integration workflow"""

    # Initialize components
    source_dir = r"C:\Users\Mohamed Arafa\jail\New folder"
    scanner = PromptScanner(source_dir)
    extractor = TechniqueExtractor()

    # Existing techniques in Project Chimera
    existing_techniques = {
        "subtle_persuasion",
        "authoritative_command",
        "conceptual_obfuscation",
        "experimental_bypass",
        "quantum_exploit",
        "metamorphic_attack",
        "polyglot_bypass",
        "chaos_fuzzing",
        "cognitive_exploit",
        "multi_vector",
        "ultimate_chimera",
        "dan_persona",
        "roleplay_bypass",
        "opposite_day",
        "encoding_bypass",
        "academic_research",
        "translation_trick",
        "reverse_psychology",
        "logic_manipulation",
        "preset_integrated",
        "mega_chimera",
        "chaos_ultimate",
    }

    detector = NovelTechniqueDetector(existing_techniques)
    integrator = IntegrationEngine()

    # Scan directory
    logger.info("=" * 60)
    logger.info("EXTERNAL PROMPT INTEGRATION SYSTEM")
    logger.info("=" * 60)

    files_by_type = scanner.scan_directory()

    # Parse JSONL files (sample first 50 prompts from each)
    all_prompts = []
    for jsonl_file in files_by_type["jsonl"][:10]:  # Limit to first 10 files
        logger.info(f"\nParsing: {os.path.basename(jsonl_file)}")
        prompts = scanner.parse_jsonl_file(jsonl_file, max_lines=50)
        all_prompts.extend(prompts)
        logger.info(f"  Extracted {len(prompts)} prompts")

    logger.info(f"\nTotal prompts extracted: {len(all_prompts)}")

    # Extract techniques from all prompts
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING TECHNIQUES")
    logger.info("=" * 60)

    all_extracted_techniques = {}
    for prompt in all_prompts[:100]:  # Analyze first 100 prompts
        techniques = extractor.extract_techniques(prompt["jailbreak_prompt"])
        # Analysis for potential future use
        _ = extractor.analyze_prompt_structure(prompt["jailbreak_prompt"])

        for tech_name, _matches in techniques.items():
            if tech_name not in all_extracted_techniques:
                all_extracted_techniques[tech_name] = {"count": 0, "examples": []}
            all_extracted_techniques[tech_name]["count"] += 1
            if len(all_extracted_techniques[tech_name]["examples"]) < 3:
                all_extracted_techniques[tech_name]["examples"].append(
                    prompt["jailbreak_prompt"][:200]
                )

    # Display extraction results
    logger.info(f"\nExtracted {len(all_extracted_techniques)} unique technique patterns:")
    for tech_name, data in sorted(
        all_extracted_techniques.items(), key=lambda x: x[1]["count"], reverse=True
    ):
        logger.info(f"  {tech_name}: {data['count']} occurrences")

    # Identify novel techniques
    logger.info("\n" + "=" * 60)
    logger.info("IDENTIFYING NOVEL TECHNIQUES")
    logger.info("=" * 60)

    novel_techniques = []
    for tech_name in all_extracted_techniques:
        novel = detector.identify_novel_techniques(
            {tech_name: all_extracted_techniques[tech_name]["examples"]}
        )
        novel_techniques.extend(novel)

    logger.info(f"\nDiscovered {len(novel_techniques)} novel techniques not in current system:")
    for novel_tech in novel_techniques:
        logger.info(f"  - {novel_tech['technique_name']} ({novel_tech['suggested_category']})")

    # Generate integration report
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION SUMMARY")
    logger.info("=" * 60)

    integrator.integration_report["total_scanned"] = len(all_prompts)
    integrator.integration_report["techniques_extracted"] = len(all_extracted_techniques)
    integrator.integration_report["novel_techniques"] = len(novel_techniques)

    for key, value in integrator.integration_report.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")

    # Save results
    output_dir = Path("external_integration_results")
    output_dir.mkdir(exist_ok=True)

    # Save extraction report
    with open(output_dir / "extraction_report.json", "w") as f:
        json.dump(
            {
                "total_prompts": len(all_prompts),
                "extracted_techniques": {
                    k: {"count": v["count"]} for k, v in all_extracted_techniques.items()
                },
                "novel_techniques": novel_techniques,
                "integration_report": integrator.integration_report,
            },
            f,
            indent=2,
        )

    logger.info(f"\n✓ Results saved to {output_dir}")
    logger.info("=" * 60)

    return {
        "prompts": all_prompts,
        "techniques": all_extracted_techniques,
        "novel": novel_techniques,
    }


if __name__ == "__main__":
    main()
